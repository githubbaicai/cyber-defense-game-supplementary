"""Serial, bounded subprocess execution; no optimizer or plotting dependency."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

from features.scalability.config import ROOT, THREAD_VARIABLES, load_config, make_schedule


def environment_metadata():
    processor = platform.processor()
    physical_memory = None
    if sys.platform == "win32":
        import ctypes
        import winreg

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                        ("avail_extended", ctypes.c_ulonglong)]

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            processor = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            physical_memory = status.total_phys
    return {
        "python_executable": sys.executable, "python_version": sys.version,
        "os": platform.platform(), "processor": processor,
        "logical_cpu_count": os.cpu_count(), "physical_memory_bytes": physical_memory,
        "libraries": {name: importlib.metadata.version(name)
                      for name in ("numpy", "scipy", "pandas", "pymoo", "openpyxl", "networkx")},
        "worker_threads": {name: "1" for name in THREAD_VARIABLES},
        "processes": 1, "peak_memory_measured": False,
    }


def code_identity():
    paths = [ROOT / "run.py"]
    for folder in ("common", "external", "features/scalability"):
        paths.extend(sorted((ROOT / folder).glob("*.py")))
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path.read_bytes()).hexdigest()
              for path in paths}
    return {"files": hashes, "aggregate_sha256": sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def run_child(task, settings, directory, timeout):
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / f"{task['Run_Key']}.request.json"
    response_path = directory / f"{task['Run_Key']}.response.json"
    request_path.write_text(json.dumps({"task": task, "settings": settings}, indent=2), encoding="utf-8")
    environment = os.environ.copy()
    environment.update({name: "1" for name in THREAD_VARIABLES})
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["MPLBACKEND"] = "Agg"
    started = perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "features.scalability.worker", str(request_path), str(response_path)],
            cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8",
            timeout=timeout,
        )
        (directory / f"{task['Run_Key']}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode != 0 or not response_path.is_file():
            row = {**task, "Status": "error", "Error": f"worker exit {completed.returncode}: {completed.stderr[-2000:]}"}
        else:
            row = json.loads(response_path.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired:
        row = {**task, "Status": "timeout", "Error": f"Child exceeded {timeout:g} seconds"}
    row["Process_Seconds"] = perf_counter() - started
    row["Applied_Timeout_Seconds"] = timeout
    return row


def solve(scalability_config=None):
    settings = load_config(scalability_config)
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output_dir = ROOT / "results" / "scalability" / experiment_id
    output_dir.mkdir(parents=True, exist_ok=False)
    work_dir = output_dir / "worker_records"
    work_dir.mkdir()
    from common.implementation import IMPLEMENTATION_VERSION, MODEL_PROTOCOL_VERSION
    manifest = {
        "experiment_id": experiment_id, "started_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings, "environment": environment_metadata(), "code": code_identity(),
        "input_hashes": {filename: sha256(Path(filename).read_bytes()).hexdigest() for filename in settings["cases"]},
        "implementation_version": IMPLEMENTATION_VERSION, "model_protocol_version": MODEL_PROTOCOL_VERSION,
        "propagation_rule": "parent action-selection mass times conditional component action mass; no success multiplier",
        "settlement_rule": "population-average detection, shared induced three-state distribution",
        "setup_boundary": "Excel loading/validation, full component-path enumeration, prefix construction, budget read, decision and problem construction; excludes process/import overhead",
        "evaluation_boundary": "one CyberDefenseProblem._evaluate call, including decision normalization, configuration application, local games, propagation, loss/cost/constraint calculation and cold cache insertion; includes one passive wrapper call retaining simulation return",
        "statistics_boundary": "workload extraction occurs after evaluation clock stops; existing production convergence diagnostics are unchanged",
        "process_boundary": "fresh process startup/imports, setup, evaluation, workload extraction and response serialization",
        "repetition_interpretation": "deterministic evaluation timing repeats, not independent threat samples or optimization runs",
        "warmup": "No extra warmup; preflight runs are separately identified and excluded. Every measured repeat starts a fresh process with cold objective cache; OS filesystem cache is uncontrolled.",
        "run_order_rule": "repeat, portfolio, mode, case; reverse cases and modes on even repeats",
        "cache_rule": "new problem per fresh subprocess; one objective submission, one cache entry, zero hits",
        "money_unit": "USD thousands",
        "search_performed": False,
    }
    (output_dir / "frozen_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    started = perf_counter()
    for task in make_schedule(settings):
        remaining = settings["total_timeout_seconds"] - (perf_counter() - started)
        if remaining <= 0:
            row = {**task, "Status": "total_timeout_not_run", "Error": "Experiment wall-time limit exhausted"}
        else:
            row = run_child(task, settings, work_dir, min(remaining, settings["per_run_timeout_seconds"]))
        rows.append(row)
        print(f"{task['Order']:02d}: {task['Run_Key']} {row['Status']} evaluation={row.get('Evaluation_Seconds')} s", flush=True)
    manifest["elapsed_process_loop_seconds"] = perf_counter() - started
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["code_after"] = code_identity()
    manifest["input_hashes_after"] = {filename: sha256(Path(filename).read_bytes()).hexdigest() for filename in settings["cases"]}
    if manifest["code_after"] != manifest["code"] or manifest["input_hashes_after"] != manifest["input_hashes"]:
        raise RuntimeError(f"Code or input changed during run; preserve and inspect {output_dir}")
    return {"output_dir": str(output_dir), "manifest": manifest, "rows": rows}


def run_and_export(scalability_config=None):
    from features.scalability.export_csv import export
    result = solve(scalability_config)
    output_dir = Path(result["output_dir"])
    (output_dir / "solve_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    export(result)
    print(f"Scalability results: {output_dir}", flush=True)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scalability-config", type=Path)
    arguments = parser.parse_args()
    run_and_export(arguments.scalability_config)


if __name__ == "__main__":
    main()
