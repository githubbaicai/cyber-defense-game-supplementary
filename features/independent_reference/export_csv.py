

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import get_feature_output_dir
from common.optimization_fronts import EmpiricalReference


FROZEN_TABLES = ("reference_front", "reference_candidates", "reference_runs", "reference_history")


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def freeze(result):
    directory = Path(result["output_dir"])
    manifest_path = directory / "frozen_reference.json"
    if manifest_path.exists():
        reference, manifest = load_frozen(directory)
        if reference.reference_id != result["reference"].reference_id or manifest["provenance"] != result["provenance"]:
            raise ValueError("The existing frozen reference differs from the current results and cannot be overwritten")
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in FROZEN_TABLES:
        path = directory / f"{name}.csv"
        result["tables"][name].to_csv(path, index=False, encoding="utf-8-sig")
        hashes[path.name] = file_hash(path)
    reference = result["reference"]
    manifest = {
        "schema_version": 1,
        "reference_type": "independent_high_budget_approximation",
        "reference_id": reference.reference_id,
        "ideal": reference.ideal.tolist(), "span": reference.span.tolist(),
        "hv_reference_raw": reference.hv_reference_raw.tolist(),
        "status": reference.status, "provenance": result["provenance"],
        "files_sha256": hashes,
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(_json_text(manifest))
    return directory


def load_frozen(directory):
    directory = Path(directory)
    with (directory / "frozen_reference.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1 or manifest.get("reference_type") != "independent_high_budget_approximation":
        raise ValueError("Unsupported frozen-reference format")
    expected_files = {f"{name}.csv" for name in FROZEN_TABLES}
    if set(manifest["files_sha256"]) != expected_files:
        raise ValueError("The frozen-file list is incomplete")
    for name, expected in manifest["files_sha256"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"A frozen reference file has changed: {name}")
    frame = pd.read_csv(directory / "reference_front.csv", float_precision="round_trip")
    front = frame[["Total_Cost", "Total_Defender_Loss"]].to_numpy(float)
    ideal, span, hv_reference = (np.asarray(manifest[name], dtype=float) for name in ("ideal", "span", "hv_reference_raw"))
    if not len(front) or not np.isfinite(front).all() or not all(values.shape == (2,) and np.isfinite(values).all() for values in (ideal, span, hv_reference)) or np.any(span <= 0):
        raise ValueError("Invalid frozen-reference objectives or normalization parameters")
    data = np.concatenate((front.ravel(), ideal, span, hv_reference)).astype("<f8")
    if sha256(data.tobytes()).hexdigest() != manifest["reference_id"]:
        raise ValueError("Frozen-reference ID verification failed")
    for values in (front, ideal, span, hv_reference):
        values.setflags(write=False)
    reference = EmpiricalReference(front, ideal, span, hv_reference, manifest["status"], manifest["reference_id"])
    return reference, manifest


def export(result):
    directory = freeze(result)
    for name, frame in result["tables"].items():
        if name not in FROZEN_TABLES:
            frame.to_csv(directory / f"{name}.csv", index=False, encoding="utf-8-sig")
    metadata = {"output_dir": str(directory), "reference_id": result["reference"].reference_id,
                "timing": result["timing"], "source_dir": result["provenance"]["source_dir"],
                "experiment_role": result["provenance"].get("experiment_role", "multi_seed_reference")}
    (directory / "evaluation_manifest.json").write_text(_json_text(metadata), encoding="utf-8")
    root = get_feature_output_dir("independent_reference", result["input_excel"])
    root.mkdir(parents=True, exist_ok=True)
    pointer = "latest_pilot.json" if metadata["experiment_role"] == "pilot" else "latest.json"
    (root / pointer).write_text(_json_text({"output_dir": str(directory)}), encoding="utf-8")
    print(f"Independent reference and reevaluation results: {directory}")
    return directory


if __name__ == "__main__":
    from common.pipeline import run_export
    run_export("independent_reference", export)
