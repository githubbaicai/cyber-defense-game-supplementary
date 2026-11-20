"""Read immutable comparison CSVs and verify their objective compatibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.implementation import IMPLEMENTATION_VERSION, MODEL_PROTOCOL_VERSION
from common.problem import CyberDefenseProblem
from common.termination import FEASIBILITY_TOL


RUN_KEY = ["Experiment_ID", "Protocol", "Budget_ID", "Algorithm", "Run", "Seed"]
GROUP_KEY = RUN_KEY[:3]
OBJECTIVES = ["Total_Cost", "Total_Defender_Loss"]
SOURCE_FILES = ("experiment_manifest.csv", "performance_metrics_raw.csv", "all_pareto_points.csv")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{name} is empty")


def _single(frame, column, name):
    values = frame[column].drop_duplicates()
    if len(values) != 1 or pd.isna(values.iloc[0]) or str(values.iloc[0]).strip() == "":
        raise ValueError(f"{name} requires one nonempty {column}")
    return values.iloc[0]


def _numeric(frame, columns, name, *, integer=False):
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} has invalid {column}") from error
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} has nonfinite {column}")
        if integer and (values != np.floor(values)).any():
            raise ValueError(f"{name} has noninteger {column}")
        frame[column] = values.astype("int64") if integer else values.astype(float)


def _valid(frame, name):
    _require(frame, ["Status", "Comparison_Valid", "Protocol_Completed"], name)
    if not frame["Status"].eq("ok").all():
        raise ValueError(f"{name} includes unsuccessful Status")
    for column in ("Comparison_Valid", "Protocol_Completed"):
        if not frame[column].astype(str).str.lower().eq("true").all():
            raise ValueError(f"{name} includes false {column}")


def _json(value, name, expected_type):
    try:
        decoded = json.loads(value)
        json.dumps(decoded, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}") from error
    if not isinstance(decoded, expected_type):
        raise ValueError(f"invalid {name}")
    return decoded


def _decision(value):
    decoded = _json(value, "X_JSON", list)
    if len(decoded) < 2 or any(isinstance(item, (bool, str)) for item in decoded):
        raise ValueError("X_JSON must retain binary genes and continuous p_response")
    try:
        decision = np.asarray(decoded, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("X_JSON must be numeric") from error
    if decision.ndim != 1 or not np.isfinite(decision).all():
        raise ValueError("X_JSON must contain a finite decision vector")
    if not np.isin(decision[:-1], [0.0, 1.0]).all() or not 0 <= decision[-1] <= 1:
        raise ValueError("X_JSON has invalid binary genes or p_response")
    return decision


def load_source(source_dir, input_excel):
    """Validate a complete upstream experiment without loading its pickle or code."""
    directory = Path(source_dir).expanduser().resolve()
    workbook = Path(input_excel).expanduser().resolve()
    paths = {name: directory / name for name in SOURCE_FILES}
    for path in [workbook, *paths.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    workbook_hash = file_sha256(workbook)
    read_options = {"keep_default_na": False, "float_precision": "round_trip",
                    "dtype": {name: str for name in [*RUN_KEY[:4], "X_JSON", "X_Bits"]}}
    manifest = pd.read_csv(paths[SOURCE_FILES[0]], **read_options)
    metrics = pd.read_csv(paths[SOURCE_FILES[1]], **read_options)
    chunks = []
    for chunk in pd.read_csv(paths[SOURCE_FILES[2]], chunksize=50000, **read_options):
        _require(chunk, ["Stage", "Front_Kind"], "all_pareto_points")
        chunks.append(chunk.loc[chunk["Stage"].eq("final") & chunk["Front_Kind"].eq("strict")].copy())
    points = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    for name, path in paths.items():
        if file_sha256(path) != hashes[name]:
            raise ValueError(f"source changed while reading: {name}")
    if file_sha256(workbook) != workbook_hash:
        raise ValueError("input workbook changed while reading")
    metadata = ["Excel_SHA256", "Implementation_Version", "Model_Protocol_Version", "Pymoo_Version",
                "Budget", "Game_Iterations", "Game_Learning_Rate", "Representative_Seed"]
    _require(manifest, [*GROUP_KEY, *metadata, "MA_Parameters_JSON", "Runs", "Algorithms_JSON"], "manifest")
    _require(metrics, [*RUN_KEY, *metadata, "Use_Pruning", "Prune_Tolerance"], "metrics")
    _require(points, [*RUN_KEY, *OBJECTIVES, "Point_ID", "X_JSON", "P_Response"], "final strict points")
    for frame, name in ((manifest, "manifest"), (metrics, "metrics"), (points, "points")):
        for column in GROUP_KEY:
            if frame[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"{name} has empty {column}")
        if str(_single(frame, "Experiment_ID", name)) != str(manifest["Experiment_ID"].iloc[0]):
            raise ValueError("cross-experiment source rows")
    for frame, name in ((manifest, "manifest"), (metrics, "metrics")):
        for column, expected in (("Excel_SHA256", workbook_hash),
                                 ("Implementation_Version", IMPLEMENTATION_VERSION),
                                 ("Model_Protocol_Version", MODEL_PROTOCOL_VERSION)):
            if str(_single(frame, column, name)) != expected:
                raise ValueError(f"{name} {column} does not match the input/current model")
        if "Input_Excel" in frame:
            names = frame["Input_Excel"].map(lambda value: Path(str(value).replace("\\", "/")).name.casefold())
            if not names.eq(workbook.name.casefold()).all():
                raise ValueError(f"{name} Input_Excel case does not match")
        _numeric(frame, ["Budget", "Game_Learning_Rate"], name)
        _numeric(frame, ["Game_Iterations", "Representative_Seed"], name, integer=True)
    for column in metadata:
        expected = _single(manifest, column, "manifest")
        if _single(metrics, column, "metrics") != expected:
            raise ValueError(f"manifest/metrics {column} conflict")
    budget = float(manifest["Budget"].iloc[0])
    if budget <= 0 or manifest["Game_Iterations"].iloc[0] < 1 or manifest["Game_Learning_Rate"].iloc[0] <= 0:
        raise ValueError("invalid source budget or game settings")
    _numeric(manifest, ["Runs"], "manifest", integer=True)
    for frame, name in ((metrics, "metrics"), (points, "points")):
        _numeric(frame, ["Run", "Seed"], name, integer=True)
        if (frame["Run"] < 1).any() or (frame["Seed"] < 0).any() or frame["Algorithm"].str.strip().eq("").any():
            raise ValueError(f"{name} has invalid run identity")
        _valid(frame, name)
    if manifest.duplicated(GROUP_KEY).any() or metrics.duplicated(RUN_KEY).any():
        raise ValueError("duplicate manifest group or run identity")
    if "Group_Status" in manifest and not manifest["Group_Status"].eq("ok").all():
        raise ValueError("manifest includes incomplete groups")
    groups = set(manifest[GROUP_KEY].itertuples(index=False, name=None))
    if set(metrics[GROUP_KEY].itertuples(index=False, name=None)) != groups:
        raise ValueError("manifest/metrics group association is incomplete")
    seeds = set(metrics["Seed"].astype(int))
    for row in manifest.to_dict("records"):
        selected = metrics.loc[(metrics[GROUP_KEY] == pd.Series({key: row[key] for key in GROUP_KEY})).all(axis=1)]
        algorithms = _json(row["Algorithms_JSON"], "Algorithms_JSON", list)
        if not algorithms or len(set(algorithms)) != len(algorithms) or set(selected["Algorithm"]) != set(algorithms):
            raise ValueError("manifest/metrics algorithm association is incomplete")
        runs = int(row["Runs"])
        for algorithm in algorithms:
            algorithm_rows = selected.loc[selected["Algorithm"].eq(algorithm)]
            if runs < 1 or len(algorithm_rows) != runs or set(algorithm_rows["Run"]) != set(range(1, runs + 1)):
                raise ValueError("manifest/metrics run association is incomplete")
            if set(algorithm_rows["Seed"]) != seeds or len(seeds) != runs:
                raise ValueError("algorithms/groups have inconsistent or duplicate seeds")
        if "Seed_Start" in row and int(row["Seed_Start"]) != min(seeds):
            raise ValueError("manifest Seed_Start conflicts with runs")
    representative_seed = int(manifest["Representative_Seed"].iloc[0])
    if representative_seed not in seeds:
        raise ValueError("representative seed is missing from source runs")
    keys = set(metrics[RUN_KEY].itertuples(index=False, name=None))
    if set(points[RUN_KEY].itertuples(index=False, name=None)) != keys:
        raise ValueError("final strict points/run association is incomplete")
    if "Pareto_Size" in metrics:
        _numeric(metrics, ["Pareto_Size"], "metrics", integer=True)
        counts = points.groupby(RUN_KEY).size()
        expected = metrics.set_index(RUN_KEY)["Pareto_Size"]
        if not np.array_equal(counts.reindex(expected.index).to_numpy(), expected.to_numpy()):
            raise ValueError("final strict points/Pareto_Size association is incomplete")
    _numeric(points, [*OBJECTIVES, "P_Response"], "points")
    _numeric(points, ["Point_ID"], "points", integer=True)
    if points.duplicated([*RUN_KEY, "Point_ID"]).any() or (points["Point_ID"] < 1).any():
        raise ValueError("duplicate or invalid final Point_ID")
    if (points["Total_Cost"] < 0).any() or (points["Total_Cost"] > budget + FEASIBILITY_TOL).any():
        raise ValueError("final points include infeasible cost")
    lengths = set()
    for row in points.to_dict("records"):
        decision = _decision(row["X_JSON"])
        lengths.add(len(decision))
        if not np.isclose(decision[-1], row["P_Response"], rtol=0, atol=1e-15):
            raise ValueError("X_JSON/P_Response conflict")
        if "X_Bits" in row and row["X_Bits"] != "".join(str(int(value)) for value in decision[:-1]):
            raise ValueError("X_JSON/X_Bits conflict")
    if len(lengths) != 1:
        raise ValueError("final points have inconsistent decision dimensions")
    if (points.groupby("X_JSON")[OBJECTIVES].nunique() > 1).any().any():
        raise ValueError("same exact X_JSON has conflicting historical objectives")
    ma_values = [_json(value, "MA_Parameters_JSON", dict) for value in manifest["MA_Parameters_JSON"]]
    if any(value != ma_values[0] for value in ma_values[1:]):
        raise ValueError("manifest has conflicting MA_Parameters_JSON")
    use_pruning = str(_single(metrics, "Use_Pruning", "metrics")).lower()
    if use_pruning not in ("true", "false"):
        raise ValueError("invalid Use_Pruning")
    if use_pruning == "true":
        _numeric(metrics, ["Prune_Tolerance"], "metrics")
        prune_tol = float(_single(metrics, "Prune_Tolerance", "metrics"))
        if prune_tol < 0:
            raise ValueError("invalid Prune_Tolerance")
    else:
        if not metrics["Prune_Tolerance"].astype(str).str.lower().isin(["", "none", "nan"]).all():
            raise ValueError("disabled pruning has a conflicting Prune_Tolerance")
        prune_tol = None
    settings = {"game_iterations": int(manifest["Game_Iterations"].iloc[0]),
                "game_learning_rate": float(manifest["Game_Learning_Rate"].iloc[0]),
                "prune_tol": prune_tol}
    return {"manifest": manifest, "experiment_manifest": manifest,
            "metrics": metrics, "performance_metrics_raw": metrics,
            "points": points, "all_pareto_points": points,
            "settings": settings, "budget": budget, "representative_seed": representative_seed,
            "seeds": seeds, "ma_parameters": ma_values[0], "hashes": hashes,
            "source_dir": directory, "input_excel": workbook, "excel_sha256": workbook_hash}


def replay_objectives(source, context, game_termination, *, rtol=1e-8, atol=1e-7, progress=None):
    """Recompute every exact saved decision; preserve historical objective values."""
    for name, value in (("rtol", rtol), ("atol", atol)):
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if isinstance(context, dict):
        items, data, graph = (context[name] for name in ("defense_items", "all_data", "ag"))
    else:
        items, data, graph = context
    problem = CyberDefenseProblem(items, data, graph, budget=source["budget"],
                                  use_constraint=True, use_cache=False,
                                  game_termination=game_termination, **source["settings"])
    unique = source["points"].drop_duplicates("X_JSON")
    rows = []
    for row in unique.to_dict("records"):
        decision = _decision(row["X_JSON"])
        if len(decision) != len(items) + 1:
            raise ValueError("saved decision does not match current defense dimension")
        expected = np.asarray([row[name] for name in OBJECTIVES], dtype=float)
        try:
            actual = np.asarray(problem.evaluate(decision, return_values_of=["F"]), dtype=float).reshape(-1)
        except Exception as error:
            raise ValueError(f"historical objective replay failed for X_JSON={row['X_JSON']}: {error}") from error
        if actual.shape != (2,) or not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=rtol, atol=atol):
            raise ValueError(f"historical objective mismatch for X_JSON={row['X_JSON']}: saved={expected.tolist()}, replayed={actual.tolist()}")
        rows.append({"X_JSON": row["X_JSON"], **dict(zip(OBJECTIVES, expected)),
                     "Replayed_Total_Cost": actual[0], "Replayed_Total_Defender_Loss": actual[1],
                     "Absolute_Cost_Error": abs(actual[0] - expected[0]),
                     "Absolute_Loss_Error": abs(actual[1] - expected[1]), "Objective_Compatible": True})
        if progress is not None and (len(rows) == 1 or len(rows) % 100 == 0 or len(rows) == len(unique)):
            progress(len(rows), len(unique))
    return pd.DataFrame(rows)
