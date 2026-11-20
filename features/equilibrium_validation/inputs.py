"""Validate E3 configuration and load complete, traceable portfolio decisions."""

from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from common.defense_config import normalize_decision_vector


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _number(value, name, lower=0.0, upper=None, integer=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(number) or number < lower or (upper is not None and number > upper):
        raise ValueError(f"{name} outside [{lower}, {upper}]")
    if integer and not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number) if integer else number


def _identifier(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError(f"{name} must be a safe nonempty identifier")
    return value


def load_configuration(equilibrium_config):
    """Read a JSON path or independent configuration dictionary; never mutate it."""
    if isinstance(equilibrium_config, dict):
        config = deepcopy(equilibrium_config)
        base_directory = Path.cwd()
        config_path = None
    elif equilibrium_config is not None:
        config_path = Path(equilibrium_config).expanduser().resolve()
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        base_directory = config_path.parent
    else:
        raise ValueError("equilibrium_config must explicitly select portfolios")
    if not isinstance(config, dict):
        raise ValueError("equilibrium configuration must be a JSON object")
    prune_tol_explicit = "prune_tol" in config
    allowed = {
        "experiment_id", "protocol", "portfolios", "attacker_initializations",
        "response_initializations", "dirichlet_seeds", "eta_multipliers",
        "iteration_checkpoints", "thresholds", "prune_tol", "max_full_graph_runs",
        "max_local_states", "trace_stride", "path_tie_tolerance", "endpoint_tolerance",
        "publication_summary",
    }
    if set(config) - allowed:
        raise ValueError(f"unknown E3 settings: {sorted(set(config) - allowed)}")
    config["experiment_id"] = _identifier(config.get("experiment_id"), "experiment_id")
    protocol = config.setdefault("protocol", "smoke")
    if protocol not in ("smoke", "full"):
        raise ValueError("protocol must be smoke or full")
    portfolios = config.get("portfolios")
    if not isinstance(portfolios, list) or not portfolios:
        raise ValueError("portfolios must be a nonempty list of explicit selections")
    seen = set()
    for portfolio in portfolios:
        if not isinstance(portfolio, dict):
            raise ValueError("each portfolio must be an object")
        identity = _identifier(portfolio.get("portfolio_id"), "portfolio_id")
        if identity in seen:
            raise ValueError(f"duplicate portfolio_id: {identity}")
        seen.add(identity)
        source = portfolio.setdefault("source", "explicit")
        if source == "explicit":
            if set(portfolio) - {"portfolio_id", "source", "selected_labels", "p_response"}:
                raise ValueError(f"unknown explicit portfolio fields: {identity}")
            labels = portfolio.get("selected_labels")
            if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
                raise ValueError("selected_labels must be a list of exact Allocation labels")
            if len(set(labels)) != len(labels):
                raise ValueError("selected_labels must not contain duplicates")
            portfolio["p_response"] = _number(portfolio.get("p_response"), "p_response", upper=1.0)
        elif source == "nsga2_ma_csv":
            if set(portfolio) - {"portfolio_id", "source", "directory", "solution_id"}:
                raise ValueError(f"unknown CSV portfolio fields: {identity}")
            if not isinstance(portfolio.get("directory"), str) or not portfolio["directory"].strip():
                raise ValueError("CSV portfolio requires directory")
            if portfolio.get("solution_id") is None or isinstance(portfolio["solution_id"], bool):
                raise ValueError("CSV portfolio requires solution_id")
            portfolio["directory"] = str((base_directory / Path(portfolio["directory"]).expanduser()).resolve())
        else:
            raise ValueError(f"unsupported portfolio source: {source}")
    smoke = protocol == "smoke"
    defaults = {
        "attacker_initializations": ["original", "uniform"] if smoke else ["original", "uniform", "dirichlet"],
        "response_initializations": ["original"] if smoke else ["original", 0.5, 0.0, 1.0, 1e-6, 1.0 - 1e-6],
        "dirichlet_seeds": [17] if smoke else [17, 29, 43, 71, 101],
        "eta_multipliers": [1.0, 0.5] if smoke else [1.0, 0.5, 2.0],
        "iteration_checkpoints": [10, 20] if smoke else [200, 500, 1000, 2000, 5000],
        "thresholds": {}, "prune_tol": 1e-9,
        "max_full_graph_runs": 8 if smoke else 40,
        "max_local_states": 2 if smoke else None,
        "trace_stride": 1 if smoke else 10,
        "path_tie_tolerance": 1e-8, "endpoint_tolerance": 1e-5,
    }
    for name, default in defaults.items():
        config.setdefault(name, default)
    for name in ("attacker_initializations", "response_initializations", "dirichlet_seeds", "eta_multipliers", "iteration_checkpoints"):
        if not isinstance(config[name], list) or not config[name]:
            raise ValueError(f"{name} must be a nonempty list")
    if any(value not in ("original", "uniform", "dirichlet") for value in config["attacker_initializations"]):
        raise ValueError("unsupported attacker initialization")
    config["dirichlet_seeds"] = [_number(value, "seed", integer=True) for value in config["dirichlet_seeds"]]
    config["eta_multipliers"] = [_number(value, "eta multiplier") for value in config["eta_multipliers"]]
    config["iteration_checkpoints"] = sorted(set(_number(value, "checkpoint", lower=1, integer=True) for value in config["iteration_checkpoints"]))
    config["response_initializations"] = [value if value == "original" else _number(value, "response initialization", upper=1) for value in config["response_initializations"]]
    for name in ("max_full_graph_runs", "trace_stride"):
        config[name] = _number(config[name], name, lower=1, integer=True)
    if config["max_local_states"] is not None:
        config["max_local_states"] = _number(config["max_local_states"], "max_local_states", lower=1, integer=True)
    for name in ("path_tie_tolerance", "endpoint_tolerance"):
        config[name] = _number(config[name], name)
    if config["prune_tol"] is not None:
        config["prune_tol"] = _number(config["prune_tol"], "prune_tol")
    allowed_thresholds = {"map_tol", "fitness_tol", "gap_tol", "support_tol", "tail_tol", "tail_window", "jacobian_step", "jacobian_tol", "stability_margin"}
    if not isinstance(config["thresholds"], dict) or set(config["thresholds"]) - allowed_thresholds:
        raise ValueError("unknown or invalid diagnostic thresholds")
    for name, value in config["thresholds"].items():
        config["thresholds"][name] = _number(value, name, lower=1 if name == "tail_window" else 0, integer=name == "tail_window")
        if name == "jacobian_step" and config["thresholds"][name] == 0:
            raise ValueError("jacobian_step must be positive")
    config["_config_path"] = str(config_path) if config_path else None
    config["_prune_tol_explicit"] = prune_tol_explicit
    publication = config.get("publication_summary")
    if publication is not None:
        if not isinstance(publication, dict) or set(publication) - {"local_iterations", "trace_stride"}:
            raise ValueError("publication_summary must specify local_iterations and optional trace_stride")
        publication["local_iterations"] = _number(publication.get("local_iterations"), "publication local_iterations", lower=1, integer=True)
        publication["trace_stride"] = _number(publication.get("trace_stride", 10), "publication trace_stride", lower=1, integer=True)
    return config


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if not columns or len(set(columns)) != len(columns) or any(not column for column in columns):
            raise ValueError(f"{path}: missing or duplicate CSV column")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path}: malformed CSV row")
    return columns, rows


def _solution_rows(rows, path):
    indexed = {}
    for row in rows:
        identity = row.get("Solution_ID", "").strip()
        if not identity or identity in indexed:
            raise ValueError(f"{path}: missing or duplicate Solution_ID")
        indexed[identity] = row
    return indexed


def load_portfolios(config, defense_items, input_hash):
    """Align source CSV decisions by label and retain raw and repaired identities."""
    labels = [label for label, _ in defense_items]
    loaded = []
    for specification in config["portfolios"]:
        source = {"type": specification["source"], "files": {}, "historical_input_status": "unknown", "historical_parameter_status": "unknown"}
        expected_cost = None
        expected_loss = None
        if specification["source"] == "explicit":
            unknown = set(specification["selected_labels"]) - set(labels)
            if unknown:
                raise ValueError(f"unknown Allocation labels: {sorted(unknown)}")
            raw = [float(label in specification["selected_labels"]) for label in labels] + [specification["p_response"]]
            source["status"] = "explicit_current_scenario"
        else:
            directory = Path(specification["directory"])
            paths = {name: directory / f"{name}.csv" for name in ("defense_selection_matrix", "pareto_front", "run_metadata")}
            sources = {name: _read_csv(path) for name, path in paths.items()}
            matrix_columns, matrix_rows = sources["defense_selection_matrix"]
            if set(matrix_columns) != {"Solution_ID", "P_Response", *labels}:
                raise ValueError("defense_selection_matrix columns must exactly match Allocation labels, Solution_ID and P_Response")
            front_columns, front_rows = sources["pareto_front"]
            if not {"Solution_ID", "P_Response", "Total_Cost", "Total_Defender_Loss"}.issubset(front_columns):
                raise ValueError("pareto_front lacks required decision/objective columns")
            matrix = _solution_rows(matrix_rows, paths["defense_selection_matrix"])
            front = _solution_rows(front_rows, paths["pareto_front"])
            if set(matrix) != set(front):
                raise ValueError("source Solution_ID sets differ between matrix and front")
            for source_id, source_row in matrix.items():
                source_bits = [_number(source_row[label], label, upper=1) for label in labels]
                if any(bit not in (0.0, 1.0) for bit in source_bits):
                    raise ValueError("source defense selections must be 0/1")
                matrix_response = _number(source_row["P_Response"], "matrix P_Response", upper=1)
                front_response = _number(front[source_id]["P_Response"], "front P_Response", upper=1)
                if not math.isclose(matrix_response, front_response, rel_tol=0, abs_tol=1e-12):
                    raise ValueError("source P_Response conflict between matrix and front")
                _number(front[source_id]["Total_Cost"], "source cost")
                _number(front[source_id]["Total_Defender_Loss"], "source loss", lower=-math.inf)
            identity = str(specification["solution_id"]).strip()
            if identity not in matrix or identity not in front:
                raise ValueError(f"Solution_ID {identity!r} not present in both source files")
            row = matrix[identity]
            bits = [_number(row[label], label, upper=1) for label in labels]
            if any(bit not in (0.0, 1.0) for bit in bits):
                raise ValueError("source defense selections must be 0/1")
            response = _number(row["P_Response"], "matrix P_Response", upper=1)
            front_response = _number(front[identity]["P_Response"], "front P_Response", upper=1)
            if not math.isclose(response, front_response, rel_tol=0, abs_tol=1e-12):
                raise ValueError("source P_Response conflict between matrix and front")
            raw = bits + [response]
            expected_cost = _number(front[identity]["Total_Cost"], "source cost")
            expected_loss = _number(front[identity]["Total_Defender_Loss"], "source loss", lower=-math.inf)
            metadata_rows = sources["run_metadata"][1]
            if len(metadata_rows) != 1:
                raise ValueError("source run_metadata must contain exactly one row")
            metadata = metadata_rows[0]
            historical_hash = metadata.get("Input_SHA256") or metadata.get("Input_Excel_SHA256")
            if historical_hash:
                if not re.fullmatch(r"[0-9a-fA-F]{64}", historical_hash):
                    raise ValueError("source input SHA256 is malformed")
                source["historical_input_status"] = "match" if historical_hash.lower() == input_hash else "mismatch"
            source.update({"solution_id": identity, "metadata": metadata, "status": "source_loaded", "historical_input_sha256": historical_hash or "unknown"})
            for name, path in paths.items():
                source["files"][name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
        normalized = normalize_decision_vector(raw, defense_items)
        loaded.append({
            "portfolio_id": specification["portfolio_id"], "raw_decision": raw,
            "decision": normalized.tolist(), "allocation_labels": labels.copy(),
            "source": source, "expected_cost": expected_cost, "expected_loss": expected_loss,
        })
    return loaded
