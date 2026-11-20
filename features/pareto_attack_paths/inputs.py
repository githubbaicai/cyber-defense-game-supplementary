"""Read an upstream result snapshot without importing its producing feature."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from common.case_config import get_input_excel
from common.config import ROOT, get_feature_output_dir


DEFAULT_SELECTIONS = ("knee_ideal_distance", "min_cost_plus_loss")


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _number(value, field, *, minimum=0, maximum=None, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if (not np.isfinite(number) or number < minimum
            or (maximum is not None and number > maximum)
            or (integer and number != int(number))):
        raise ValueError(f"Invalid {field}: {value!r}")
    return int(number) if integer else number


def load_options(input_excel, configuration=None):
    """Paths are project-relative; no option changes shared case defaults."""
    path = get_input_excel(input_excel)
    if configuration is None:
        options = {}
    elif isinstance(configuration, dict):
        options = dict(configuration)
    else:
        config_path = Path(configuration)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        options = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(options, dict):
        raise ValueError("attack_path_config must be a JSON object")
    unknown = set(options) - {"source_dir", "source_case", "selections", "game_iterations", "game_learning_rate"}
    if unknown:
        raise ValueError(f"Unknown attack-path options: {sorted(unknown)}")
    selections = options.get("selections", DEFAULT_SELECTIONS)
    if (not isinstance(selections, (list, tuple)) or len(selections) != 2
            or any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value)
                   or value == "undefended" for value in selections)
            or len(set(selections)) != 2):
        raise ValueError("selections must contain two distinct Selection_Type names")
    source = Path(options.get("source_dir", get_feature_output_dir("pareto_path_analysis", path)))
    if not source.is_absolute():
        source = ROOT / source
    source = source.resolve()
    declared_case = options.get("source_case", source.parent.name)
    if declared_case != path.stem:
        raise ValueError(f"Source case {declared_case!r} does not match {path.stem!r}; provide the correct source_case")
    return {**options, "source_dir": source, "selections": tuple(selections)}


def read_snapshot(input_excel, configuration=None):
    options = load_options(input_excel, configuration)
    directory = options["source_dir"]
    tables, fingerprints = {}, []
    for name in ("selected_points", "selected_defenses", "node_risk", "run_metadata"):
        path = directory / f"{name}.csv"
        if not path.is_file():
            if name in ("selected_points", "selected_defenses"):
                raise FileNotFoundError(f"Missing upstream result {path}; run pareto_path_analysis solve/export explicitly first")
            continue
        raw = path.read_bytes()
        fingerprints.append({"Source_File": str(path), "SHA256": sha256(raw).hexdigest()})
        tables[name] = pd.read_csv(BytesIO(raw), keep_default_na=False, float_precision="round_trip")
    points, defenses = tables["selected_points"], tables["selected_defenses"]
    required = {"Selection_Type", "Solution_ID", "Selected_Defenses", "Number_of_Defenses",
                "Total_Cost", "P_Response", "Game_Iterations", "Game_Learning_Rate"}
    if not required <= set(points):
        raise ValueError(f"selected_points.csv missing columns: {sorted(required - set(points))}")
    required = {"Selection_Type", "Solution_ID", "Defense_Label", "Defense_Cost", "Total_Cost"}
    if not required <= set(defenses):
        raise ValueError(f"selected_defenses.csv missing columns: {sorted(required - set(defenses))}")
    selected = []
    for selection_type in options["selections"]:
        rows = points.loc[points.Selection_Type.eq(selection_type)]
        if len(rows) != 1:
            raise ValueError(f"Selection_Type={selection_type!r}: expected one selected point, found {len(rows)}")
        row = rows.iloc[0].to_dict()
        row["Solution_ID"] = _number(row["Solution_ID"], "Solution_ID", minimum=1, integer=True)
        row["Total_Cost"] = _number(row["Total_Cost"], "Total_Cost")
        row["P_Response"] = _number(row["P_Response"], "P_Response", maximum=1)
        row["Game_Iterations"] = _number(row["Game_Iterations"], "Game_Iterations", minimum=1, integer=True)
        row["Game_Learning_Rate"] = _number(row["Game_Learning_Rate"], "Game_Learning_Rate")
        count = _number(row["Number_of_Defenses"], "Number_of_Defenses", integer=True)
        labels = [label.strip() for label in str(row["Selected_Defenses"]).split("|") if label.strip()]
        if len(labels) != count or len(set(labels)) != len(labels):
            raise ValueError(f"{selection_type}: invalid Selected_Defenses/Number_of_Defenses")
        linked = defenses.loc[defenses.Selection_Type.eq(selection_type)].copy()
        if any(_number(value, "Solution_ID", minimum=1, integer=True) != row["Solution_ID"]
               for value in linked.Solution_ID):
            raise ValueError(f"{selection_type}: selected_defenses Solution_ID conflicts with selected_points")
        if len(linked) != count or set(linked.Defense_Label) != set(labels):
            raise ValueError(f"{selection_type}: selected_defenses does not match selected point measures")
        for value in linked.Total_Cost:
            if not np.isclose(_number(value, "Total_Cost"), row["Total_Cost"], rtol=1e-7, atol=1e-6):
                raise ValueError(f"{selection_type}: selected_defenses Total_Cost mismatch")
        costs = [_number(value, "Defense_Cost") for value in linked.Defense_Cost]
        if not np.isclose(sum(costs), row["Total_Cost"], rtol=1e-7, atol=1e-6):
            raise ValueError(f"{selection_type}: Defense_Cost sum mismatch")
        row["labels"] = labels
        row["source_defenses"] = linked
        selected.append(row)
    settings = {}
    for key, field in (("game_iterations", "Game_Iterations"), ("game_learning_rate", "Game_Learning_Rate")):
        if key in options:
            settings[field] = _number(options[key], field, minimum=1 if key == "game_iterations" else 0,
                                      integer=key == "game_iterations")
        else:
            values = {row[field] for row in selected}
            if len(values) != 1:
                raise ValueError(f"Upstream {field} differs; explicitly unify it in attack_path_config")
            settings[field] = values.pop()
    assert_unchanged(fingerprints)
    return options, selected, settings, tables, fingerprints


def assert_unchanged(fingerprints):
    for row in fingerprints:
        if not Path(row["Source_File"]).is_file() or file_hash(row["Source_File"]) != row["SHA256"]:
            raise ValueError(f"Upstream file changed during solve: {row['Source_File']}")


def decision_vector(point, defense_items):
    labels = [label for label, _ in defense_items]
    if len(labels) != len(set(labels)):
        raise ValueError("Allocation.Label must be unique")
    unknown = set(point["labels"]) - set(labels)
    if unknown:
        raise ValueError(f"Unknown source defenses in current Allocation: {sorted(unknown)}")
    selected = set(point["labels"])
    responses = selected.intersection({"RP1", "RP2", "RP3"})
    if len(responses) > 1 or (not responses and point["P_Response"] != 0):
        raise ValueError("Response decision is inconsistent with selected RP resources")
    costs = dict(defense_items)
    for row in point["source_defenses"].itertuples(index=False):
        if not np.isclose(float(row.Defense_Cost), costs[row.Defense_Label], rtol=1e-7, atol=1e-6):
            raise ValueError(f"Source/workbook cost mismatch for {row.Defense_Label}")
    vector = np.asarray([label in selected for label in labels], dtype=bool)
    total = sum(float(cost) for (label, cost), enabled in zip(defense_items, vector) if enabled)
    if not np.isclose(total, point["Total_Cost"], rtol=1e-7, atol=1e-6):
        raise ValueError("Source total cost does not match current Allocation")
    return vector
