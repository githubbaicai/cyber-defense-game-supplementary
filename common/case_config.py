

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import warnings

import pandas as pd

from common import config


class CaseParameterWarning(UserWarning):
    pass


def get_input_excel(input_excel=None) -> Path:
    
    path = Path(config.INPUT_EXCEL if input_excel is None else input_excel).expanduser()
    return (path if path.is_absolute() else config.ROOT / path).resolve()


def _normalize(value) -> str:
    return re.sub(r"[\s_-]+", "", str(value)).casefold()



_PARAMETERS = {
    "Use_Pruning": ("USE_PRUNING", bool, None),
    "Budget": ("BUDGET", float, 0),
    "Sensitivity_Budget": ("SENSITIVITY_BUDGET", float, 0),
    "Game_Iterations": ("GAME_OPTIMIZATION_ITERATIONS", int, 1),
    "Game_Learning_Rate": ("GAME_OPTIMIZATION_LEARNING_RATE", float, 0),
    "Population_Size": ("NSGA2_MA_POPULATION_SIZE", int, 2),
    "Max_Generations": ("NSGA2_MA_MAX_GENERATIONS", int, 1),
    "Termination_Tolerance": ("TERMINATION_TOLERANCE", float, 0),
    "Termination_Stable_Generations": ("TERMINATION_STABLE_GENERATIONS", int, 1),
    "Termination_Robust_Period": ("TERMINATION_ROBUST_PERIOD", int, 1),
    "Multi_Algorithm_Runs": ("MULTI_ALGORITHM_RUNS", int, 1),
    "Multi_Algorithm_Population_Size": ("MULTI_ALGORITHM_POPULATION_SIZE", int, 3),
    "Multi_Algorithm_Default_Max_Generations": ("MULTI_ALGORITHM_DEFAULT_MAX_GENERATIONS", int, 1),
    "Multi_Algorithm_Reference_Loss": ("MULTI_ALGORITHM_REFERENCE_LOSS", float, 0),
    "Multi_Algorithm_Reference_Cost_Factor": ("MULTI_ALGORITHM_REFERENCE_COST_FACTOR", float, 1),
    "Macomp_Runs": ("MACOMP_RUNS", int, 2),
    "Macomp_FE_Budgets": ("MACOMP_FE_BUDGETS", tuple, 1),
    "Macomp_Seed_Start": ("MACOMP_SEED_START", int, 0),
    "Macomp_Workers": ("MACOMP_WORKERS", int, 1),
}


_TERMINATION_PARAMETERS = {
    "Termination_Metric": ("TERMINATION_METRIC", str, ("front", "hv", "ideal", "both")),
    "Termination_Min_Generations": ("TERMINATION_MIN_GENERATIONS", int, 1),
    "Termination_Min_Front_Size": ("TERMINATION_MIN_FRONT_SIZE", int, 1),
    "Termination_Loss_Scale": (None, float, 0),
    "Termination_HV_Reference_Cost": (None, float, 1),
    "Termination_HV_Reference_Loss": (None, float, 1),
}


def _parse(value, name, kind, minimum):
    if kind is bool:
        if isinstance(value, bool):
            return value
        if type(value).__name__ == "bool_":
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"true", "false"}:
                return normalized == "true"
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
            return bool(value)
        raise ValueError("value must be a boolean true/false")
    if kind is str:
        if not isinstance(value, str) or value.strip().casefold() not in minimum:
            raise ValueError(f"value must be one of {minimum}")
        return value.strip().casefold()
    if kind is tuple:
        if isinstance(value, str):
            raw = value.strip()
            if raw.startswith("["):
                value = json.loads(raw)
            else:
                value = re.split(r"[,;\s]+", raw.strip("()"))
        elif not isinstance(value, (list, tuple)):
            value = [value]
        parsed = tuple(_parse(item, name, int, minimum) for item in value)
        if not parsed or list(parsed) != sorted(set(parsed)):
            raise ValueError("FE budgets must be nonempty, strictly increasing integers")
        return parsed
    if isinstance(value, bool):
        raise ValueError("boolean values are not numeric parameters")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("value must be finite")
    if kind is int:
        if not number.is_integer() or number < minimum:
            raise ValueError(f"value must be an integer >= {minimum}")
        return int(number)
    if name == "Termination_Tolerance" and number == 0:
        return number
    if number <= minimum:
        raise ValueError(f"value must be > {minimum}")
    return number


def load_case_parameters(input_excel=None, *, include_termination=False, overrides=None, names=None,
                         lenient_names=()) -> dict:
    
    path = get_input_excel(input_excel)
    specifications = dict(_PARAMETERS)
    if include_termination:
        specifications.update(_TERMINATION_PARAMETERS)
    if names is not None:
        requested = set(names)
        unknown = requested - set(specifications)
        if unknown:
            raise ValueError(f"unknown case parameters: {sorted(unknown)}")
        specifications = {name: value for name, value in specifications.items() if name in requested}
    values = {}
    known = {_normalize(name): name for name in specifications}
    with pd.ExcelFile(path) as workbook:
        matches = [sheet for sheet in workbook.sheet_names if _normalize(sheet) == "strategy"]
        if len(matches) > 1:
            raise ValueError(f"{path}: multiple Strategy sheets")
        if matches:
            frame = pd.read_excel(workbook, sheet_name=matches[0]).dropna(how="all")
            columns = {}
            for column in frame.columns:
                key = _normalize(column)
                if key in columns:
                    raise ValueError(f"{path}: duplicate Strategy column {column!r}")
                columns[key] = column
            if not {"parameter", "value"}.issubset(columns):
                raise ValueError(f"{path}: Strategy must contain Parameter and Value columns")
            for _, row in frame.iterrows():
                name = known.get(_normalize(row[columns["parameter"]]))
                if name is None:
                    continue
                if name in values:
                    raise ValueError(f"{path}: duplicate Strategy parameter {name}")
                values[name] = row[columns["value"]]

    if overrides:
        unknown = set(overrides) - set(specifications)
        if unknown:
            raise ValueError(f"unknown parameter overrides: {sorted(unknown)}")
        values.update(overrides)

    parameters = {}
    missing = []
    invalid_fallbacks = []
    for name, (constant, kind, minimum) in specifications.items():
        value = values.get(name)
        if value is None or (isinstance(value, str) and not value.strip()) or (
            not isinstance(value, (list, tuple)) and pd.isna(value)
        ):
            if constant is None:
                parameters[name] = None
                continue
            value = getattr(config, constant)
            missing.append(f"{name}={value!r}")
        try:
            parameters[name] = _parse(value, name, kind, minimum)
        except (TypeError, ValueError, OverflowError) as error:
            if name in set(lenient_names):
                fallback = getattr(config, constant) if constant is not None else None
                parameters[name] = fallback
                invalid_fallbacks.append(f"{name}={value!r} -> {fallback!r}")
                continue
            raise ValueError(f"{path}: invalid Strategy {name}={value!r}: {error}") from error
    if missing or invalid_fallbacks:
        warnings.warn(
            f"{path.name}: missing Strategy parameters; using common.config defaults: "
            + ", ".join([*missing, *invalid_fallbacks]),
            CaseParameterWarning,
            stacklevel=2,
        )
    return parameters
