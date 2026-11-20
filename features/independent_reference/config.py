

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.config import ROOT
from common.game_termination import GameTermination


ALGORITHMS = ("NSGA2_MA", "NSGA2", "SPEA2", "MOEA/D")
DEFAULT_CONFIG = Path(__file__).with_name("revision_20261102.json")


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_config(path=None, *, input_excel=None, workers=None):
    config_path = project_path(DEFAULT_CONFIG if path is None else path)
    with config_path.open(encoding="utf-8-sig") as stream:
        settings = json.load(stream)
    required = {
        "source_dir", "input_excel", "game_termination", "seeds", "algorithms",
        "max_fe", "min_actual_fe", "max_gen", "min_gen", "pop_size",
        "tol", "n_stable", "loss_scale", "workers",
    }
    if not required.issubset(settings) or set(settings) - required - {"pilot"}:
        raise ValueError(f"Configuration fields do not match: missing {sorted(required - set(settings))}; extra {sorted(set(settings) - required - {'pilot'})}")
    settings.setdefault("pilot", False)
    if not isinstance(settings["pilot"], bool):
        raise ValueError("pilot must be boolean")
    settings["source_dir"] = project_path(settings["source_dir"])
    settings["input_excel"] = project_path(input_excel or settings["input_excel"])
    settings["config_path"] = config_path
    if workers is not None:
        settings["workers"] = workers
    for name in ("max_fe", "min_actual_fe", "max_gen", "min_gen", "pop_size", "n_stable", "workers"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if settings["pop_size"] < 4 or settings["min_actual_fe"] < settings["pop_size"]:
        raise ValueError("pop_size must be at least 4, and min_actual_fe must not be smaller than the initial population")
    if settings["max_fe"] < settings["min_actual_fe"] or settings["max_gen"] < settings["min_gen"]:
        raise ValueError("Safety caps must not be lower than the minimum search requirements")
    for name in ("tol", "loss_scale"):
        value = settings[name]
        if isinstance(value, (bool, str)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    seeds, methods = settings["seeds"], settings["algorithms"]
    minimum_seeds = 1 if settings["pilot"] else 2
    if not isinstance(seeds, list) or len(set(seeds)) != len(seeds) or len(seeds) < minimum_seeds:
        raise ValueError("A formal reference search requires at least two distinct seeds; set pilot=true for a single-seed pilot")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("seeds must be valid nonnegative integers")
    if not isinstance(methods, list) or len(set(methods)) != len(methods) or len(methods) < 2 or set(methods) - set(ALGORITHMS):
        raise ValueError(f"A reference search requires at least two distinct methods from {ALGORITHMS}")
    if not isinstance(settings["game_termination"], dict) or set(settings["game_termination"]) != set(GameTermination.__dataclass_fields__):
        raise ValueError("Complete game_termination settings must be provided explicitly to prevent changes in model defaults")
    settings["game_termination"] = GameTermination(**settings["game_termination"])
    return settings
