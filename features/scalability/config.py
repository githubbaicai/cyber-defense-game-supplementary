"""Explicit, immutable-by-copy settings for the evaluator benchmark."""

from copy import deepcopy
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
DEFAULTS = {
    "cases": ["game-theory.xlsx", "game-theory-purdue.xlsx"],
    "portfolios": ["none", "DR", "PDR"],
    "game_modes": ["evolutionary", "fixed_policy"],
    "repeats": 3,
    "phase": "measurement",
    "game_iterations": 500,
    "game_learning_rate": 0.5,
    "prune_tol": 1e-9,
    "p_response": 0.5,
    "seed": 42,
    "per_run_timeout_seconds": 120.0,
    "total_timeout_seconds": 1800.0,
    "game_termination": {
        "min_iterations": 100, "stable_iterations": 10, "check_interval": 10,
        "probability_tolerance": 1e-7, "residual_tolerance": 1e-6,
        "invasion_tolerance": 1e-6,
    },
    "model_options": {
        "history_mode": "elapsed", "alternative_trigger": 0.0,
        "exposure_penalty_ratio": 0.3, "response_effectiveness_multiplier": 1.0,
        "response_penalty_ratio": 0.3,
    },
    "preflight_reference": None,
}


def load_config(path=None):
    settings = deepcopy(DEFAULTS)
    if path is not None:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(supplied) - set(settings)
        if unknown:
            raise ValueError(f"Unknown scalability settings: {sorted(unknown)}")
        settings.update(supplied)
    for name in ("cases", "portfolios", "game_modes"):
        values = settings[name]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be a nonempty list of distinct values")
    if set(settings["portfolios"]) - {"none", "DR", "PDR"}:
        raise ValueError("portfolios must use none, DR or PDR")
    if set(settings["game_modes"]) - {"evolutionary", "fixed_policy"}:
        raise ValueError("Invalid game_modes")
    if settings["phase"] not in ("preflight", "measurement", "warmup"):
        raise ValueError("Invalid phase")
    for name in ("repeats", "game_iterations"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(settings["seed"]) is not int or settings["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name in ("per_run_timeout_seconds", "total_timeout_seconds", "game_learning_rate"):
        if type(settings[name]) not in (int, float) or not math.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 <= settings["p_response"] <= 1:
        raise ValueError("p_response must be within [0, 1]")
    if settings["prune_tol"] is not None and (
        not math.isfinite(settings["prune_tol"]) or settings["prune_tol"] < 0
    ):
        raise ValueError("prune_tol must be None or finite and nonnegative")
    from common.game_termination import GameTermination
    from common.model_options import ModelOptions
    GameTermination(**settings["game_termination"])
    ModelOptions(**settings["model_options"])
    settings["cases"] = [str((ROOT / filename).resolve()) for filename in settings["cases"]]
    if len({Path(filename).stem for filename in settings["cases"]}) != len(settings["cases"]):
        raise ValueError("Case filenames must have distinct stems")
    for filename in settings["cases"]:
        if not Path(filename).is_file():
            raise FileNotFoundError(filename)
    return settings


def make_schedule(settings):
    tasks = []
    for repeat in range(1, settings["repeats"] + 1):
        cases = settings["cases"] if repeat % 2 else list(reversed(settings["cases"]))
        modes = settings["game_modes"] if repeat % 2 else list(reversed(settings["game_modes"]))
        for portfolio in settings["portfolios"]:
            for mode in modes:
                for filename in cases:
                    case = Path(filename).stem
                    tasks.append({
                        "Run_Key": f"{settings['phase']}_{case}_{portfolio}_{mode}_{repeat}",
                        "Case": case, "Input_Excel": filename, "Portfolio": portfolio,
                        "Game_Mode": mode, "Repeat": repeat, "Phase": settings["phase"],
                        "Order": len(tasks) + 1, "Seed": settings["seed"],
                    })
    return tasks
