"""Resolve only E7 settings; never read another feature's configuration."""

from copy import deepcopy
import json
import math

from common.case_config import get_input_excel, load_case_parameters
from common.memetic import MA_PARAMETER_DEFAULTS, MemeticNSGA2
from common.model_options import ModelOptions
from common.game_termination import (
    DEFAULT_GAME_TERMINATION,
    resolve_game_termination,
    termination_metadata,
)
from common.simulation import resolve_prune_tolerance


def scenarios():
    return (
        ("baseline", ModelOptions(), True),
        ("no_history", ModelOptions(history_mode="local_step"), True),
        ("no_egt", ModelOptions(game_mode="fixed_policy"), True),
        ("exposure_zero", ModelOptions(exposure_penalty_ratio=0.0), True),
        ("response_half", ModelOptions(response_effectiveness_multiplier=0.5), True),
    )


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def resolve_config(input_excel, structural_config=None, **overrides):
    settings = {
        "use_pruning": None,
        "seeds": [42, 43, 44], "max_fe": 2000,
        "workers": 1, "fixed_only": False,
        "crossover_prob": 0.9, "mutation_prob": 0.05,
        "fixed_portfolios": [], "ma_parameters": {},
        "game_termination": termination_metadata(DEFAULT_GAME_TERMINATION),
    }
    optional = {"budget", "pop_size", "game_iterations", "game_learning_rate", "use_pruning"}
    supplied = {}
    if structural_config is not None:
        if isinstance(structural_config, dict):
            supplied = deepcopy(structural_config)
        else:
            supplied = json.loads(get_input_excel(structural_config).read_text(encoding="utf-8-sig"))
        if not isinstance(supplied, dict):
            raise ValueError("structural_config must contain a JSON object")
    unknown = set(supplied) - set(settings) - optional
    if unknown:
        raise ValueError(f"unknown structural settings: {sorted(unknown)}")
    settings.update(supplied)
    unknown = set(overrides) - set(settings) - optional - {"seed"}
    if unknown:
        raise ValueError(f"unknown structural overrides: {sorted(unknown)}")
    settings.update({key: value for key, value in overrides.items() if value is not None and key != "seed"})
    if overrides.get("seed") is not None:
        settings["seeds"] = [overrides["seed"]]
    case_names = {
        "use_pruning": "Use_Pruning",
        "budget": "Budget", "pop_size": "Population_Size",
        "game_iterations": "Game_Iterations", "game_learning_rate": "Game_Learning_Rate",
    }
    if settings["use_pruning"] is not None and type(settings["use_pruning"]) is not bool:
        raise ValueError("use_pruning must be a boolean")
    case = load_case_parameters(
        input_excel, names=tuple(case_names.values()),
        overrides={name: settings[key] for key, name in case_names.items() if settings.get(key) is not None},
    )
    settings.update({key: case[name] for key, name in case_names.items()})
    settings["prune_tol"] = resolve_prune_tolerance(settings["use_pruning"])
    settings["game_termination"] = termination_metadata(resolve_game_termination(settings["game_termination"]))
    settings["workers"] = _integer(settings["workers"], "workers", 1)
    if type(settings["fixed_only"]) is not bool:
        raise ValueError("fixed_only must be a boolean")
    seeds = settings["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a nonempty list")
    settings["seeds"] = [_integer(seed, "seed", 0) for seed in seeds]
    if any(seed >= 2**32 for seed in settings["seeds"]):
        raise ValueError("seeds must be smaller than 2**32")
    if len(set(settings["seeds"])) != len(seeds):
        raise ValueError("seeds must be unique")
    settings["max_fe"] = _integer(settings["max_fe"], "max_fe", settings["pop_size"])
    for name in ("crossover_prob", "mutation_prob"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a probability")
    parameters = settings["ma_parameters"]
    if not isinstance(parameters, dict) or set(parameters) - set(MA_PARAMETER_DEFAULTS):
        raise ValueError("ma_parameters must contain only common MemeticNSGA2 parameters")
    settings["ma_parameters"] = {
        **deepcopy(MA_PARAMETER_DEFAULTS), "max_local_search_iters": 5,
        "max_local_fe_ratio": 0.2, **parameters,
    }
    MemeticNSGA2(pop_size=settings["pop_size"], max_fe=settings["max_fe"], **settings["ma_parameters"])
    if not isinstance(settings["fixed_portfolios"], list):
        raise ValueError("fixed_portfolios must be a list")
    return settings
