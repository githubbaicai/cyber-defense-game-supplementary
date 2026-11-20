"""Feature-local probability-mapping protocol and validated case settings."""

from copy import deepcopy
from dataclasses import asdict
import json
import math
from numbers import Real
from pathlib import Path

from common.case_config import get_input_excel, load_case_parameters
from common.config import ROOT
from common.game_termination import DEFAULT_GAME_TERMINATION, GameTermination
from common.memetic import MemeticNSGA2
from common.simulation import resolve_prune_tolerance


PROTOCOL_VERSION = "probability-mapping-decision-robustness-v1"
MAPPINGS = ("baseline", "square", "sqrt", "scale_0_8")
MAPPING_SPECS = {
    "baseline": {"label": "EPSS", "formula": "p=e"},
    "square": {"label": "Squared EPSS", "formula": "p=e^2"},
    "sqrt": {"label": "Square-root EPSS", "formula": "p=sqrt(e)"},
    "scale_0_8": {"label": "0.8 EPSS", "formula": "p=0.8*e"},
}


def _defaults():
    return {
        "seeds": [42, 43, 44], "workers": 9, "pop_size": 50, "max_fe": 1000,
        "lambda_loss": 1.0, "smoke_only": False, "budget": None,
        "game_iterations": None, "game_learning_rate": None, "use_pruning": None,
        "game_termination": asdict(DEFAULT_GAME_TERMINATION),
        "ma_parameters": {
            "selection_mode": "elite", "neighborhood": "all", "local_search_prob": 0.1,
            "max_local_search_iters": 5, "no_improve_rounds": 2, "elite_size": None,
            "max_local_fe_ratio": 0.2, "max_local_fe_per_gen": None,
            "local_search_trigger": "every_gen", "local_search_start_ratio": 0.5,
            "hv_stagnation_generations": 5, "front_stagnation_generations": 3,
            "trigger_hv_tolerance": 1e-6, "acceptance_mode": "current",
            "acceptance_epsilon": 0.001, "trigger_loss_scale": 3000.0,
            "trigger_hv_reference_point": [1.1, 1.1], "last_gen_local_fe_ratio": 0.2,
            "use_archive": True, "archive_max_size": None,
            "use_greedy_operators": False, "greedy_max_steps": 200,
            "use_ma_specific_init": False,
        },
    }


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def load_config(mapping_config=None, *, input_excel=None, **overrides):
    """Resolve explicit overrides, feature JSON and workbook values without mutation."""
    supplied = {}
    config_path = None
    if isinstance(mapping_config, dict):
        supplied = deepcopy(mapping_config)
    elif mapping_config is not None:
        config_path = Path(mapping_config).expanduser()
        config_path = (config_path if config_path.is_absolute() else ROOT / config_path).resolve()
        with config_path.open(encoding="utf-8-sig") as stream:
            supplied = json.load(stream)
    if not isinstance(supplied, dict):
        raise ValueError("mapping_config must contain a JSON object")
    settings = _defaults()
    allowed = set(settings)
    unknown = (set(supplied) | set(overrides)) - allowed
    if unknown:
        raise ValueError(f"unknown probability-mapping settings: {sorted(unknown)}")
    supplied.update({name: value for name, value in overrides.items() if value is not None})
    ma_overrides = supplied.pop("ma_parameters", {})
    if not isinstance(ma_overrides, dict) or set(ma_overrides) - set(settings["ma_parameters"]):
        raise ValueError("ma_parameters must contain only the supported MA settings")
    settings["ma_parameters"].update(deepcopy(ma_overrides))
    settings.update(supplied)
    settings["input_excel"] = get_input_excel(input_excel)
    settings["config_path"] = config_path
    for name, minimum in (("workers", 1), ("pop_size", 4), ("max_fe", 4)):
        settings[name] = _integer(settings[name], name, minimum)
    if settings["max_fe"] < settings["pop_size"]:
        raise ValueError("max_fe must be at least pop_size")
    if type(settings["smoke_only"]) is not bool:
        raise ValueError("smoke_only must be bool")
    seeds = settings["seeds"]
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("seeds must be a nonempty list")
    seeds = [_integer(seed, "seed", 0) for seed in seeds]
    if len(set(seeds)) != len(seeds) or any(seed >= 2**32 for seed in seeds):
        raise ValueError("seeds must be unique uint32 values")
    settings["seeds"] = seeds
    value = settings["lambda_loss"]
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("lambda_loss must be finite and positive")
    settings["lambda_loss"] = float(value)
    names = {"budget": "Budget", "game_iterations": "Game_Iterations",
             "game_learning_rate": "Game_Learning_Rate", "use_pruning": "Use_Pruning"}
    parameters = load_case_parameters(
        settings["input_excel"], names=list(names.values()),
        overrides={target: settings[name] for name, target in names.items() if settings[name] is not None},
    )
    settings.update({name: parameters[target] for name, target in names.items()})
    settings["prune_tol"] = resolve_prune_tolerance(settings["use_pruning"])
    termination = settings["game_termination"]
    if termination is not None:
        if not isinstance(termination, dict) or set(termination) != set(GameTermination.__dataclass_fields__):
            raise ValueError("game_termination must specify all GameTermination fields or be null")
        settings["game_termination"] = GameTermination(**termination)
    MemeticNSGA2(pop_size=settings["pop_size"], max_fe=settings["max_fe"], **settings["ma_parameters"])
    return settings
