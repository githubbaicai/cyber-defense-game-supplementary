"""Resolve this feature's experiment protocol without changing shared defaults."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Real
from pathlib import Path

from features.sensitivity_analysis.parameters import (
    PARAMETER_NAMES,
    normalize_ranges,
)


PROTOCOL_VERSION = "sensitivity-epss-decision-robustness-v3"

_MODES = {
    "fast": {
        "n_samples": 8, "n_levels": (8,), "optimization_seeds": (42, 43),
        "pop_size": 20, "max_fe": 100, "max_generations": 5,
    },
    "standard": {
        "n_samples": 128, "n_levels": (128,), "optimization_seeds": (42, 43, 44),
        "pop_size": 100, "max_fe": 2000, "max_generations": 50,
    },
    "comprehensive": {
        "n_samples": 256, "n_levels": (64, 128, 256),
        "optimization_seeds": (42, 43, 44), "pop_size": 100,
        "max_fe": 2000, "max_generations": 50,
    },
}


@dataclass(frozen=True)
class SensitivityConfig:
    mode: str = "fast"
    n_samples: int = 8
    n_levels: tuple = (8,)
    optimization_seeds: tuple = (42, 43)
    noise_seeds: tuple = (1042, 1043)
    noise_scenarios: int = 2
    workers: int = 9
    calc_second_order: bool = False
    scramble: bool = True
    sampling_seed: int = 142
    bootstrap_seed: int = 242
    bootstrap_resamples: int = 1000
    confidence_level: float = 0.95
    pop_size: int = 20
    max_fe: int = 100
    max_generations: int = 5
    crossover_prob: float = 0.9
    mutation_prob: float = 0.05
    game_iterations: int | None = None
    game_learning_rate: float | None = None
    use_pruning: bool | None = None
    budget: float | None = None
    lambda_loss: float = 1.0
    budget_fractions: tuple = (0.25, 0.5, 1.0)
    parameter_ranges: dict = field(default_factory=dict)
    ma_parameters: dict = field(default_factory=dict)

    @property
    def repeats(self):
        return len(self.optimization_seeds)

    @property
    def smoke_only(self):
        return self.mode == "fast"

    def to_dict(self):
        return asdict(self)

    def planned_workload(self, dimension=len(PARAMETER_NAMES)):
        dimension = _integer(dimension, "dimension", 1)
        block_size = 2 * dimension + 2 if self.calc_second_order else dimension + 2
        sample_rows = self.n_samples * block_size
        optimization_runs = sample_rows * self.repeats
        nominal_runs = self.repeats
        noise_runs = min(self.noise_scenarios, 2 * self.n_samples) * len(self.noise_seeds)
        return {
            "Base_N": self.n_samples, "D": dimension,
            "Saltelli_Block_Size": block_size, "Sample_Rows": sample_rows,
            "Optimization_Repeats": self.repeats,
            "Sample_Optimization_Runs": optimization_runs,
            "Nominal_Optimization_Runs": nominal_runs,
            "Noise_Optimization_Runs": noise_runs,
            "Planned_Optimization_FE_Upper_Bound":
                (optimization_runs + nominal_runs + noise_runs) * self.max_fe,
            "Additional_Evaluations": "fixed portfolios and candidate-pool evaluations are recorded separately",
        }


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name, minimum=0.0, *, allow_equal=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (not allow_equal and value == minimum):
        raise ValueError(f"{name} must be {'>=' if allow_equal else '>'} {minimum}")
    return float(value)


def _seeds(values, name, *, allow_empty=False):
    if not isinstance(values, (list, tuple)) or (not values and not allow_empty):
        raise ValueError(f"{name} must be a {'possibly empty' if allow_empty else 'nonempty'} list")
    seeds = tuple(_integer(value, name, 0) for value in values)
    if len(set(seeds)) != len(seeds) or any(value >= 2**32 for value in seeds):
        raise ValueError(f"{name} must contain unique seeds below 2**32")
    return seeds


def _load_settings(sensitivity_config):
    if sensitivity_config is None:
        return {}
    if isinstance(sensitivity_config, SensitivityConfig):
        return sensitivity_config.to_dict()
    if isinstance(sensitivity_config, dict):
        return deepcopy(sensitivity_config)
    from common.config import ROOT

    path = Path(sensitivity_config).expanduser()
    path = path if path.is_absolute() else ROOT / path
    settings = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(settings, dict):
        raise ValueError("sensitivity_config must contain a JSON object")
    return settings


def resolve_config(
    mode="fast", n_samples=None, workers=None, sensitivity_config=None, *,
    budget=None, game_iterations=None, game_learning_rate=None,
):
    """Return validated, per-call settings; case values may be resolved later."""
    supplied = _load_settings(sensitivity_config)
    from common import config as common_config

    unknown = set(supplied) - set(SensitivityConfig.__dataclass_fields__)
    if "mappings" in supplied:
        raise ValueError("mappings is obsolete; remove it and configure parameter_ranges.epss")
    if unknown:
        raise ValueError(f"unknown sensitivity settings: {sorted(unknown)}")
    selected_mode = supplied.get("mode", mode)
    if selected_mode not in _MODES:
        raise ValueError("mode must be fast, standard or comprehensive")
    settings = SensitivityConfig().to_dict()
    if "workers" not in supplied and workers is None:
        settings["workers"] = common_config.DEFAULT_WORKERS
    settings.update(deepcopy(_MODES[selected_mode]))
    settings.update(supplied)
    settings["mode"] = selected_mode
    for name, value in (
        ("n_samples", n_samples), ("workers", workers), ("budget", budget),
        ("game_iterations", game_iterations), ("game_learning_rate", game_learning_rate),
    ):
        if value is not None:
            settings[name] = value
    if n_samples is not None or ("n_samples" in supplied and "n_levels" not in supplied):
        settings["n_levels"] = (settings["n_samples"],)
    elif "n_levels" in supplied and "n_samples" not in supplied:
        levels = supplied["n_levels"]
        if isinstance(levels, (list, tuple)) and levels:
            settings["n_samples"] = max(levels)
    for name, minimum in (
        ("n_samples", 2), ("workers", 1), ("bootstrap_resamples", 2),
        ("pop_size", 2), ("max_fe", 2), ("max_generations", 1), ("noise_scenarios", 0),
    ):
        settings[name] = _integer(settings[name], name, minimum)
    levels = settings["n_levels"]
    if not isinstance(levels, (list, tuple)) or not levels:
        raise ValueError("n_levels must be a nonempty list of powers of two")
    levels = tuple(_integer(value, "n_levels", 2) for value in levels)
    if list(levels) != sorted(set(levels)) or any(value & (value - 1) for value in levels):
        raise ValueError("n_levels must contain strictly increasing powers of two")
    if settings["n_samples"] & (settings["n_samples"] - 1) or levels[-1] != settings["n_samples"]:
        raise ValueError("n_samples must be a power of two and equal max(n_levels)")
    settings["n_levels"] = levels
    if settings["max_fe"] < settings["pop_size"]:
        raise ValueError("max_fe must be at least pop_size")
    settings["optimization_seeds"] = _seeds(settings["optimization_seeds"], "optimization_seeds")
    settings["noise_seeds"] = _seeds(settings["noise_seeds"], "noise_seeds", allow_empty=True)
    if set(settings["optimization_seeds"]) & set(settings["noise_seeds"]):
        raise ValueError("noise_seeds must be separate from optimization_seeds")
    if settings["noise_scenarios"] and not settings["noise_seeds"]:
        raise ValueError("noise_scenarios requires noise_seeds")
    for name in ("sampling_seed", "bootstrap_seed"):
        settings[name] = _seeds((settings[name],), name)[0]
    for name in ("calc_second_order", "scramble"):
        if type(settings[name]) is not bool:
            raise ValueError(f"{name} must be bool")
    for name in ("confidence_level", "crossover_prob", "mutation_prob"):
        settings[name] = _number(settings[name], name, allow_equal=name != "confidence_level")
        if settings[name] > 1 or (name == "confidence_level" and settings[name] == 1):
            raise ValueError(f"{name} must be {'strictly ' if name == 'confidence_level' else ''}between zero and one")
    for name in ("budget", "game_learning_rate"):
        if settings[name] is not None:
            settings[name] = _number(settings[name], name)
    if settings["game_iterations"] is not None:
        settings["game_iterations"] = _integer(settings["game_iterations"], "game_iterations", 1)
    settings["lambda_loss"] = _number(settings["lambda_loss"], "lambda_loss")
    fractions = settings["budget_fractions"]
    if not isinstance(fractions, (list, tuple)) or not fractions:
        raise ValueError("budget_fractions must be a nonempty list")
    fractions = tuple(_number(value, "budget_fractions") for value in fractions)
    if list(fractions) != sorted(set(fractions)) or fractions[-1] > 1:
        raise ValueError("budget_fractions must be strictly increasing and no greater than one")
    settings["budget_fractions"] = fractions
    settings["parameter_ranges"] = normalize_ranges(settings["parameter_ranges"])
    from common.memetic import MA_PARAMETER_DEFAULTS, MemeticNSGA2

    parameters = settings["ma_parameters"]
    if not isinstance(parameters, dict) or set(parameters) - set(MA_PARAMETER_DEFAULTS):
        raise ValueError("ma_parameters must contain only common MemeticNSGA2 settings")
    parameters = {
        **deepcopy(MA_PARAMETER_DEFAULTS), "max_local_search_iters": 5,
        "max_local_fe_ratio": 0.2, "use_archive": True, **deepcopy(parameters),
    }
    MemeticNSGA2(pop_size=settings["pop_size"], max_fe=settings["max_fe"], **parameters)
    settings["ma_parameters"] = parameters
    return SensitivityConfig(**settings)
