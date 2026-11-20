"""Per-run local-game stopping rules, independent of optimizer termination."""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GameTermination:
    min_iterations: int = 100
    stable_iterations: int = 10
    check_interval: int = 10
    probability_tolerance: float = 1e-7
    residual_tolerance: float = 1e-6
    invasion_tolerance: float = 1e-6

    def __post_init__(self):
        for name in ("min_iterations", "stable_iterations", "check_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("probability_tolerance", "residual_tolerance", "invasion_tolerance"):
            value = getattr(self, name)
            try:
                valid = not isinstance(value, (bool, str)) and math.isfinite(value) and value >= 0
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise ValueError(f"{name} must be finite and non-negative")


# Shared local-game default.  Callers may still pass ``None`` explicitly to
# request the historical fixed-iteration behavior.
DEFAULT_GAME_TERMINATION = GameTermination()


def resolve_game_termination(value=None):
    if value is None or isinstance(value, GameTermination):
        return value
    if isinstance(value, dict):
        try:
            return GameTermination(**value)
        except TypeError as error:
            raise ValueError(f"invalid game_termination settings: {error}") from error
    raise TypeError("game_termination must be None, a mapping or GameTermination")


def convergence_diagnostics(p, r, payoffs, fitness, payoff_scale, loss_scale, has_response):
    """Check stationarity and profitable deviations, including zero support."""
    advantage = (payoffs - float(p @ payoffs)) / payoff_scale
    residual = float(np.max(np.abs(p * advantage)))
    gap = max(0.0, float(np.max(advantage)))
    if has_response:
        advantage = (fitness - float(r @ fitness)) / loss_scale
        residual = max(residual, float(np.max(np.abs(r * advantage))))
        gap = max(gap, float(np.max(advantage)))
    return {"stationarity_residual": residual, "invasion_gap": gap}


def check_game_convergence(
    settings, p_history, r_history, diagnostics, iteration, learning_rate, *, iteration_limit=None
):
    if settings is None:
        return False, {"window_complete": False, "probability_span": None}
    if iteration < settings.min_iterations or (
        (iteration - settings.min_iterations) % settings.check_interval != 0
        and iteration != iteration_limit
    ):
        return False, {"window_complete": False, "probability_span": None}
    count = settings.stable_iterations + 1
    complete = len(p_history) >= count and iteration >= settings.min_iterations
    span = None
    if complete:
        span = max(
            float(np.ptp(np.asarray(p_history[-count:]), axis=0).max()),
            float(np.ptp(np.asarray(r_history[-count:]), axis=0).max()),
        )
    converged = (
        complete and learning_rate > 0
        and span <= settings.probability_tolerance
        and diagnostics["stationarity_residual"] <= settings.residual_tolerance
        and diagnostics["invasion_gap"] <= settings.invasion_tolerance
    )
    return bool(converged), {"window_complete": complete, "probability_span": span}


def termination_metadata(settings):
    return None if settings is None else asdict(settings)
