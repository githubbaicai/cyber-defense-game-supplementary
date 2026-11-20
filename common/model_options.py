"""Immutable, per-run choices for the shared threat model."""

from dataclasses import dataclass
import math

from common.config import EXPOSURE_PENALTY_RATIO, RESPONSE_PENALTY_RATIO


@dataclass(frozen=True)
class ModelOptions:
    game_mode: str = "evolutionary"
    history_mode: str = "elapsed"
    alternative_trigger: float = 0.0
    exposure_penalty_ratio: float = EXPOSURE_PENALTY_RATIO
    response_effectiveness_multiplier: float = 1.0
    response_penalty_ratio: float = RESPONSE_PENALTY_RATIO
    defender_initialization: str = "configured"

    def __post_init__(self):
        if self.game_mode not in ("evolutionary", "fixed_policy"):
            raise ValueError("game_mode must be evolutionary or fixed_policy")
        if self.history_mode not in ("elapsed", "local_step"):
            raise ValueError("history_mode must be elapsed or local_step")
        if self.defender_initialization not in ("configured", "inverse_loss"):
            raise ValueError("defender_initialization must be configured or inverse_loss")
        for name in ("alternative_trigger", "exposure_penalty_ratio", "response_effectiveness_multiplier", "response_penalty_ratio"):
            value = getattr(self, name)
            if isinstance(value, (bool, str)):
                raise ValueError(f"{name} must be a finite non-negative number")
            try:
                valid = math.isfinite(value) and value >= 0
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.alternative_trigger != 0:
            raise ValueError("alternative_trigger must be 0: response requires detection")


def resolve_model_options(model_options=None):
    if model_options is None:
        return ModelOptions()
    if not isinstance(model_options, ModelOptions):
        raise TypeError("model_options must be ModelOptions or None")
    return model_options
