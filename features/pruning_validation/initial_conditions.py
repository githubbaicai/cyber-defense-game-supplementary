"""Compatibility helpers after moving sensitivity diagnostics to E3."""

import numpy as np

from common.case_config import get_input_excel, load_case_parameters
from common.game_theory import replicator_dynamics
from common.problem_factory import load_problem_context


def _local_game(initial_p, initial_r, attacker, defender, detection, has_response,
                iterations, learning_rate):
    """Run one shared local game for callers that still need a regression helper."""
    result = replicator_dynamics(
        initial_p, attacker, defender, detection, iterations=iterations,
        learning_rate=learning_rate, has_response=has_response,
        p_response=float(initial_r[1]) if has_response else 0.0,
    )
    result["r_history"] = np.asarray(result["response_policy_history"])
    for name in ("p_history", "r_history", "q_history"):
        result[name] = np.asarray(result[name])
        history = result[name]
        if not np.isfinite(history).all() or np.any(history < 0) or not np.allclose(
            history.sum(axis=1), 1.0, atol=1e-12, rtol=0.0
        ):
            raise ValueError(f"Invalid normalized probabilities in {name}")
    return result


def _batched_local_games(initializations, task):
    """Preserve the old regression helper without a second dynamics engine."""
    outputs = []
    for _, _, initial_p, initial_r, _ in initializations:
        outputs.append(_local_game(
            initial_p, initial_r, task["attacker"], task["defender"],
            task["detection"], task["has_response"], task["iterations"],
            task["learning_rate"],
        ))
    return outputs


def run_initial_probability_experiments(input_excel=None, *args, **kwargs):
    """Reject the removed duplicate sensitivity experiment after resolving the case."""
    path = get_input_excel(input_excel)
    load_case_parameters(path, names=("Game_Iterations", "Game_Learning_Rate"))
    load_problem_context(path)
    raise RuntimeError(
        "pruning_validation.initial_conditions was removed; use "
        "features.equilibrium_validation for initialization sensitivity"
    )


__all__ = ["_local_game", "_batched_local_games", "run_initial_probability_experiments"]
