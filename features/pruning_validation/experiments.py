"""Compatibility entry point for the removed duplicate equilibrium experiments.

Equilibrium diagnostics, initialization sensitivity, and learning-rate sensitivity
now belong to ``features.equilibrium_validation``. This module keeps the old import
path detectable while refusing to run a second implementation under the pruning
feature.
"""

from common.case_config import get_input_excel, load_case_parameters
from common.problem_factory import load_problem_context
from features.pruning_validation.matrices import build_matrices, _matrices


def run_experiments(input_excel=None, case_id=4):
    """Reject the removed duplicate experiment suite after resolving the case."""
    path = get_input_excel(input_excel)
    load_case_parameters(path, names=("Game_Iterations", "Game_Learning_Rate"))
    load_problem_context(path)
    raise RuntimeError(
        "pruning_validation.experiments was removed; use "
        "features.equilibrium_validation for equilibrium and sensitivity diagnostics"
    )


__all__ = ["build_matrices", "_matrices", "run_experiments"]
