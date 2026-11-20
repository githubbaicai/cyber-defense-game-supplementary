"""Build local payoff matrices for pruning-specific diagnostics."""

import numpy as np

from common.config import EXPOSURE_PENALTY_RATIO


def build_matrices(strategies, old_matrix=False):
    attacker, defender, detection, initial = [], [], [], []
    total_benefit = sum(strategy.attack_benefit for strategy in strategies) or 1.0
    for strategy in strategies:
        if hasattr(strategy, "vulnerability"):
            if old_matrix:
                attacker.append([strategy.get_expected_utility()] * 3)
                defender.append([strategy.get_defend_expected_loss()] * 3)
            else:
                attacker.append([
                    strategy.get_utility_not_detected(),
                    strategy.get_utility_detected_no_response(EXPOSURE_PENALTY_RATIO),
                    strategy.get_utility_detected_with_response(EXPOSURE_PENALTY_RATIO),
                ])
                defender.append([
                    strategy.get_defend_loss_not_detected(),
                    strategy.get_defend_loss_detected_no_response(),
                    strategy.get_defend_loss_detected_with_response(
                        strategy.response_operational_disruption_cost
                    ),
                ])
            initial.append(strategy.attack_benefit / total_benefit)
        else:
            attacker.append([0.0, 0.0, 0.0])
            defender.append([0.0, 0.0, 0.0])
            initial.append(0.01)
        detection.append(strategy.detected_prob)
    initial = np.asarray(initial, dtype=float)
    return np.asarray(attacker), np.asarray(defender), np.asarray(detection), initial / initial.sum()


_matrices = build_matrices
