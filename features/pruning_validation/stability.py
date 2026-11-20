"""Attacker equilibria with the final defender response policy frozen.

This is not a coupled two-population Nash/ESS certificate.
Pure points and isolated two-strategy points are enumerated. Degenerate
multi-strategy continua are reported, not presented as a complete enumeration.
"""

import numpy as np
from common.game_theory import conditional_expected_values


def induced_matrix(attacker_matrix, detection, has_response=True, p_response=0.0):
    matrix = np.asarray(attacker_matrix, dtype=float)
    detection = np.asarray(detection, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("Expected three-state attacker matrix")
    values = conditional_expected_values(matrix, detection, p_response, has_response)
    return np.repeat(values[:, None], len(values), axis=1)


def diagnose_distribution(p, attacker_matrix, detection, has_response=True, p_response=0.0):
    p = np.asarray(p, dtype=float)
    payoff = induced_matrix(attacker_matrix, detection, has_response, p_response) @ p
    average = float(p @ payoff)
    positive = p[p > 0]
    return {
        "Effective_Strategies": float(1.0 / np.sum(p ** 2)),
        "Entropy": float(-np.sum(positive * np.log(positive))),
        "Max_Probability": float(p.max()),
        "Min_Positive_Probability": float(positive.min()),
        "Significant_Strategies_1pct": int((p >= 0.01).sum()),
        "Replicator_Residual": float(np.max(np.abs(p * (payoff - average)))),
        "Invasion_Gap": float(payoff.max() - average),
        "Last_Point_Type": "quasi_pure" if p.max() >= 0.99 else "finite_iteration_mixture",
    }


