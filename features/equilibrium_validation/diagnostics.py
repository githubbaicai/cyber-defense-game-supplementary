"""Diagnostics for the synchronous map and frozen-response induced game."""

from __future__ import annotations

import json

import numpy as np

from common.game_theory import (
    action_event_probabilities, marginal_expected_values, evolutionary_step, game_scales,
    initialize_dynamics_state, response_fitness,
)


DEFAULT_THRESHOLDS = {
    "map_tol": 1e-8, "fitness_tol": 1e-8, "gap_tol": 1e-8,
    "support_tol": 1e-8, "tail_tol": 1e-8, "tail_window": 10,
    "jacobian_step": 1e-6, "jacobian_tol": 1e-5, "stability_margin": 1e-4,
}


def resolve_thresholds(values=None):
    supplied = values or {}
    if set(supplied) - set(DEFAULT_THRESHOLDS):
        raise ValueError("unknown diagnostic thresholds")
    resolved = {**DEFAULT_THRESHOLDS, **supplied}
    for name, value in resolved.items():
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid {name}")
    if resolved["tail_window"] < 1 or int(resolved["tail_window"]) != resolved["tail_window"]:
        raise ValueError("tail_window must be a positive integer")
    if not 0 < resolved["jacobian_step"] < 0.1:
        raise ValueError("jacobian_step must be in (0, 0.1)")
    if resolved["support_tol"] >= 0.5:
        raise ValueError("support_tol must be less than 0.5")
    return resolved


def _state(p, r, has_response, log_p=None, log_r=None):
    p, r = np.asarray(p, dtype=float), np.asarray(r, dtype=float)
    if p.ndim != 1 or not p.size or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
        raise ValueError("p must be a probability vector")
    if r.shape != (2,) or not np.isfinite(r).all() or (r < 0).any() or not np.isclose(r.sum(), 1):
        raise ValueError("r must be a two-action probability vector")
    if not has_response and not np.array_equal(r, [1.0, 0.0]):
        raise ValueError("no-response games require r=(1,0)")
    state = initialize_dynamics_state(p, r[1], has_response)
    state["p"], state["r"] = p.copy(), r.copy()
    for name, value in (("log_p", log_p), ("log_r", log_r)):
        if value is not None:
            array = np.asarray(value, dtype=float)
            expected = p if name == "log_p" else r
            if array.shape != expected.shape or np.isnan(array).any() or np.isposinf(array).any():
                raise ValueError(f"invalid {name}")
            if not np.allclose(np.exp(array), expected, atol=1e-12, rtol=1e-8):
                raise ValueError(f"{name} disagrees with displayed probabilities")
            state[name] = array.copy()
    return state


def map_residual(state, attacker, defender, detection, learning_rate, has_response):
    updated = evolutionary_step(state, attacker, defender, detection,
                                learning_rate=learning_rate, has_response=has_response)
    residual_p = float(np.max(np.abs(updated["p"] - state["p"])))
    residual_r = float(np.max(np.abs(updated["r"] - state["r"]))) if has_response else 0.0
    return max(residual_p, residual_r), residual_p, residual_r


def induced_attacker_matrix(attacker, detection, response):
    """Frozen response: M @ p equals A @ q(p) under average detection."""
    events = action_event_probabilities(detection, response)
    return np.asarray(attacker, dtype=float) @ events.T


def _jacobian(p, r, attacker, defender, detection, eta, has_response, step, support_only, true_support_p, true_support_r):
    pivot = int(np.argmax(p))
    attackers = [index for index in range(len(p)) if index != pivot and (true_support_p[index] or not support_only)]
    response_coordinate = has_response and (not support_only or np.all(true_support_r))
    coordinates = [("p", index) for index in attackers] + ([("r", 1)] if response_coordinate else [])

    def evaluate(current_p, current_r):
        state = _state(current_p, current_r, has_response)
        updated = evolutionary_step(state, attacker, defender, detection,
                                    learning_rate=eta, has_response=has_response)
        return np.asarray([updated["p"][index] if population == "p" else updated["r"][1]
                           for population, index in coordinates])

    base = evaluate(p, r)
    jacobian = np.empty((len(coordinates), len(coordinates)))
    schemes = []
    for column, (population, index) in enumerate(coordinates):
        plus_p, minus_p, plus_r, minus_r = p.copy(), p.copy(), r.copy(), r.copy()
        if population == "p":
            can_plus, can_minus = p[pivot] >= step, p[index] >= step
            plus_p[index] += step
            plus_p[pivot] -= step
            minus_p[index] -= step
            minus_p[pivot] += step
        else:
            can_plus, can_minus = r[0] >= step, r[1] >= step
            plus_r += np.array([-step, step])
            minus_r += np.array([step, -step])
        if can_plus and can_minus:
            jacobian[:, column] = (evaluate(plus_p, plus_r) - evaluate(minus_p, minus_r)) / (2 * step)
            schemes.append("central")
        elif can_plus:
            jacobian[:, column] = (evaluate(plus_p, plus_r) - base) / step
            schemes.append("forward_feasible")
        elif can_minus:
            jacobian[:, column] = (base - evaluate(minus_p, minus_r)) / step
            schemes.append("backward_feasible")
        else:
            raise FloatingPointError("no feasible finite-difference step")
    return jacobian, schemes


def _stability(state, attacker, defender, detection, eta, has_response, threshold, stationary, underflow):
    p, r = state["p"], state["r"]
    true_support_p, true_support_r = np.isfinite(state["log_p"]), np.isfinite(state["log_r"])
    rows, full_radius = [], None
    status, reason = "neutral_or_undetermined", ""
    for scope, support_only in (("full_feasible_simplex", False), ("within_true_support", True)):
        step = threshold["jacobian_step"]
        coarse, schemes = _jacobian(p, r, attacker, defender, detection, eta, has_response, step, support_only, true_support_p, true_support_r)
        fine, _ = _jacobian(p, r, attacker, defender, detection, eta, has_response, step / 2, support_only, true_support_p, true_support_r)
        error = float(np.max(np.abs(coarse - fine))) if fine.size else 0.0
        reliable = error <= threshold["jacobian_tol"] * max(1.0, float(np.max(np.abs(fine))) if fine.size else 1.0)
        reliable = reliable and not underflow
        eigenvalues = np.linalg.eigvals(fine) if fine.size else np.array([], dtype=complex)
        radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
        if not support_only:
            full_radius = radius
            if not reliable:
                status, reason = "numerically_undetermined", "Jacobian step sensitivity or underflow"
            elif eta == 0:
                status, reason = "neutral_or_undetermined", "zero learning rate is the identity map"
            elif stationary != "yes":
                status, reason = "not_stationary", "Jacobian at a finite nonstationary state cannot certify equilibrium stability"
            elif not eigenvalues.size:
                status, reason = "locally_asymptotically_stable", "empty feasible tangent space: isolated feasible state"
            elif radius < 1 - threshold["stability_margin"]:
                status, reason = "locally_asymptotically_stable", "reliable discrete spectral radius below one"
            elif radius > 1 + threshold["stability_margin"]:
                status, reason = "unstable", "reliable discrete spectral radius above one"
            else:
                reason = "spectral radius near one; linearization is inconclusive"
        for value in eigenvalues:
            rows.append({
                "Jacobian_Type": "Discrete", "Coordinate_Scope": scope,
                "Eigenvalue_Index": len(rows), "Eigenvalue_Real": float(value.real),
                "Eigenvalue_Imag": float(value.imag), "Spectral_Radius": radius,
                "Step_Size": step / 2, "Step_Comparison_Error": error,
                "Difference_Schemes": json.dumps(schemes), "Status": "reliable" if reliable else "numerically_undetermined",
            })
    return status, full_radius, reason, rows


def diagnose_game(attacker_gain_matrix, defender_loss_matrix, f, p, r, *, learning_rate,
                  has_response=True, log_p=None, log_r=None, p_history=None,
                  r_history=None, thresholds=None):
    """Evaluate independent properties without promoting stationarity to Nash or ESS."""
    threshold = resolve_thresholds(thresholds)
    attacker, defender, detection = (np.asarray(value, dtype=float) for value in (attacker_gain_matrix, defender_loss_matrix, f))
    state = _state(p, r, has_response, log_p, log_r)
    p, r = state["p"], state["r"]
    residual, residual_p, residual_r = map_residual(state, attacker, defender, detection, learning_rate, has_response)
    payoffs = marginal_expected_values(p, attacker, detection, r[1], has_response)
    fitness = response_fitness(p, defender, detection, has_response)
    attacker_scale, loss_scale = game_scales(attacker, defender)
    avg_payoff, avg_fitness = float(p @ payoffs), float(r @ fitness)
    attacker_gap = max(0.0, float(np.max(payoffs) - avg_payoff))
    response_gap = max(0.0, float(np.max(fitness) - avg_fitness)) if has_response else 0.0
    residual_fitness_p = float(np.max(np.abs(p * (payoffs - avg_payoff)))) / attacker_scale
    residual_fitness_r = float(np.max(np.abs(r * (fitness - avg_fitness)))) / loss_scale if has_response else 0.0
    residual_fitness = max(residual_fitness_p, residual_fitness_r)
    underflow = bool(np.any((p == 0) & np.isfinite(state["log_p"])) or
                     (has_response and np.any((r == 0) & np.isfinite(state["log_r"]))))
    stationary = "yes" if residual <= threshold["map_tol"] and residual_fitness <= threshold["fitness_tol"] else "no"
    if learning_rate == 0:
        stationary = "undetermined"
    if underflow:
        stationary = "numerically_undetermined"
    induced = induced_attacker_matrix(attacker, detection, r[1])
    induced_payoffs = induced @ p
    induced_gap = max(0.0, float(np.max(induced_payoffs) - p @ induced_payoffs))
    nash = "yes" if induced_gap / attacker_scale <= threshold["gap_tol"] else "no"
    ess, ess_reason = "undetermined", "only the strict pure sufficient condition is implemented"
    if nash == "no":
        ess, ess_reason = "no", "induced Nash condition fails"
    elif np.count_nonzero(p) == 1:
        resident = int(np.argmax(p))
        mutants = np.delete(induced[:, resident], resident)
        if not mutants.size or np.all((induced[resident, resident] - mutants) / attacker_scale > threshold["gap_tol"]):
            ess, ess_reason = "yes", "strict pure ESS sufficient condition in the frozen-response attacker game"
    if underflow:
        nash = "numerically_undetermined"
        ess, ess_reason = "undetermined", "displayed zero has finite internal log mass"
    stability, radius, stability_reason, eigenvalues = _stability(
        state, attacker, defender, detection, learning_rate, has_response, threshold, stationary, underflow)
    tail_change, tail_span = None, None
    if p_history is not None and len(p_history) > 1:
        count = min(int(threshold["tail_window"]) + 1, len(p_history))
        tail = np.asarray(p_history[-count:])
        tail_change = float(np.max(np.abs(np.diff(tail, axis=0))))
        tail_span = float(np.max(np.ptp(tail, axis=0)))
        if r_history is not None and has_response:
            tail_r = np.asarray(r_history[-count:])
            tail_change = max(tail_change, float(np.max(np.abs(np.diff(tail_r, axis=0)))))
            tail_span = max(tail_span, float(np.max(np.ptp(tail_r, axis=0))))
    diagnostics = {
        "R_map": residual, "R_map_p": residual_p, "R_map_r": residual_r,
        "Fitness_Residual": residual_fitness, "Attacker_Fitness_Residual": residual_fitness_p,
        "Response_Fitness_Residual": residual_fitness_r,
        "Tail_Variation": tail_change, "Tail_Span": tail_span,
        "Tail_Status": "not_available" if tail_span is None else "stable" if tail_span <= threshold["tail_tol"] else "changing",
        "Invasion_Gap": max(attacker_gap, response_gap), "Attacker_Invasion_Gap": attacker_gap,
        "Response_Invasion_Gap": response_gap if has_response else None,
        "Normalized_Invasion_Gap": max(attacker_gap / attacker_scale, response_gap / loss_scale),
        "Invasion_Status": "resistant_within_tolerance" if max(attacker_gap / attacker_scale, response_gap / loss_scale) <= threshold["gap_tol"] else "invadable",
        "Induced_Nash_Gap": induced_gap, "Stationary_Status": stationary, "Induced_Nash_Status": nash,
        "Stationary_Reason": "identity map at eta=0" if learning_rate == 0 else "unresolved log-domain underflow" if underflow else "map and normalized weighted fitness residuals checked independently of invasion and tail",
        "Jacobian_Coordinate_Dimension": len(p) - 1 + int(has_response),
        "Response_Action_Count": 2 if has_response else 1,
        "Detection_Probability": float(p @ detection),
        "Map_Learning_Rate": learning_rate,
        "Joint_Nash_Status": "not_established", "ESS_Status": ess, "ESS_Reason": ess_reason,
        "Discrete_Stability_Status": stability, "Spectral_Radius": radius,
        "Support_Size": int(np.count_nonzero(p > threshold["support_tol"])),
        "True_Support_Size": int(np.count_nonzero(np.isfinite(state["log_p"]))),
        "Attack_Advantages_JSON": json.dumps((payoffs - avg_payoff).tolist()),
        "Response_Advantages_JSON": json.dumps((fitness - avg_fitness).tolist()) if has_response else "[]",
        "Initial_Zero_Actions_Checked": True, "Numerical_State_Status": "underflow" if underflow else "resolved",
        "Game_Scope": "frozen_response_population_average_attacker_game; stability=joint_synchronous_map",
        "Reason": ess_reason + "; " + stability_reason,
    }
    return {"diagnostics": diagnostics, "eigenvalues": eigenvalues}
