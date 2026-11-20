"""Three-state evolutionary dynamics using population-average detection."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from common.config import (
    GAME_OPTIMIZATION_ITERATIONS,
    GAME_OPTIMIZATION_LEARNING_RATE,
)
from common.model_options import resolve_model_options
from common.game_termination import (
    DEFAULT_GAME_TERMINATION,
    check_game_convergence, convergence_diagnostics, resolve_game_termination,
    termination_metadata,
)


DEFENDER_STATE_NAMES = (
    "not_detected",
    "detected_no_response",
    "detected_with_response",
)
ALTERNATIVE_STATE_NAMES = (
    "not_detected_no_response",
    "not_detected_with_response",
    "detected_no_response",
    "detected_with_response",
)


def calculate_defender_probabilities(
    p, f, p_response=0.0, has_response=True, *, alternative_trigger=0.0
):
    """Return induced probabilities for the three detection/response states."""
    p = np.asarray(p, dtype=float)
    f = np.asarray(f, dtype=float)
    if p.shape != f.shape:
        raise ValueError("p and f must have the same shape")
    if not np.isfinite(alternative_trigger) or alternative_trigger != 0:
        raise ValueError("alternative_trigger must be 0: response requires detection")
    action_event_probabilities(f, p_response, has_response)
    detection = float(np.clip(np.sum(p * f), 0.0, 1.0))
    response = float(p_response) if has_response else 0.0
    return np.array([
        1.0 - detection,
        detection * (1.0 - response),
        detection * response,
    ], dtype=float)


def calculate_expected_values(q, matrix):
    return np.asarray(matrix, dtype=float) @ np.asarray(q, dtype=float)


def action_event_probabilities(f, p_response=0.0, has_response=True):
    """P(event | attack i); response is possible only after detection."""
    detection = np.asarray(f, dtype=float).reshape(-1)
    if not np.isfinite(detection).all() or np.any((detection < 0) | (detection > 1)):
        raise ValueError("detection probabilities must be in [0, 1]")
    if not np.isfinite(p_response) or not 0 <= p_response <= 1:
        raise ValueError("p_response must be finite and in [0, 1]")
    response = float(p_response) if has_response else 0.0
    return np.column_stack((1.0 - detection, detection * (1.0 - response), detection * response))


def conditional_expected_values(matrix, f, p_response=0.0, has_response=True):
    """Explicit action-conditional comparison; not the population settlement rule."""
    matrix = np.asarray(matrix, dtype=float)
    events = action_event_probabilities(f, p_response, has_response)
    if matrix.shape != events.shape:
        raise ValueError("payoff matrix must have shape (n_strategies, 3)")
    return np.sum(matrix * events, axis=1)


def marginal_expected_values(p, matrix, f, p_response=0.0, has_response=True):
    """Settle all actions against the same population-induced event mix."""
    q = calculate_defender_probabilities(p, f, p_response, has_response)
    return calculate_expected_values(q, matrix)


def response_fitness(p, defender, f, has_response=True):
    """Legacy rule: average detection times population-average losses."""
    if not has_response:
        return np.zeros(2)
    p = np.asarray(p, dtype=float)
    detection = float(p @ np.asarray(f, dtype=float))
    return -detection * (p @ np.asarray(defender, dtype=float)[:, 1:])


def game_scales(attacker, defender):
    """Legacy normalization using every entry of each payoff matrix."""
    return (
        max(float(np.max(np.abs(np.asarray(attacker)))), 1.0),
        max(float(np.max(np.abs(np.asarray(defender)))), 1.0),
    )


def loss_weighted_response_probability(p, defender_loss_matrix):
    """Initialize detected-state frequencies by inverse expected losses."""
    probabilities = np.asarray(p, dtype=float)
    losses = np.asarray(defender_loss_matrix, dtype=float)
    if (
        probabilities.ndim != 1
        or losses.shape != (probabilities.size, 3)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or probabilities.sum() <= 0
        or not np.isfinite(losses).all()
        or np.any(losses < 0)
    ):
        raise ValueError("inverse-loss initialization requires nonnegative finite losses and probabilities")
    scale = float(np.max(losses[:, 1:]))
    if scale == 0:
        return 0.5
    average_losses = (probabilities / probabilities.sum()) @ (losses[:, 1:] / scale)
    total = float(average_losses.sum())
    return float(average_losses[0] / total) if total > 0 else 0.5


def _normalise(values):
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    log_probabilities = shifted - np.log(np.exp(shifted).sum())
    return log_probabilities, np.exp(log_probabilities)


def initialize_dynamics_state(
    p_init, p_response=0.0, has_response=True, *,
    defender_loss_matrix=None, model_options=None,
):
    """Create a log-domain state for the shared synchronous update."""
    p = np.asarray(p_init, dtype=float)
    if p.ndim != 1 or p.size == 0 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("p_init must be a non-negative finite vector")
    if p.sum() <= 0:
        raise ValueError("p_init must contain positive mass")
    p = p / p.sum()
    if not np.isfinite(p_response) or not 0.0 <= p_response <= 1.0:
        raise ValueError("p_response must be finite and in [0, 1]")
    response = float(p_response) if has_response else 0.0
    options = resolve_model_options(model_options)
    if has_response and options.defender_initialization == "inverse_loss":
        response = loss_weighted_response_probability(p, defender_loss_matrix)
    r = np.array([1.0 - response, response], dtype=float)
    log_p = np.full_like(p, -np.inf)
    log_p[p > 0] = np.log(p[p > 0])
    log_r = np.full_like(r, -np.inf)
    log_r[r > 0] = np.log(r[r > 0])
    return {
        "p": p,
        "r": r,
        "log_p": log_p,
        "log_r": log_r,
    }


def _validate_game_arrays(attacker_gain_matrix, defender_loss_matrix, f, p, *, state_count=3):
    attacker = np.asarray(attacker_gain_matrix, dtype=float)
    defender = np.asarray(defender_loss_matrix, dtype=float)
    detection = np.asarray(f, dtype=float).reshape(-1)
    probabilities = np.asarray(p, dtype=float).reshape(-1)
    if (
        attacker.ndim != 2
        or attacker.shape[1] != state_count
        or defender.shape != attacker.shape
        or detection.shape != (attacker.shape[0],)
        or probabilities.shape != detection.shape
    ):
        raise ValueError(f"game matrices must have shape (n_strategies, {state_count})")
    if not np.isfinite(attacker).all() or not np.isfinite(defender).all():
        raise ValueError("game matrices must be finite")
    if not np.isfinite(detection).all() or np.any((detection < 0) | (detection > 1)):
        raise ValueError("detection probabilities must be in [0, 1]")
    return attacker, defender, detection, probabilities


def evolutionary_step(
    state,
    attacker_gain_matrix,
    defender_loss_matrix,
    f,
    *,
    learning_rate,
    has_response=True,
    payoff_scale=None,
    loss_scale=None,
    model_options=None,
):
    """Apply one synchronous log-domain evolutionary update.

    The returned mapping contains the updated ``p``, ``r``, ``log_p`` and
    ``log_r`` together with the payoffs and induced probabilities computed
    from the old state.  Keeping the logs in the state is important for zero
    support and underflow behavior.
    """
    resolve_model_options(model_options)
    if not np.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("learning_rate must be finite and non-negative")
    p = np.asarray(state["p"], dtype=float).reshape(-1)
    r = np.asarray(state["r"], dtype=float).reshape(-1)
    log_p = np.asarray(state["log_p"], dtype=float).reshape(-1)
    log_r = np.asarray(state["log_r"], dtype=float).reshape(-1)
    if r.shape != (2,) or log_p.shape != p.shape or log_r.shape != r.shape:
        raise ValueError("invalid evolutionary state shape")
    attacker, defender, detection, p = _validate_game_arrays(
        attacker_gain_matrix, defender_loss_matrix, f, p,
    )
    if not np.isfinite(r).all() or np.any(r < 0) or not np.isclose(r.sum(), 1.0):
        raise ValueError("response policy must be a probability vector")
    if np.any(np.isnan(log_p)) or np.any(np.isposinf(log_p)) or np.any(np.isnan(log_r)) or np.any(np.isposinf(log_r)):
        raise ValueError("log state must be finite or -inf")
    default_payoff_scale, default_loss_scale = game_scales(attacker, defender)
    if payoff_scale is None:
        payoff_scale = default_payoff_scale
    if loss_scale is None:
        loss_scale = default_loss_scale
    if not np.isfinite(payoff_scale) or payoff_scale <= 0:
        raise ValueError("payoff_scale must be positive and finite")
    if not np.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError("loss_scale must be positive and finite")

    q = calculate_defender_probabilities(p, detection, r[1], has_response)
    attacker_payoffs = calculate_expected_values(q, attacker)
    losses = calculate_expected_values(q, defender)
    avg_payoff = float(p @ attacker_payoffs)
    detection_probability = float(p @ detection)
    fitness = response_fitness(p, defender, detection, has_response)
    avg_fitness = float(r @ fitness)

    next_log_p = log_p + learning_rate * (attacker_payoffs - avg_payoff) / payoff_scale
    next_log_p, next_p = _normalise(next_log_p)
    if has_response:
        next_log_r = log_r + learning_rate * (fitness - avg_fitness) / loss_scale
        next_log_r, next_r = _normalise(next_log_r)
    else:
        next_log_r = log_r
        next_r = np.array([1.0, 0.0], dtype=float)

    q_next = calculate_defender_probabilities(next_p, detection, next_r[1], has_response)
    next_attacker_payoffs = calculate_expected_values(q_next, attacker)
    next_losses = calculate_expected_values(q_next, defender)
    return {
        "p": next_p,
        "r": next_r,
        "log_p": next_log_p,
        "log_r": next_log_r,
        "q": q,
        "q_next": q_next,
        "attacker_payoffs": attacker_payoffs,
        "defender_losses": losses,
        "fitness": fitness,
        "avg_payoff": avg_payoff,
        "avg_fitness": avg_fitness,
        "detection_probability": detection_probability,
        "next_attacker_payoffs": next_attacker_payoffs,
        "next_defender_losses": next_losses,
    }


def build_game_inputs(strategies, *, model_options=None):
    """Extract the matrices, detection vector, and legacy initial ``p``."""
    options = resolve_model_options(model_options)
    attacker_gain = []
    defender_loss = []
    detection = []
    initial = []
    attack_benefits = [
        float(s.attack_benefit)
        for s in strategies
        if hasattr(s, "vulnerability")
    ]
    total_benefit = sum(attack_benefits) or 1.0
    scaled_initialisation = (
        not np.isfinite(total_benefit)
        and bool(attack_benefits)
        and all(np.isfinite(benefit) and benefit >= 0 for benefit in attack_benefits)
    )
    if scaled_initialisation:
        max_benefit = max(attack_benefits)
        scaled_total = sum(benefit / max_benefit for benefit in attack_benefits)
        do_nothing_weight = 0.01 * scaled_total
    else:
        max_benefit = 1.0
        do_nothing_weight = 0.01
    for strategy in strategies:
        if hasattr(strategy, "vulnerability"):
            attacker_row = [
                strategy.get_utility_not_detected(),
                strategy.get_utility_detected_no_response(options.exposure_penalty_ratio),
                strategy.get_utility_detected_with_response(options.exposure_penalty_ratio),
            ]
            defender_row = [
                strategy.get_defend_loss_not_detected(),
                strategy.get_defend_loss_detected_no_response(),
                strategy.get_defend_loss_detected_with_response(
                    strategy.response_operational_disruption_cost
                ),
            ]
            attacker_gain.append(attacker_row)
            defender_loss.append(defender_row)
            detection.append(strategy.detected_prob)
            if scaled_initialisation:
                initial.append(float(strategy.attack_benefit) / max_benefit)
            else:
                initial.append(strategy.attack_benefit / total_benefit)
        else:
            attacker_gain.append([0.0] * 3)
            defender_loss.append([0.0] * 3)
            detection.append(0.0)
            initial.append(do_nothing_weight)
    p_init = np.asarray(initial, dtype=float)
    p_init /= p_init.sum()
    return {
        "attacker_gain_matrix": np.asarray(attacker_gain, dtype=float),
        "defender_loss_matrix": np.asarray(defender_loss, dtype=float),
        "f": np.asarray(detection, dtype=float),
        "p_init": p_init,
    }


def replicator_dynamics(
    p_init,
    attacker_gain_matrix,
    defender_loss_matrix,
    f,
    iterations=GAME_OPTIMIZATION_ITERATIONS,
    learning_rate=GAME_OPTIMIZATION_LEARNING_RATE,
    has_response=True,
    p_response=0.0,
    *,
    capture_diagnostics=False,
    model_options=None,
    game_termination=DEFAULT_GAME_TERMINATION,
):
    """Evolve attacker p and defender conditional response policy r.

    ``p_response`` is the initial probability of responding after detection.
    The response policy evolves from population-average defender losses.  ``q``
    is the induced three-state distribution derived from p, detection f and r.
    Losses are positive damage magnitudes internally; defender fitness is -loss.
    """
    options = resolve_model_options(model_options)
    settings = resolve_game_termination(game_termination)
    state_count = 3
    p = np.asarray(p_init, dtype=float)
    attacker = np.asarray(attacker_gain_matrix, dtype=float)
    defender = np.asarray(defender_loss_matrix, dtype=float)
    f = np.asarray(f, dtype=float)
    if p.ndim != 1 or p.size == 0 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("p_init must be a non-negative finite vector")
    if p.sum() <= 0:
        raise ValueError("p_init must contain positive mass")
    p = p / p.sum()
    if (
        attacker.ndim != 2
        or attacker.shape[1] != state_count
        or defender.shape != attacker.shape
        or f.shape != (attacker.shape[0],)
        or p.shape != f.shape
    ):
        raise ValueError(f"game matrices must have shape (n_strategies, {state_count})")
    if not np.isfinite(attacker).all() or not np.isfinite(defender).all():
        raise ValueError("game matrices must be finite")
    if np.any((f < 0) | (f > 1)) or not np.isfinite(f).all():
        raise ValueError("detection probabilities must be in [0, 1]")
    if int(iterations) != iterations or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("learning_rate must be finite and non-negative")

    if not np.isfinite(p_response) or not 0.0 <= p_response <= 1.0:
        raise ValueError("p_response must be finite and in [0, 1]")
    initial_state = initialize_dynamics_state(
        p, p_response, has_response,
        defender_loss_matrix=defender, model_options=options,
    )
    r = initial_state["r"]
    payoff_scale, loss_scale = game_scales(attacker, defender)
    log_p = np.full_like(p, -np.inf)
    log_p[p > 0] = np.log(p[p > 0])
    log_r = np.full_like(r, -np.inf)
    log_r[r > 0] = np.log(r[r > 0])
    log_p_history = [log_p.copy()] if capture_diagnostics else None
    log_r_history = [log_r.copy()] if capture_diagnostics else None
    initial_p = p.copy()

    def state_q(current_p, current_r):
        return calculate_defender_probabilities(
            current_p, f, current_r[1], has_response,
            alternative_trigger=options.alternative_trigger,
        )

    q_history = [state_q(p, r)]
    p_history = [p.copy()]
    response_policy_history = [r.copy()]
    attacker_payoff_history = [float(p @ attacker @ q_history[0])]
    defender_loss_history = [float(p @ defender @ q_history[0])]
    defender_fitness_history = [-defender_loss_history[0]]

    executed_iterations = 0
    converged = False
    window = {"window_complete": False, "probability_span": None}
    stop_reason = "fixed_policy" if options.game_mode == "fixed_policy" else (
        "iteration_limit" if settings is not None else "fixed_iterations"
    )
    maximum = 0 if options.game_mode == "fixed_policy" else int(iterations)
    for iteration in range(1, maximum + 1):
        step = evolutionary_step(
            {"p": p, "r": r, "log_p": log_p, "log_r": log_r},
            attacker, defender, f,
            learning_rate=learning_rate,
            has_response=has_response,
            payoff_scale=payoff_scale,
            loss_scale=loss_scale,
            model_options=options,
        )
        p, r = step["p"], step["r"]
        log_p, log_r = step["log_p"], step["log_r"]
        q_next = step["q_next"]
        next_attacker = step["next_attacker_payoffs"]
        next_losses = step["next_defender_losses"]
        attacker_payoff_history.append(float(p @ next_attacker))
        defender_loss_history.append(float(p @ next_losses))
        defender_fitness_history.append(float(-(p @ next_losses)))
        p_history.append(p.copy())
        q_history.append(q_next)
        response_policy_history.append(r.copy())
        if capture_diagnostics:
            log_p_history.append(log_p.copy())
            log_r_history.append(log_r.copy())
        executed_iterations = iteration
        should_check = settings is not None and iteration >= settings.min_iterations and (
            (iteration - settings.min_iterations) % settings.check_interval == 0
            or iteration == maximum
        )
        if should_check:
            diagnostics = convergence_diagnostics(
                p, r, next_attacker, response_fitness(p, defender, f, has_response),
                payoff_scale, loss_scale, has_response,
            )
            converged, window = check_game_convergence(
                settings, p_history, response_policy_history, diagnostics, iteration, learning_rate,
                iteration_limit=maximum,
            )
            if converged:
                stop_reason = "converged"
                break

    final_q = q_history[-1]
    final_attacker_payoffs = calculate_expected_values(final_q, attacker)
    final_defender_losses = calculate_expected_values(final_q, defender)
    detection_probability = float(np.clip(p @ f, 0.0, 1.0))
    trigger_probability = detection_probability if has_response else 0.0
    diagnostics = convergence_diagnostics(
        p, r, final_attacker_payoffs, response_fitness(p, defender, f, has_response),
        payoff_scale, loss_scale, has_response,
    )
    result = {
        "final_p": p,
        "final_q": final_q,
        "initial_q": q_history[0],
        "final_response_policy": r,
        "final_response_probability": float(r[1]),
        "final_attacker_payoffs": final_attacker_payoffs,
        "final_defender_losses": final_defender_losses,
        "attacker_payoff": float(p @ final_attacker_payoffs),
        "defender_loss": float(p @ final_defender_losses),
        "p_history": p_history,
        "q_history": q_history,
        "response_policy_history": response_policy_history,
        "attacker_payoff_history": attacker_payoff_history,
        "defender_loss_history": defender_loss_history,
        "defender_fitness_history": defender_fitness_history,
        "payoff_scale": payoff_scale,
        "loss_scale": loss_scale,
        "has_response": bool(has_response),
        "p_response": float(p_response),
        "initial_response_probability": float(response_policy_history[0][1]),
        "defender_initialization": options.defender_initialization,
        "state_names": DEFENDER_STATE_NAMES,
        "iterations_executed": executed_iterations,
        "iteration_limit": int(iterations),
        "stop_reason": stop_reason,
        "converged": converged,
        "convergence_checked": settings is not None and options.game_mode != "fixed_policy",
        "game_termination": termination_metadata(settings),
        **diagnostics,
        **window,
        "detection_probability": detection_probability,
        "response_trigger_probability": trigger_probability,
        "response_execution_probability": trigger_probability * float(r[1]),
        "model_options": asdict(options),
        "payoff_rule": "population_average_detection",
    }
    if capture_diagnostics:
        result.update({
            "game_inputs": {
                "attacker_gain_matrix": attacker.copy(),
                "defender_loss_matrix": defender.copy(),
                "f": f.copy(),
                "p_init": initial_p,
            },
            "log_p_history": log_p_history,
            "log_r_history": log_r_history,
        })
    return result


def run_evolutionary_game(
    strategies,
    has_response=True,
    iterations=GAME_OPTIMIZATION_ITERATIONS,
    learning_rate=GAME_OPTIMIZATION_LEARNING_RATE,
    p_response=0.0,
    *,
    attacker_initializer=None,
    capture_diagnostics=False,
    model_options=None,
    game_termination=DEFAULT_GAME_TERMINATION,
):
    options = resolve_model_options(model_options)
    inputs = build_game_inputs(strategies, model_options=options)
    p_init = inputs["p_init"].copy()
    if attacker_initializer is not None:
        if not callable(attacker_initializer):
            raise TypeError("attacker_initializer must be callable")
        p_init = np.asarray(attacker_initializer(strategies, p_init.copy()), dtype=float)
        if p_init.shape != (len(strategies),) or not np.isfinite(p_init).all() or np.any(p_init < 0) or p_init.sum() <= 0:
            raise ValueError("attacker_initializer must return a valid strategy distribution")
        p_init /= p_init.sum()
    return replicator_dynamics(
        p_init,
        inputs["attacker_gain_matrix"],
        inputs["defender_loss_matrix"],
        inputs["f"],
        iterations=iterations,
        learning_rate=learning_rate,
        has_response=has_response,
        p_response=p_response,
        capture_diagnostics=capture_diagnostics,
        model_options=options,
        game_termination=game_termination,
    )
