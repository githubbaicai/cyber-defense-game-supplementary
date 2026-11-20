"""Independently reconcile the current choice_history calculation."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math

import numpy as np

from common.models import DoNothingStrategy
from common.simulation import compute_total_expected_values


def _finite_number(value, name):
    if isinstance(value, (bool, str)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _vector(value, size, name):
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite vector of length {size}") from exc
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return values


def validate_scenario(scenario_id, results, reference_loss=None, *, atol=1e-10,
                      rtol=1e-9, max_trajectories=10000, trajectory_mode="actions"):
    """Compare production totals with independent fixed-policy trajectories.

    Histories describe component choices, not successful exploits. Local losses
    are already conditional expected losses from each frozen game. Enumerating
    a DoNothing action or a history without strategies ends a trajectory.
    Every attack action must have its complete child history in ``results``;
    production integrations should therefore disable pruning explicitly.
    Component trajectories merge actions sharing the same child history using
    their conditional expected loss, preserving the total expected loss.
    """
    atol = _finite_number(atol, "atol")
    rtol = _finite_number(rtol, "rtol")
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")
    if isinstance(max_trajectories, bool) or not isinstance(max_trajectories, (int, np.integer)) or max_trajectories < 1:
        raise ValueError("max_trajectories must be a positive integer")
    if trajectory_mode not in ("actions", "components"):
        raise ValueError("trajectory_mode must be actions or components")
    if reference_loss is not None:
        reference_loss = _finite_number(reference_loss, "reference_loss")
    if not isinstance(results, Mapping) or "S" not in results:
        raise ValueError("results must contain the root history S")

    histories = {}
    children = {history: set() for history in results}
    for history, information in results.items():
        if not isinstance(history, str) or not isinstance(information, Mapping):
            raise ValueError("each history must be a string with a result mapping")
        components = history.split(" → ")
        if components[0] != "S" or any(not component or "→" in component for component in components):
            raise ValueError(f"invalid component history: {history!r}")
        if len(set(components)) != len(components):
            raise ValueError(f"cycle in component history: {history}")
        if history != "S":
            parent_history = " → ".join(components[:-1])
            if parent_history not in results:
                raise ValueError(f"orphan history: {history}")
            children[parent_history].add(history)
        histories[history] = components

    ordered_histories = sorted(histories, key=lambda history: (len(histories[history]), history))
    actions_by_history = {}
    action_rows = []
    for history in ordered_histories:
        information = results[history]
        strategies = information.get("strategies", [])
        if not isinstance(strategies, (list, tuple)):
            raise ValueError(f"{history}: strategies must be a list or tuple")
        probabilities = _vector(information.get("attacker_strategy", []), len(strategies),
                                f"{history}: attacker_strategy")
        if np.any(probabilities < 0):
            raise ValueError(f"{history}: attacker probabilities must be non-negative")
        if strategies and not math.isclose(math.fsum(probabilities), 1.0, abs_tol=atol, rel_tol=rtol):
            raise ValueError(f"{history}: attacker probabilities must sum to one")
        game_result = information.get("game_result")
        if strategies or game_result is not None:
            if not isinstance(game_result, Mapping) or "final_defender_losses" not in game_result:
                raise ValueError(f"{history}: missing game_result.final_defender_losses")
            losses = _vector(game_result["final_defender_losses"], len(strategies),
                             f"{history}: final_defender_losses")
        else:
            losses = np.empty(0, dtype=float)
        if not strategies and children[history]:
            raise ValueError(f"{history}: a history without strategies has child histories")

        actions = []
        referenced_children = set()
        for action_id, (strategy, probability, loss) in enumerate(zip(strategies, probabilities, losses), start=1):
            probability, loss = float(probability), float(loss)
            if isinstance(strategy, DoNothingStrategy):
                if loss != 0.0:
                    raise ValueError(f"{history}: DoNothing must have zero local loss")
                target_component, child_history, action_name = "", None, "Do Nothing"
            else:
                target_component = getattr(strategy, "target_component", None)
                if not isinstance(target_component, str) or not target_component or "→" in target_component:
                    raise ValueError(f"{history}: unsupported attack action {action_id}")
                if target_component in histories[history]:
                    raise ValueError(f"{history}: attack action {action_id} creates a cycle")
                child_history = f"{history} → {target_component}"
                if child_history not in results:
                    raise ValueError(f"{history}: missing child history {child_history}")
                referenced_children.add(child_history)
                vulnerability = getattr(getattr(strategy, "vulnerability", None), "name", None)
                action_name = f"Attack {target_component}"
                if vulnerability is not None:
                    action_name += f" ({vulnerability})"
            actions.append({
                "id": action_id, "name": action_name, "target": target_component,
                "child": child_history, "probability": probability, "loss": loss,
            })
            action_rows.append({
                "Scenario_ID": scenario_id, "History": history, "Action_ID": action_id,
                "Action": action_name, "Target_Component": target_component,
                "Conditional_Probability": probability, "Local_Expected_Loss": loss,
            })
        if referenced_children != children[history]:
            unreferenced = sorted(children[history] - referenced_children)
            raise ValueError(f"{history}: unreachable child histories {unreferenced}")
        actions_by_history[history] = actions

    summary_rows = []

    def compare(check, actual, expected, detail):
        actual = _finite_number(actual, f"{check}: actual")
        expected = _finite_number(expected, f"{check}: expected")
        tolerance = atol + rtol * abs(expected)
        error = abs(actual - expected)
        summary_rows.append({
            "Scenario_ID": scenario_id, "Check": check, "Actual": actual,
            "Expected": expected, "Abs_Error": error, "Tolerance": tolerance,
            "Status": "PASS" if error <= tolerance else "FAIL",
            "Detail": f"trajectory_mode={trajectory_mode}; {detail}",
        })

    _, production_loss, production_reach = compute_total_expected_values(results)
    reach = {"S": 1.0}
    prefix_rows = []
    compare("reach_probability", production_reach.get("S", float("nan")), 1.0,
            "choice_history root S")
    for history in ordered_histories:
        if history not in reach:
            raise ValueError(f"unreachable history: {history}")
        grouped_actions = {}
        for action in actions_by_history[history]:
            if action["child"] is not None:
                grouped_actions.setdefault(action["child"], []).append(action)
        for child_history, actions in sorted(grouped_actions.items()):
            probability = math.fsum(action["probability"] for action in actions)
            weighted_loss = math.fsum(action["probability"] * action["loss"] for action in actions)
            child_probability = reach[history] * probability
            reach[child_history] = child_probability
            contribution = reach[history] * weighted_loss
            for name, value in (("child probability", child_probability), ("weighted loss", weighted_loss),
                                ("expected contribution", contribution)):
                _finite_number(value, f"{child_history}: {name}")
            prefix_rows.append({
                "Scenario_ID": scenario_id, "Parent_History": history,
                "Target_Component": histories[child_history][-1], "Child_History": child_history,
                "Parent_Probability": reach[history], "Conditional_Probability": probability,
                "Child_Probability": child_probability, "Local_Weighted_Loss": weighted_loss,
                "Expected_Contribution": contribution,
            })
            if "defender_loss" not in results[child_history]:
                raise ValueError(f"{child_history}: missing stored defender_loss")
            compare("prefix_storage", results[child_history]["defender_loss"], weighted_loss,
                    f"choice_history local weighted loss: {child_history}")
            compare("reach_probability", production_reach.get(child_history, float("nan")), child_probability,
                    f"choice_history prefix: {child_history}")

    trajectory_actions = actions_by_history
    if trajectory_mode == "components":
        trajectory_actions = {}
        for history, actions in actions_by_history.items():
            groups = {}
            for action in actions:
                groups.setdefault(action["child"], []).append(action)
            merged = []
            for child_history, group in groups.items():
                probability = math.fsum(action["probability"] for action in group)
                if probability == 0.0:
                    continue
                loss = math.fsum(action["probability"] * action["loss"] for action in group) / probability
                _finite_number(loss, f"{history}: conditional component expected loss")
                target = group[0]["target"]
                merged.append({
                    "id": len(merged) + 1, "child": child_history,
                    "probability": probability, "loss": loss,
                    "name": f"Component {target} (conditional expected loss)" if child_history is not None else "Do Nothing",
                })
            trajectory_actions[history] = merged

    terminal_rows = []
    pending = [("S", 1.0, 0.0, ())]
    while pending:
        history, trajectory_probability, cumulative_loss, trajectory = pending.pop()
        actions = trajectory_actions[history]
        if not actions:
            endings = [(trajectory_probability, cumulative_loss, trajectory, "candidate_history_end")]
        else:
            endings = []
            for action in reversed(actions):
                if action["probability"] == 0.0:
                    continue
                next_probability = trajectory_probability * action["probability"]
                if next_probability == 0.0:
                    raise ValueError(f"trajectory probability underflow at {history}")
                next_loss = cumulative_loss + action["loss"]
                _finite_number(next_loss, f"{history}: cumulative expected loss")
                next_trajectory = (*trajectory, f"{history} :: {action['id']}: {action['name']}")
                if action["child"] is None:
                    endings.append((next_probability, next_loss, next_trajectory, "do_nothing"))
                else:
                    pending.append((action["child"], next_probability, next_loss, next_trajectory))
        for probability, loss, trajectory, reason in endings:
            if len(terminal_rows) >= max_trajectories:
                raise ValueError(f"trajectory enumeration exceeds max_trajectories={max_trajectories}")
            contribution = probability * loss
            _finite_number(contribution, f"{history}: terminal expected contribution")
            terminal_rows.append({
                "Scenario_ID": scenario_id, "Trajectory_ID": len(terminal_rows) + 1,
                "Trajectory_Mode": trajectory_mode,
                "Action_Trajectory": json.dumps(trajectory, ensure_ascii=False),
                "Component_History": history, "Stop_Reason": reason, "Probability": probability,
                "Cumulative_Expected_Loss": loss, "Expected_Contribution": contribution,
            })

    terminal_loss = math.fsum(row["Expected_Contribution"] for row in terminal_rows)
    prefix_loss = math.fsum(row["Expected_Contribution"] for row in prefix_rows)
    terminal_probability = math.fsum(row["Probability"] for row in terminal_rows)
    compare("production_vs_terminal", production_loss, terminal_loss,
            "choice_history production total versus independent action trajectories")
    compare("prefix_vs_terminal", prefix_loss, terminal_loss,
            "choice_history unique prefix increments versus independent action trajectories")
    compare("terminal_probability_sum", terminal_probability, 1.0,
            "choice_history DoNothing and candidate-history-end trajectories")
    if reference_loss is not None:
        compare("reference_vs_production", reference_loss, production_loss,
                "choice_history supplied hand calculation versus production total")
        compare("reference_vs_terminal", reference_loss, terminal_loss,
                "choice_history supplied hand calculation versus independent trajectories")
    return {
        "action_probabilities": action_rows, "prefix_contributions": prefix_rows,
        "terminal_trajectories": terminal_rows, "validation_summary": summary_rows,
    }
