"""Find the most probable complete successful vulnerability sequence.

The production model conditions games on component histories, so vulnerability
choices within a fixed component history do not change its later games. We can
maximize each step within every complete component path, then compare paths.
This is not a greedy choice between different next components.
"""

import json
import math

import pandas as pd


STEP_COLUMNS = [
    "Step", "State_Path", "Source_Component", "Target_Component",
    "Vulnerability_ID", "Action_Probability", "Final_Response_Probability",
    "Effective_Success_Probability", "Step_Probability", "Cumulative_Probability",
    "Source_Node_ID", "Vulnerability_Node_ID", "Target_Node_ID", "Is_Display_Only",
]
CANDIDATE_COLUMNS = [
    "Component_Path", "Vulnerability_Path", "Path_Probability", "Log_Path_Probability",
    "Component_Selection_Probability", "Selected", "Status",
]


def _node_id(kind, *names):
    return json.dumps([kind, *names], ensure_ascii=False, separators=(",", ":"))


def _probability(value, context):
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context} must be a finite probability") from exc
    if not math.isfinite(probability) or not -1e-12 <= probability <= 1 + 1e-12:
        raise ValueError(f"{context} must be finite and in [0, 1]")
    return min(1.0, max(0.0, probability))


def _best_action(info, target, state):
    strategies = info.get("strategies", [])
    probabilities = info.get("attacker_strategy", [])
    if len(strategies) != len(probabilities):
        raise ValueError(f"{state}: strategy and probability lengths differ")
    probabilities = [_probability(value, f"{state} attack probability") for value in probabilities]
    if sum(probabilities) > 1 + 1e-9:
        raise ValueError(f"{state}: attack probability mass exceeds one")
    actions = [(strategy, probability) for strategy, probability in zip(strategies, probabilities)
               if getattr(strategy, "target_component", None) == target]
    if not actions:
        return None
    response = _probability(info.get("game_result", {}).get("final_response_probability"),
                            f"{state} final conditional response probability")
    choices = []
    for strategy, probability in actions:
        success = _probability(strategy.compute_success_probability(p_response=response),
                               f"{state} effective success probability")
        name = str(strategy.vulnerability.name)
        weight = probability * success
        log_weight = (math.log(probability) + math.log(success)
                      if probability > 0 and success > 0 else -math.inf)
        choices.append((log_weight, name, probability, success, weight))
    # Exact ties use the vulnerability identity, independent of input row order.
    log_weight, name, probability, success, weight = min(choices, key=lambda row: (-row[0], row[1]))
    return name, probability, response, success, weight, log_weight


def analyze_paths(results, attack_paths, availability):
    """Compare complete S-to-T paths without renormalizing stopped attacks.

    ``attack_paths`` is the original topology's complete component paths, not an
    arbitrary list of reachable prefixes. T is appended as a display-only transition
    only when a recorded terminal game has no attack action targeting T. The shared
    simulation creates such a game for a vulnerability-free topology endpoint.

    ``availability`` is recorded as ``Component_Selection_Probability``: under the
    current shared model it multiplies component action masses, not success rates.
    """
    candidates = []
    complete = sorted({tuple(path) for path in attack_paths
                       if len(path) >= 2 and path[0] == "S" and path[-1] == "T"})
    for path in complete:
        if len(path) != len(set(path)):
            raise ValueError("Complete component paths must not contain a cycle")
        steps, vulnerabilities = [], []
        log_probability = 0.0
        status = "ok"
        last_simulated_state = "S"
        for index, (source, target) in enumerate(zip(path, path[1:]), start=1):
            state = " → ".join(path[:index])
            info = results.get(state)
            if info is None:
                status, log_probability = "target_unreachable", -math.inf
                break
            action = _best_action(info, target, state)
            display_only = (target == "T" and action is None and "game_result" in info)
            if action is None and not display_only:
                status, log_probability = "target_unreachable", -math.inf
                break
            if display_only:
                name, probability, response, success, weight, log_weight = "", 1.0, 0.0, 1.0, 1.0, 0.0
            else:
                name, probability, response, success, weight, log_weight = action
                vulnerabilities.append(name)
                last_simulated_state = " → ".join(path[:index + 1])
            if not math.isfinite(log_weight):
                log_probability = -math.inf
                status = "zero_probability"
            elif math.isfinite(log_probability):
                log_probability += log_weight
            steps.append({
                "Step": index,
                "State_Path": state,
                "Source_Component": source,
                "Target_Component": target,
                "Vulnerability_ID": name,
                "Action_Probability": probability,
                "Final_Response_Probability": response,
                "Effective_Success_Probability": success,
                "Step_Probability": weight,
                "Cumulative_Probability": math.exp(log_probability),
                "Source_Node_ID": _node_id("component", source),
                "Vulnerability_Node_ID": _node_id("vulnerability", target, name) if name else "",
                "Target_Node_ID": _node_id("component", target),
                "Is_Display_Only": display_only,
            })
        component_probability = _probability(availability.get(last_simulated_state, 0.0),
                                             "Component selection probability")
        row = {
            "Component_Path": " → ".join(path),
            "Vulnerability_Path": " → ".join(vulnerabilities),
            "Path_Probability": math.exp(log_probability),
            "Log_Path_Probability": log_probability,
            "Component_Selection_Probability": component_probability,
            "Selected": False,
            "Status": status,
        }
        candidates.append((row, steps))

    positive = [candidate for candidate in candidates
                if math.isfinite(candidate[0]["Log_Path_Probability"])]
    if positive:
        selected, steps = min(positive, key=lambda item: (
            -item[0]["Log_Path_Probability"], item[0]["Component_Path"], item[0]["Vulnerability_Path"]))
        selected["Selected"] = True
        summary = {key: selected[key] for key in CANDIDATE_COLUMNS if key != "Selected"}
    else:
        status = ("zero_probability" if any(row["Status"] == "zero_probability" for row, _ in candidates)
                  else "target_unreachable")
        summary = {
            "Status": status, "Path_Probability": 0.0, "Log_Path_Probability": -math.inf,
            "Component_Path": "", "Vulnerability_Path": "", "Component_Selection_Probability": 0.0,
        }
        steps = []
    return {
        "summary": summary,
        "path_steps": pd.DataFrame(steps, columns=STEP_COLUMNS),
        "candidate_paths": pd.DataFrame([row for row, _ in candidates], columns=CANDIDATE_COLUMNS),
    }
