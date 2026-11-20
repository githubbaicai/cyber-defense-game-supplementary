

from __future__ import annotations

import numpy as np
import pandas as pd

from common.defense_config import create_selected_configuration
from common.parsers import defense_categories
from common.simulation import compute_total_expected_values, run_simulation


REFERENCE_POINTS_FILENAME = "cost_breakdown_reference_points.csv"
SELECTED_POINTS_FILENAME = "cost_breakdown_selected_points.csv"

COST_COLUMNS = [
    "Prevention", "Detection", "Response", "TotalCost", "DefenderLoss", "P_Response",
]
SELECTED_POINT_COLUMNS = [
    "Selection", "Solution_ID", *COST_COLUMNS, "Ideal_Distance", "Cost_Plus_Loss",
]
REFERENCE_POINT_COLUMNS = ["Reference", *COST_COLUMNS]


def select_cost_breakdown_points(cost_data):
    
    if cost_data.empty:
        return pd.DataFrame(columns=SELECTED_POINT_COLUMNS)

    required = ["Solution_ID", *COST_COLUMNS]
    missing = set(required) - set(cost_data.columns)
    if missing:
        raise ValueError(f"Cost-breakdown data are missing columns: {sorted(missing)}")
    ordered = cost_data[required].copy()
    if not np.isfinite(ordered.to_numpy(dtype=float)).all():
        raise ValueError("Cost-breakdown selection data must contain finite numbers")
    if not ordered["Solution_ID"].is_unique:
        raise ValueError("Solution_ID values in cost-breakdown data must be unique")
    ordered = (
        ordered.sort_values(["TotalCost", "DefenderLoss", "Solution_ID"], kind="stable")
        .drop_duplicates(["TotalCost", "DefenderLoss"])
        .reset_index(drop=True)
    )
    losses = ordered["DefenderLoss"].to_numpy(dtype=float)
    previous_best = np.minimum.accumulate(np.concatenate(([np.inf], losses[:-1])))
    ordered = ordered.loc[losses < previous_best].reset_index(drop=True)

    values = ordered[["TotalCost", "DefenderLoss"]].to_numpy(dtype=float)
    minimum = values.min(axis=0)
    span = values.max(axis=0) - minimum
    normalized = (values - minimum) / np.where(span > 0, span, 1.0)
    ordered["Ideal_Distance"] = np.linalg.norm(normalized, axis=1)
    ordered["Cost_Plus_Loss"] = values.sum(axis=1)

    rows = []
    for selection, score in (
        ("knee_ideal_distance", "Ideal_Distance"),
        ("min_cost_plus_loss", "Cost_Plus_Loss"),
    ):
        ranked = ordered.sort_values([score, "Solution_ID"], kind="stable") if selection == "min_cost_plus_loss" else ordered
        chosen = ranked.iloc[int(np.argmin(ranked[score].to_numpy(dtype=float)))]
        rows.append({"Selection": selection, **chosen.to_dict()})
    output = pd.DataFrame(rows, columns=SELECTED_POINT_COLUMNS)
    output["Solution_ID"] = output["Solution_ID"].astype(int)
    return output


def evaluate_reference_points(
    items,
    all_data,
    graph,
    *,
    game_iterations,
    game_learning_rate,
    prune_tol,
):
    
    categories = defense_categories(items, all_data)
    if len(categories) != len(items) or set(categories) - {"Prevention", "Detection", "Response"}:
        raise ValueError("Invalid measure categories in a fixed reference portfolio")
    costs = np.asarray([cost for _, cost in items], dtype=float)
    if not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError("Fixed-reference portfolio costs must be nonnegative and finite")
    response_indices = [
        index for index, ((label, _), category) in enumerate(zip(items, categories))
        if category == "Response" and label in ("RP1", "RP2", "RP3")
    ]
    if not response_indices or "Prevention" not in categories or "Detection" not in categories:
        raise ValueError("A fixed reference portfolio requires prevention, detection and RP response measures")
    highest_response = max(
        response_indices,
        key=lambda index: int(str(items[index][0])[2:]),
    )

    rows = []
    for reference, domain, response_probability in (
        ("all_prevention", "Prevention", 0.0),
        ("all_detection_max_response", "Detection", 0.5),
    ):
        prefix = np.asarray([category == domain for category in categories], dtype=bool)
        if reference == "all_detection_max_response":
            prefix[highest_response] = True
        configuration = create_selected_configuration(prefix, items, all_data)
        simulated = run_simulation(
            state_graph=graph.state_paths,
            vuln_data=configuration["vuln_data_sim"],
            component_vuln_dist=all_data["Vulnerabilities distribution"],
            ids_config=configuration["ids_config"],
            firewall_config=configuration["firewall_config"],
            patch_config=configuration["patch_config"],
            iterations=game_iterations,
            learning_rate=game_learning_rate,
            p_response=response_probability,
            prune_tol=prune_tol,
        )
        _, defender_loss, _ = compute_total_expected_values(simulated)
        if not np.isfinite(defender_loss):
            raise ValueError(f"Fixed reference portfolio {reference} returned a nonfinite loss")
        category_costs = {
            category: float(sum(
                costs[index] for index, actual in enumerate(categories)
                if actual == category and prefix[index]
            ))
            for category in ("Prevention", "Detection", "Response")
        }
        rows.append({
            "Reference": reference,
            **category_costs,
            "TotalCost": float(sum(category_costs.values())),
            "DefenderLoss": float(defender_loss),
            "P_Response": response_probability,
        })
    return pd.DataFrame(rows, columns=REFERENCE_POINT_COLUMNS)
