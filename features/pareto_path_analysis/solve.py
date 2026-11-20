

import argparse
import warnings
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

from common.case_config import get_input_excel, load_case_parameters
from common.config import CSV_FILES, get_feature_output_dir
from common.defense_config import create_selected_configuration
from common.game_theory import (
    ALTERNATIVE_STATE_NAMES,
    DEFENDER_STATE_NAMES,
    build_game_inputs,
    calculate_defender_probabilities,
)
from common.model_options import ModelOptions
from common.simulation import (
    compute_total_expected_values,
    resolve_prune_tolerance,
    run_simulation,
)
from external.attack_graph import AttackGraph
from external.excel_processor import ExcelProcessor


DEFAULT_GAME_ITERATIONS = 1000
TAIL_WINDOW = 20
TAIL_TOLERANCE = 1e-6

DEFENDER_STATE_LABELS = {
    "not_detected": "Not detected",
    "detected_no_response": "Detected, no response",
    "detected_with_response": "Detected, with response",
    "not_detected_no_response": "Not detected, no response",
    "not_detected_with_response": "Not detected, with response",
}


def _select_knee(front):
    ordered = front.sort_values(
        ["Total_Cost", "Total_Defender_Loss", "Solution_ID"]
    ).reset_index(drop=True)

    values = ordered[
        ["Total_Cost", "Total_Defender_Loss"]
    ].to_numpy(dtype=float)

    minimum = values.min(axis=0)
    span = values.max(axis=0) - minimum
    normalized = (values - minimum) / np.where(span > 0, span, 1.0)

    direction = normalized[-1] - normalized[0]
    length = np.linalg.norm(direction)

    if length <= 1e-12:
        index = int(np.argmin(np.linalg.norm(normalized, axis=1)))
    else:
        relative = normalized - normalized[0]
        distances = np.abs(
            direction[0] * relative[:, 1]
            - direction[1] * relative[:, 0]
        ) / length

        if distances.max() <= 1e-12:
            index = int(np.argmin(np.linalg.norm(normalized, axis=1)))
        else:
            index = int(np.argmax(distances))

    return int(ordered.iloc[index]["Solution_ID"])


def _normalized_selection_front(front):
    
    if front.empty:
        raise ValueError("Pareto data are empty; point selection is unavailable")

    ordered = front.sort_values(
        ["Total_Cost", "Total_Defender_Loss", "Solution_ID"]
    ).drop_duplicates(
        ["Total_Cost", "Total_Defender_Loss"]
    ).reset_index(drop=True)

    losses = ordered["Total_Defender_Loss"].to_numpy(dtype=float)
    previous_best = np.minimum.accumulate(
        np.concatenate(([np.inf], losses[:-1]))
    )
    ordered = ordered.loc[
        losses < previous_best
    ].reset_index(drop=True)

    values = ordered[
        ["Total_Cost", "Total_Defender_Loss"]
    ].to_numpy(dtype=float)

    minimum = values.min(axis=0)
    span = values.max(axis=0) - minimum
    normalized = (values - minimum) / np.where(span > 0, span, 1.0)

    return ordered, normalized


def _select_knee_ideal_distance(front):
    
    ordered, normalized = _normalized_selection_front(front)
    index = int(np.argmin(np.linalg.norm(normalized, axis=1)))

    return int(ordered.iloc[index]["Solution_ID"])


def _select_knee_angle(front):
    
    ordered, normalized = _normalized_selection_front(front)

    if len(ordered) >= 3:
        incoming = normalized[1:-1] - normalized[:-2]
        outgoing = normalized[2:] - normalized[1:-1]

        valid = (
            (np.linalg.norm(incoming, axis=1) > 1e-12)
            & (np.linalg.norm(outgoing, axis=1) > 1e-12)
        )

        cross = (
            incoming[:, 0] * outgoing[:, 1]
            - incoming[:, 1] * outgoing[:, 0]
        )
        dot = np.einsum("ij,ij->i", incoming, outgoing)
        angles = np.arctan2(cross, dot)

        candidates = np.flatnonzero(
            valid & (angles > 1e-12)
        )

        if len(candidates):
            index = int(
                candidates[np.argmax(angles[candidates])]
            ) + 1

            return int(ordered.iloc[index]["Solution_ID"])

    warnings.warn(
        "No positive local turning angle is available; falling back to the nearest-ideal-point method.",
        UserWarning,
        stacklevel=2,
    )

    index = int(np.argmin(np.linalg.norm(normalized, axis=1)))
    return int(ordered.iloc[index]["Solution_ID"])


def _strategy_name(strategy):
    if not hasattr(strategy, "target_component"):
        return "Do Nothing"

    name = f"Attack {strategy.target_component}"

    if hasattr(strategy, "vulnerability"):
        name += f" ({strategy.vulnerability.name})"

    return name


def _load_input_csv(input_excel=None):
    directory = get_feature_output_dir(
        "nsga2_ma", get_input_excel(input_excel)
    )
    paths = {
        name: directory / CSV_FILES["nsga2_ma"][name]
        for name in ("pareto_front", "defense_selection_matrix")
    }

    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; run nsga2_ma solve/export first"
            )

    front = pd.read_csv(paths["pareto_front"])
    matrix = pd.read_csv(paths["defense_selection_matrix"])

    required = {
        "Solution_ID",
        "Total_Cost",
        "Total_Defender_Loss",
    }

    if not required.issubset(front.columns):
        raise ValueError("pareto_front.csv is missing required columns")

    if "Solution_ID" not in matrix.columns:
        raise ValueError(
            "defense_selection_matrix.csv is missing Solution_ID"
        )

    if front.empty or matrix.empty:
        raise ValueError("Pareto data are empty; resimulation is unavailable")

    for frame in (front, matrix):
        identities = pd.to_numeric(
            frame["Solution_ID"], errors="raise"
        ).to_numpy(dtype=float)

        if (
            not np.isfinite(identities).all()
            or np.any(identities < 1)
            or np.any(identities != np.floor(identities))
        ):
            raise ValueError("Solution_ID must be a positive integer")

        frame["Solution_ID"] = identities.astype(np.int64)

        if frame["Solution_ID"].duplicated().any():
            raise ValueError(
                "Solution_ID values must be unique in the input CSV"
            )

    if not set(front["Solution_ID"]).issubset(
        matrix["Solution_ID"]
    ):
        raise ValueError(
            "A Pareto point has no matching Solution_ID in the selection matrix"
        )

    for column in ("Total_Cost", "Total_Defender_Loss"):
        front[column] = pd.to_numeric(
            front[column], errors="raise"
        )

        if not np.isfinite(
            front[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(f"{column} contains nonfinite values")

    return front, matrix


def _response_probability(point, selection):
    probabilities = []

    for row in (point, selection):
        if "P_Response" not in row.index:
            raise ValueError(
                "The input CSV is missing P_Response; "
                "re-export the nsga2_ma results"
            )

        probability = float(row["P_Response"])

        if (
            not np.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError(
                "P_Response must be a finite probability in [0, 1]"
            )

        probabilities.append(probability)

    if not np.isclose(
        probabilities[0],
        probabilities[1],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            "pareto_front and defense_selection_matrix "
            "have different P_Response values"
        )

    return probabilities[1]


def _probability_history(values, shape, name):
    history = np.asarray(values, dtype=float)

    if (
        history.shape != shape
        or not np.isfinite(history).all()
        or np.any(history < 0)
        or np.any(history > 1)
        or not np.allclose(
            history.sum(axis=1),
            1,
            rtol=0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            f"{name} has invalid probability-history dimensions or values"
        )

    return history


def _node_evolution_rows(identity, node, info):
    values = info.get("evolution_history", [])

    if not len(values):
        return []

    strategies = info["strategies"]
    game = info["game_result"]
    states = tuple(game["state_names"])

    if states not in (
        DEFENDER_STATE_NAMES,
        ALTERNATIVE_STATE_NAMES,
    ):
        raise ValueError(
            f"{node} has unsupported defense-state names"
        )

    count = len(values)

    attacker = _probability_history(
        values,
        (count, len(strategies)),
        f"{node} attack strategy",
    )
    defender = _probability_history(
        info["defender_evolution_history"],
        (count, len(states)),
        f"{node} defense state",
    )
    response = _probability_history(
        info["response_policy_history"],
        (count, 2),
        f"{node} conditional response strategy",
    )

    initial_response = (
        game.get("initial_response_probability", game["p_response"])
        if game["has_response"]
        else 0.0
    )

    if not np.allclose(
        response[0],
        [1 - initial_response, initial_response],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{node} initial conditional response probabilities disagree with the simulation parameters"
        )

    inputs = build_game_inputs(
        strategies,
        model_options=ModelOptions(
            **game["model_options"]
        ),
    )

    expected_defender = np.asarray([
        calculate_defender_probabilities(
            probabilities,
            inputs["f"],
            policy[1],
            game["has_response"],
            alternative_trigger=(
                game["model_options"]["alternative_trigger"]
            ),
        )
        for probabilities, policy in zip(
            attacker, response
        )
    ])

    if not np.allclose(
        defender,
        expected_defender,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{node} defense states disagree with the attack, detection and response probabilities"
        )

    distributions = (
        (
            "Attacker",
            "attacker_strategy",
            [
                _strategy_name(strategy)
                for strategy in strategies
            ],
            attacker,
        ),
        (
            "Defender",
            "defender_state",
            [
                DEFENDER_STATE_LABELS[state]
                for state in states
            ],
            defender,
        ),
    )

    rows = []

    for player, probability_type, names, history in distributions:
        for generation, probabilities in enumerate(history):
            for name, probability in zip(
                names, probabilities
            ):
                rows.append({
                    **identity,
                    "Node_Path": node,
                    "Generation": generation,
                    "Strategy": name,
                    "Probability": float(probability),
                    "Player": player,
                    "Probability_Type": probability_type,
                })

    return rows


def _summary_step_rows(identity, results, availability):
    rows = []
    step_order = 0

    for node in sorted(
        set(results) | set(availability),
        key=lambda path: (path.count("→"), path),
    ):
        info = results.get(node, {})
        reach = float(availability.get(node, 0.0))

        if (
            not np.isfinite(reach)
            or not 0 <= reach <= 1
        ):
            raise ValueError(
                f"{node} has an incorrect global arrival probability"
            )

        rounded_reach = float(
            Decimal(str(reach)).quantize(
                Decimal("0.01"),
                rounding=ROUND_HALF_UP,
            )
        )

        strategies = info.get("strategies", [])
        final_probabilities = np.asarray(
            info.get("attacker_strategy", []),
            dtype=float,
        )

        if len(final_probabilities) != len(strategies):
            raise ValueError(
                f"{node} final attack probabilities do not match the number of strategies"
            )

        final_attack_mass = float(sum(
            probability
            for strategy, probability in zip(
                strategies, final_probabilities
            )
            if hasattr(strategy, "target_component")
        ))

        has_attack = any(
            hasattr(strategy, "target_component")
            for strategy in strategies
        )
        has_active_attack = final_attack_mass > 0
        has_history = bool(
            len(info.get("evolution_history", []))
        )

        if rounded_reach == 0:
            reason = "Rounded reach probability is zero"
        elif not has_active_attack:
            reason = "No positive-probability attack strategy"
        elif not has_history:
            reason = "No evolution history"
        else:
            reason = ""
            step_order += 1

        rows.append({
            **identity,
            "Node_Path": node,
            "Reach_Probability": reach,
            "Rounded_Reach_Probability": rounded_reach,
            "Has_Attack_Strategy": has_attack,
            "Final_Attack_Mass": final_attack_mass,
            "Has_Active_Attack": has_active_attack,
            "Has_Evolution_History": has_history,
            "Included": not reason,
            "Step_Order": (
                step_order if not reason else None
            ),
            "Exclusion_Reason": reason,
        })

    return rows


def _tail_diagnostics(info):
    histories = [
        np.asarray(
            info.get(name, []),
            dtype=float,
        )
        for name in (
            "evolution_history",
            "response_policy_history",
            "defender_evolution_history",
        )
    ]

    if any(
        history.ndim != 2
        or len(history) < TAIL_WINDOW + 1
        for history in histories
    ):
        return np.nan, False

    changes = [
        float(
            np.max(
                np.ptp(
                    history[-TAIL_WINDOW - 1:],
                    axis=0,
                )
            )
        )
        for history in histories
    ]

    if not np.isfinite(changes).all():
        return np.nan, False

    change = max(changes)
    return change, bool(change <= TAIL_TOLERANCE)


def solve(
    input_excel=None,
    *,
    game_iterations=DEFAULT_GAME_ITERATIONS,
    game_learning_rate=None,
):
    input_excel = get_input_excel(input_excel)

    if game_iterations is None:
        game_iterations = DEFAULT_GAME_ITERATIONS

    overrides = {
        name: value
        for name, value in (
            ("Game_Iterations", game_iterations),
            ("Game_Learning_Rate", game_learning_rate),
        )
        if value is not None
    }

    parameters = load_case_parameters(
        input_excel,
        names=(
            "Use_Pruning",
            "Game_Iterations",
            "Game_Learning_Rate",
        ),
        overrides=overrides,
    )

    front, matrix = _load_input_csv(input_excel)
    nodes, adjacency, all_data = (
        ExcelProcessor.import_excel(input_excel)
    )

    graph = AttackGraph(nodes, adjacency)
    graph.build_state_transition_graph()

    labels = list(all_data["Allocation"]["Label"])
    costs = np.asarray(
        all_data["Allocation"]["Cost"],
        dtype=float,
    )

    if len(labels) != len(set(labels)):
        raise ValueError(
            "Allocation.Label contains duplicate names"
        )

    missing = set(labels) - set(matrix.columns)

    if missing:
        raise ValueError(
            f"The defense-selection matrix is missing measure columns: {sorted(missing)}"
        )

    defense_items = list(zip(labels, costs))

    sum_front = front.assign(
        Cost_Plus_Loss=(
            front["Total_Cost"]
            + front["Total_Defender_Loss"]
        )
    ).sort_values([
        "Cost_Plus_Loss",
        "Solution_ID",
    ])

    selections = (
        (
            "knee",
            _select_knee(front),
        ),
        (
            "min_cost_plus_loss",
            int(sum_front.iloc[0]["Solution_ID"]),
        ),
        (
            "knee_angle",
            _select_knee_angle(front),
        ),
        (
            "knee_ideal_distance",
            _select_knee_ideal_distance(front),
        ),
    )

    front_by_id = front.set_index("Solution_ID")
    matrix_by_id = matrix.set_index("Solution_ID")

    selected_rows = []
    defense_rows = []
    risk_rows = []
    evolution_rows = []
    summary_rows = []
    simulated = {}

    prune_tol = resolve_prune_tolerance(
        parameters["Use_Pruning"]
    )

    for selection_type, solution_id in selections:
        point = front_by_id.loc[solution_id]
        selection = matrix_by_id.loc[solution_id]
        p_response = _response_probability(
            point, selection
        )

        raw_vector = pd.to_numeric(
            selection[labels],
            errors="raise",
        ).to_numpy(dtype=float)

        if not np.isin(
            raw_vector, [0.0, 1.0]
        ).all():
            raise ValueError(
                f"Solution_ID={solution_id} has selection values other than 0/1"
            )

        vector = raw_vector.astype(bool)
        total_cost = float(costs[vector].sum())
        original_cost = float(point["Total_Cost"])
        original_loss = float(
            point["Total_Defender_Loss"]
        )

        if not np.isclose(
            total_cost,
            original_cost,
            rtol=1e-7,
            atol=1e-6,
        ):
            raise ValueError(
                f"Solution_ID={solution_id} has inconsistent costs: "
                f"CSV={original_cost}, measure total={total_cost}"
            )

        configuration = create_selected_configuration(
            vector,
            defense_items,
            all_data,
        )

        selected_labels = [
            label
            for label, enabled in zip(labels, vector)
            if enabled
        ]

        if set(
            configuration["selected"]
        ) != set(selected_labels):
            raise ValueError(
                f"Solution_ID={solution_id} "
                "Noncanonical mutually exclusive response selections"
            )

        if solution_id not in simulated:
            results = run_simulation(
                state_graph=graph.state_paths,
                vuln_data=configuration["vuln_data_sim"],
                component_vuln_dist=(
                    all_data["Vulnerabilities distribution"]
                ),
                ids_config=configuration["ids_config"],
                firewall_config=(
                    configuration["firewall_config"]
                ),
                patch_config=configuration["patch_config"],
                iterations=parameters["Game_Iterations"],
                learning_rate=(
                    parameters["Game_Learning_Rate"]
                ),
                p_response=p_response,
                prune_tol=prune_tol,
                # Export the complete requested trajectory for reproducible plots.
                game_termination=None,
            )

            payoff, loss, availability = (
                compute_total_expected_values(results)
            )

            simulated[solution_id] = (
                results,
                payoff,
                loss,
                availability,
            )

        results, payoff, loss, availability = (
            simulated[solution_id]
        )

        diagnostics = {
            node: _tail_diagnostics(info)
            for node, info in results.items()
            if len(
                info.get("evolution_history", [])
            )
        }

        changes = [
            change
            for change, stable in diagnostics.values()
        ]

        max_change = (
            max(changes)
            if changes and np.isfinite(changes).all()
            else np.nan
        )

        tail_stable = bool(diagnostics) and all(
            stable
            for change, stable in diagnostics.values()
        )

        identity = {
            "Selection_Type": selection_type,
            "Solution_ID": solution_id,
        }

        selected_rows.append({
            **identity,
            "Selected_Defenses": " | ".join(selected_labels),
            "Number_of_Defenses": len(selected_labels),
            "Total_Cost": original_cost,
            "Original_Defender_Loss": original_loss,
            "Original_Cost_Plus_Loss": (
                original_cost + original_loss
            ),
            "Resimulated_Attacker_Payoff": float(payoff),
            "Resimulated_Defender_Loss": float(loss),
            "Resimulated_Cost_Plus_Loss": (
                original_cost + float(loss)
            ),
            "P_Response": p_response,
            "Game_Iterations": parameters["Game_Iterations"],
            "Game_Learning_Rate": (
                parameters["Game_Learning_Rate"]
            ),
            "Loss_Difference": float(loss) - original_loss,
            "Max_Tail_Change": max_change,
            "Tail_Stable": tail_stable,
        })

        for label, cost, enabled in zip(
            labels, costs, vector
        ):
            if enabled:
                defense_rows.append({
                    **identity,
                    "Defense_Label": label,
                    "Defense_Cost": float(cost),
                    "Total_Cost": original_cost,
                    "Original_Defender_Loss": original_loss,
                    "Resimulated_Defender_Loss": float(loss),
                })

        for node in sorted(
            set(results) | set(availability)
        ):
            info = results.get(node, {})
            change, stable = diagnostics.get(
                node, (np.nan, None)
            )

            risk_rows.append({
                **identity,
                "Node_Path": node,
                "Compromise_Probability": float(
                    availability.get(node, np.nan)
                ),
                "Attacker_Payoff": float(
                    info.get("attacker_payoff", np.nan)
                ),
                "Defender_Loss": float(
                    info.get("defender_loss", np.nan)
                ),
                "Tail_Change": change,
                "Tail_Stable": stable,
            })

        for node, info in results.items():
            evolution_rows.extend(
                _node_evolution_rows(
                    identity, node, info
                )
            )

        summary_rows.extend(
            _summary_step_rows(
                identity,
                results,
                availability,
            )
        )

    selected = pd.DataFrame(selected_rows)
    step_selection = pd.DataFrame(summary_rows)
    history = pd.DataFrame(evolution_rows)

    included = step_selection.loc[
        step_selection["Included"],
        [
            "Selection_Type",
            "Solution_ID",
            "Node_Path",
            "Step_Order",
        ],
    ]

    useful_history = history.merge(
        included,
        on=[
            "Selection_Type",
            "Solution_ID",
            "Node_Path",
        ],
        how="inner",
        validate="many_to_one",
    )

    for row in selected.itertuples(index=False):
        print(
            f"\n[{row.Selection_Type}] "
            f"Solution_ID={row.Solution_ID}\n"
            f"  Defense measures: {row.Selected_Defenses or 'none'}\n"
            f"  Cost: {row.Total_Cost:.6f}\n"
            f"  Original optimization loss: {row.Original_Defender_Loss:.6f}\n"
            f"  Resimulated loss: "
            f"{row.Resimulated_Defender_Loss:.6f}\n"
            f"  Loss difference (recomputed minus original): "
            f"{row.Loss_Difference:+.6f}\n"
            f"  Initial conditional response probability after detection: "
            f"{row.P_Response:.12f}\n"
            f"  Simulation generations: {row.Game_Iterations}\n"
            f"  Maximum probability change over the final {TAIL_WINDOW} generations: "
            f"{row.Max_Tail_Change:.3e}\n"
            f"  Tail stability: {row.Tail_Stable}"
        )

    if not selected["Tail_Stable"].all():
        warnings.warn(
            "Some nodes have not passed the tail-stability check or have insufficient history. "
            "The current results are endpoints at the specified generation count.",
            UserWarning,
            stacklevel=2,
        )

    print(
        "\nNote: loss-consistency comparisons require the same workbook, initial response probability, "
        "generation count, learning rate and pruning settings. Changing the generation count can change the loss; "
        "tail stability is a numerical diagnostic and does not establish a Nash equilibrium or ESS."
    )

    return {
        "input_excel": str(input_excel),
        "use_pruning": parameters["Use_Pruning"],
        "prune_tol": prune_tol,
        "selected_points": selected,
        "selected_defenses": pd.DataFrame(defense_rows),
        "node_risk": pd.DataFrame(risk_rows),
        "game_strategy_history": history,
        "summary_step_selection": step_selection,
        "useful_game_strategy_history": useful_history,
    }


if __name__ == "__main__":
    from common.pipeline import run_solve
    from features.pareto_path_analysis.export_csv import export

    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument("--input-excel")
    parser.add_argument(
        "--game-iterations",
        type=int,
        default=DEFAULT_GAME_ITERATIONS,
        help="Number of evolutionary updates for this feature; default: 1000 generations.",
    )
    parser.add_argument(
        "--game-learning-rate",
        type=float,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
    )

    arguments = parser.parse_args()

    run_solve(
        "pareto_path_analysis",
        solve,
        export,
        input_excel=arguments.input_excel,
        game_iterations=arguments.game_iterations,
        game_learning_rate=arguments.game_learning_rate,
    )

    print("✅ Pareto-analysis CSV files and effective-step Excel workbook exported.")

    if arguments.plot:
        from features.pareto_path_analysis.plot import plot

        plot(
            input_excel=get_input_excel(
                arguments.input_excel
            )
        )
        print("✅ Pareto-analysis plots updated.")
