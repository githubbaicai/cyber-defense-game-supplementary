

import pandas as pd

from common.config import (
    CSV_FILES,
    get_feature_output_dir,
)


NODE_COLUMNS = [
    "Node_Path",
    "Attack_Time_h",
    "Attacker_Payoff",
    "Defender_Loss",
    "Attacker_Strategy",
    "Defender_Strategy",
]

AVAILABILITY_COLUMNS = [
    "Node_Path",
    "Probability",
]

GLOBAL_COLUMNS = [
    "Total_Attacker_Expected_Payoff",
    "Total_Defender_Expected_Loss",
]

STRATEGY_COLUMNS = [
    "Node_Path",
    "Strategy",
    "Probability",
    "Detected_Probability",
    "Success_Probability",
]

HISTORY_COLUMNS = [
    "Node_Path",
    "Generation",
    "Strategy",
    "Probability",
]


def strategy_name(strategy):
    
    if hasattr(strategy, "target_component"):
        name = f"Attack {strategy.target_component}"
        if hasattr(strategy, "vulnerability"):
            name += (
                f" ({strategy.vulnerability.name})"
            )
        return name
    return "Do Nothing"


def export(result):
    
    directory = get_feature_output_dir("initial_simulation", result.get("input_excel"))
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    node_rows = []
    availability_rows = []
    strategy_rows = []
    history_rows = []

    for node, info in result["results"].items():
        node_rows.append({
            "Node_Path": node,
            "Attack_Time_h": info.get(
                "attack_time",
                0.0
            ),
            "Attacker_Payoff": info.get(
                "attacker_payoff",
                0.0
            ),
            "Defender_Loss": info.get(
                "defender_loss",
                0.0
            ),
            "Attacker_Strategy": ", ".join(
                str(x)
                for x in info.get(
                    "attacker_strategy",
                    []
                )
            ),
            "Defender_Strategy": ", ".join(
                str(x)
                for x in info.get(
                    "defender_strategy",
                    []
                )
            ),
        })

        strategies = info.get("strategies", [])
        game = info.get("game_result", {})
        if strategies and not {"final_response_probability", "has_response"}.issubset(game):
            raise ValueError("Missing solved response policy; rerun initial_simulation before export")
        probabilities = info.get(
            "attacker_strategy",
            []
        )
        for strategy, probability in zip(
            strategies,
            probabilities
        ):
            strategy_rows.append({
                "Node_Path": node,
                "Strategy": strategy_name(strategy),
                "Probability": probability,
                "Detected_Probability": (
                    strategy.detected_prob
                ),
                "Success_Probability": (
                    strategy.compute_success_probability(
                        game["final_response_probability"], has_response=game["has_response"]
                    )
                ),
            })

        history = info.get(
            "evolution_history",
            []
        )
        for generation, probabilities_at_generation in enumerate(
            history
        ):
            for strategy, probability in zip(
                strategies,
                probabilities_at_generation
            ):
                history_rows.append({
                    "Node_Path": node,
                    "Generation": generation,
                    "Strategy": strategy_name(strategy),
                    "Probability": float(
                        probability
                    ),
                })

    for node, probability in result[
        "availability"
    ].items():
        availability_rows.append({
            "Node_Path": node,
            "Probability": probability,
        })

    output_frames = {
        "node_metrics": pd.DataFrame(
            node_rows,
            columns=NODE_COLUMNS
        ),
        "availability": pd.DataFrame(
            availability_rows,
            columns=AVAILABILITY_COLUMNS
        ),
        "global_totals": pd.DataFrame(
            [{
                "Total_Attacker_Expected_Payoff": result[
                    "total_attacker_payoff"
                ],
                "Total_Defender_Expected_Loss": result[
                    "total_defender_loss"
                ],
            }],
            columns=GLOBAL_COLUMNS
        ),
        "strategy_probabilities": pd.DataFrame(
            strategy_rows,
            columns=STRATEGY_COLUMNS
        ),
        "game_strategy_history": pd.DataFrame(
            history_rows,
            columns=HISTORY_COLUMNS
        ),
    }

    for key, frame in output_frames.items():
        output_path = (
            directory
            / CSV_FILES["initial_simulation"][key]
        )
        frame.to_csv(
            output_path,
            index=False
        )

    return output_frames


if __name__ == "__main__":
    from common.pipeline import run_export

    run_export(
        "initial_simulation",
        export
    )
    print("✅ Initial simulation CSV files exported.")
