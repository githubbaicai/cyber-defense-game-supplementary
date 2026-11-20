"""Complete-decision evaluation and comparisons within each target mapping."""

from hashlib import sha256
import json

import numpy as np
import pandas as pd

from common.defense_config import normalize_decision_vector
from common.problem import CyberDefenseProblem
from common.termination import FEASIBILITY_TOL

from .config import MAPPINGS, MAPPING_SPECS


def decision_record(items, decision, objectives, weight):
    vector = np.asarray(decision, dtype=float).reshape(-1)
    if vector.size != len(items) + 1 or not np.isfinite(vector).all():
        raise ValueError("A complete finite decision including P_Response is required")
    normalized = normalize_decision_vector(vector, items)
    if not np.array_equal(vector, normalized):
        raise ValueError("Decision must already use canonical binary and response encoding")
    objectives = np.asarray(objectives, dtype=float)
    if objectives.shape != (2,) or not np.isfinite(objectives).all():
        raise ValueError("Expected two finite objectives")
    labels = [str(label) for label, _ in items]
    bits = [int(value) for value in normalized[:-1]]
    resource_json = json.dumps([labels, bits], separators=(",", ":"), ensure_ascii=False)
    resource_id = sha256(resource_json.encode("utf-8")).hexdigest()
    identifier = sha256((resource_id + ":" + float(normalized[-1]).hex()).encode()).hexdigest()
    return {
        "Decision_ID": identifier, "Resource_ID": resource_id,
        "Decision_Vector": json.dumps(normalized.tolist(), separators=(",", ":")),
        "Allocation_Labels": json.dumps(labels, ensure_ascii=False),
        "Selected_Defenses": json.dumps([label for label, enabled in zip(labels, bits) if enabled], ensure_ascii=False),
        "P_Response": float(normalized[-1]), "Cost": float(objectives[0]),
        "Loss": float(objectives[1]), "J": float(objectives[0] + weight * objectives[1]),
    }


def evaluate_pool(context, candidates, settings):
    items, data, graph = context
    problem = CyberDefenseProblem(
        items, data, graph, budget=settings["budget"],
        game_iterations=settings["game_iterations"],
        game_learning_rate=settings["game_learning_rate"],
        prune_tol=settings["prune_tol"], game_termination=settings["game_termination"],
    )
    records = []
    for candidate in candidates:
        vector = json.loads(candidate["Decision_Vector"])
        output = {}
        problem._evaluate(vector, output)
        record = decision_record(items, vector, output["F"], settings["lambda_loss"])
        if record["Decision_ID"] != candidate["Decision_ID"]:
            raise ValueError("Candidate identity changed during evaluation")
        if float(output["G"][0]) > FEASIBILITY_TOL:
            raise ValueError("Cross-evaluation candidate violates the fixed budget")
        records.append(record)
    return records, {"Evaluation_FE": problem.total_evals,
                     "Unique_Simulations": len(problem.eval_cache), "Cache_Hits": problem.cache_hits}


def select_best(records):
    if not records:
        raise ValueError("Cannot select from an empty feasible candidate set")
    return min(records, key=lambda row: (row["J"], row["Cost"], row["Loss"], row["Decision_ID"]))


def summarize(candidates, cross_evaluations, no_defense_id):
    baseline = select_best([row for row in candidates if row["Mapping_ID"] == "baseline"])
    summaries, decisions = [], []
    for order, mapping in enumerate(MAPPINGS):
        evaluated = [row for row in cross_evaluations if row["Mapping_ID"] == mapping]
        lookup = {row["Decision_ID"]: row for row in evaluated}
        own_ids = {row["Decision_ID"] for row in candidates if row["Mapping_ID"] == mapping}
        fixed = lookup[baseline["Decision_ID"]]
        reoptimized = select_best([lookup[identifier] for identifier in sorted(own_ids)])
        best = select_best(evaluated)
        undefended = lookup[no_defense_id]
        row = {"Mapping_ID": mapping, "Mapping_Label": MAPPING_SPECS[mapping]["label"],
               "Mapping_Order": order, "No_Defense_Loss": undefended["Loss"],
               "Candidate_Gap": fixed["J"] - best["J"],
               "Reoptimized_Search_Gap": reoptimized["J"] - best["J"],
               "Candidate_Count": len(evaluated), "Baseline_Decision_ID": baseline["Decision_ID"]}
        fixed_vector = np.asarray(json.loads(fixed["Decision_Vector"]))
        chosen_vector = np.asarray(json.loads(reoptimized["Decision_Vector"]))
        row["Resource_Changes"] = int(np.count_nonzero(fixed_vector[:-1] != chosen_vector[:-1]))
        row["Response_Change"] = abs(float(fixed_vector[-1] - chosen_vector[-1]))
        for role, record in (("Fixed", fixed), ("Reoptimized", reoptimized), ("Pool_Best", best)):
            for field in ("Cost", "Loss", "J", "Decision_ID", "P_Response", "Selected_Defenses"):
                row[f"{role}_{field}"] = record[field]
            row[f"{role}_Loss_Reduction"] = undefended["Loss"] - record["Loss"]
            row[f"{role}_Loss_Reduction_Percent"] = (
                100 * (undefended["Loss"] - record["Loss"]) / undefended["Loss"]
                if undefended["Loss"] > 0 else np.nan
            )
            decisions.append({**record, "Mapping_ID": mapping, "Role": role})
        summaries.append(row)
    return pd.DataFrame(summaries), pd.DataFrame(decisions)
