

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_EXCEL = Path(os.environ.get("GAME_THEORY_EXCEL", ROOT / "game-theory.xlsx"))
RESULTS_ROOT = ROOT / "results"
CASE_RESULTS_ROOT = RESULTS_ROOT / INPUT_EXCEL.stem


def get_feature_output_dir(feature, input_excel=None):
    
    path = Path(INPUT_EXCEL if input_excel is None else input_excel).expanduser()
    path = (path if path.is_absolute() else ROOT / path).resolve()
    return RESULTS_ROOT / path.stem / feature

INITIAL_SIMULATION_DIR = CASE_RESULTS_ROOT / "initial_simulation"
MACOMP_DIR = CASE_RESULTS_ROOT / "macomp"
NSGA2_MA_DIR = CASE_RESULTS_ROOT / "nsga2_ma"
NSGA2_DIR = CASE_RESULTS_ROOT / "nsga2"
PARETO_PATH_ANALYSIS_DIR = CASE_RESULTS_ROOT / "pareto_path_analysis"
PARETO_ATTACK_PATHS_DIR = CASE_RESULTS_ROOT / "pareto_attack_paths"
MULTI_ALGORITHM_DIR = CASE_RESULTS_ROOT / "multi_algorithm"
STATE_VISUALIZATION_DIR = CASE_RESULTS_ROOT / "state_visualization"
SENSITIVITY_ANALYSIS_DIR = CASE_RESULTS_ROOT / "sensitivity_analysis"
PRUNING_VALIDATION_DIR = CASE_RESULTS_ROOT / "pruning_validation"
EQUILIBRIUM_VALIDATION_DIR = CASE_RESULTS_ROOT / "equilibrium_validation"
REVISION_REGISTRY_DIR = CASE_RESULTS_ROOT / "revision_registry"
STRUCTURAL_ABLATION_DIR = CASE_RESULTS_ROOT / "structural_ablation"
MODEL_VALIDATION_DIR = CASE_RESULTS_ROOT / "model_validation"
INDEPENDENT_REFERENCE_DIR = CASE_RESULTS_ROOT / "independent_reference"
PROBABILITY_MAPPING_DIR = CASE_RESULTS_ROOT / "probability_mapping"

FEATURE_OUTPUT_DIRS = {
    "initial_simulation": INITIAL_SIMULATION_DIR,
    "macomp": MACOMP_DIR,
    "nsga2_ma": NSGA2_MA_DIR,
    "nsga2": NSGA2_DIR,
    "pareto_path_analysis": PARETO_PATH_ANALYSIS_DIR,
    "pareto_attack_paths": PARETO_ATTACK_PATHS_DIR,
    "multi_algorithm": MULTI_ALGORITHM_DIR,
    "state_visualization": STATE_VISUALIZATION_DIR,
    "sensitivity_analysis": SENSITIVITY_ANALYSIS_DIR,
    "pruning_validation": PRUNING_VALIDATION_DIR,
    "equilibrium_validation": EQUILIBRIUM_VALIDATION_DIR,
    "revision_registry": REVISION_REGISTRY_DIR,
    "structural_ablation": STRUCTURAL_ABLATION_DIR,
    "model_validation": MODEL_VALIDATION_DIR,
    "independent_reference": INDEPENDENT_REFERENCE_DIR,
    "probability_mapping": PROBABILITY_MAPPING_DIR,
}



BUDGET = 60
SENSITIVITY_BUDGET = 500
DEFAULT_WORKERS = 9


GAME_OPTIMIZATION_ITERATIONS = 200
GAME_OPTIMIZATION_LEARNING_RATE = 0.3
EXPOSURE_PENALTY_RATIO = 0.3
RESPONSE_PENALTY_RATIO = 0.3


USE_PRUNING = True
PRUNE_TOL_DEFAULT = 1e-9


GAME_SENSITIVITY_ITERATIONS = 100
GAME_SENSITIVITY_LEARNING_RATE = 0.1


GAME_STATE_VISUALIZATION_ITERATIONS = 1000
GAME_STATE_VISUALIZATION_LEARNING_RATE = 0.1


NSGA2_MA_POPULATION_SIZE = 100
NSGA2_MA_SEED = 42
NSGA2_MA_CROSSOVER_PROBABILITY = 0.9
NSGA2_MA_MUTATION_PROBABILITY = 0.05
NSGA2_MA_LOCAL_SEARCH_PROBABILITY = 0.1
NSGA2_MA_LOCAL_SEARCH_ITERATIONS = 5


NSGA2_MA_MAX_GENERATIONS = 50


#




TERMINATION_METRIC = "both"


TERMINATION_TOLERANCE = 0.01


TERMINATION_STABLE_GENERATIONS = 5


TERMINATION_MIN_GENERATIONS = 20



TERMINATION_MIN_FRONT_SIZE = 2



#   loss_norm = loss / TERMINATION_LOSS_SCALE
#




TERMINATION_LOSS_SCALE = 3000.0



TERMINATION_HV_REFERENCE_POINT = (1.1, 1.1)




TERMINATION_HV_TARGET = None
TERMINATION_IDEAL_DISTANCE_TARGET = None


TERMINATION_ROBUST_PERIOD = 5


NSGA2_MA_TERMINATION_STRATEGY = "combined_or"


MULTI_ALGORITHM_RUNS = 1
MULTI_ALGORITHM_POPULATION_SIZE = 100
MULTI_ALGORITHM_DEFAULT_MAX_GENERATIONS = 50
MULTI_ALGORITHM_REFERENCE_LOSS = 300
MULTI_ALGORITHM_REFERENCE_COST_FACTOR = 1.1


MACOMP_RUNS = 2
MACOMP_FE_BUDGETS = (2000,)
MACOMP_SEED_START = NSGA2_MA_SEED
MACOMP_WORKERS = DEFAULT_WORKERS


SENSITIVITY_SAMPLE_MODES = {
    "fast": (64, True),
    "standard": (128, False),
    "comprehensive": (256, False),
}


SENSITIVITY_DECLARED_FAST_CONFIG = {
    "pop_size": 10,
    "n_gen": 3,
    "local_search_prob": 0.1,
    "local_search_iters": 2,
    "mutation_prob": 0.1,
    "crossover_prob": 0.9,
}


SENSITIVITY_EFFECTIVE_FAST_CONFIG = {
    "pop_size": 20,
    "n_gen": 5,
    "local_search_prob": 0.05,
    "local_search_iters": 1,
    "mutation_prob": 0.1,
    "crossover_prob": 0.9,
}

SENSITIVITY_STANDARD_CONFIG = {
    "pop_size": 100,
    "n_gen": 50,
    "local_search_prob": 0.1,
    "local_search_iters": 5,
    "mutation_prob": 0.05,
    "crossover_prob": 0.9,
}


CSV_FILES = {
    "pareto_attack_paths": {
        name: f"{name}.csv" for name in (
            "attack_graph_nodes", "attack_graph_edges", "scenarios", "decisions",
            "path_steps", "candidate_paths", "defense_annotations", "run_metadata",
            "source_comparison",
        )
    },
    "nsga2": {
        name: f"{name}.csv" for name in (
            "pareto_front", "defense_selection_matrix", "detailed_items_list",
            "cost_breakdown_data", "optimization_history", "run_metadata",
            "termination_history",
        )
    },
    "model_validation": {
        name: f"{name}.csv" for name in (
            "validation_summary", "action_probabilities", "prefix_contributions",
            "terminal_trajectories", "run_metadata",
        )
    },
    "structural_ablation": {
        "experiment_manifest": "experiment_manifest.csv",
        "structural_ablation_runs": "structural_ablation_runs.csv",
        "fixed_portfolio_results": "fixed_portfolio_results.csv",
        "ablation_fronts": "ablation_fronts.csv",
        "cross_evaluation": "cross_evaluation.csv",
        "ablation_paths": "ablation_paths.csv",
        "model_checks": "model_checks.csv",
    },
    "equilibrium_validation": {
        name: f"{name}.csv" for name in (
            "game_definition", "equilibrium_diagnostics", "jacobian_eigenvalues",
            "initialization_sensitivity", "strategy_trace", "path_sensitivity",
            "run_metadata",
        )
    },
    "initial_simulation": {
        "node_metrics": "node_metrics.csv",
        "availability": "availability.csv",
        "global_totals": "global_totals.csv",
        "strategy_probabilities": "strategy_probabilities.csv",
        "game_strategy_history": "game_strategy_history.csv",
    },
    "macomp": {
        "runs": "runs.csv",
        "history": "history.csv",
        "final_decisions": "final_decisions.csv",
        "summary": "summary.csv",
        "ranking": "ranking.csv",
        "reference_front": "reference_front.csv",
        "experiment_manifest": "experiment_manifest.csv",
        "variants": "variants.csv",
    },
    "nsga2_ma": {
        "pareto_front": "pareto_front.csv",
        "defense_selection_matrix": "defense_selection_matrix.csv",
        "detailed_items_list": "detailed_items_list.csv",
        "cost_breakdown_data": "cost_breakdown_data.csv",
        "optimization_history": "optimization_history.csv",
        "game_strategy_history": "game_strategy_history.csv",
        "path_vulnerability_details": "path_vulnerability_details.csv",
        "run_metadata": "run_metadata.csv",
        "termination_history": "termination_history.csv",
    },
    "pareto_path_analysis": {
        "selected_points": "selected_points.csv",
        "selected_defenses": "selected_defenses.csv",
        "node_risk": "node_risk.csv",
        "game_strategy_history": "game_strategy_history.csv",
    },
    "multi_algorithm": {
        "performance_metrics_raw": "performance_metrics_raw.csv",
        "performance_summary": "performance_summary.csv",
        "all_pareto_points": "all_pareto_points.csv",
        "best_pareto_fronts": "best_pareto_fronts.csv",
        "runtime_statistics": "runtime_statistics.csv",
        "statistical_tests": "statistical_tests.csv",
        "optimization_history": "optimization_history.csv",
        "reference_front": "reference_front.csv",
        "experiment_manifest": "experiment_manifest.csv",
    },
    "state_visualization": {
        "attack_graph_nodes": "attack_graph_nodes.csv",
        "attack_graph_edges": "attack_graph_edges.csv",
        "node_metrics": "node_metrics.csv",
        "edge_probabilities": "edge_probabilities.csv",
        "main_attack_path": "main_attack_path.csv",
        "state_graph_edges": "state_graph_edges.csv",
        "availability": "availability.csv",
    },
    "sensitivity_analysis": {
        "parameters": "parameters.csv",
        "parameter_samples": "parameter_samples.csv",
        "sample_metrics": "sample_metrics.csv",
        "sobol_indices": "sobol_indices.csv",
        "critical_parameters": "critical_parameters.csv",
        "run_metadata": "run_metadata.csv",
        "fronts": "fronts.csv",
        "portfolio_evaluations": "portfolio_evaluations.csv",
        "sobol_diagnostics": "sobol_diagnostics.csv",
    },
    "pruning_validation": {
        "case_summary": "case_summary.csv",
        "case_details": "case_details.csv",
    },
    "revision_registry": {
        "resolved_case_config": "resolved_case_config.csv",
        "artifact_provenance": "artifact_provenance.csv",
        "component_paths": "component_paths.csv",
        "component_paths_purdue": "component_paths_purdue.csv",
        "graph_paths": "graph_paths.csv",
        "graph_paths_purdue": "graph_paths_purdue.csv",
        "parameter_provenance": "parameter_provenance.csv",
        "case_validation": "case_validation.csv",
        "purdue_component_path_summary": "purdue_component_path_summary.csv",
    },
}
