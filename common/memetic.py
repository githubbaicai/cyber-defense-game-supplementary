

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from common.config import (
    get_feature_output_dir,
    TERMINATION_HV_REFERENCE_POINT,
    TERMINATION_LOSS_SCALE,
)
from common.defense_config import normalize_decision_vector, repair_response_selection
from common.sampling import (
    ResponseAwareCrossover,
    ResponseAwareMutation,
    defense_categories,
    response_probability,
)
from common.problem import compose_decision_vector, decision_key, split_decision_vector
from common.parsers import vulnerability_patch_labels
from common.implementation import IMPLEMENTATION_VERSION, MODEL_PROTOCOL_VERSION
from common.termination import (
    FEASIBILITY_TOL,
    calculate_front_metrics,
    feasible_pareto_front,
)


ACCEPTANCE_TOL = 1e-12

MA_PARAMETER_DEFAULTS = {
    "selection_mode": "elite",
    "neighborhood": "all",
    "local_search_prob": 0.1,
    "max_local_search_iters": 10,
    "no_improve_rounds": 2,
    "elite_size": None,
    "max_local_fe_ratio": None,
    "max_local_fe_per_gen": None,
    "local_search_trigger": "every_gen",
    "local_search_start_ratio": 0.5,
    "hv_stagnation_generations": 5,
    "front_stagnation_generations": 3,
    "trigger_hv_tolerance": 1e-6,
    "acceptance_mode": "current",
    "acceptance_epsilon": 1e-3,
    "trigger_loss_scale": TERMINATION_LOSS_SCALE,
    "trigger_hv_reference_point": TERMINATION_HV_REFERENCE_POINT,
    "last_gen_local_fe_ratio": 0.2,
    "use_archive": False,
    "archive_max_size": None,
    "use_greedy_operators": False,
    "greedy_max_steps": 200,
    "use_ma_specific_init": False,
}


_COST_EPS = 1e-12


MA_NEIGHBORHOOD_CHOICES = (
    "all",
    "swap",
    "add_remove",
    "add+swap",
    "remove+swap",
    "structured",
)

_STAT_ATTRIBUTES = {
    "Selected_Elites_Total": "n_elite_selected",
    "Local_Accepted": "n_local_accepted",
    "Completed_Local_Rounds": "n_local_rounds",
    "Interrupted_Local_Rounds": "n_local_interrupted_rounds",
    "Neighbor_Scans": "n_neighbor_scans",
    "Local_Improved_Center": "n_local_improved_center",
    "Local_Entered_Final_Front": "n_local_entered_final_front",
    "Local_Dominates_Center": "n_local_dominates_center",
    "Final_Refill_Genetic_FE": "n_final_refill_genetic_evals",
    "Archive_Size": "n_archive_size",
    "Archive_Entered_Final_Front": "n_archive_entered_final_front",
    "Archive_Reinjected": "n_archive_reinjected",
    "Archive_Survived": "n_archive_survived",
    "Greedy_Fill_Used": "n_greedy_fill_used",
    "Greedy_Repair_Used": "n_greedy_repair_used",
    "Greedy_Accepted": "n_greedy_accepted",
}

LOCAL_STAT_COLUMNS = (
    "Local_FE",
    "Local_FE_Limit",
    "Local_FE_Reference",
    *_STAT_ATTRIBUTES,
)


def _integer(value, name: str, minimum: int = 1) -> int:
    
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if (
        isinstance(value, (bool, np.bool_))
        or result != value
        or result < minimum
    ):
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")

    return result


def _finite_float(value, name: str) -> float:
    
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc

    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalise_ma_parameters(parameters: dict) -> dict:
    
    values = {
        name: parameters.get(name, default)
        for name, default in MA_PARAMETER_DEFAULTS.items()
    }

    choices = {
        "selection_mode": ("elite", "random"),
        "neighborhood": MA_NEIGHBORHOOD_CHOICES,
        "local_search_trigger": (
            "every_gen",
            "last_gen",
            "after_ratio",
            "hv_stagnation",
            "front_stagnation",
        ),
        "acceptance_mode": (
            "current",
            "dominance",
            "weighted_sum",
            "first_improvement",
            "epsilon",
        ),
    }
    for name, allowed in choices.items():
        if values[name] not in allowed:
            raise ValueError(f"{name} must be one of {allowed}")

    for name in (
        "local_search_prob",
        "local_search_start_ratio",
        "last_gen_local_fe_ratio",
    ):
        values[name] = _finite_float(values[name], name)
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    if values["max_local_fe_ratio"] is not None:
        values["max_local_fe_ratio"] = _finite_float(
            values["max_local_fe_ratio"],
            "max_local_fe_ratio",
        )
        if not 0.0 <= values["max_local_fe_ratio"] <= 1.0:
            raise ValueError("max_local_fe_ratio must be in [0, 1] or None")

    for name in (
        "max_local_search_iters",
        "no_improve_rounds",
        "hv_stagnation_generations",
        "front_stagnation_generations",
    ):
        values[name] = _integer(values[name], name)

    if values["elite_size"] is not None:
        values["elite_size"] = _integer(
            values["elite_size"],
            "elite_size",
        )
    if values["max_local_fe_per_gen"] is not None:
        values["max_local_fe_per_gen"] = _integer(
            values["max_local_fe_per_gen"],
            "max_local_fe_per_gen",
            minimum=0,
        )

    if not isinstance(values["use_archive"], (bool, np.bool_)):
        raise ValueError("use_archive must be bool")
    values["use_archive"] = bool(values["use_archive"])

    if values["archive_max_size"] is not None:
        values["archive_max_size"] = _integer(
            values["archive_max_size"],
            "archive_max_size",
        )

    if not isinstance(
        values["use_greedy_operators"],
        (bool, np.bool_),
    ):
        raise ValueError("use_greedy_operators must be bool")
    values["use_greedy_operators"] = bool(
        values["use_greedy_operators"]
    )

    values["greedy_max_steps"] = _integer(
        values["greedy_max_steps"],
        "greedy_max_steps",
    )

    if not isinstance(
        values["use_ma_specific_init"],
        (bool, np.bool_),
    ):
        raise ValueError("use_ma_specific_init must be bool")
    values["use_ma_specific_init"] = bool(
        values["use_ma_specific_init"]
    )

    for name in ("trigger_hv_tolerance", "acceptance_epsilon"):
        values[name] = _finite_float(values[name], name)
        if values[name] < 0:
            raise ValueError(f"{name} must be nonnegative")

    values["trigger_loss_scale"] = _finite_float(
        values["trigger_loss_scale"],
        "trigger_loss_scale",
    )
    if values["trigger_loss_scale"] <= 0:
        raise ValueError("trigger_loss_scale must be greater than zero")

    reference = np.asarray(
        values["trigger_hv_reference_point"],
        dtype=float,
    )
    if (
        reference.shape != (2,)
        or not np.isfinite(reference).all()
        or np.any(reference <= 1.0)
    ):
        raise ValueError(
            "trigger_hv_reference_point must contain two finite numbers, both greater than 1"
        )
    values["trigger_hv_reference_point"] = tuple(
        float(value) for value in reference
    )
    return values


def sha256_file(path) -> str:
    
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_selected_ma_file(input_excel) -> Path:
    
    return get_feature_output_dir("macomp", input_excel) / "selected_ma.json"


def load_selected_ma_parameters(
    *,
    budget,
    pop_size,
    game_iterations,
    game_learning_rate,
    input_excel=None,
) -> tuple:
    
    from common.case_config import get_input_excel
    input_excel = get_input_excel(input_excel)
    selected_ma_file = get_selected_ma_file(input_excel)
    if not selected_ma_file.is_file():
        raise FileNotFoundError(
            f"MA ablation selection file not found: {selected_ma_file}\n"
            "Run python -m features.macomp.solve first"
        )

    document = json.loads(
        selected_ma_file.read_text(encoding="utf-8")
    )
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported selected_ma.json version")

    if not isinstance(document.get("context"), dict):
        raise ValueError("selected_ma.json is missing context")
    if not isinstance(document.get("selected"), dict):
        raise ValueError("selected_ma.json is missing selected")

    expected = {
        "budget": float(budget),
        "game_iterations": int(game_iterations),
        "game_learning_rate": float(game_learning_rate),
        "excel_sha256": sha256_file(input_excel),
    }
    if pop_size is not None:
        expected["pop_size"] = int(pop_size)

    from common.selection_override import (
        get_selection_override_file, read_selection_override, selection_metadata,
    )
    override_metadata = selection_metadata(
        document["context"].get("excel_sha256"), expected["excel_sha256"],
    )
    for name, value in expected.items():
        actual = document["context"].get(name)
        if isinstance(value, float):
            try:
                matches = math.isclose(
                    float(actual),
                    value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            except (TypeError, ValueError, OverflowError):
                matches = False
        else:
            matches = actual == value

        if not matches:
            if name == "excel_sha256":
                override_path = get_selection_override_file(selected_ma_file)
                if not override_path.is_file():
                    raise ValueError(
                        f"The ablation experiment and current solver settings differ: excel_sha256; "
                        f"experiment={actual!r}, current={value!r}. "
                        f"For author authorization, provide an override file: {override_path.resolve()}"
                    )
                override = read_selection_override(
                    override_path, value, run_budget=budget,
                    context_budget=document["context"].get("budget"),
                )
                if override["allow_downstream_run"] is not True:
                    raise ValueError(f"The override does not allow downstream execution: {override_path.resolve()}")
                warnings.warn(
                    f"selected_ma.json input hash mismatch; using author authorization: {override_path.resolve()}; "
                    f"reason={override['reason']}; bypassing only the excel_sha256 check.",
                    RuntimeWarning, stacklevel=2,
                )
                override_metadata = selection_metadata(
                    override["original_input_sha256"], value, override_path,
                )
                continue
            raise ValueError(
                f"The ablation experiment and current solver settings differ: {name}; "
                f"experiment={actual!r}, current={value!r}. "
                "Rerun features.macomp.solve with the current settings."
            )

    versions = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "model_protocol_version": MODEL_PROTOCOL_VERSION,
    }
    for name, expected_version in versions.items():
        actual_version = document["context"].get(name)
        if actual_version is not None and actual_version != expected_version:
            raise ValueError(
                f"The selected {name}={actual_version!r} differs from the current {expected_version!r}; "
                "rerun macomp; a hash override cannot authorize implementation-version differences."
            )
    version_match = all(document["context"].get(name) == value for name, value in versions.items())
    if not version_match:
        warnings.warn(
            "selected_ma.json is missing implementation/model protocol versions and has status unverified_legacy; "
            "matching input hashes do not establish compatibility of a legacy selection; rerun macomp before formal experiments.",
            RuntimeWarning, stacklevel=2,
        )
    override_metadata.update({
        "Implementation_Version": IMPLEMENTATION_VERSION,
        "Model_Protocol_Version": MODEL_PROTOCOL_VERSION,
        "Selection_Implementation_Version": document["context"].get("implementation_version", "unknown"),
        "Selection_Model_Protocol_Version": document["context"].get("model_protocol_version", "unknown"),
        "Selection_Implementation_Status": "version_match" if version_match else "unverified_legacy",
    })

    required_selected = {
        "selection_mode",
        "neighborhood",
        "local_search_prob",
        "max_local_search_iters",
        "no_improve_rounds",
        "elite_size",
    }
    missing = required_selected - set(document["selected"])
    if missing:
        raise ValueError(f"selected is missing fields: {sorted(missing)}")

    return _normalise_ma_parameters(document["selected"]), override_metadata


def get_local_search_statistics(algorithm) -> dict:
    
    result = {
        "Local_FE": int(getattr(
            algorithm,
            "n_local_fe_used",
            getattr(algorithm, "n_local_evals", 0),
        )),
        "Local_FE_Limit": getattr(
            algorithm,
            "n_local_fe_limit",
            None,
        ),
        "Local_FE_Reference": getattr(
            algorithm,
            "_fe_reference",
            None,
        ),
    }

    for column, attribute in _STAT_ATTRIBUTES.items():
        result[column] = int(getattr(algorithm, attribute, 0))
    return result


def is_feasible_individual(individual, budget) -> bool:
    
    if individual is None:
        return False

    f = np.asarray(individual.F, dtype=float).reshape(-1)
    if (
        f.size != 2
        or not np.isfinite(f).all()
        or f[0] > float(budget) + FEASIBILITY_TOL
    ):
        return False

    for raw_values in (individual.G, individual.CV):
        if raw_values is None:
            continue

        values = np.asarray(raw_values, dtype=float).reshape(-1)
        if values.size and (
            not np.isfinite(values).all()
            or np.any(values > FEASIBILITY_TOL)
        ):
            return False

    return True


def _dominates(first, second) -> bool:
    
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    return bool(np.all(a <= b) and np.any(a < b))


def _crowding_distance(values) -> np.ndarray:
    
    values = np.asarray(values, dtype=float)
    distances = np.zeros(len(values), dtype=float)

    if len(values) <= 2:
        distances[:] = np.inf
        return distances

    for objective in range(values.shape[1]):
        order = np.argsort(values[:, objective], kind="mergesort")
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        span = (
            values[order[-1], objective]
            - values[order[0], objective]
        )
        if span > 0:
            distances[order[1:-1]] += (
                values[order[2:], objective]
                - values[order[:-2], objective]
            ) / span

    return distances


class MemeticNSGA2(NSGA2):
    

    def __init__(
        self,
        local_search_prob=0.1,
        max_local_search_iters=10,
        no_improve_rounds=2,
        selection_mode="elite",
        neighborhood="all",
        elite_size=None,
        use_archive=False,
        archive_max_size=None,
        use_greedy_operators=False,
        greedy_max_steps=200,
        use_ma_specific_init=False,
        max_fe=None,
        local_search_iters=None,
        max_local_fe_ratio=None,
        max_local_fe_per_gen=None,
        local_search_trigger="every_gen",
        acceptance_mode="current",
        local_search_start_ratio=0.5,
        hv_stagnation_generations=5,
        front_stagnation_generations=3,
        trigger_hv_tolerance=1e-6,
        acceptance_epsilon=1e-3,
        trigger_loss_scale=TERMINATION_LOSS_SCALE,
        trigger_hv_reference_point=TERMINATION_HV_REFERENCE_POINT,
        last_gen_local_fe_ratio=0.2,
        local_fe_reference=None,
        **kwargs,
    ):
        if local_search_iters is not None:
            if max_local_search_iters != 10:
                raise ValueError(
                    "Do not specify both local_search_iters and "
                    "max_local_search_iters"
                )
            max_local_search_iters = local_search_iters

        parameters = _normalise_ma_parameters({
            "selection_mode": selection_mode,
            "neighborhood": neighborhood,
            "local_search_prob": local_search_prob,
            "max_local_search_iters": max_local_search_iters,
            "no_improve_rounds": no_improve_rounds,
            "elite_size": elite_size,
            "use_archive": use_archive,
            "archive_max_size": archive_max_size,
            "use_greedy_operators": use_greedy_operators,
            "greedy_max_steps": greedy_max_steps,
            "use_ma_specific_init": use_ma_specific_init,
            "max_local_fe_ratio": max_local_fe_ratio,
            "max_local_fe_per_gen": max_local_fe_per_gen,
            "local_search_trigger": local_search_trigger,
            "local_search_start_ratio": local_search_start_ratio,
            "hv_stagnation_generations": hv_stagnation_generations,
            "front_stagnation_generations": front_stagnation_generations,
            "trigger_hv_tolerance": trigger_hv_tolerance,
            "acceptance_mode": acceptance_mode,
            "acceptance_epsilon": acceptance_epsilon,
            "trigger_loss_scale": trigger_loss_scale,
            "trigger_hv_reference_point": trigger_hv_reference_point,
            "last_gen_local_fe_ratio": last_gen_local_fe_ratio,
        })

        super().__init__(**kwargs)

        for name, value in parameters.items():
            setattr(self, name, value)

        self.max_fe = (
            None
            if max_fe is None
            else _integer(max_fe, "max_fe", minimum=self.pop_size)
        )
        self.local_fe_reference = (
            None
            if local_fe_reference is None
            else _integer(local_fe_reference, "local_fe_reference")
        )

        self.n_local_evals = 0
        self.n_local_fe_used = 0
        self.n_local_fe_limit = None
        self.max_local_fe = None

        for attribute in _STAT_ATTRIBUTES.values():
            setattr(self, attribute, 0)

        self._costs = None
        self._rp_indices = frozenset()
        self._fe_reference = None
        self._stage_local_end = None

        self._effectiveness = None
        self._value_per_cost = None
        self._rp_levels = None
        self._structured_add_order = None
        self._structured_remove_order = None

        
        
        self.pareto_archive = []
        self.pareto_archive_keys = set()
        self.n_archive_size = 0
        self.n_archive_entered_final_front = 0
        self.n_archive_reinjected = 0
        self.n_archive_survived = 0

        self.n_greedy_fill_used = 0
        self.n_greedy_repair_used = 0
        self.n_greedy_accepted = 0

        self._local_accepted_keys = set()
        self._dominance_pairs = set()

        self._last_gen_pending = False
        self._last_gen_done = False
        self._last_gen_reserved_fe = 0

        self.trigger_hv = np.nan
        self._best_trigger_hv = -np.inf
        self._best_front_size = 0
        self._hv_stagnant_generations = 0
        self._front_stagnant_generations = 0

    def _setup(self, problem, **kwargs):
        
        
        
        self.problem = problem
        super()._setup(problem, **kwargs)

        if isinstance(self.mating.crossover, TwoPointCrossover) and not isinstance(self.mating.crossover, ResponseAwareCrossover):
            self.mating.crossover = ResponseAwareCrossover(
                prob=float(self.mating.crossover.prob.value)
            )
        if isinstance(self.mating.mutation, BitflipMutation) and not isinstance(self.mating.mutation, ResponseAwareMutation):
            self.mating.mutation = ResponseAwareMutation(
                prob=float(self.mating.mutation.prob.value)
            )

        items = getattr(problem, "defense_items", None)
        budget = getattr(problem, "budget", None)

        if items is None or budget is None or problem.n_var not in (len(items), len(items) + 1):
            raise ValueError(
                "MemeticNSGA2 requires problem.defense_items and "
                "problem.budget to be consistent with the variable dimension"
            )

        self._costs = np.asarray(
            [item[1] for item in items],
            dtype=float,
        )
        if (
            not np.isfinite(self._costs).all()
            or np.any(self._costs < 0)
        ):
            raise ValueError("Defense-measure costs must be finite and nonnegative")

        self._rp_indices = frozenset(
            index
            for index, (label, _) in enumerate(items)
            if label in ("RP1", "RP2", "RP3")
        )

        self._prepare_structured_orders(
            items,
            getattr(problem, "all_data", None),
        )

        self._fe_reference = (
            self.max_fe
            if self.max_fe is not None
            else self.local_fe_reference
        )

        needs_reference = (
            self.max_local_fe_ratio is not None
            or self.local_search_trigger == "after_ratio"
        )
        if needs_reference and self._fe_reference is None:
            raise ValueError(
                "When a local FE ratio or after_ratio is configured, "
                "max_fe or local_fe_reference must be provided"
            )

        if self.max_local_fe_ratio is not None:
            self.n_local_fe_limit = math.floor(
                self._fe_reference * self.max_local_fe_ratio
            )
        self.max_local_fe = self.n_local_fe_limit

        
        
        
        if self.use_ma_specific_init:
            initialization = getattr(self, "initialization", None)
            sampling = getattr(initialization, "sampling", None)
            if sampling is not None and hasattr(sampling, "extra_injected"):
                extra = self._generate_structured_initial_solutions(
                    n_solutions=8
                )
                if extra:
                    sampling.extra_injected = [
                        np.asarray(vector, dtype=float).reshape(-1)
                        for vector in extra
                    ]

    @staticmethod
    def _effectiveness_lookup(sheet, column) -> dict:
        
        if not isinstance(sheet, dict):
            return {}

        result = {}
        for label, value in zip(
            sheet.get("Label", []),
            sheet.get(column, []),
        ):
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if np.isfinite(number):
                result[str(label)] = max(0.0, number)
        return result

    def _prepare_structured_orders(
        self,
        defense_items,
        all_data,
    ) -> None:
        
        n_items = len(defense_items)
        effectiveness = np.zeros(n_items, dtype=float)
        rp_levels = np.zeros(n_items, dtype=int)

        lookup = {}
        if isinstance(all_data, dict):
            for sheet_name, column in (
                ("Patch vulnerability", "Patching effectiveness"),
                ("Set firewall", "Firewall effectiveness"),
                ("Set IDS", "Detection effectiveness"),
            ):
                lookup.update(
                    self._effectiveness_lookup(
                        all_data.get(sheet_name),
                        column,
                    )
                )

        for index, (label, _) in enumerate(defense_items):
            name = str(label)
            if name in ("RP1", "RP2", "RP3"):
                rp_levels[index] = int(name[-1])
                effectiveness[index] = float(rp_levels[index])
            else:
                effectiveness[index] = lookup.get(name, 0.0)

        costs = np.asarray(self._costs, dtype=float)
        value_per_cost = effectiveness / np.maximum(costs, _COST_EPS)

        other = [
            index for index in range(n_items) if rp_levels[index] == 0
        ]
        responses = [
            index for index in range(n_items) if rp_levels[index] > 0
        ]

        add_order = sorted(
            other,
            key=lambda index: (-value_per_cost[index], index),
        )
        add_order += sorted(
            responses,
            key=lambda index: (-rp_levels[index], index),
        )

        remove_order = sorted(
            other,
            key=lambda index: (value_per_cost[index], index),
        )
        remove_order += sorted(
            responses,
            key=lambda index: (rp_levels[index], index),
        )

        self._effectiveness = effectiveness
        self._value_per_cost = value_per_cost
        self._rp_levels = rp_levels
        self._structured_add_order = np.asarray(add_order, dtype=int)
        self._structured_remove_order = np.asarray(
            remove_order,
            dtype=int,
        )

    def _greedy_fill_with_priority(self, preferred, target_budget):
        
        candidate = np.zeros(self._costs.size, dtype=bool)
        total = 0.0
        rp_selected = False

        preferred_set = {int(index) for index in preferred}
        order = [int(index) for index in preferred]
        order += [
            int(index)
            for index in self._structured_add_order
            if int(index) not in preferred_set
        ]

        for index in order:
            if index in self._rp_indices and rp_selected:
                continue
            if (
                total + self._costs[index]
                > float(target_budget) + FEASIBILITY_TOL
            ):
                continue
            candidate[index] = True
            total += self._costs[index]
            if index in self._rp_indices:
                rp_selected = True

        return candidate

    def _select_subset_within_budget(self, indices, target_budget):
        
        candidate = np.zeros(self._costs.size, dtype=bool)
        total = 0.0
        rp_selected = False

        ordered = sorted(
            {int(index) for index in indices},
            key=lambda index: (-self._value_per_cost[index], index),
        )
        for index in ordered:
            if index in self._rp_indices and rp_selected:
                continue
            if (
                total + self._costs[index]
                > float(target_budget) + FEASIBILITY_TOL
            ):
                continue
            candidate[index] = True
            total += self._costs[index]
            if index in self._rp_indices:
                rp_selected = True

        return candidate

    def _category_balanced_solution(self):
        
        n = self._costs.size
        categories = defense_categories(
            self.problem.defense_items, self.problem.all_data
        )

        prevention = [
            index
            for index, category in enumerate(categories)
            if category == "Prevention"
        ]
        detection = [
            index
            for index, category in enumerate(categories)
            if category == "Detection"
        ]
        response = sorted(self._rp_indices)

        for pool in (prevention, detection, response):
            pool.sort(key=lambda index: -self._value_per_cost[index])

        pools = {
            "prevention": prevention,
            "detection": detection,
            "response": response,
        }
        positions = {"prevention": 0, "detection": 0, "response": 0}

        candidate = np.zeros(n, dtype=bool)
        total = 0.0
        budget = float(self.problem.budget)
        rp_selected = False

        progressed = True
        while progressed:
            progressed = False
            for key in ("prevention", "detection", "response"):
                while positions[key] < len(pools[key]):
                    index = pools[key][positions[key]]
                    positions[key] += 1
                    if key == "response" and rp_selected:
                        continue
                    if total + self._costs[index] <= budget + FEASIBILITY_TOL:
                        candidate[index] = True
                        total += self._costs[index]
                        if key == "response":
                            rp_selected = True
                        progressed = True
                        break

        return candidate

    def _attack_path_solutions(self):
        
        graph = getattr(self.problem, "ag", None)
        paths = getattr(graph, "attack_paths", None) or []
        key_paths = [path for path in paths if len(path) >= 3][:2]
        if not key_paths:
            return []

        all_data = getattr(self.problem, "all_data", None) or {}
        distribution = all_data.get("Vulnerabilities distribution", {})
        component_map = {}
        for component, value in zip(
            distribution.get("Component", []),
            distribution.get("Vulnerability", []),
        ):
            if value is None or value == 0:
                component_map[str(component)] = []
                continue
            component_map[str(component)] = [
                token.strip()
                for token in str(value).split(",")
                if token.strip()
            ]

        label_to_index = {
            str(item[0]): index
            for index, item in enumerate(self.problem.defense_items)
        }
        patch_labels = vulnerability_patch_labels(
            all_data["Vulnerability"], all_data["Patch vulnerability"]
        )

        solutions = []
        for path in key_paths:
            preferred = []
            for component in path[1:-1]:
                for vulnerability in component_map.get(
                    str(component), []
                ):
                    index = label_to_index.get(patch_labels[vulnerability])
                    if index is not None:
                        preferred.append(index)

            if not preferred:
                continue

            preferred = sorted(
                set(preferred),
                key=lambda index: (
                    -self._value_per_cost[index],
                    index,
                ),
            )
            solutions.append(
                self._select_subset_within_budget(
                    preferred,
                    self.problem.budget,
                )
            )

        return solutions

    def _perturbed_greedy_solution(self, factor):
        
        n = self._costs.size
        budget = float(self.problem.budget)
        perturbation = self._random_generator().uniform(-abs(factor), abs(factor), n)
        perturbed = self._value_per_cost * (1.0 + perturbation)

        other = [
            index for index in range(n) if index not in self._rp_indices
        ]
        other.sort(key=lambda index: (-perturbed[index], index))
        responses = sorted(
            self._rp_indices,
            key=lambda index: (-self._rp_levels[index], index),
        )

        candidate = np.zeros(n, dtype=bool)
        total = 0.0
        rp_selected = False

        for index in other + responses:
            if index in self._rp_indices and rp_selected:
                continue
            if total + self._costs[index] <= budget + FEASIBILITY_TOL:
                candidate[index] = True
                total += self._costs[index]
                if index in self._rp_indices:
                    rp_selected = True

        return candidate

    def _generate_structured_initial_solutions(self, n_solutions=8):
        
        budget = float(self.problem.budget)
        n = len(self.problem.defense_items)

        candidates = []

        
        candidates.append(self._greedy_fill_with_priority([], budget))

        
        for ratio in (0.25, 0.5, 0.75):
            candidates.append(
                self._greedy_fill_with_priority([], budget * ratio)
            )

        
        candidates.append(self._category_balanced_solution())

        
        candidates.extend(self._attack_path_solutions())

        
        for _ in range(2):
            candidates.append(self._perturbed_greedy_solution(0.2))

        seen = set()
        solutions = []
        for candidate in candidates:
            if candidate is None:
                continue
            vector = np.asarray(candidate, dtype=bool).reshape(-1)
            if vector.size != n:
                continue

            repair_response_selection(
                vector,
                self.problem.defense_items,
            )
            if float(np.dot(vector, self._costs)) > budget + FEASIBILITY_TOL:
                continue

            key = tuple(bool(value) for value in vector)
            if key in seen:
                continue
            seen.add(key)
            solutions.append(
                compose_decision_vector(vector, response_probability(vector, self.problem.defense_items))
                if self.problem.n_var > n else vector.astype(float)
            )
            if len(solutions) >= n_solutions:
                break

        return solutions

    def _initialize_advance(self, infills=None, **kwargs):
        
        super()._initialize_advance(infills=infills, **kwargs)
        self._monitor_front()

    def _operator_kwargs(self, kwargs=None) -> dict:
        
        options = {} if kwargs is None else dict(kwargs)
        random_state = getattr(self, "random_state", None)
        if random_state is not None:
            options.setdefault("random_state", random_state)
        return options

    def _random_generator(self):
        
        random_state = getattr(self, "random_state", None)
        return np.random if random_state is None else random_state

    def _quota(self) -> int:
        
        if self.local_search_prob == 0:
            return 0
        if self.elite_size is not None:
            return self.elite_size
        return max(
            1,
            math.ceil(self.pop_size * self.local_search_prob),
        )

    def _local_fe_available(self) -> bool:
        
        if (
            self.max_fe is not None
            and self.evaluator.n_eval >= self.max_fe
        ):
            return False
        if (
            self.n_local_fe_limit is not None
            and self.n_local_evals >= self.n_local_fe_limit
        ):
            return False
        if (
            self._stage_local_end is not None
            and self.n_local_evals >= self._stage_local_end
        ):
            return False
        return True

    def _last_generation_reserve(self, remaining: int) -> int:
        
        if (
            self.local_search_trigger != "last_gen"
            or self._last_gen_done
            or self._quota() == 0
            or not self._local_fe_available()
        ):
            return 0

        if self.n_local_fe_limit is None:
            allowance = math.floor(
                self.max_fe * self.last_gen_local_fe_ratio
            )
        else:
            allowance = self.n_local_fe_limit - self.n_local_evals

        if self.max_local_fe_per_gen is not None:
            allowance = min(allowance, self.max_local_fe_per_gen)

        return max(0, min(allowance, remaining - 1))

    def _infill(self):
        
        if self.max_fe is None:
            return super()._infill()

        remaining = self.max_fe - int(self.evaluator.n_eval)
        if remaining <= 0:
            self.termination.force_termination = True
            return None

        original_count = self.n_offsprings
        count = min(original_count, remaining)

        reserved = self._last_generation_reserve(remaining)
        if reserved > 0 and remaining <= original_count + reserved:
            self._last_gen_pending = True
            self._last_gen_reserved_fe = reserved
            count = min(original_count, remaining - reserved)

        try:
            self.n_offsprings = count
            return super()._infill()
        finally:
            self.n_offsprings = original_count

    @property
    def early_termination_ready(self) -> bool:
        
        if (
            self._quota() == 0
            or self.max_local_fe_per_gen == 0
            or not self._local_fe_available()
            or self.local_search_trigger == "last_gen"
        ):
            return True
        return self.n_elite_selected > 0

    def _should_local_search(self) -> bool:
        
        if self._quota() == 0 or not self._local_fe_available():
            return False
        if self.max_local_fe_per_gen == 0:
            return False

        if self.local_search_trigger == "every_gen":
            return True
        if self.local_search_trigger == "last_gen":
            return self._last_gen_pending and not self._last_gen_done
        if self.local_search_trigger == "after_ratio":
            return (
                self.evaluator.n_eval
                >= self._fe_reference * self.local_search_start_ratio
            )
        if self.local_search_trigger == "hv_stagnation":
            return (
                np.isfinite(self.trigger_hv)
                and self._hv_stagnant_generations
                >= self.hv_stagnation_generations
            )
        return (
            self._front_stagnant_generations
            >= self.front_stagnation_generations
        )

    def _select_search_targets(self, merged) -> list:
        
        quota = self._quota()
        if quota == 0:
            return []

        feasible = []
        seen = set()

        for individual in merged:
            if not is_feasible_individual(
                individual,
                self.problem.budget,
            ):
                continue

            key = decision_key(individual.X, len(self.problem.defense_items), self.problem.defense_items)
            if key not in seen:
                feasible.append(individual)
                seen.add(key)

        quota = min(quota, len(feasible))
        if quota == 0:
            return []

        if self.selection_mode == "random":
            indices = self._random_generator().choice(
                len(feasible),
                size=quota,
                replace=False,
            )
            return [feasible[int(index)] for index in indices]

        values = np.asarray(
            [individual.F for individual in feasible],
            dtype=float,
        )
        fronts = NonDominatedSorting().do(values)
        selected = []

        for front in fronts:
            front = np.asarray(front, dtype=int)
            crowding = _crowding_distance(values[front])
            order = sorted(
                range(len(front)),
                key=lambda index: (
                    -float(crowding[index]),
                    int(front[index]),
                ),
            )

            for index in order:
                selected.append(feasible[int(front[index])])
                if len(selected) == quota:
                    return selected

        return selected

    def _neighbors(self, x):
        
        x = normalize_decision_vector(x, self.problem.defense_items)
        prefix, p_response = split_decision_vector(x, len(self.problem.defense_items))

        if self.neighborhood == "structured":
            add_order = self._structured_add_order
            remove_order = self._structured_remove_order
            if add_order is None or remove_order is None:
                add_order = np.arange(len(prefix))
                remove_order = np.arange(len(prefix))
            add_indices = [
                int(index) for index in add_order if not x[index]
            ]
            remove_indices = [
                int(index) for index in remove_order if x[index]
            ]
        else:
            add_indices = [int(index) for index in np.flatnonzero(~prefix)]
            remove_indices = [int(index) for index in np.flatnonzero(prefix)]

        if self.neighborhood in (
            "all", "add_remove", "add+swap", "structured",
        ):
            for add_index in add_indices:
                candidate = x.copy()
                candidate[add_index] = True
                yield normalize_decision_vector(candidate, self.problem.defense_items)

        if self.neighborhood in (
            "all", "add_remove", "remove+swap", "structured",
        ):
            for remove_index in remove_indices:
                candidate = x.copy()
                candidate[remove_index] = False
                yield normalize_decision_vector(candidate, self.problem.defense_items)

        if self.neighborhood in (
            "all", "swap", "add+swap", "remove+swap", "structured",
        ):
            for remove_index in remove_indices:
                for add_index in add_indices:
                    candidate = x.copy()
                    candidate[remove_index] = False
                    candidate[add_index] = True
                    yield normalize_decision_vector(candidate, self.problem.defense_items)

        if len(x) > len(prefix) and any(prefix[index] for index in self._rp_indices):
            for response_value in (max(0.0, p_response - 0.1), min(1.0, p_response + 0.1)):
                if response_value == p_response:
                    continue
                candidate = x.copy()
                candidate[-1] = response_value
                yield candidate

    def _greedy_fill(self, x):
        
        if self._structured_add_order is None:
            return None

        budget = float(self.problem.budget)
        candidate = normalize_decision_vector(x, self.problem.defense_items)
        prefix, p_response = split_decision_vector(candidate, len(self.problem.defense_items))
        total_cost = float(np.dot(prefix, self._costs))
        selected_response = any(
            candidate[index] for index in self._rp_indices
        )
        changed = False

        for step, index in enumerate(self._structured_add_order):
            if step >= self.greedy_max_steps:
                break

            index = int(index)
            if candidate[index]:
                continue
            if index in self._rp_indices and selected_response:
                continue
            if (
                total_cost + self._costs[index]
                > budget + FEASIBILITY_TOL
            ):
                continue

            candidate[index] = True
            total_cost += self._costs[index]
            if index in self._rp_indices:
                selected_response = True
            changed = True

        if not changed:
            return None

        repair_response_selection(
            candidate,
            self.problem.defense_items,
        )
        return normalize_decision_vector(candidate, self.problem.defense_items)

    def _greedy_repair(self, x):
        
        if self._structured_remove_order is None:
            return None

        budget = float(self.problem.budget)
        candidate = normalize_decision_vector(x, self.problem.defense_items)
        prefix, p_response = split_decision_vector(candidate, len(self.problem.defense_items))
        total_cost = float(np.dot(prefix, self._costs))

        if total_cost <= budget + FEASIBILITY_TOL:
            return None

        changed = False

        for step, index in enumerate(self._structured_remove_order):
            if step >= self.greedy_max_steps:
                break

            index = int(index)
            if not candidate[index]:
                continue

            candidate[index] = False
            total_cost -= self._costs[index]
            changed = True

            if total_cost <= budget + FEASIBILITY_TOL:
                break

        if not changed or total_cost > budget + FEASIBILITY_TOL:
            return None

        
        repair_response_selection(
            candidate,
            self.problem.defense_items,
        )
        return normalize_decision_vector(candidate, self.problem.defense_items)

    def _greedy_candidates(self, center_x):
        
        if not self.use_greedy_operators:
            return

        filled = self._greedy_fill(center_x)
        if filled is not None:
            self.n_greedy_fill_used += 1
            yield filled

        repaired = self._greedy_repair(center_x)
        if repaired is not None:
            self.n_greedy_repair_used += 1
            yield repaired

    def _response_activation_candidates(self, center_x, candidate):
        
        items = self.problem.defense_items
        candidate = normalize_decision_vector(candidate, items)
        yield candidate
        if (
            len(center_x) == len(items) + 1
            and not any(center_x[index] for index in self._rp_indices)
            and any(candidate[index] for index in self._rp_indices)
        ):
            joint = candidate.copy()
            joint[-1] = response_probability(candidate, items)
            yield joint

    def _round_candidates(self, center_x):
        
        center_x = normalize_decision_vector(center_x, self.problem.defense_items)
        for candidate in self._greedy_candidates(center_x):
            for expanded in self._response_activation_candidates(center_x, candidate):
                yield expanded, True

        for candidate in self._neighbors(center_x):
            for expanded in self._response_activation_candidates(center_x, candidate):
                yield expanded, False

    @staticmethod
    def _choose_next(accepted, current, budget):
        
        unique = {
            tuple(float(value) for value in individual.X): individual
            for individual in accepted
        }
        loss_scale = max(1.0, abs(float(current.F[1])))

        def priority(individual):
            distance = (
                abs(float(individual.F[0] - current.F[0]))
                / max(float(budget), 1.0)
                + abs(float(individual.F[1] - current.F[1]))
                / loss_scale
            )
            return (
                -distance,
                float(individual.F[0]),
                float(individual.F[1]),
                tuple(float(value) for value in individual.X),
            )

        return min(unique.values(), key=priority)

    def _accepts(self, current, neighbor, weights) -> bool:
        
        if self.acceptance_mode == "current":
            return not _dominates(current.F, neighbor.F)

        if self.acceptance_mode in ("dominance", "first_improvement"):
            return _dominates(neighbor.F, current.F)

        scale = np.asarray([
            max(float(self.problem.budget), 1.0),
            self.trigger_loss_scale,
        ])
        center = np.asarray(current.F, dtype=float) / scale
        candidate = np.asarray(neighbor.F, dtype=float) / scale

        if self.acceptance_mode == "weighted_sum":
            return (
                float(np.dot(weights, candidate))
                < float(np.dot(weights, center)) - ACCEPTANCE_TOL
            )

        
        
        return bool(
            np.all(candidate <= center + self.acceptance_epsilon)
            and np.any(candidate < center - self.acceptance_epsilon)
        )

    def _search_from(self, starting_point, known, added, pool_keys):
        
        current = starting_point
        budget = float(self.problem.budget)
        visited_centers = {
            decision_key(current.X, len(self.problem.defense_items), self.problem.defense_items)
        }
        stagnant_rounds = 0
        weights = None

        if self.acceptance_mode == "weighted_sum":
            first_weight = float(self._random_generator().random())
            weights = np.asarray(
                [first_weight, 1.0 - first_weight],
                dtype=float,
            )

        for _ in range(self.max_local_search_iters):
            accepted_for_move = []
            new_accepted = 0
            complete_round = True

            center_x = np.asarray(current.X, dtype=float)
            center_key = decision_key(center_x, len(self.problem.defense_items), self.problem.defense_items)

            for neighbor_x, from_greedy in self._round_candidates(
                center_x
            ):
                if not self._local_fe_available():
                    complete_round = False
                    break

                if not from_greedy:
                    self.n_neighbor_scans += 1

                neighbor_x = normalize_decision_vector(neighbor_x, self.problem.defense_items)
                if (
                    float(np.dot(neighbor_x[:len(self._costs)], self._costs))
                    > budget + FEASIBILITY_TOL
                ):
                    continue

                key = decision_key(neighbor_x, len(self.problem.defense_items), self.problem.defense_items)
                if key == center_key:
                    continue

                neighbor = known.get(key)
                if neighbor is None:
                    
                    
                    if not self._local_fe_available():
                        complete_round = False
                        break

                    neighbor = Individual(X=neighbor_x.copy())
                    self.evaluator.eval(
                        self.problem,
                        neighbor,
                        algorithm=self,
                        skip_already_evaluated=False,
                    )
                    self.n_local_evals += 1
                    self.n_local_fe_used += 1
                    known[key] = neighbor

                if not is_feasible_individual(neighbor, budget):
                    continue
                if not self._accepts(current, neighbor, weights):
                    continue

                if _dominates(neighbor.F, current.F):
                    pair = (center_key, key)
                    if pair not in self._dominance_pairs:
                        self._dominance_pairs.add(pair)
                        self.n_local_dominates_center += 1

                if key not in pool_keys:
                    added.append(neighbor)
                    pool_keys.add(key)
                    self._local_accepted_keys.add(key)
                    self.n_local_accepted += 1
                    new_accepted += 1
                    if from_greedy:
                        self.n_greedy_accepted += 1

                if key not in visited_centers:
                    accepted_for_move.append(neighbor)

                    if self.acceptance_mode == "first_improvement":
                        
                        break

            if not complete_round:
                self.n_local_interrupted_rounds += 1
                break

            self.n_local_rounds += 1

            
            
            progress = new_accepted > 0
            if self.acceptance_mode != "current":
                progress = progress or bool(accepted_for_move)

            stagnant_rounds = 0 if progress else stagnant_rounds + 1
            if stagnant_rounds >= self.no_improve_rounds:
                break

            if not accepted_for_move:
                continue

            if self.acceptance_mode == "weighted_sum":
                scale = np.asarray([
                    max(budget, 1.0),
                    self.trigger_loss_scale,
                ])
                next_center = min(
                    accepted_for_move,
                    key=lambda individual: (
                        float(np.dot(
                            weights,
                            np.asarray(individual.F) / scale,
                        )),
                        decision_key(individual.X, len(self.problem.defense_items), self.problem.defense_items),
                    ),
                )
            elif self.acceptance_mode == "first_improvement":
                next_center = accepted_for_move[0]
            else:
                next_center = self._choose_next(
                    accepted_for_move,
                    current,
                    budget,
                )

            improved = (
                _dominates(next_center.F, current.F)
                if self.acceptance_mode == "current"
                else self._accepts(current, next_center, weights)
            )
            if improved:
                self.n_local_improved_center += 1

            current = next_center
            visited_centers.add(
                decision_key(current.X, len(self.problem.defense_items), self.problem.defense_items)
            )

    def _run_local_stage(self, merged, stage_cap=None):
        
        limits = [
            limit
            for limit in (self.max_local_fe_per_gen, stage_cap)
            if limit is not None
        ]
        self._stage_local_end = (
            self.n_local_evals + min(limits)
            if limits
            else None
        )

        try:
            if not self._local_fe_available():
                return merged

            targets = self._select_search_targets(merged)
            self.n_elite_selected += len(targets)

            known = {
                decision_key(individual.X, len(self.problem.defense_items), self.problem.defense_items): individual
                for individual in merged
            }
            pool_keys = set(known)
            added = []

            for starting_point in targets:
                if not self._local_fe_available():
                    break
                self._search_from(
                    starting_point,
                    known,
                    added,
                    pool_keys,
                )

            if added:
                return Population.merge(
                    merged,
                    Population.create(*added),
                )
            return merged
        finally:
            self._stage_local_end = None

    def _fill_final_genetic_fe(self, merged):
        
        while self.evaluator.n_eval < self.max_fe:
            remaining = self.max_fe - int(self.evaluator.n_eval)
            count = min(self.n_offsprings, remaining)

            offspring = self.mating.do(
                self.problem,
                self.pop,
                count,
                algorithm=self,
                **self._operator_kwargs(),
            )
            if offspring is None or len(offspring) == 0:
                raise RuntimeError(
                    "Unable to generate actual genetic offspring in the final generation; "
                    "strict FE experiments must disable duplicate elimination"
                )

            offspring = offspring[:remaining]
            offspring.set("n_gen", self.n_gen)
            offspring.set("n_iter", self.n_gen)

            before = int(self.evaluator.n_eval)
            self.evaluator.eval(
                self.problem,
                offspring,
                algorithm=self,
                skip_already_evaluated=False,
            )
            used = int(self.evaluator.n_eval) - before

            if used != len(offspring):
                raise RuntimeError("Unexpected FE count for final-generation genetic offspring")

            self.n_final_refill_genetic_evals += used
            merged = Population.merge(merged, offspring)

        return merged

    def _advance(self, infills=None, **kwargs):
        
        if infills is None:
            return super()._advance(infills=infills, **kwargs)

        merged = Population.merge(self.pop, infills)
        final_stage = self._last_gen_pending

        if self._should_local_search():
            merged = self._run_local_stage(
                merged,
                stage_cap=(
                    self._last_gen_reserved_fe
                    if final_stage
                    else None
                ),
            )

        if final_stage:
            self._last_gen_done = True
            self._last_gen_pending = False
            merged = self._fill_final_genetic_fe(merged)

        if self.use_archive:
            
            if self.pareto_archive:
                existing_keys = {
                    decision_key(individual.X, len(self.problem.defense_items), self.problem.defense_items)
                    for individual in merged
                }
                to_inject = []

                for individual in self.pareto_archive:
                    key = decision_key(individual.X, len(self.problem.defense_items), self.problem.defense_items)
                    if key in existing_keys:
                        continue
                    existing_keys.add(key)
                    to_inject.append(individual)

                if to_inject:
                    self.n_archive_reinjected += len(to_inject)
                    merged = Population.merge(
                        merged,
                        Population.create(*to_inject),
                    )

            
            merged = self._deduplicate_by_decision(merged)

        self.pop = self.survival.do(
            self.problem,
            merged,
            n_survive=self.pop_size,
            algorithm=self,
            **self._operator_kwargs(kwargs),
        )

        if self.use_archive and self.pareto_archive:
            
            survived = 0
            for individual in self.pop:
                key = decision_key(individual.X, len(self.problem.defense_items), self.problem.defense_items)
                if key in self.pareto_archive_keys:
                    survived += 1
            self.n_archive_survived += survived

        self._update_archive(merged)
        self._refresh_local_front_count()
        self._monitor_front()

    def _monitor_front(self) -> None:
        
        if self.local_search_trigger not in (
            "hv_stagnation",
            "front_stagnation",
        ):
            return

        front = feasible_pareto_front(
            self.pop,
            self.problem.budget,
        )

        if len(front) == 0:
            self.trigger_hv = np.nan
            self._best_trigger_hv = -np.inf
            self._best_front_size = 0
            self._hv_stagnant_generations = 0
            self._front_stagnant_generations = 0
            return

        if len(front) > self._best_front_size:
            self._best_front_size = len(front)
            self._front_stagnant_generations = 0
        else:
            self._front_stagnant_generations += 1

        if self.local_search_trigger != "hv_stagnation":
            return

        metrics = calculate_front_metrics(
            front,
            self.problem.budget,
            self.trigger_loss_scale,
            self.trigger_hv_reference_point,
        )
        self.trigger_hv = float(metrics["hv"])

        if not metrics["reference_valid"]:
            self._best_trigger_hv = -np.inf
            self._hv_stagnant_generations = 0
            return

        if (
            self.trigger_hv
            > self._best_trigger_hv + self.trigger_hv_tolerance
        ):
            self._best_trigger_hv = self.trigger_hv
            self._hv_stagnant_generations = 0
        else:
            self._hv_stagnant_generations += 1

    def _refresh_local_front_count(self) -> None:
        
        if not self._local_accepted_keys:
            self.n_local_entered_final_front = 0
            return

        feasible = [
            individual
            for individual in self.pop
            if is_feasible_individual(
                individual,
                self.problem.budget,
            )
        ]
        if not feasible:
            self.n_local_entered_final_front = 0
            return

        values = np.asarray(
            [individual.F for individual in feasible],
            dtype=float,
        )
        indices = NonDominatedSorting().do(
            values,
            only_non_dominated_front=True,
        )
        final_keys = {
            decision_key(feasible[int(index)].X, len(self.problem.defense_items), self.problem.defense_items)
            for index in indices
        }
        self.n_local_entered_final_front = len(
            final_keys & self._local_accepted_keys
        )

    def _update_archive(self, merged) -> None:
        
        if not self.use_archive:
            return

        budget = float(self.problem.budget)
        candidates = list(self.pareto_archive)
        seen = set(self.pareto_archive_keys)

        for individual in merged:
            if not is_feasible_individual(individual, budget):
                continue

            key = decision_key(individual.X, len(self.problem.defense_items), self.problem.defense_items)
            if key in seen:
                continue

            seen.add(key)
            candidates.append(individual)

        if not candidates:
            self.pareto_archive = []
            self.pareto_archive_keys = set()
            self.n_archive_size = 0
            return

        values = np.asarray(
            [individual.F for individual in candidates],
            dtype=float,
        )
        indices = NonDominatedSorting().do(
            values,
            only_non_dominated_front=True,
        )

        archive = []
        keys = set()

        for index in indices:
            individual = candidates[int(index)]
            key = decision_key(individual.X, len(self.problem.defense_items), self.problem.defense_items)
            if key in keys:
                continue
            keys.add(key)
            archive.append(individual)

        if (
            self.archive_max_size is not None
            and len(archive) > self.archive_max_size
        ):
            front_values = np.asarray(
                [individual.F for individual in archive],
                dtype=float,
            )
            crowding = _crowding_distance(front_values)
            order = sorted(
                range(len(archive)),
                key=lambda position: (
                    -float(crowding[position]),
                    tuple(
                        float(value)
                        for value in archive[position].X
                    ),
                ),
            )
            archive = [
                archive[position]
                for position in order[: self.archive_max_size]
            ]
            keys = {
                decision_key(individual.X, len(self.problem.defense_items), self.problem.defense_items)
                for individual in archive
            }

        self.pareto_archive = archive
        self.pareto_archive_keys = keys
        self.n_archive_size = len(archive)

    def _apply_archive_to_finalize(self, candidates=None) -> None:
        
        pool = Population.merge(
            self.pop if candidates is None else candidates,
            Population.create(*self.pareto_archive),
        )
        pool = self._deduplicate_by_decision(pool)
        self.pop = self.survival.do(
            self.problem,
            pool,
            n_survive=self.pop_size,
            algorithm=self,
            **self._operator_kwargs(),
        )

        self._set_optimum()
        self._refresh_local_front_count()
        self.n_archive_size = len(self.pareto_archive)
        self.n_archive_entered_final_front = sum(
            1
            for individual in self.opt
            if decision_key(individual.X, len(self.problem.defense_items), self.problem.defense_items)
            in self.pareto_archive_keys
        )

        
        original_generation = self.n_gen
        self.n_gen = max(1, int(original_generation) - 1)
        try:
            self._set_optimum()
            self.callback(self)
        finally:
            self.n_gen = original_generation

    def _deduplicate_by_decision(self, population):
        
        seen = set()
        unique = []

        for individual in population:
            key = decision_key(individual.X, len(self.problem.defense_items), self.problem.defense_items)
            if key in seen:
                continue
            seen.add(key)
            unique.append(individual)

        return Population.create(*unique)

    def _finalize(self):
        
        merged = self.pop
        before = int(self.evaluator.n_eval)
        if (
            self.local_search_trigger == "last_gen"
            and not self._last_gen_done
            and self._quota() > 0
        ):
            self._last_gen_done = True

            if self._local_fe_available():
                merged = self._run_local_stage(self.pop)

        if len(merged) > len(self.pop) or not self.pareto_archive:
            self._update_archive(merged)
        if self.use_archive and self.pareto_archive:
            self._apply_archive_to_finalize(merged)
        elif int(self.evaluator.n_eval) != before or len(merged) > len(self.pop):
            if len(merged) > len(self.pop):
                self.pop = self.survival.do(
                    self.problem, merged, n_survive=self.pop_size,
                    algorithm=self, **self._operator_kwargs(),
                )
            self._refresh_local_front_count()
            original_generation = self.n_gen
            self.n_gen = max(1, int(original_generation) - 1)
            try:
                self._set_optimum()
                self.callback(self)
            finally:
                self.n_gen = original_generation

        return super()._finalize()
