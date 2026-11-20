

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from common.config import (
    GAME_OPTIMIZATION_ITERATIONS,
    GAME_OPTIMIZATION_LEARNING_RATE,
)
from common.defense_config import create_selected_configuration, normalize_decision_vector
from common.parsers import parse_firewall_config, parse_ids_config
from common.model_options import resolve_model_options
from common.game_termination import DEFAULT_GAME_TERMINATION, resolve_game_termination
from common.simulation import (
    compute_total_expected_values,
    PRUNE_TOL_DEFAULT,
    run_simulation,
)
from common.termination import FEASIBILITY_TOL


def split_decision_vector(x, defense_count):
    """Return the binary defense prefix and continuous response propensity."""
    values = np.asarray(x, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("The decision vector must contain finite numbers")
    if values.size not in (defense_count, defense_count + 1):
        raise ValueError(
            "Decision-vector length must equal the number of defense measures, optionally plus p_response"
        )
    prefix = values[:defense_count] >= 0.5
    p_response = (
        float(np.clip(values[-1], 0.0, 1.0))
        if values.size == defense_count + 1
        else 0.0
    )
    return prefix, p_response


def decision_key(x, defense_count, defense_items=None):
    """Stable cache/deduplication key retaining the continuous response gene."""
    if defense_items is not None:
        if len(defense_items) != defense_count:
            raise ValueError("defense_items and defense_count disagree")
        x = normalize_decision_vector(x, defense_items)
    prefix, p_response = split_decision_vector(x, defense_count)
    return tuple(bool(value) for value in prefix) + (p_response,)


def compose_decision_vector(prefix, p_response=0.0):
    prefix = np.asarray(prefix, dtype=float).reshape(-1)
    if not np.isfinite(prefix).all() or not np.isfinite(p_response):
        raise ValueError("The decision vector must contain finite numbers")
    return np.concatenate((prefix, [float(np.clip(p_response, 0.0, 1.0))]))


class CyberDefenseProblem(ElementwiseProblem):
    def __init__(
        self,
        defense_items,
        all_data,
        ag,
        budget=1000,
        use_constraint=True,
        use_cache=True,
        game_iterations=GAME_OPTIMIZATION_ITERATIONS,
        game_learning_rate=GAME_OPTIMIZATION_LEARNING_RATE,
        *,
        model_options=None,
        infer_response_from_effectiveness=False,
        prune_tol=PRUNE_TOL_DEFAULT,
        game_termination=DEFAULT_GAME_TERMINATION,
    ):
        defense_count = len(defense_items)
        super().__init__(
            n_var=defense_count + 1,
            n_obj=2,
            n_ieq_constr=1 if use_constraint else 0,
            xl=np.zeros(defense_count + 1),
            xu=np.ones(defense_count + 1),
            vtype=float
        )
        self.defense_items = defense_items
        self.defense_count = defense_count
        self.all_data = all_data
        self.ag = ag
        self.budget = budget
        self.use_constraint = use_constraint
        self.use_cache = use_cache

        self.game_iterations = game_iterations
        self.game_learning_rate = game_learning_rate
        self.game_termination = resolve_game_termination(game_termination)
        self.model_options = None if model_options is None else resolve_model_options(model_options)
        if prune_tol is not None:
            if isinstance(prune_tol, bool):
                raise ValueError("prune_tol must be None or a nonnegative finite number")
            try:
                prune_tol = float(prune_tol)
            except (TypeError, ValueError) as error:
                raise ValueError("prune_tol must be None or a nonnegative finite number") from error
            if not np.isfinite(prune_tol) or prune_tol < 0:
                raise ValueError("prune_tol must be None or a nonnegative finite number")
        self.prune_tol = prune_tol
        if type(infer_response_from_effectiveness) is not bool:
            raise ValueError("infer_response_from_effectiveness must be bool")
        self.infer_response_from_effectiveness = infer_response_from_effectiveness

        self.base_ids = parse_ids_config(
            all_data["Set IDS"]
        )
        self.base_fw = parse_firewall_config(
            all_data["Set firewall"]
        )

        self.total_evals = 0
        self.cache_hits = 0
        if self.use_cache:
            self.eval_cache = {}

    def _evaluate(self, x, out, *args, **kwargs):
        normalized = normalize_decision_vector(x, self.defense_items)
        prefix, p_response = split_decision_vector(normalized, self.defense_count)
        out["X"] = normalized
        self.total_evals += 1

        if self.use_cache:
            cache_key = decision_key(normalized, self.defense_count, self.defense_items)

            
            if cache_key in self.eval_cache:
                self.cache_hits += 1
                cached_result = self.eval_cache[cache_key]
                out["F"] = cached_result["F"].copy()
                if self.use_constraint and "G" in cached_result:
                    out["G"] = cached_result["G"].copy()
                if self.use_constraint and "CV" in cached_result:
                    out["CV"] = cached_result["CV"].copy()
                return

        
        costs = [item[1] for item in self.defense_items]
        total_cost = float(np.dot(prefix.astype(float), costs))

        # Run simulation
        try:
            configuration = create_selected_configuration(
                prefix, self.defense_items, self.all_data
            )
            model_kwargs = {}
            if self.game_termination is not DEFAULT_GAME_TERMINATION:
                model_kwargs["game_termination"] = self.game_termination
            if self.model_options is not None:
                model_kwargs["model_options"] = self.model_options
                if not self.infer_response_from_effectiveness:
                    model_kwargs["has_response"] = any(
                        label in ("RP1", "RP2", "RP3") for label in configuration["selected"]
                    )
            results = run_simulation(
                state_graph=self.ag.state_paths,
                vuln_data=configuration["vuln_data_sim"],
                component_vuln_dist=self.all_data[
                    "Vulnerabilities distribution"
                ],
                ids_config=configuration["ids_config"],
                firewall_config=configuration["firewall_config"],
                patch_config=configuration["patch_config"],
                iterations=self.game_iterations,
                learning_rate=self.game_learning_rate,
                p_response=p_response,
                prune_tol=self.prune_tol,
                **model_kwargs,
            )
            _, total_def, _ = compute_total_expected_values(
                results
            )
            if not np.isfinite(total_def):
                raise ValueError(f"The simulation returned a nonfinite loss: {total_def!r}")
        except Exception as exc:
            raise RuntimeError(
                f"Simulation failed: decision={normalized.tolist()!r}; "
                f"budget={self.budget!r}; game_iterations={self.game_iterations!r}; "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        # Output result
        if self.use_constraint:
            out["F"] = np.array([
                total_cost,
                total_def
            ])
            out["G"] = np.array([
                total_cost - self.budget
            ])
            out["CV"] = np.where(out["G"] <= FEASIBILITY_TOL, 0.0, out["G"])
        else:
            penalty = max(
                0.0,
                total_cost - self.budget
            ) * 10000
            out["F"] = np.array([
                total_cost,
                total_def + penalty
            ])

        
        if self.use_cache:
            cache_result = {
                "F": out["F"].copy()
            }
            if self.use_constraint:
                cache_result["G"] = out["G"].copy()
                cache_result["CV"] = out["CV"].copy()
            self.eval_cache[cache_key] = cache_result

    def print_cache_stats(self):
        
        if self.use_cache:
            hit_rate = (
                self.cache_hits / self.total_evals * 100
                if self.total_evals > 0
                else 0
            )
            print("📊 Cache Statistics:")
            print(f"   Total evaluations: {self.total_evals}")
            print(f"   Cache hits: {self.cache_hits}")
            print(f"   Cache hit rate: {hit_rate:.2f}%")
            print(
                "   Unique solutions evaluated: "
                f"{len(self.eval_cache)}"
            )
