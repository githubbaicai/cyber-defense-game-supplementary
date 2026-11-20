

from collections import deque

import numpy as np

from pymoo.core.termination import Termination
from pymoo.indicators.hv import HV
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.robust import RobustTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from common.config import (
    TERMINATION_HV_REFERENCE_POINT,
    TERMINATION_HV_TARGET,
    TERMINATION_IDEAL_DISTANCE_TARGET,
    TERMINATION_LOSS_SCALE,
    TERMINATION_METRIC,
    TERMINATION_MIN_FRONT_SIZE,
    TERMINATION_MIN_GENERATIONS,
    TERMINATION_ROBUST_PERIOD,
    TERMINATION_STABLE_GENERATIONS,
    TERMINATION_TOLERANCE,
)


FEASIBILITY_TOL = 1e-8
_FEASIBILITY_TOL = FEASIBILITY_TOL


def _generation_number(algorithm):
    generation = getattr(algorithm, "n_gen", None)
    if (
        isinstance(generation, (bool, np.bool_))
        or not isinstance(generation, (int, np.integer))
        or generation < 1
    ):
        raise ValueError("algorithm.n_gen must be an integer greater than or equal to 1")
    return int(generation)


def _evaluation_count(algorithm, *, required=False):
    evaluator = getattr(algorithm, "evaluator", None)
    count = getattr(evaluator, "n_eval", None)
    if count is None and not required:
        return 0
    if (
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, (int, np.integer))
        or count < 0
    ):
        raise ValueError("algorithm.evaluator.n_eval must be a nonnegative integer; max_fe requires a valid FE count")
    return int(count)


def _positive_finite(value, name):
    
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc

    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")

    return result


def _reference_array(ref_point):
    
    reference = np.asarray(ref_point, dtype=float)

    if (
        reference.shape != (2,)
        or not np.isfinite(reference).all()
        or np.any(reference <= 1.0)
    ):
        raise ValueError(
            "ref_point must contain two finite components, both greater than 1"
        )

    return reference.copy()


def _constraint_feasibility(raw_values, number_of_points):
    
    if raw_values is None:
        return None

    values = np.asarray(raw_values)

    if values.size == 0 or all(
        item is None
        for item in values.flat
    ):
        return None

    values = np.asarray(raw_values, dtype=float)

    if (
        values.ndim == 0
        or values.shape[0] != number_of_points
    ):
        raise ValueError("Constraint and objective row counts in the population differ")

    values = values.reshape(number_of_points, -1)

    return (
        np.isfinite(values).all(axis=1)
        & (values <= _FEASIBILITY_TOL).all(axis=1)
    )


def feasible_pareto_front(population, budget):
    
    budget = _positive_finite(budget, "budget")
    empty = np.empty((0, 2), dtype=float)

    if population is None or len(population) == 0:
        return empty

    raw_f = population.get("F")
    if raw_f is None:
        return empty

    values = np.asarray(raw_f, dtype=float)

    if values.size == 0:
        return empty

    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Pareto convergence requires F to be an (n, 2) array")

    number_of_points = len(values)

    valid = (
        np.isfinite(values).all(axis=1)
        & (values[:, 0] <= budget + _FEASIBILITY_TOL)
    )

    
    cv_valid = _constraint_feasibility(
        population.get("CV"),
        number_of_points
    )
    g_valid = _constraint_feasibility(
        population.get("G"),
        number_of_points
    )

    if cv_valid is not None:
        valid &= cv_valid

    if g_valid is not None:
        valid &= g_valid

    feasible = values[valid]
    if len(feasible) == 0:
        return empty

    
    unique = np.unique(feasible, axis=0)

    indices = NonDominatedSorting().do(
        unique,
        only_non_dominated_front=True
    )
    front = unique[indices]

    
    order = np.lexsort(
        (front[:, 1], front[:, 0])
    )
    return front[order]


def calculate_front_metrics(
    front,
    budget,
    loss_scale,
    ref_point
):
    
    budget = _positive_finite(budget, "budget")
    loss_scale = _positive_finite(
        loss_scale,
        "loss_scale"
    )
    reference = _reference_array(ref_point)

    metrics = {
        "front_size": 0,
        "hv": np.nan,
        "ideal_distance": np.nan,
        "ideal_cost_norm": np.nan,
        "ideal_loss_norm": np.nan,
        "closest_cost": np.nan,
        "closest_loss": np.nan,
        "reference_valid": False,
        "valid_ideal": False,
    }

    if front is None:
        return metrics

    values = np.asarray(front, dtype=float)
    if values.size == 0:
        return metrics

    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("front must be an (n, 2) array")

    metrics["front_size"] = len(values)

    normalized = values / np.array(
        [budget, loss_scale],
        dtype=float
    )

    if not np.isfinite(normalized).all():
        return metrics

    distances = np.linalg.norm(
        normalized,
        axis=1
    )
    best_index = int(np.argmin(distances))

    metrics["ideal_distance"] = float(
        distances[best_index]
    )
    metrics["ideal_cost_norm"] = float(
        normalized[best_index, 0]
    )
    metrics["ideal_loss_norm"] = float(
        normalized[best_index, 1]
    )
    metrics["closest_cost"] = float(
        values[best_index, 0]
    )
    metrics["closest_loss"] = float(
        values[best_index, 1]
    )

    
    metrics["valid_ideal"] = bool(
        np.all(normalized >= 0.0)
    )

    
    
    metrics["reference_valid"] = bool(
        metrics["valid_ideal"]
        and np.all(normalized < reference)
    )

    if metrics["reference_valid"]:
        raw_hv = HV(
            ref_point=reference
        ).do(normalized)

        metrics["hv"] = float(
            raw_hv / np.prod(reference)
        )

    return metrics


def front_hausdorff_distance(first, second):
    
    fronts = []
    for name, values in (("first", first), ("second", second)):
        try:
            front = np.asarray(values, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a nonempty finite (n, 2) front") from exc
        if (
            front.ndim != 2
            or front.shape[1] != 2
            or len(front) == 0
            or not np.isfinite(front).all()
        ):
            raise ValueError(f"{name} must be a nonempty finite (n, 2) front")
        fronts.append(front)

    first_front, second_front = fronts
    distances = np.max(
        np.abs(first_front[:, None, :] - second_front[None, :, :]),
        axis=2,
    )
    return float(max(
        np.min(distances, axis=1).max(),
        np.min(distances, axis=0).max(),
    ))


class FrontWindowTermination(Termination):
    

    def __init__(
        self,
        *,
        max_gen,
        budget,
        loss_scale,
        tol=0.001,
        n_stable=10,
        min_gen=20,
        min_front_size=2,
        max_fe=None,
        ref_point=(1.1, 1.1),
    ):
        super().__init__()
        for name, value in (
            ("max_gen", max_gen),
            ("n_stable", n_stable),
            ("min_gen", min_gen),
            ("min_front_size", min_front_size),
            ("max_fe", max_fe),
        ):
            if name == "max_fe" and value is None:
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be an integer greater than or equal to 1")
        for name, value in (("budget", budget), ("loss_scale", loss_scale)):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be positive and finite")
        try:
            tolerance = float(tol)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("tol must be nonnegative and finite") from exc
        if isinstance(tol, (bool, np.bool_)) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tol must be nonnegative and finite")

        self.max_gen = int(max_gen)
        self.max_fe = None if max_fe is None else int(max_fe)
        self.budget = _positive_finite(budget, "budget")
        self.loss_scale = _positive_finite(loss_scale, "loss_scale")
        self.ref_point = _reference_array(ref_point)
        self.tol = tolerance
        self.n_stable = int(n_stable)
        self.min_gen = int(min_gen)
        self.min_front_size = int(min_front_size)
        self.metric = "front"
        self.history = []
        self._recent = deque(maxlen=self.n_stable + 1)
        self.last_front = np.empty((0, 2), dtype=float)
        self.stop_reason = None
        self._reported = False

    def _update(self, algorithm):
        generation = _generation_number(algorithm)
        total_fe = _evaluation_count(algorithm, required=self.max_fe is not None)
        cap_reason = (
            "max_fe"
            if self.max_fe is not None and total_fe >= self.max_fe
            else "max_generations" if generation >= self.max_gen else None
        )
        previous_generation = self.history[-1]["generation"] if self.history else None
        if previous_generation is not None and generation < previous_generation:
            raise ValueError("Generation count moved backwards; use a new FrontWindowTermination instance for a new optimization run")

        if generation == previous_generation:
            record = self.history[-1]
            record["total_fe"] = total_fe
            record["ma_ready"] = bool(getattr(algorithm, "early_termination_ready", True))
        else:
            if previous_generation is not None and generation != previous_generation + 1:
                self._recent.clear()
            monitor_error = ""
            try:
                front = feasible_pareto_front(algorithm.pop, self.budget)
            except Exception as exc:
                if cap_reason is None:
                    raise
                front = np.empty((0, 2), dtype=float)
                monitor_error = f"{type(exc).__name__}: {exc}"
            self.last_front = front.copy()
            with np.errstate(over="ignore", invalid="ignore"):
                normalized = front / np.array([self.budget, self.loss_scale])
            valid = len(front) >= self.min_front_size and np.isfinite(normalized).all()
            if valid:
                self._recent.append(normalized.copy())
            else:
                self._recent.clear()

            record = {
                "generation": generation,
                "total_fe": total_fe,
                "front_size": len(front),
                "window_ready": len(self._recent) == self.n_stable + 1,
                "front_window_distance": np.nan,
                "min_generation_reached": generation >= self.min_gen,
                "ma_ready": bool(getattr(algorithm, "early_termination_ready", True)),
                "hv": np.nan,
                "hv_status": "empty_front",
                "hv_detail": "",
                "reference_valid": False,
                "stop_reason": None,
                "monitor_error": monitor_error,
            }
            if record["window_ready"]:
                try:
                    record["front_window_distance"] = max(
                        front_hausdorff_distance(first, second)
                        for index, first in enumerate(self._recent)
                        for second in list(self._recent)[index + 1:]
                    )
                except Exception as exc:
                    if cap_reason is None:
                        raise
                    self._recent.clear()
                    record["window_ready"] = False
                    record["monitor_error"] = f"{type(exc).__name__}: {exc}"
            try:
                metrics = calculate_front_metrics(
                    front, self.budget, self.loss_scale, self.ref_point,
                )
                record["reference_valid"] = bool(metrics["reference_valid"])
                if len(front):
                    if not np.isfinite(normalized).all():
                        record["hv_status"] = "non_finite_normalization"
                    elif np.any(normalized < 0.0):
                        record["hv_status"] = "negative_objectives"
                    elif not metrics["reference_valid"]:
                        record["hv_status"] = "reference_not_covering_front"
                    elif not np.isfinite(metrics["hv"]):
                        record["hv_status"] = "invalid_hv"
                    else:
                        record["hv"] = float(metrics["hv"])
                        record["hv_status"] = "ok"
            except Exception as exc:
                record["hv_status"] = f"error:{type(exc).__name__}"
                record["hv_detail"] = str(exc)
            self.history.append(record)

        if cap_reason is not None:
            self.stop_reason = cap_reason
        elif (
            record["min_generation_reached"]
            and record["window_ready"]
            and record["front_window_distance"] <= self.tol
            and record["ma_ready"]
        ):
            self.stop_reason = "front_stable"
        else:
            self.stop_reason = None
        record["stop_reason"] = self.stop_reason
        if self.stop_reason is not None and not self._reported:
            print(
                f"[termination] generation={generation}; total_fe={total_fe}; "
                f"front_size={record['front_size']}; window_ready={record['window_ready']}; "
                f"front_window_distance={record['front_window_distance']}; "
                f"min_generation_reached={record['min_generation_reached']}; "
                f"ma_ready={record['ma_ready']}; hv={record['hv']}; "
                f"hv_status={record['hv_status']}; stop_reason={self.stop_reason}"
            )
            self._reported = True
        return 1.0 if self.stop_reason is not None else 0.0


class FitnessConvergenceTermination(Termination):
    

    def __init__(
        self,
        tol=TERMINATION_TOLERANCE,
        n_stable=TERMINATION_STABLE_GENERATIONS,
        metric=TERMINATION_METRIC,
        *,
        budget=None,
        loss_scale=TERMINATION_LOSS_SCALE,
        ref_point=TERMINATION_HV_REFERENCE_POINT,
        min_gen=TERMINATION_MIN_GENERATIONS,
        min_front_size=TERMINATION_MIN_FRONT_SIZE,
        hv_target=TERMINATION_HV_TARGET,
        ideal_distance_target=(
            TERMINATION_IDEAL_DISTANCE_TARGET
        )
    ):
        super().__init__()

        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be nonnegative and finite")

        if metric not in ("hv", "ideal", "both"):
            raise ValueError(
                "metric must be 'hv', 'ideal' or 'both'"
            )

        for name, value in (
            ("n_stable", n_stable),
            ("min_gen", min_gen),
            ("min_front_size", min_front_size),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be an integer greater than or equal to 1"
                )

        if budget is not None:
            budget = _positive_finite(
                budget,
                "budget"
            )

        loss_scale = _positive_finite(
            loss_scale,
            "loss_scale"
        )
        reference = _reference_array(ref_point)

        if hv_target is not None:
            hv_target = float(hv_target)
            if (
                not np.isfinite(hv_target)
                or not 0.0 <= hv_target <= 1.0
            ):
                raise ValueError(
                    "hv_target must be in [0, 1]"
                )

        if ideal_distance_target is not None:
            ideal_distance_target = float(
                ideal_distance_target
            )
            if (
                not np.isfinite(ideal_distance_target)
                or ideal_distance_target < 0.0
            ):
                raise ValueError(
                    "ideal_distance_target must be nonnegative and finite"
                )

        self.tol = float(tol)
        self.n_stable = int(n_stable)
        self.metric = metric
        self.budget = budget
        self.loss_scale = loss_scale
        self.ref_point = reference
        self.min_gen = int(min_gen)
        self.min_front_size = int(min_front_size)
        self.hv_target = hv_target
        self.ideal_distance_target = ideal_distance_target

        
        self.history = []

        
        self._recent = deque(
            maxlen=self.n_stable + 1
        )
        self.last_front = np.empty((0, 2), dtype=float)
        self.stop_reason = None

    def _budget_for(self, algorithm):
        if self.budget is not None:
            return self.budget

        problem = getattr(
            algorithm,
            "problem",
            None
        )
        problem_budget = getattr(
            problem,
            "budget",
            None
        )

        if problem_budget is None:
            raise ValueError(
                "budget was not provided, and algorithm.problem "
                "has no budget attribute"
            )

        return _positive_finite(
            problem_budget,
            "algorithm.problem.budget"
        )

    def _update(self, algorithm):
        generation = _generation_number(algorithm)
        total_fe = _evaluation_count(algorithm)
        previous_generation = self.history[-1]["generation"] if self.history else None
        if previous_generation is not None and generation < previous_generation:
            raise ValueError("Generation count moved backwards; use a new FitnessConvergenceTermination instance for a new optimization run")

        
        if (
            previous_generation == generation
        ):
            record = self.history[-1]
            record["total_fe"] = total_fe
            record["ma_ready"] = bool(getattr(algorithm, "early_termination_ready", True))
            record["converged"] = bool(
                generation >= self.min_gen
                and record["window_ready"]
                and record["stability_met"]
                and record["quality_met"]
                and record["ma_ready"]
            )
            self.stop_reason = "stability" if record["converged"] else None
            record["stop_reason"] = self.stop_reason
            return 1.0 if record["converged"] else 0.0

        if previous_generation is not None and generation != previous_generation + 1:
            self._recent.clear()

        budget = self._budget_for(algorithm)

        front = feasible_pareto_front(
            algorithm.pop,
            budget
        )
        self.last_front = front.copy()
        metrics = calculate_front_metrics(
            front,
            budget,
            self.loss_scale,
            self.ref_point
        )

        record = {
            "generation": generation,
            **metrics,
            "hv_range": np.nan,
            "ideal_coordinate_range": np.nan,
            "window_ready": False,
            "stability_met": False,
            "quality_met": False,
            "converged": False,
            "total_fe": total_fe,
            "ma_ready": bool(getattr(algorithm, "early_termination_ready", True)),
            "monitor_error": "",
            "stop_reason": None,
        }

        valid = (
            metrics["front_size"] >= self.min_front_size
            and metrics["valid_ideal"]
        )

        
        
        if self.metric in ("hv", "both"):
            valid = (
                valid
                and metrics["reference_valid"]
            )

        if self.hv_target is not None:
            valid = (
                valid
                and metrics["reference_valid"]
            )

        if valid:
            self._recent.append(record)
        else:
            
            
            self._recent.clear()

        if len(self._recent) == self.n_stable + 1:
            record["window_ready"] = True

            hv_values = np.array(
                [item["hv"] for item in self._recent],
                dtype=float
            )
            cost_values = np.array(
                [
                    item["ideal_cost_norm"]
                    for item in self._recent
                ],
                dtype=float
            )
            loss_values = np.array(
                [
                    item["ideal_loss_norm"]
                    for item in self._recent
                ],
                dtype=float
            )

            if np.isfinite(hv_values).all():
                record["hv_range"] = float(
                    np.ptp(hv_values)
                )

            
            
            record["ideal_coordinate_range"] = float(
                max(
                    np.ptp(cost_values),
                    np.ptp(loss_values)
                )
            )

            hv_stable = (
                self.metric == "ideal"
                or (
                    np.isfinite(record["hv_range"])
                    and record["hv_range"] <= self.tol
                )
            )

            ideal_stable = (
                self.metric == "hv"
                or (
                    record["ideal_coordinate_range"]
                    <= self.tol
                )
            )

            record["stability_met"] = bool(
                hv_stable and ideal_stable
            )

        hv_quality_met = (
            self.hv_target is None
            or (
                np.isfinite(metrics["hv"])
                and metrics["hv"] >= self.hv_target
            )
        )
        ideal_quality_met = (
            self.ideal_distance_target is None
            or (
                np.isfinite(metrics["ideal_distance"])
                and metrics["ideal_distance"]
                <= self.ideal_distance_target
            )
        )
        record["quality_met"] = bool(
            hv_quality_met and ideal_quality_met
        )

        record["converged"] = bool(
            generation >= self.min_gen
            and record["window_ready"]
            and record["stability_met"]
            and record["quality_met"]
            and record["ma_ready"]
        )

        self.stop_reason = "stability" if record["converged"] else None
        record["stop_reason"] = self.stop_reason

        self.history.append(record)

        return 1.0 if record["converged"] else 0.0


def _find_pareto_monitor(termination):
    
    if isinstance(termination, (FitnessConvergenceTermination, FrontWindowTermination)):
        return termination
    if isinstance(termination, RobustTermination):
        return _find_pareto_monitor(termination.termination)
    for child in getattr(termination, "terminations", ()):
        monitor = _find_pareto_monitor(child)
        if monitor is not None:
            return monitor
    return None


class _GenerationLimit(MaximumGenerationTermination):
    def __init__(self, max_gen):
        super().__init__(max_gen)
        self.stop_reason = None
        self.history = []

    def _update(self, algorithm):
        generation = _generation_number(algorithm)
        progress = super()._update(algorithm)
        self.stop_reason = "max_generations" if progress >= 1.0 else None
        record = {
            "generation": generation,
            "total_fe": _evaluation_count(algorithm),
            "stop_reason": self.stop_reason,
        }
        if self.history and self.history[-1]["generation"] == generation:
            self.history[-1].update(record)
        else:
            if self.history and generation < self.history[-1]["generation"]:
                raise ValueError("Generation count moved backwards; use a new termination instance for a new optimization run")
            self.history.append(record)
        return progress


class _EvaluationLimit(MaximumFunctionCallTermination):
    def _update(self, algorithm):
        return _evaluation_count(algorithm, required=True) / self.n_max_evals


class _GenerationAwareRobustTermination(RobustTermination):
    

    def __init__(self, termination, period):
        super().__init__(termination, period=period)
        self._last_generation = None

    def _update(self, algorithm):
        generation = _generation_number(algorithm)
        if self._last_generation is not None and generation < self._last_generation:
            raise ValueError("Generation count moved backwards; use a new termination instance for a new optimization run")
        progress = self.termination.update(algorithm)
        if generation == self._last_generation:
            self.history[-1] = progress
        else:
            if self._last_generation is not None and generation != self._last_generation + 1:
                self.history.clear()
            self.history.append(progress)
            self._last_generation = generation
        return min(self.history)


class MyTerminationSelector:
    

    @staticmethod
    def get_termination(
        strategy="convergence",
        max_gen=100,
        tol=TERMINATION_TOLERANCE,
        n_stable=TERMINATION_STABLE_GENERATIONS,
        period=TERMINATION_ROBUST_PERIOD,
        *,
        metric=TERMINATION_METRIC,
        budget=None,
        loss_scale=TERMINATION_LOSS_SCALE,
        ref_point=TERMINATION_HV_REFERENCE_POINT,
        min_gen=TERMINATION_MIN_GENERATIONS,
        min_front_size=TERMINATION_MIN_FRONT_SIZE,
        hv_target=TERMINATION_HV_TARGET,
        ideal_distance_target=(
            TERMINATION_IDEAL_DISTANCE_TARGET
        ),
        max_fe=None,
    ):
        
        if (
            isinstance(max_gen, (bool, np.bool_))
            or not isinstance(max_gen, (int, np.integer))
            or max_gen < 1
        ):
            raise ValueError(
                "max_gen must be an integer greater than or equal to 1"
            )

        if (
            isinstance(period, (bool, np.bool_))
            or not isinstance(period, (int, np.integer))
            or period < 1
        ):
            raise ValueError(
                "period must be an integer greater than or equal to 1"
            )

        if max_fe is not None and (
            isinstance(max_fe, (bool, np.bool_))
            or not isinstance(max_fe, (int, np.integer))
            or max_fe < 1
        ):
            raise ValueError("max_fe must be an integer greater than or equal to 1")

        if strategy not in (
            "convergence",
            "max_gen",
            "combined_or",
            "combined_and",
            "robust",
        ):
            print(
                f"⚠️ Unknown strategy {strategy!r}; "
                "using convergence with a max_gen safety cap"
            )
            strategy = "convergence"

        max_gen_term = _GenerationLimit(int(max_gen))
        safety_terms = [max_gen_term]
        if max_fe is not None:
            safety_terms.append(_EvaluationLimit(int(max_fe)))

        if strategy == "max_gen":
            print(
                "📊 Using only the maximum-generation stopping criterion: "
                f"max_gen={max_gen}"
            )
            return (
                max_gen_term if max_fe is None
                else MyCombinedTermination(*safety_terms)
            )

        convergence_term = FitnessConvergenceTermination(
            tol=tol,
            n_stable=n_stable,
            metric=metric,
            budget=budget,
            loss_scale=loss_scale,
            ref_point=ref_point,
            min_gen=min_gen,
            min_front_size=min_front_size,
            hv_target=hv_target,
            ideal_distance_target=(
                ideal_distance_target
            )
        )

        if strategy == "robust":
            convergence_term = _GenerationAwareRobustTermination(
                convergence_term,
                period=int(period)
            )

        print(
            "📊 Using feasible Pareto-front convergence: "
            f"strategy={strategy}, "
            f"metric={metric}, "
            f"tol={tol}, "
            f"n_stable={n_stable}, "
            f"min_gen={min_gen}, "
            f"max_gen={max_gen}"
        )

        if strategy == "combined_and":
            return MyAndTermination(
                convergence_term,
                *safety_terms,
            )

        
        return MyCombinedTermination(
            convergence_term,
            *safety_terms,
        )


class MyCombinedTermination(Termination):
    

    def __init__(self, *terminations):
        super().__init__()
        if not terminations:
            raise ValueError(
                "At least one stopping criterion is required"
            )
        self.terminations = tuple(
            terminations
        )
        self._reported = False
        self.stop_reason = None
        self._history = []

    @property
    def history(self):
        monitor = _find_pareto_monitor(self)
        return monitor.history if monitor is not None else self._history

    @property
    def last_front(self):
        monitor = _find_pareto_monitor(self)
        return monitor.last_front if monitor is not None else np.empty((0, 2), dtype=float)

    def _update(self, algorithm):
        generation = _generation_number(algorithm)
        safety_terms = [
            term for term in self.terminations
            if isinstance(term, (MaximumGenerationTermination, MaximumFunctionCallTermination))
        ]
        requires_fe = any(isinstance(term, MaximumFunctionCallTermination) for term in safety_terms)
        total_fe = _evaluation_count(algorithm, required=requires_fe)
        for term in safety_terms:
            term.update(algorithm)
        reached_fe = any(
            isinstance(term, MaximumFunctionCallTermination) and term.has_terminated()
            for term in safety_terms
        )
        reached_generation = any(
            isinstance(term, MaximumGenerationTermination) and term.has_terminated()
            for term in safety_terms
        )
        cap_reason = "max_fe" if reached_fe else "max_generations" if reached_generation else None
        other_terms = [term for term in self.terminations if term not in safety_terms]
        for term in other_terms:
            try:
                term.update(algorithm)
            except Exception as exc:
                if cap_reason is None:
                    raise
                monitor = _find_pareto_monitor(term)
                if monitor is not None:
                    monitor._recent.clear()
                    monitor.last_front = np.empty((0, 2), dtype=float)
                    record = {
                        "generation": generation, "total_fe": total_fe,
                        "front_size": 0, "window_ready": False,
                        "hv": np.nan, "ideal_distance": np.nan,
                        "reference_valid": False,
                        "monitor_error": f"{type(exc).__name__}: {exc}",
                        "stop_reason": cap_reason,
                    }
                    if monitor.history and monitor.history[-1]["generation"] == generation:
                        monitor.history[-1].update(record)
                    else:
                        monitor.history.append(record)
        converged = bool(other_terms) and (
            all(term.has_terminated() for term in other_terms)
            if isinstance(self, MyAndTermination)
            else any(term.has_terminated() for term in other_terms)
        )
        self.stop_reason = cap_reason or ("stability" if converged else None)
        record = {"generation": generation, "total_fe": total_fe, "stop_reason": self.stop_reason}
        if self.history and self.history[-1]["generation"] == generation:
            self.history[-1].update(record)
        else:
            self.history.append(record)
        if self.stop_reason is not None and not self._reported:
            print(
                f"[termination] generation={generation}; total_fe={total_fe}; "
                f"stop_reason={self.stop_reason}"
            )
            self._reported = True
        if self.stop_reason is not None:
            return 1.0
        return (
            0.0 if isinstance(self, MyAndTermination)
            else max(term.perc for term in self.terminations)
        )


class MyAndTermination(MyCombinedTermination):
    pass

