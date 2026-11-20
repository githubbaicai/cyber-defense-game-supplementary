

from __future__ import annotations

import numpy as np

from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from scipy.stats import t

from common.memetic import get_local_search_statistics
from common.termination import feasible_pareto_front

from features.macomp.config import CONFIDENCE_LEVEL


class FrontTrace(Callback):
    

    def __init__(self, budget: float):
        super().__init__()
        self.budget = float(budget)
        self.data["steps"] = []

    def notify(self, algorithm) -> None:
        front = feasible_pareto_front(algorithm.pop, self.budget)
        statistics = get_local_search_statistics(algorithm)
        total_fe = int(algorithm.evaluator.n_eval)

        record = {
            "Generation": int(algorithm.n_gen),
            "Total_FE": total_fe,
            "Genetic_FE": total_fe - statistics["Local_FE"],
            **statistics,
            "Trigger_HV": float(
                getattr(algorithm, "trigger_hv", np.nan)
            ),
            "Front": front.copy(),
        }

        steps = self.data["steps"]

        
        
        if steps and steps[-1]["Generation"] == record["Generation"]:
            if total_fe < steps[-1]["Total_FE"]:
                raise RuntimeError("FE cannot decrease within the same generation")
            steps[-1] = record
            return

        if steps and total_fe <= steps[-1]["Total_FE"]:
            raise RuntimeError("FE must increase strictly between generations")

        steps.append(record)


class EmpiricalReference:
    

    def __init__(self, observed_fronts, budget: float):
        usable = []

        for front in observed_fronts:
            if front is None:
                continue

            values = np.asarray(front, dtype=float).reshape(-1, 2)
            valid = (
                np.isfinite(values).all(axis=1)
                & (values[:, 0] <= float(budget) + 1e-8)
            )
            values = values[valid]

            if len(values):
                usable.append(values)

        if not usable:
            raise RuntimeError("No feasible front is available to construct the HV/IGD reference")

        pooled = np.unique(np.vstack(usable), axis=0)
        indices = NonDominatedSorting().do(
            pooled,
            only_non_dominated_front=True,
        )

        self.front = pooled[indices]
        order = np.lexsort((self.front[:, 1], self.front[:, 0]))
        self.front = self.front[order]

        
        
        self.ideal = pooled.min(axis=0)
        nadir = pooled.max(axis=0)
        self.span = np.where(
            nadir > self.ideal,
            nadir - self.ideal,
            1.0,
        )

        self.hv_reference_raw = np.maximum(
            nadir + 0.1 * self.span,
            np.asarray(
                [
                    float(budget) * 1.1,
                    float(nadir[1]) + 0.1 * self.span[1],
                ],
                dtype=float,
            ),
        )

        self._hv = HV(
            ref_point=(
                self.hv_reference_raw - self.ideal
            ) / self.span
        )
        self._igd = IGD((self.front - self.ideal) / self.span)

    def metrics(self, front) -> dict:
        
        values = np.asarray(front, dtype=float).reshape(-1, 2)

        if not len(values):
            return {"HV": 0.0, "IGD": np.nan}

        if not np.isfinite(values).all():
            raise ValueError("Metric calculation received nonfinite objective values")

        normalized = (values - self.ideal) / self.span
        return {
            "HV": float(self._hv.do(normalized)),
            "IGD": float(self._igd.do(normalized)),
        }

    def metadata(self) -> dict:
        
        return {
            "Reference_Type": "pooled_empirical_observed_front",
            "Reference_Size": int(len(self.front)),
            "Normalization_Ideal_Cost": float(self.ideal[0]),
            "Normalization_Ideal_Loss": float(self.ideal[1]),
            "Normalization_Span_Cost": float(self.span[0]),
            "Normalization_Span_Loss": float(self.span[1]),
            "HV_Reference_Cost": float(self.hv_reference_raw[0]),
            "HV_Reference_Loss": float(self.hv_reference_raw[1]),
        }


def mean_confidence_interval(
    values,
    confidence: float = CONFIDENCE_LEVEL,
) -> tuple[float, float, float, float]:
    
    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]

    if len(array) == 0:
        return np.nan, np.nan, np.nan, np.nan

    mean = float(array.mean())
    if len(array) < 2:
        return mean, np.nan, np.nan, np.nan

    standard_deviation = float(array.std(ddof=1))
    critical_value = float(
        t.ppf((1.0 + confidence) / 2.0, df=len(array) - 1)
    )
    half_width = (
        critical_value * standard_deviation / np.sqrt(len(array))
    )

    return (
        mean,
        standard_deviation,
        mean - half_width,
        mean + half_width,
    )