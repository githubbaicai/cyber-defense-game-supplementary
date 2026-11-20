

from dataclasses import dataclass
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pymoo.core.individual import Individual
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from common.memetic import is_feasible_individual
from common.problem import split_decision_vector


@dataclass(frozen=True)
class FrontSnapshot:
    

    front: np.ndarray
    X: np.ndarray
    defense_count: int
    kind: str
    resource_configuration_count: int
    display_compatible_size: int
    individuals: tuple = ()

    @property
    def F(self):
        return self.front

    @property
    def decisions(self):
        result = []
        for decision in self.X:
            prefix, probability = split_decision_vector(decision, self.defense_count)
            result.append({
                "X_Bits": "".join(str(int(value)) for value in prefix),
                "P_Response": probability,
                "X_JSON": json.dumps(decision.tolist(), separators=(",", ":")),
            })
        return result

    def __len__(self):
        return len(self.front)


def _objective_array(front):
    values = np.asarray(front, dtype=float)
    if values.size == 0:
        return np.empty((0, 2), dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("front must be a finite (n, 2) array")
    return values


def select_front(population, budget, defense_count, kind="strict"):
    
    if kind not in ("strict", "display_compatible"):
        raise ValueError("kind must be strict or display_compatible")
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("budget must be positive and finite")
    if isinstance(defense_count, bool) or not isinstance(defense_count, (int, np.integer)) or defense_count < 0:
        raise ValueError("defense_count must be a nonnegative integer")

    candidates = []
    for individual in ([] if population is None else population):
        raw = individual.get("F_True") if hasattr(individual, "get") else getattr(individual, "F_True", None)
        values = np.asarray(individual.F if raw is None else raw, dtype=float).reshape(-1)
        view = SimpleNamespace(F=values, G=getattr(individual, "G", None), CV=getattr(individual, "CV", None))
        if not is_feasible_individual(view, budget):
            continue
        decision = np.asarray(individual.X, dtype=float).reshape(-1).copy()
        prefix, probability = split_decision_vector(decision, defense_count)
        copy = Individual(X=decision.copy(), F=values.copy())
        copy.G = np.asarray(view.G, dtype=float).copy() if view.G is not None else np.empty(0)
        copy.CV = np.asarray(view.CV, dtype=float).copy() if view.CV is not None else np.zeros(1)
        candidates.append((values.copy(), decision, tuple(bool(value) for value in prefix), probability, copy))

    if not candidates:
        return FrontSnapshot(np.empty((0, 2)), np.empty((0, defense_count + 1)), defense_count, kind, 0, 0)

    values = np.vstack([candidate[0] for candidate in candidates])
    indices = NonDominatedSorting().do(values, only_non_dominated_front=True)
    nondominated = [candidates[int(index)] for index in indices]
    legacy_resources = {}
    for candidate in sorted(nondominated, key=lambda candidate: (float(candidate[0][0]), float(candidate[0][1]))):
        legacy_resources.setdefault(candidate[2], candidate)
    legacy_representatives = list(legacy_resources.values())
    display = [
        candidate for candidate in legacy_representatives
        if not any(
            other is not candidate
            and other[0][1] <= candidate[0][1] + 1e-3
            and other[0][0] < candidate[0][0] - 1e-3
            for other in legacy_representatives
        )
    ]
    selected = nondominated.copy()
    selected.sort(key=lambda candidate: (
        float(candidate[0][0]), float(candidate[0][1]), candidate[2], candidate[3], tuple(candidate[1]),
    ))
    resources = {}
    for candidate in selected:
        resources.setdefault(candidate[2], candidate)
    representatives = list(resources.values())
    if kind == "display_compatible":
        selected = display
    else:
        objectives = {}
        for candidate in representatives:
            objectives.setdefault(tuple(candidate[0]), candidate)
        selected = list(objectives.values())
    front = np.vstack([candidate[0] for candidate in selected]) if selected else np.empty((0, 2))
    decisions = np.vstack([candidate[1] for candidate in selected]) if selected else np.empty((0, defense_count + 1))
    return FrontSnapshot(front, decisions, defense_count, kind, len(representatives), len(display), tuple(candidate[4] for candidate in selected))


@dataclass(frozen=True)
class EmpiricalReference:
    

    front: np.ndarray
    ideal: np.ndarray
    span: np.ndarray
    hv_reference_raw: np.ndarray
    status: str
    reference_id: str

    def metadata(self):
        return {
            "Reference_ID": self.reference_id,
            "Reference_Type": "empirical_observed_reference",
            "Reference_Status": self.status,
            "Reference_Front_Size": len(self.front),
            "HV_Reference_Cost": float(self.hv_reference_raw[0]),
            "HV_Reference_Loss": float(self.hv_reference_raw[1]),
            "Normalization_Ideal_Cost": float(self.ideal[0]),
            "Normalization_Ideal_Loss": float(self.ideal[1]),
            "Normalization_Span_Cost": float(self.span[0]),
            "Normalization_Span_Loss": float(self.span[1]),
        }

    @property
    def records(self):
        return [
            {
                "Point_ID": point_id,
                "Total_Cost": float(point[0]),
                "Total_Defender_Loss": float(point[1]),
                **self.metadata(),
            }
            for point_id, point in enumerate(self.front, start=1)
        ]

    def frame(self):
        columns = ["Point_ID", "Total_Cost", "Total_Defender_Loss", *self.metadata()]
        return pd.DataFrame(self.records, columns=columns)

    def evaluate(self, front):
        values = _objective_array(front)
        status = self.status if self.status != "ok" else "empty_front" if not len(values) else "ok"
        result = {"HV": np.nan, "IGD": np.nan, "GD": np.nan, "Metric_Status": status, "HV_Status": status}
        if status != "ok":
            return result
        normalized = (values - self.ideal) / self.span
        normalized_reference = (self.front - self.ideal) / self.span
        hv_reference = (self.hv_reference_raw - self.ideal) / self.span
        if not np.isfinite(normalized).all() or not np.isfinite(hv_reference).all():
            result.update(Metric_Status="non_finite_normalization", HV_Status="non_finite_normalization")
            return result
        result["IGD"] = float(IGD(normalized_reference).do(normalized))
        result["GD"] = float(GD(normalized_reference).do(normalized))
        if np.all(normalized < hv_reference):
            result["HV"] = float(HV(ref_point=hv_reference).do(normalized))
        else:
            result.update(Metric_Status="reference_not_covering_front", HV_Status="reference_not_covering_front")
        return result


def build_empirical_reference(fronts, budget, reference_cost_factor=1.1, reference_loss_floor=None):
    
    for name, value in (("budget", budget), ("reference_cost_factor", reference_cost_factor)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if reference_loss_floor is not None and not np.isfinite(reference_loss_floor):
        raise ValueError("reference_loss_floor must be finite")
    available = []
    for front in fronts:
        values = _objective_array(front.front if isinstance(front, FrontSnapshot) else front)
        if len(values):
            available.append(values)
    if not available:
        return EmpiricalReference(np.empty((0, 2)), np.full(2, np.nan), np.full(2, np.nan), np.full(2, np.nan), "no_feasible_reference", "")
    pooled = np.unique(np.vstack(available), axis=0)
    indices = NonDominatedSorting().do(pooled, only_non_dominated_front=True)
    empirical = pooled[indices]
    empirical = empirical[np.lexsort((empirical[:, 1], empirical[:, 0]))]
    ideal = pooled.min(axis=0)
    nadir = pooled.max(axis=0)
    span = np.where(nadir > ideal, nadir - ideal, 1.0)
    hv_reference_raw = nadir + 0.1 * span
    hv_reference_raw[0] = max(hv_reference_raw[0], budget * reference_cost_factor)
    if reference_loss_floor is not None:
        hv_reference_raw[1] = max(hv_reference_raw[1], reference_loss_floor)
    data = np.concatenate((empirical.ravel(), ideal, span, hv_reference_raw)).astype("<f8")
    reference_id = sha256(data.tobytes()).hexdigest()
    return EmpiricalReference(empirical, ideal, span, hv_reference_raw, "ok", reference_id)
