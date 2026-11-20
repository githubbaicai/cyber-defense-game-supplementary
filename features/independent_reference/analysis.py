"""Freeze independent search observations and reevaluate saved comparison fronts."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from common.optimization_fronts import build_empirical_reference
from common.termination import FEASIBILITY_TOL


RUN_KEY = ["Experiment_ID", "Protocol", "Budget_ID", "Algorithm", "Run", "Seed"]
GROUP_KEY = RUN_KEY[:3]
OBJECTIVES = ["Total_Cost", "Total_Defender_Loss"]
REFERENCE_TYPE = "independent_high_budget_approximation"
COMPARATORS = ("NSGA2", "SPEA2", "MOEA/D")


def _require(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _values(frame, name):
    _require(frame, OBJECTIVES, name)
    try:
        values = frame[OBJECTIVES].to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} has invalid objectives") from error
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} requires finite nonnegative objectives")
    return values


def _metadata(reference):
    return {**reference.metadata(), "Reference_Type": REFERENCE_TYPE}


def freeze_reference(candidates, budget, *, hv_reference=None):
    """Use only independent candidates for the front, normalization and HV point."""
    if isinstance(budget, bool) or not np.isfinite(budget) or budget <= 0:
        raise ValueError("budget must be finite and positive")
    values = _values(candidates, "independent candidates")
    if not len(values):
        raise ValueError("independent candidates are empty")
    if (values[:, 0] > budget + FEASIBILITY_TOL).any():
        raise ValueError("independent candidates exceed the defense budget")
    reference = build_empirical_reference([values], budget)
    if hv_reference is not None:
        point = np.asarray(hv_reference, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all() or not np.all(reference.front < point):
            raise ValueError("hv_reference must strictly cover the independent front")
        reference = replace(reference, hv_reference_raw=point.copy())
    content = np.concatenate((reference.front.ravel(), reference.ideal,
                              reference.span, reference.hv_reference_raw)).astype("<f8")
    reference = replace(reference, reference_id=sha256(content.tobytes()).hexdigest())
    for array in (reference.front, reference.ideal, reference.span, reference.hv_reference_raw):
        array.setflags(write=False)
    frame = reference.frame()
    frame["Reference_Type"] = REFERENCE_TYPE
    return reference, frame


def _comparison_mask(frame):
    valid = pd.Series(True, index=frame.index)
    if "Status" in frame:
        valid &= frame["Status"].eq("ok")
    for column in ("Comparison_Valid", "Protocol_Completed"):
        if column in frame:
            valid &= frame[column].astype(str).str.lower().eq("true")
    return valid


def _summary(metrics):
    names = ("HV", "IGD", "GD", "Runtime_Seconds", "Run_Wall_Seconds",
             "Pareto_Size", "Generations", "Total_FE", "Genetic_FE", "Local_FE")
    rows = []
    for keys, group in metrics.groupby([*GROUP_KEY, "Algorithm"], sort=True):
        valid = group.loc[_comparison_mask(group)]
        row = dict(zip([*GROUP_KEY, "Algorithm"], keys))
        row.update(Runs=len(group), Valid_Runs=len(valid), Failed_Runs=len(group) - len(valid))
        for name in names:
            if name not in group:
                continue
            values = pd.to_numeric(valid[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
            row.update({f"{name}_Mean": values.mean(), f"{name}_Std": values.std(ddof=1),
                        f"{name}_Valid_Runs": int(values.notna().sum()),
                        f"{name}_Invalid_Runs": len(group) - int(values.notna().sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def _statistical_tests(metrics):
    generator = np.random.default_rng(20261104)
    rows = []
    for keys, group in metrics.groupby(GROUP_KEY, sort=True):
        for metric in ("HV", "IGD"):
            for comparator in COMPARATORS:
                samples = []
                for algorithm in ("NSGA2_MA", comparator):
                    values = pd.to_numeric(group.loc[group["Algorithm"].eq(algorithm)
                                           & _comparison_mask(group), metric], errors="coerce")
                    samples.append(np.sort(values[np.isfinite(values)].to_numpy(dtype=float)))
                first, second = samples
                row = {**dict(zip(GROUP_KEY, keys)), "Metric": metric,
                       "Algorithm_A": "NSGA2_MA", "Algorithm_B": comparator,
                       "N_A": len(first), "N_B": len(second),
                       "Statistic_U": np.nan, "P_Value_Unadjusted": np.nan,
                       "Mean_Difference_A_Minus_B": np.nan,
                       "Mean_Difference_CI_Lower": np.nan, "Mean_Difference_CI_Upper": np.nan,
                       "Rank_Biserial_A_Minus_B": np.nan, "Status": "insufficient_samples",
                       "Better_Direction": "higher" if metric == "HV" else "lower",
                       "Statistical_Design": "independent_samples; two-sided Mann-Whitney U",
                       "CI_Design": "independent bootstrap; pointwise 95%; 1000 resamples",
                       "Bootstrap_Seed": 20261104, "Bootstrap_Resamples": 1000}
                if min(len(first), len(second)) >= 2:
                    test = mannwhitneyu(first, second, alternative="two-sided")
                    differences = np.empty(1000)
                    for sample_index in range(len(differences)):
                        differences[sample_index] = (generator.choice(first, len(first), replace=True).mean()
                                                     - generator.choice(second, len(second), replace=True).mean())
                    lower, upper = np.quantile(differences, [0.025, 0.975])
                    row.update(Statistic_U=float(test.statistic), P_Value_Unadjusted=float(test.pvalue),
                               Mean_Difference_A_Minus_B=float(first.mean() - second.mean()),
                               Mean_Difference_CI_Lower=float(lower), Mean_Difference_CI_Upper=float(upper),
                               Rank_Biserial_A_Minus_B=float(2 * test.statistic / (len(first) * len(second)) - 1),
                               Status="ok")
                rows.append(row)
    tests = pd.DataFrame(rows)
    tests["P_Value_Holm"] = np.nan
    tests["Holm_Family_Size"] = len(tests)
    tests["Holm_Family"] = "all_groups; MA versus three comparators; HV and IGD"
    previous = 0.0
    for rank, index in enumerate(tests["P_Value_Unadjusted"].dropna().sort_values(kind="stable").index):
        previous = max(previous, min(1.0, (len(tests) - rank) * tests.at[index, "P_Value_Unadjusted"]))
        tests.at[index, "P_Value_Holm"] = previous
    tests["Reject_Holm_0_05"] = tests["P_Value_Holm"].lt(0.05)
    return tests


def _diagnostic(values, reference):
    weakly_covered = np.any(np.all(reference.front[:, None, :] <= values[None, :, :], axis=2), axis=0)
    dominates = (np.all(values[:, None, :] <= reference.front[None, :, :], axis=2)
                 & np.any(values[:, None, :] < reference.front[None, :, :], axis=2))
    dominated_count = int(np.any(dominates, axis=0).sum())
    uncovered_count = int((~weakly_covered).sum())
    outside_count = int(np.any(values >= reference.hv_reference_raw, axis=1).sum())
    return {"Evaluated_Front_Size": len(values),
            "Reference_Points_Dominated": dominated_count,
            "Evaluated_Points_Not_Weakly_Dominated": uncovered_count,
            "Evaluated_Points_Outside_HV_Reference": outside_count,
            "Reference_Coverage_Status": "coverage_gap" if dominated_count or uncovered_count else "covered",
            "Evaluated_Points_Added_To_Reference": 0,
            **_metadata(reference)}


def reevaluate(source, reference):
    """Recompute final strict-run metrics while keeping saved objectives and runtime."""
    if reference.status != "ok" or not len(reference.front):
        raise ValueError("a nonempty frozen reference is required")
    metrics = source["performance_metrics_raw"].copy(deep=True)
    points = source["all_pareto_points"].copy(deep=True)
    _require(metrics, [*RUN_KEY, "HV", "IGD", "GD"], "performance_metrics_raw")
    _require(points, [*RUN_KEY, *OBJECTIVES, "Stage", "Front_Kind"], "all_pareto_points")
    if metrics.empty or metrics.duplicated(RUN_KEY).any() or metrics[RUN_KEY].isna().any().any():
        raise ValueError("metrics require unique nonempty run identities")
    strict = points.loc[points["Stage"].eq("final") & points["Front_Kind"].eq("strict")]
    _values(strict, "final strict points")
    known_keys = set(metrics[RUN_KEY].itertuples(index=False, name=None))
    if not set(strict[RUN_KEY].itertuples(index=False, name=None)).issubset(known_keys):
        raise ValueError("final strict points have unknown run identities")
    fronts = {keys: group[OBJECTIVES].to_numpy(dtype=float, copy=True)
              for keys, group in strict.groupby(RUN_KEY, sort=False)}
    for name in ("HV", "IGD", "GD", "Reference_ID"):
        metrics[f"Original_{name}"] = metrics[name] if name in metrics else ""
    diagnostics = []
    for index, row in metrics.iterrows():
        keys = tuple(row[name] for name in RUN_KEY)
        values = fronts.get(keys, np.empty((0, 2)))
        result = {**reference.evaluate(values), **_metadata(reference)}
        for name, value in result.items():
            metrics.at[index, name] = value
        diagnostics.append({**dict(zip(RUN_KEY, keys)), **_diagnostic(values, reference)})
    if "best_pareto_fronts" in source:
        best = source["best_pareto_fronts"].copy(deep=True)
    else:
        representative_seed = source.get("representative_seed")
        if representative_seed is None:
            raise ValueError("source requires best_pareto_fronts or representative_seed")
        best = strict.loc[strict["Seed"].eq(representative_seed)].copy(deep=True)
        best["Selection_Rule"] = "fixed_seed_before_running"
    return {"performance_metrics_raw": metrics, "performance_summary": _summary(metrics),
            "all_pareto_points": points, "best_pareto_fronts": best,
            "statistical_tests": _statistical_tests(metrics),
            "reference_diagnostics": pd.DataFrame(diagnostics)}
