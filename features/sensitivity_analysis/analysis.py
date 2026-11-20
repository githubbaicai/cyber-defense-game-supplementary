"""Saltelli designs and output-specific Sobol inference without optimization."""

from importlib.metadata import version
from itertools import combinations
import warnings

import numpy as np
from SALib.analyze import sobol as sobol_analyzer
from SALib.sample import sobol as sobol_sampler


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _base_n(value):
    result = _integer(value, "base_n", 2)
    if result & (result - 1):
        raise ValueError("base_n must be a power of two")
    return result


def _problem_definition(registry):
    if isinstance(registry, dict):
        if registry.get("groups"):
            raise ValueError("Use explicit shared factors instead of SALib groups")
        if registry.get("dists") and any(item != "unif" for item in registry["dists"]):
            raise ValueError("Only independent uniform latent factors are supported")
        names = list(registry.get("names", []))
        bounds = registry.get("bounds", [])
        if registry.get("num_vars", len(names)) != len(names):
            raise ValueError("num_vars does not match factor names")
    else:
        entries = list(registry)
        names = [item["name"] for item in entries]
        bounds = [item.get("bounds", [item.get("lower"), item.get("upper")]) for item in entries]
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Factor names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Factor names must be unique")
    try:
        limits = np.asarray(bounds, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Each factor needs two finite bounds") from error
    if limits.shape != (len(names), 2) or not np.isfinite(limits).all():
        raise ValueError("Each factor needs two finite bounds")
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Factor lower bounds must be below upper bounds")
    return {"num_vars": len(names), "names": names, "bounds": limits.tolist()}


def build_design(
    registry, base_n, *, calc_second_order=False, sample_seed=42,
    scramble=True, design_id=None,
):
    """Generate the maximum-N design with explicit, zero-based block metadata."""
    base_n = _base_n(base_n)
    sample_seed = _integer(sample_seed, "sample_seed")
    if not isinstance(calc_second_order, bool) or not isinstance(scramble, bool):
        raise ValueError("calc_second_order and scramble must be boolean")
    problem = _problem_definition(registry)
    dimension = problem["num_vars"]
    block_size = (2 * dimension + 2) if calc_second_order else (dimension + 2)
    design_id = str(design_id) if design_id is not None else (
        f"saltelli_N{base_n}_D{dimension}_S{sample_seed}_O{2 if calc_second_order else 1}"
    )
    samples = sobol_sampler.sample(
        problem, base_n, calc_second_order=calc_second_order,
        scramble=scramble, seed=sample_seed,
    )
    roles = [("A", None)] + [(f"AB_{index}", index) for index in range(dimension)]
    if calc_second_order:
        roles += [(f"BA_{index}", index) for index in range(dimension)]
    roles += [("B", None)]
    rows = []
    for base_index in range(base_n):
        for role, factor_index in roles:
            rows.append({
                "Design_ID": design_id, "Base_N": base_n,
                "Sample_ID": len(rows) + 1, "Saltelli_Block_ID": base_index,
                "Base_Index": base_index, "Sample_Role": role,
                "Factor_Index": factor_index,
                "Factor_Name": problem["names"][factor_index] if factor_index is not None else None,
                "D": dimension, "Calc_Second_Order": calc_second_order,
            })
    return {
        "problem": problem, "X": samples, "rows": rows,
        "base_n": base_n, "factor_names": list(problem["names"]),
        "calc_second_order": calc_second_order, "design_id": design_id,
        "block_size": block_size, "sample_seed": sample_seed,
        "scramble": scramble, "salib_version": version("SALib"),
    }


def base_sample_mask(design):
    """Identify A/B observations; hybrid rows are not independent replicates."""
    return np.asarray([row["Sample_Role"] in ("A", "B") for row in design["rows"]], dtype=bool)


def _validate_design(design):
    problem = _problem_definition(design["problem"])
    base_n = _base_n(design["base_n"])
    second_order = design["calc_second_order"]
    if not isinstance(second_order, bool):
        raise ValueError("Design second-order flag must be boolean")
    dimension = problem["num_vars"]
    block_size = (2 * dimension + 2) if second_order else (dimension + 2)
    expected_rows = base_n * block_size
    if design["block_size"] != block_size:
        raise ValueError("Design block_size does not match its factor/order definition")
    samples = np.asarray(design["X"], dtype=float)
    if samples.shape != (expected_rows, dimension) or not np.isfinite(samples).all():
        raise ValueError("Design X has an invalid shape or nonfinite input")
    if len(design["rows"]) != expected_rows:
        raise ValueError("Design metadata does not contain every Saltelli row")
    roles = [("A", None)] + [(f"AB_{index}", index) for index in range(dimension)]
    if second_order:
        roles += [(f"BA_{index}", index) for index in range(dimension)]
    roles += [("B", None)]
    for row_index, metadata in enumerate(design["rows"]):
        role, factor_index = roles[row_index % block_size]
        block_index = row_index // block_size
        expected = {
            "Sample_ID": row_index + 1, "Saltelli_Block_ID": block_index,
            "Base_Index": block_index, "Sample_Role": role,
            "Factor_Index": factor_index, "D": dimension,
            "Calc_Second_Order": second_order, "Base_N": base_n,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Saltelli metadata order is invalid at row {row_index + 1}")
        block_start = block_index * block_size
        first = samples[block_start]
        last = samples[block_start + block_size - 1]
        if factor_index is not None:
            expected_sample = first.copy() if role.startswith("AB_") else last.copy()
            expected_sample[factor_index] = last[factor_index] if role.startswith("AB_") else first[factor_index]
            if not np.array_equal(samples[row_index], expected_sample):
                raise ValueError(f"Saltelli hybrid pairing is invalid at row {row_index + 1}")
    return problem, base_n, block_size


def _undefined_indices(names, metric, design, base_n, status):
    common = {
        "Design_ID": design["design_id"], "Base_N": base_n,
        "Metric": metric, "Status": status,
        "Output_Is_Constant": status == "undefined_zero_variance",
    }
    rows = [{
        **common, "Record_Type": "S1_ST", "Parameter": name, "Parameter_2": None,
        "S1": np.nan, "S1_Conf": np.nan, "S1_CI_Lower": np.nan, "S1_CI_Upper": np.nan,
        "ST": np.nan, "ST_Conf": np.nan, "ST_CI_Lower": np.nan, "ST_CI_Upper": np.nan,
        "S2": np.nan, "S2_Conf": np.nan, "S2_CI_Lower": np.nan, "S2_CI_Upper": np.nan,
    } for name in names]
    if design["calc_second_order"]:
        rows += [{
            **common, "Record_Type": "S2", "Parameter": names[first], "Parameter_2": names[second],
            "S1": np.nan, "S1_Conf": np.nan, "ST": np.nan, "ST_Conf": np.nan,
            "S2": np.nan, "S2_Conf": np.nan, "S2_CI_Lower": np.nan, "S2_CI_Upper": np.nan,
        } for first, second in combinations(range(len(names)), 2)]
    return rows


def _one_level(problem, design, metric, values, base_n, ci_seed, num_resamples, conf_level):
    rows = _undefined_indices(problem["names"], metric, design, base_n, "invalid_design")
    if values is None or values.shape != (design["base_n"] * design["block_size"],):
        return rows, "Output length/type does not match the complete Saltelli design"
    prefix = values[:base_n * design["block_size"]]
    failed = np.flatnonzero(~np.isfinite(prefix))
    if failed.size:
        return rows, f"Nonfinite output at Sample_ID {','.join(str(int(index + 1)) for index in failed)}"
    centered = prefix - prefix.mean()
    output_scale = float(np.max(np.abs(centered)))
    if output_scale == 0.0:
        return _undefined_indices(problem["names"], metric, design, base_n, "undefined_zero_variance"), "Output is constant"
    normalized = centered / output_scale
    base_values = normalized.reshape(base_n, design["block_size"])[:, [0, -1]]
    if np.var(base_values) == 0.0:
        return _undefined_indices(problem["names"], metric, design, base_n, "undefined_zero_variance"), "A/B output variance is zero"
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            estimates = sobol_analyzer.analyze(
                problem, normalized, calc_second_order=design["calc_second_order"],
                num_resamples=num_resamples, conf_level=conf_level,
                print_to_console=False, seed=ci_seed,
            )
        for row_index, name in enumerate(problem["names"]):
            row = rows[row_index]
            row["Status"] = "ok"
            for order in ("S1", "ST"):
                value = float(estimates[order][row_index])
                halfwidth = float(estimates[f"{order}_conf"][row_index])
                row.update({
                    order: value, f"{order}_Conf": halfwidth,
                    f"{order}_CI_Lower": value - halfwidth,
                    f"{order}_CI_Upper": value + halfwidth,
                })
            if not all(np.isfinite(row[key]) for key in ("S1", "ST", "S1_Conf", "ST_Conf")):
                row["Status"] = "unstable_estimate"
        if design["calc_second_order"]:
            for offset, (first, second) in enumerate(combinations(range(problem["num_vars"]), 2)):
                row = rows[problem["num_vars"] + offset]
                value = float(estimates["S2"][first, second])
                halfwidth = float(estimates["S2_conf"][first, second])
                row.update({
                    "Status": "ok" if np.isfinite(value) and np.isfinite(halfwidth) else "unstable_estimate",
                    "S2": value, "S2_Conf": halfwidth,
                    "S2_CI_Lower": value - halfwidth, "S2_CI_Upper": value + halfwidth,
                })
        return rows, " | ".join(dict.fromkeys(str(item.message) for item in captured))
    except (ValueError, RuntimeError, FloatingPointError) as error:
        return rows, f"Sobol estimation failed: {error}"


def analyze_outputs(
    design, outputs, *, bootstrap_seed=2026, ci_seed=None, num_resamples=1000,
    conf_level=0.95, n_levels=None,
):
    """Analyze complete scalar outputs separately and diagnose complete N prefixes.

    The final-N estimates are returned in sobol_indices. diagnostics contains
    every requested N and factor, including failures and adjacent-N changes.
    SALib bootstrap resamples base-block indices, retaining A/B/hybrid pairing.
    """
    problem, maximum_n, block_size = _validate_design(design)
    bootstrap_seed = _integer(bootstrap_seed if ci_seed is None else ci_seed, "bootstrap_seed")
    num_resamples = _integer(num_resamples, "num_resamples", 2)
    if not np.isfinite(conf_level) or not 0.0 < conf_level < 1.0:
        raise ValueError("conf_level must be between zero and one")
    levels = sorted(set(_base_n(value) for value in (n_levels if n_levels is not None else [maximum_n])))
    if not levels or levels[-1] != maximum_n:
        raise ValueError("n_levels must include the design maximum N and cannot exceed it")
    actual_ci_seed = bootstrap_seed or int(np.random.default_rng(0).integers(1, 2**31 - 1))
    indices = []
    diagnostics = []
    for metric, raw_values in outputs.items():
        try:
            values = np.asarray(raw_values, dtype=float)
        except (TypeError, ValueError):
            values = None
        previous = {}
        for base_n in levels:
            level_rows, reason = _one_level(
                problem, design, metric, values, base_n, actual_ci_seed,
                num_resamples, conf_level,
            )
            for row in level_rows:
                row.update(CI_Level=float(conf_level),
                           CI_Convention="normal-approximation bootstrap halfwidth; estimate +/- halfwidth")
            current = {}
            for row in level_rows:
                if row["Record_Type"] != "S1_ST":
                    continue
                diagnostic = {
                    **row, "Record_Type": "convergence", "Reason": reason,
                    "Expanded_Rows": base_n * block_size,
                    "Previous_Base_N": previous.get(row["Parameter"], {}).get("Base_N"),
                    "S1_Abs_Delta": np.nan, "ST_Abs_Delta": np.nan,
                }
                prior = previous.get(row["Parameter"])
                if prior and prior["Status"] == "ok" and row["Status"] == "ok":
                    diagnostic["S1_Abs_Delta"] = abs(row["S1"] - prior["S1"])
                    diagnostic["ST_Abs_Delta"] = abs(row["ST"] - prior["ST"])
                diagnostics.append(diagnostic)
                current[row["Parameter"]] = row
            previous = current
            if base_n == maximum_n:
                indices.extend(level_rows)
    return {
        "sobol_indices": indices, "diagnostics": diagnostics,
        "metadata": {
            "SALib_Version": version("SALib"), "Estimator_S1_ST": "Saltelli et al. 2010",
            "Estimator_S2": "Saltelli 2002" if design["calc_second_order"] else None,
            "Bootstrap_Resamples": num_resamples, "Confidence_Level": float(conf_level),
            "Bootstrap_Seed": bootstrap_seed, "Effective_Bootstrap_Seed": actual_ci_seed,
            "CI_Meaning": "normal-approximation bootstrap halfwidth; estimate +/- halfwidth",
            "N_Levels": levels, "Base_Index_Origin": 0, "Sample_ID_Origin": 1,
        },
    }
