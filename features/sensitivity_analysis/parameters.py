"""Scenario-local sensitivity factors, provenance, and model options."""

from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import dataclass, replace
import json
import math
from numbers import Real
from pathlib import Path

from common.model_options import ModelOptions
from common.parsers import defense_categories


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    targets: tuple
    transformation: str = "multiply"
    unit: str = "currency"
    category: str | None = None


PARAMETER_SPECS = (
    ParameterSpec("epss", (("Vulnerability", "Attack Success Probability"),), "odds", "dimensionless"),
    ParameterSpec("attack_duration", (("Vulnerability", "Attack time (h)"),), unit="h"),
    ParameterSpec("attacker_cost", (("Vulnerability", "Attack cost (monetary)"),)),
    ParameterSpec("attacker_monetary_benefit", (("Vulnerability", "Attack Benefit (monetary)"),)),
    ParameterSpec("attacker_destruction_benefit", (("Vulnerability", "Attack benefit (destruction)"),)),
    ParameterSpec("attacker_preference", (("Vulnerability", "Weight_ABM"), ("Vulnerability", "Weight_ABD")), "complementary_odds", "dimensionless"),
    ParameterSpec("patch_effectiveness", (("Patch vulnerability", "Patching effectiveness"),), "odds", "dimensionless"),
    ParameterSpec("firewall_effectiveness", (("Set firewall", "Firewall effectiveness"),), "odds", "dimensionless"),
    ParameterSpec("ids_effectiveness", (("Set IDS", "Detection effectiveness"),), "odds", "dimensionless"),
    ParameterSpec("response_effectiveness", tuple(("Vulnerability", f"Defense Success Probability after detecting ({level})") for level in (1, 2, 3)), "odds", "dimensionless"),
    ParameterSpec("ids_detection_cycle", (("Set IDS", "Detection Cycle (h)"),), unit="h"),
    ParameterSpec("prevention_cost", (("Allocation", "Cost"),), category="Prevention"),
    ParameterSpec("detection_cost", (("Allocation", "Cost"),), category="Detection"),
    ParameterSpec("response_cost", (("Allocation", "Cost"),), category="Response"),
    ParameterSpec("defender_monetary_loss", (("Vulnerability", "Defend loss (monetary)"),)),
    ParameterSpec("defender_destruction_loss", (("Vulnerability", "Defend loss (destruction)"),)),
    ParameterSpec("attacker_exposure_penalty_scale", tuple(), "model_option", "dimensionless"),
    ParameterSpec("response_penalty_scale", tuple(), "model_option", "dimensionless"),
)
PARAMETER_NAMES = tuple(spec.name for spec in PARAMETER_SPECS)
PARAMETER_INDEX = {name: index for index, name in enumerate(PARAMETER_NAMES)}
MODEL_OPTION_FACTORS = {
    "attacker_exposure_penalty_scale": (
        "exposure_penalty_ratio", "EXPOSURE_PENALTY_RATIO",
        "Detection-state attacker exposure penalty applied to attack cost",
    ),
    "response_penalty_scale": (
        "response_penalty_ratio", "RESPONSE_PENALTY_RATIO",
        "Defender response disruption coefficient applied to weighted destruction loss",
    ),
}


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric value")
    return float(value)


def odds_transform(value, multiplier):
    """Apply an odds multiplier while preserving endpoints and the neutral point."""
    probability = _finite(value, "probability")
    multiplier = _finite(multiplier, "multiplier")
    if not 0 <= probability <= 1 or multiplier <= 0:
        raise ValueError("probability must be in [0, 1] and multiplier must be positive")
    if multiplier == 1 or probability in (0.0, 1.0):
        return probability
    return multiplier * probability / (1.0 - probability + multiplier * probability)


def normalize_ranges(parameter_ranges=None):
    """Copy and validate uniform latent-factor bounds in registered order."""
    supplied = {} if parameter_ranges is None else parameter_ranges
    if not isinstance(supplied, dict) or set(supplied) - set(PARAMETER_NAMES):
        raise ValueError("parameter_ranges must map registered factor names to [lower, upper]")
    ranges = {}
    for name in PARAMETER_NAMES:
        limits = supplied.get(name, (0.8, 1.2))
        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
            raise ValueError(f"{name}: bounds must be [lower, upper]")
        lower, upper = (_finite(value, f"{name} bounds") for value in limits)
        if not 0 < lower < upper:
            raise ValueError(f"{name}: bounds must satisfy 0 < lower < upper")
        ranges[name] = (lower, upper)
    return ranges


def parameter_problem(config=None):
    ranges = normalize_ranges(getattr(config, "parameter_ranges", config))
    return {"num_vars": len(PARAMETER_NAMES), "names": list(PARAMETER_NAMES), "bounds": [list(ranges[name]) for name in PARAMETER_NAMES]}


def _column(data, sheet, column):
    try:
        values = data[sheet][column]
    except KeyError as error:
        raise ValueError(f"missing sensitivity input: {sheet}.{column}") from error
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{sheet}.{column} must be a nonempty column")
    return values


def validate_parameters(data, *, weight_tolerance=1e-10):
    """Validate fields used by these transformations without repairing input."""
    for spec in PARAMETER_SPECS:
        for sheet, column in spec.targets:
            values = _column(data, sheet, column)
            key = "Vulnerability" if sheet == "Vulnerability" else "Label"
            if len(values) != len(_column(data, sheet, key)):
                raise ValueError(f"{sheet}.{column} is not aligned with {key}")
            for value in values:
                number = _finite(value, f"{sheet}.{column}")
                if spec.transformation not in ("multiply", "model_option") and not 0 <= number <= 1:
                    raise ValueError(f"{sheet}.{column} must be in [0, 1]")
                if column in ("Cost", "Attack cost (monetary)", "Attack time (h)") and number < 0:
                    raise ValueError(f"{sheet}.{column} must be nonnegative")
                if column == "Detection Cycle (h)" and number <= 0:
                    raise ValueError("IDS detection cycles must be positive")
    vulnerabilities = data["Vulnerability"]
    for first, second in zip(vulnerabilities["Weight_ABM"], vulnerabilities["Weight_ABD"]):
        if not math.isclose(first + second, 1.0, rel_tol=0.0, abs_tol=weight_tolerance):
            raise ValueError("attacker_preference requires complementary Weight_ABM/Weight_ABD; input was not repaired")
    response = [vulnerabilities[f"Defense Success Probability after detecting ({level})"] for level in (1, 2, 3)]
    if any(not first <= second <= third for first, second, third in zip(*response)):
        raise ValueError("response_effectiveness requires RP1 <= RP2 <= RP3; input was not repaired")
    categories = defense_categories(zip(data["Allocation"]["Label"], data["Allocation"]["Cost"]), data)
    if any(category not in ("Prevention", "Detection", "Response") for category in categories):
        raise ValueError("sensitivity costs require known defense categories")


def _factor_values(sampled_values):
    if sampled_values is None:
        values = dict.fromkeys(PARAMETER_NAMES, 1.0)
    elif isinstance(sampled_values, dict):
        unknown = set(sampled_values) - set(PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"unknown sensitivity factors: {sorted(unknown)}")
        values = {**dict.fromkeys(PARAMETER_NAMES, 1.0), **sampled_values}
    else:
        values = list(sampled_values)
        if len(values) != len(PARAMETER_NAMES):
            raise ValueError(f"sampled_values must have exactly {len(PARAMETER_NAMES)} registered factors")
        values = dict(zip(PARAMETER_NAMES, values))
    for name, value in values.items():
        values[name] = _finite(value, name)
        if values[name] <= 0:
            raise ValueError(f"{name} must be a positive multiplier")
    return values


def _targets(data, spec):
    categories = None
    if spec.category:
        categories = defense_categories(zip(data["Allocation"]["Label"], data["Allocation"]["Cost"]), data)
    for sheet, column in spec.targets:
        for index, value in enumerate(data[sheet][column]):
            if categories is None or categories[index] == spec.category:
                yield sheet, column, index, float(value)


def transform_data(data, sampled_values=None, *, parameter_ranges=None):
    """Return independent scenario data and costs rebuilt from Allocation."""
    validate_parameters(data)
    factors = _factor_values(sampled_values)
    if parameter_ranges is not None:
        ranges = normalize_ranges(parameter_ranges)
        for name, value in factors.items():
            if not ranges[name][0] <= value <= ranges[name][1]:
                raise ValueError(f"{name} multiplier lies outside its configured bounds")
    scenario = deepcopy(data)
    for spec in PARAMETER_SPECS:
        if spec.transformation == "model_option":
            continue
        multiplier = factors[spec.name]
        for sheet, column, index, nominal in _targets(data, spec):
            if multiplier == 1.0:
                transformed = nominal
            elif spec.transformation == "complementary_odds":
                alpha = odds_transform(data[sheet]["Weight_ABM"][index], multiplier)
                transformed = alpha if column == "Weight_ABM" else 1.0 - alpha
            elif spec.transformation == "odds":
                transformed = odds_transform(nominal, multiplier)
            else:
                transformed = nominal * multiplier
            if not math.isfinite(transformed):
                raise ValueError(f"{sheet}.{column}: transformation overflow")
            scenario[sheet][column][index] = transformed
    defense_items = list(zip(scenario["Allocation"]["Label"], scenario["Allocation"]["Cost"]))
    return scenario, defense_items


def scenario_model_options(sampled_values=None):
    """Return independent, immutable exposure and response penalty coefficients."""
    factors = _factor_values(sampled_values)
    nominal = ModelOptions()
    return replace(nominal, **{
        option: getattr(nominal, option) * factors[name]
        for name, (option, _, _) in MODEL_OPTION_FACTORS.items()
    })


def _value(table, column, index, default="unknown"):
    values = table.get(column, ())
    value = values[index] if index < len(values) else None
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return value


def _source(table, column, index):
    if column == "Cost":
        return str(_value(table, "Cost_Source", index))
    if "Parameter_Source" in table:
        try:
            sources = json.loads(_value(table, "Parameter_Source", index, "{}"))
            if isinstance(sources, dict):
                weight_key = "attacker_weights" if column in ("Weight_ABM", "Weight_ABD") else "weights"
                return str(sources.get(column, sources.get(weight_key, "unknown")))
        except (ValueError, TypeError):
            return "invalid_parameter_source_metadata"
    return str(_value(table, "Effectiveness_Source", index))


def _currency(data):
    strategy = data.get("Strategy", {})
    for index, name in enumerate(strategy.get("Parameter", ())):
        if str(name).strip() in ("Budget", "Sensitivity_Budget") and "USD thousands" in str(_value(strategy, "Notes", index)):
            return "USD", "USD thousands"
    return "unknown", "unknown"


def _base_record(data, sheet, column, index, input_excel):
    table = data[sheet]
    label_key = "Vulnerability" if sheet == "Vulnerability" else "Label"
    entity = _value(table, label_key, index)
    currency, amount_unit = _currency(data)
    source_file = str(Path(input_excel).resolve()) if input_excel is not None else "unknown"
    return {
        "Entity_ID": entity, "Sheet": sheet, "Column": column, "Applied_To": f"{sheet}.{column}",
        "CVE": _value(table, "CVE", index), "Component": _value(table, "Position", index),
        "Currency": currency, "Amount_Unit": amount_unit, "Evaluation_Period": "unknown",
        "Source_Type": _source(table, column, index), "Source_File": source_file,
        "Source_Location": f"{source_file}::{sheet}.{column}[{entity}]", "Source_Reference": _value(table, "Source_Reference", index),
        "Extraction_Rule": "current workbook column matched by entity label; no empirical range inferred",
        "Calibration_Status": _value(table, "Calibration_Status", index), "Validation_Status": "input_semantics_checked",
        "Source_Notes": " | ".join(str(_value(table, key, index)) for key in ("Cost_Note", "Effectiveness_Note", "Expert_Judgement_Note") if _value(table, key, index) != "unknown") or "unknown",
    }


def _epss_records(input_excel):
    if input_excel is None:
        return {}
    path = Path(input_excel).resolve().parent / "cve_epss.csv"
    if not path.is_file():
        return {}
    records = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (str(row.get("cve", "")).strip(), str(row.get("date", "")).strip())
            records.setdefault(key, []).append(row)
    return records


def _epss_provenance(vulnerabilities, index, nominal, snapshots):
    epss = _value(vulnerabilities, "EPSS", index)
    date = str(_value(vulnerabilities, "EPSS_Date", index))
    cve = str(_value(vulnerabilities, "CVE", index))
    status = "snapshot_unavailable"
    if isinstance(epss, Real) and math.isfinite(epss):
        status = "workbook_epss_matches_proxy" if math.isclose(nominal, epss, rel_tol=0, abs_tol=1e-9) else "workbook_epss_differs_from_proxy"
    matches = snapshots.get((cve, date), ())
    if len(matches) == 1:
        try:
            snapshot = float(matches[0].get("epss", ""))
            agrees = math.isfinite(snapshot) and math.isclose(nominal, snapshot, rel_tol=0, abs_tol=1e-9)
            status += "; local_snapshot_matches" if agrees else "; local_snapshot_conflict"
        except (ValueError, TypeError):
            status += "; invalid_local_snapshot"
    elif len(matches) > 1:
        status += "; ambiguous_local_snapshot"
    return {
        "EPSS": epss, "EPSS_Date": date, "Validation_Status": status,
        "Calibration_Status": "EPSS_event_and_time_window_differ_from_local_conditional_success",
    }


def parameter_records(data, config=None, *, input_excel=None):
    """Describe source entries, ranges, and model constants in display order."""
    validate_parameters(data)
    ranges = normalize_ranges(getattr(config, "parameter_ranges", config))
    snapshots = _epss_records(input_excel)
    records = []
    for spec in PARAMETER_SPECS:
        lower, upper = ranges[spec.name]
        if spec.transformation == "model_option":
            option, constant, notes = MODEL_OPTION_FACTORS[spec.name]
            nominal = getattr(ModelOptions(), option)
            records.append({
                "Record_Type": "parameter", "Parameter": spec.name, "Factor_Index": PARAMETER_INDEX[spec.name],
                "Entity_ID": f"ModelOptions.{option}", "Sheet": "common.model_options", "Column": option,
                "Applied_To": f"ModelOptions.{option}", "Nominal_Value": nominal,
                "Lower_Bound": lower, "Upper_Bound": upper, "Effective_Lower": nominal * lower, "Effective_Upper": nominal * upper,
                "Unit": "dimensionless", "Range_Source": "scenario_assumption", "Distribution": "uniform",
                "Distribution_Target": "latent_multiplier", "Transformation": f"nominal ModelOptions.{option} * latent_multiplier",
                "Dependence_Group": spec.name, "Dependence_Assumption": "independent across factors; no workbook row",
                "Source_Type": "code_constant", "Source_File": str(Path(__file__).resolve().parents[2] / "common" / "config.py"),
                "Source_Location": f"common.config.{constant} -> common.model_options.ModelOptions.{option}", "Source_Reference": "unknown",
                "Calibration_Status": "uncalibrated_code_default", "Validation_Status": "model_option_source_checked",
                "Extraction_Rule": "read immutable ModelOptions default", "Source_Notes": notes,
            })
            continue
        for sheet, column, index, nominal in _targets(data, spec):
            if spec.transformation in ("odds", "complementary_odds"):
                base = data[sheet]["Weight_ABM"][index] if spec.transformation == "complementary_odds" else nominal
                endpoints = [odds_transform(base, bound) for bound in (lower, upper)]
                if spec.transformation == "complementary_odds" and column == "Weight_ABD":
                    endpoints = [1.0 - value for value in endpoints]
                transformation = "T(alpha,k); Weight_ABD=1-T(alpha,k), T(p,k)=k*p/(1-p+k*p)" if spec.transformation == "complementary_odds" else "T(p,k)=k*p/(1-p+k*p); 0/1 endpoints preserved"
            else:
                endpoints = [nominal * lower, nominal * upper]
                transformation = "nominal_value * latent_multiplier; sign preserved"
            record = _base_record(data, sheet, column, index, input_excel)
            extra = {}
            if spec.name == "epss":
                extra = {
                    **_epss_provenance(data["Vulnerability"], index, nominal, snapshots),
                    "EPSS_Probability_Lower": endpoints[0], "EPSS_Probability_Upper": endpoints[1],
                }
            records.append({
                **record, **extra, "Record_Type": "parameter", "Parameter": spec.name, "Factor_Index": PARAMETER_INDEX[spec.name],
                "Nominal_Value": nominal, "Lower_Bound": lower, "Upper_Bound": upper,
                "Effective_Lower": min(endpoints), "Effective_Upper": max(endpoints),
                "Unit": "dimensionless" if spec.unit == "dimensionless" else (record["Amount_Unit"] if spec.unit == "currency" else spec.unit),
                "Range_Source": "scenario_assumption", "Distribution": "uniform", "Distribution_Target": "latent_multiplier",
                "Transformation": transformation, "Dependence_Group": spec.name,
                "Dependence_Assumption": "one common factor within group; independent across factors",
            })
    return records


