"""Strict EPSS inputs and independently transformed scenario data."""

from copy import deepcopy
import csv
from datetime import date, datetime
import math
from numbers import Real
from pathlib import Path

from .config import MAPPINGS, MAPPING_SPECS


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return float(value)


def _date(value):
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d")
    try:
        return date.fromisoformat(str(value).strip()).isoformat()
    except ValueError as error:
        raise ValueError("EPSS_Date must be an explicit ISO calendar date") from error


def validate_inputs(data):
    """Require explicit dated EPSS values matching the nominal model input."""
    table = data.get("Vulnerability", {})
    columns = ("Vulnerability", "CVE", "EPSS", "EPSS_Date", "Attack Success Probability")
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"missing EPSS input columns: {missing}; no CVSS fallback is permitted")
    lengths = [len(table[column]) for column in columns]
    if not lengths[0] or len(set(lengths)) != 1:
        raise ValueError("EPSS input column lengths must agree and be nonzero")
    labels = [str(value).strip() for value in table["Vulnerability"]]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Vulnerability labels must be nonempty and unique")
    for index, label in enumerate(labels):
        cve = str(table["CVE"][index]).strip()
        if not cve.startswith("CVE-"):
            raise ValueError(f"{label}: an explicit CVE identifier is required")
        _date(table["EPSS_Date"][index])
        epss = _probability(table["EPSS"][index], f"{label} EPSS")
        original = _probability(table["Attack Success Probability"][index], f"{label} nominal probability")
        if not math.isclose(epss, original, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"{label}: EPSS differs from nominal Attack Success Probability")
    return table


def mapped_probability(epss, mapping_id):
    """Apply a prespecified scenario proxy, without claiming empirical calibration."""
    if mapping_id not in MAPPINGS:
        raise ValueError(f"unknown mapping: {mapping_id}")
    epss = _probability(epss, "EPSS")
    if mapping_id == "square":
        return epss**2
    if mapping_id == "sqrt":
        return math.sqrt(epss)
    return 0.8 * epss if mapping_id == "scale_0_8" else epss


def transform_data(data, mapping_id):
    """Copy all scenario data; preserve provenance and alter only model probabilities."""
    table = validate_inputs(data)
    result = deepcopy(data)
    result["Vulnerability"]["Attack Success Probability"] = [
        mapped_probability(value, mapping_id) for value in table["EPSS"]
    ]
    return result


def _snapshots(input_excel):
    path = Path(input_excel).resolve().parent / "cve_epss.csv"
    records = {}
    if path.is_file():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for record in csv.DictReader(stream):
                key = (str(record.get("cve", "")).strip(), str(record.get("date", "")).strip())
                records.setdefault(key, []).append(record)
    return path, records


def mapping_input_records(data, input_excel):
    """Export every transformed probability and its dated workbook provenance."""
    table = validate_inputs(data)
    snapshot_path, snapshots = _snapshots(input_excel)
    records = []
    for mapping_id in MAPPINGS:
        for index, label in enumerate(table["Vulnerability"]):
            epss = float(table["EPSS"][index])
            cve, snapshot_date = str(table["CVE"][index]).strip(), _date(table["EPSS_Date"][index])
            matches = snapshots.get((cve, snapshot_date), [])
            status = "local_snapshot_unavailable"
            if len(matches) > 1:
                status = "ambiguous_local_snapshot"
            elif matches:
                try:
                    snapshot = float(matches[0]["epss"])
                    valid = math.isfinite(snapshot) and 0 <= snapshot <= 1 and matches[0].get("status", "ok") == "ok"
                    status = "local_snapshot_matches" if valid and math.isclose(snapshot, epss, rel_tol=0, abs_tol=1e-9) else "local_snapshot_conflict"
                except (KeyError, ValueError, TypeError):
                    status = "invalid_local_snapshot"
            records.append({
                "Mapping_ID": mapping_id, "Mapping_Label": MAPPING_SPECS[mapping_id]["label"],
                "Mapping_Formula": MAPPING_SPECS[mapping_id]["formula"], "Vulnerability": label,
                "CVE": cve, "EPSS_Date": snapshot_date, "EPSS": epss,
                "Original_Probability": float(table["Attack Success Probability"][index]),
                "Mapped_Probability": mapped_probability(epss, mapping_id),
                "Input_Excel": str(Path(input_excel).resolve()), "Snapshot_File": str(snapshot_path),
                "Snapshot_Status": status, "EPSS_Time_Horizon_Days": 30,
                "Calibration_Status": "scenario_proxy_not_calibrated_conditional_attack_success",
                "Mapping_Assumption": "prespecified sensitivity scenario; not an empirical calibration",
            })
    return records
