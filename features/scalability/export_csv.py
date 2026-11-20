"""Validate and export raw timings before deriving the summary from that CSV."""

import csv
import json
import math
from pathlib import Path
from statistics import median


STATUSES = {"ok", "error", "timeout", "total_timeout_not_run", "nonfinite", "infeasible"}
DIMENSION_COLUMNS = (
    "Assets", "Vulnerabilities", "Component_Nodes", "Component_Edges", "Component_Paths",
    "Potential_Vulnerability_Combinations", "Constructed_Prefix_Nodes", "Constructed_Prefix_Edges",
    "Stored_State_Edge_Entries", "Binary_Measures", "Continuous_Variables",
)


def write_rows(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(rows):
    keys = set()
    for row in rows:
        if row["Run_Key"] in keys:
            raise ValueError(f"Duplicate run key: {row['Run_Key']}")
        keys.add(row["Run_Key"])
        if row["Status"] not in STATUSES:
            raise ValueError(f"Invalid status: {row['Status']}")
        if row["Status"] == "ok":
            for name in ("Setup_Seconds", "Evaluation_Seconds", "Cost_USD_Thousands", "Loss_USD_Thousands"):
                if not math.isfinite(float(row[name])) or float(row[name]) < 0:
                    raise ValueError(f"Invalid {name}")
            if row["Cache_Hits"] != 0 or row["Uncached_Evaluations"] != 1 or row["Total_FE"] != 1:
                raise ValueError("Successful timing must contain one cold evaluation")


def summary_from_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    groups = {}
    for row in records:
        key = (row["Case"], row["Portfolio"], row["Game_Mode"], row["Phase"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, rows in groups.items():
        valid = [row for row in rows if row["Status"] == "ok"]
        summary = dict(zip(("Case", "Portfolio", "Game_Mode", "Phase"), key))
        summary.update({
            "Attempts": len(rows), "Valid_Timings": len(valid),
            "Timeouts": sum(row["Status"] in ("timeout", "total_timeout_not_run") for row in rows),
            "Other_Failures": sum(row["Status"] not in ("ok", "timeout", "total_timeout_not_run") for row in rows),
            "Limited_Diagnostic_Evaluations": sum(row.get("Diagnostic_Status") == "limited_unconverged" for row in valid),
        })
        for column in ("Setup_Seconds", "Evaluation_Seconds", "Process_Seconds"):
            values = [float(row[column]) for row in valid]
            summary.update({column + "_Median": median(values) if values else None,
                            column + "_Min": min(values) if values else None,
                            column + "_Max": max(values) if values else None})
        for column in ("Visited_Prefixes", "Local_Games", "Total_Updates", "Iteration_Limit_Games",
                       "Unconverged_Games", "Cost_USD_Thousands", "Loss_USD_Thousands"):
            values = [float(row[column]) for row in valid]
            summary[column + "_Min"] = min(values) if values else None
            summary[column + "_Max"] = max(values) if values else None
        summaries.append(summary)
    return summaries


def export(result):
    rows = result["rows"]
    validate_rows(rows)
    directory = Path(result["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    write_rows(directory / "raw_timings.csv", rows)
    write_rows(directory / "timing_summary.csv", summary_from_csv(directory / "raw_timings.csv"))
    dimension_rows = {}
    for row in rows:
        if all(column in row for column in DIMENSION_COLUMNS):
            dimensions = {"Case": row["Case"], "Input_SHA256": row["Input_SHA256"],
                          **{column: row[column] for column in DIMENSION_COLUMNS}}
            if row["Case"] in dimension_rows and dimension_rows[row["Case"]] != dimensions:
                raise ValueError("Case dimensions or input hash changed across repeats")
            dimension_rows[row["Case"]] = dimensions
    write_rows(directory / "case_dimensions.csv", list(dimension_rows.values()))
    manifest = result["manifest"]
    (directory / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rows(directory / "experiment_manifest.csv", [{"Field": key, "Value": json.dumps(value, ensure_ascii=False)}
                                                       for key, value in manifest.items()])
    return directory


def export_existing(directory):
    path = Path(directory) / "solve_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if Path(result["output_dir"]).resolve() != Path(directory).resolve():
        raise ValueError("Saved result belongs to another output directory")
    return export(result)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scalability-results-dir", type=Path, required=True)
    export_existing(parser.parse_args().scalability_results_dir)


if __name__ == "__main__":
    main()
