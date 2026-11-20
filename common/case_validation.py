

import numpy as np

from common.parsers import component_vulnerability_names, defense_categories, vulnerability_patch_labels


def _unique_column(data, sheet, column):
    values = data[sheet][column]
    if any(not isinstance(value, str) or not value.strip() for value in values) or len(set(values)) != len(values):
        raise ValueError(f"{sheet}.{column} labels must be nonempty and unique")
    return set(values)


def _numeric_column(data, sheet, column, lower=None, upper=None, positive=False):
    raw = data[sheet][column]
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{sheet}.{column} must contain finite numbers") from error
    if (
        values.ndim != 1 or not np.isfinite(values).all()
        or any(isinstance(value, (str, bool, np.bool_)) for value in raw)
        or (lower is not None and np.any(values < lower))
        or (upper is not None and np.any(values > upper))
        or (positive and np.any(values <= 0))
    ):
        raise ValueError(f"Invalid {sheet}.{column}: finite numbers required in [{lower}, {upper}], positive={positive}")


def validate_case_data(nodes, adjacency, data, *, allow_no_attack=False):
    
    if type(allow_no_attack) is not bool:
        raise ValueError("allow_no_attack must be bool")
    assets = set(nodes)
    for sheet, column in (
        ("Label", "Label"), ("Vulnerability", "Vulnerability"),
        ("Allocation", "Label"), ("Patch vulnerability", "Label"),
        ("Set firewall", "Label"), ("Set IDS", "Label"),
    ):
        _unique_column(data, sheet, column)
    vulnerability_patch_labels(data["Vulnerability"], data["Patch vulnerability"])
    distribution = component_vulnerability_names(data["Vulnerability"], data["Vulnerabilities distribution"])
    if set(distribution) != assets:
        raise ValueError("Vulnerabilities distribution.Component must match the Attack graph node set")
    allocation = set(data["Allocation"]["Label"])
    measures = set()
    for sheet in ("Patch vulnerability", "Set firewall", "Set IDS"):
        labels = set(data[sheet]["Label"])
        if measures & labels:
            raise ValueError(f"{sheet}.Label duplicates labels in other measure sheets")
        measures.update(labels)
    if measures & {"RP1", "RP2", "RP3"} or not measures.issubset(allocation) or allocation - measures - {"RP1", "RP2", "RP3"}:
        raise ValueError("Allocation.Label must reference only patches, firewalls, IDSs or RP1/RP2/RP3")
    defense_categories(zip(data["Allocation"]["Label"], data["Allocation"]["Cost"]), data)
    for sheet in ("Set firewall", "Set IDS"):
        positions = data[sheet]["Position"]
        if len(set(positions)) != len(positions) or any(position not in assets | {"network"} for position in positions):
            raise ValueError(f"{sheet}.Position values must be unique and reference network or an Attack graph node")
    for sheet, column in (
        ("Vulnerability", "Attack Success Probability"),
        *(("Vulnerability", f"Defense Success Probability after detecting ({level})") for level in range(4)),
        *(("Vulnerability", column) for column in ("Weight_ABM", "Weight_ABD", "Weight_DLM", "Weight_DLD")),
        ("Patch vulnerability", "Patching effectiveness"),
        ("Set firewall", "Firewall effectiveness"), ("Set IDS", "Detection effectiveness"),
    ):
        _numeric_column(data, sheet, column, 0, 1)
    for sheet, column in (("Allocation", "Cost"), ("Vulnerability", "Attack cost (monetary)"),
                          ("Vulnerability", "Attack time (h)")):
        _numeric_column(data, sheet, column, lower=0)
    _numeric_column(data, "Set IDS", "Detection Cycle (h)", positive=True)
    for column in ("Attack Benefit (monetary)", "Attack benefit (destruction)",
                   "Defend loss (monetary)", "Defend loss (destruction)"):
        _numeric_column(data, "Vulnerability", column)
    vulnerabilities = data["Vulnerability"]
    with np.errstate(over="ignore", invalid="ignore"):
        attack_benefits = (
            np.asarray(vulnerabilities["Attack Benefit (monetary)"], dtype=float)
            * np.asarray(vulnerabilities["Weight_ABM"], dtype=float)
            + np.asarray(vulnerabilities["Attack benefit (destruction)"], dtype=float)
            * np.asarray(vulnerabilities["Weight_ABD"], dtype=float)
        )
    for name, benefit in zip(vulnerabilities["Vulnerability"], attack_benefits):
        if not np.isfinite(benefit) or benefit < 0:
            raise ValueError(
                f"Vulnerability {name!r}: weighted attack gains must be finite and nonnegative; "
                f"Attack Benefit (monetary) * Weight_ABM + "
                f"Attack benefit (destruction) * Weight_ABD = {benefit}"
            )
    if allow_no_attack:
        return
    if not {"S", "T"}.issubset(assets):
        raise ValueError("Attack graph is missing S/T; a no-attack case requires explicit allow_no_attack=True")
    node_indices = {node: index for index, node in enumerate(nodes)}
    reached = {"S"}
    pending = ["S"]
    while pending:
        source = pending.pop()
        for target, edge in zip(nodes, adjacency[node_indices[source]]):
            if edge != 1 or target in reached or (source == "S" and target == "T"):
                continue
            if target == "T":
                return
            if distribution[target]:
                reached.add(target)
                pending.append(target)
    raise ValueError("Vulnerabilities distribution has no executable vulnerability attack path; a no-attack case requires explicit allow_no_attack=True")
