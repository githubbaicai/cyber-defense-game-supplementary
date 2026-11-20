


# ============================================

# ============================================

LEGACY_IDS_POSITIONS = dict(zip(
    ("IDS0", "IDS1", "IDS2", "IDS3", "IDS4"),
    ("network", "A", "B", "C", "D"),
))
LEGACY_FIREWALL_POSITIONS = dict(zip(
    ("F0", "F1", "F2", "F3", "F4"),
    ("network", "A", "B", "C", "D"),
))


def defense_categories(defense_items, all_data=None):
    """Use measure-sheet membership; retain prefix inference for incomplete legacy data."""
    data = all_data or {}
    measure_sheets = (
        ("Patch vulnerability", "Prevention"),
        ("Set firewall", "Prevention"),
        ("Set IDS", "Detection"),
    )
    complete = all(isinstance(data.get(sheet), dict) and "Label" in data[sheet]
                   for sheet, _ in measure_sheets)
    membership = {label: "Response" for label in ("RP1", "RP2", "RP3")}
    for sheet, category in measure_sheets:
        for label in data.get(sheet, {}).get("Label", []):
            if label in membership:
                raise ValueError(f"{sheet}.Label duplicates another measure-sheet label or an RP label: {label!r}")
            membership[label] = category
    allocation = data.get("Allocation", {})
    labels = allocation.get("Label", [])
    values = allocation.get("Category")
    explicit = {}
    if values is not None:
        if len(values) != len(labels):
            raise ValueError("Allocation.Category and Label row counts differ")
        for label, category in zip(labels, values):
            if category is None or (isinstance(category, str) and not category.strip()):
                continue
            if category not in ("Prevention", "Detection", "Response"):
                raise ValueError(f"Invalid Allocation.Category: {label!r} -> {category!r}")
            if label in membership and category != membership[label]:
                raise ValueError(f"Allocation.Category conflicts with measure membership: {label!r} -> {category!r}, expected {membership[label]}")
            explicit[label] = category
    categories = []
    for label, _ in defense_items:
        if label in membership:
            category = membership[label]
        elif complete:
            raise ValueError(f"Allocation.Label has no associated measure: {label!r}")
        elif label in explicit:
            category = explicit[label]
        elif label.startswith(("V", "F")):
            category = "Prevention"
        elif label.startswith("IDS"):
            category = "Detection"
        elif label.startswith("RP"):
            category = "Response"
        else:
            category = "Unknown"
        categories.append(category)
    return categories


def position_by_label(sheet: dict, legacy_positions: dict) -> dict:
    labels = sheet["Label"]
    positions = sheet.get("Position")
    if positions is None:
        return {label: legacy_positions[label] for label in labels
                if label in legacy_positions}
    if len(positions) != len(labels):
        raise ValueError("Position and Label row counts differ")
    return {str(label): str(position) for label, position in zip(labels, positions)}


def parse_ids_config(ids_sheet: dict) -> dict:
    positions = position_by_label(ids_sheet, LEGACY_IDS_POSITIONS)
    return {
        positions[label]: {"eff": float(eff), "cycle": float(cycle), "label": label}
        for label, cycle, eff in zip(
            ids_sheet["Label"],
            ids_sheet["Detection Cycle (h)"],
            ids_sheet["Detection effectiveness"],
        ) if label in positions
    }


def parse_firewall_config(firewall_sheet: dict) -> dict:
    positions = position_by_label(firewall_sheet, LEGACY_FIREWALL_POSITIONS)
    return {
        positions[label]: {"eff": float(eff), "label": label}
        for label, eff in zip(
            firewall_sheet["Label"], firewall_sheet["Firewall effectiveness"]
        ) if label in positions
    }


def vulnerability_patch_labels(vuln_sheet: dict, patch_sheet: dict) -> dict:
    
    names = vuln_sheet["Vulnerability"]
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("Vulnerability.Vulnerability values must be nonempty and unique")
    patch_labels = patch_sheet["Label"]
    if len(set(patch_labels)) != len(patch_labels):
        raise ValueError("Patch vulnerability.Label values must be unique")
    if len(patch_labels) != len(patch_sheet["Patching effectiveness"]):
        raise ValueError("Patch vulnerability Label and effectiveness row counts differ")
    explicit = vuln_sheet.get("Label")
    if explicit is not None and len(explicit) != len(names):
        raise ValueError("Vulnerability.Label and Vulnerability row counts differ")
    mapping = {}
    for index, name in enumerate(names):
        label = explicit[index] if explicit is not None else (name if name in patch_labels else None)
        if label is None or label == "":
            mapping[name] = None
        elif label not in patch_labels:
            raise ValueError(f"Vulnerability.Label references an unknown patch: {name!r} -> {label!r}")
        else:
            mapping[name] = label
    linked = [label for label in mapping.values() if label is not None]
    if len(linked) != len(set(linked)):
        raise ValueError("Nonempty patch references in Vulnerability.Label must be unique")
    unlinked = set(patch_labels) - set(linked)
    if unlinked:
        raise ValueError(f"Patch vulnerability contains patches with no associated vulnerability: {sorted(unlinked)}")
    return mapping


def component_vulnerability_names(vuln_sheet: dict, distribution: dict) -> dict:
    
    known = set(vuln_sheet["Vulnerability"])
    components = distribution["Component"]
    values = distribution["Vulnerability"]
    if len(components) != len(values) or len(set(components)) != len(components):
        raise ValueError("Vulnerabilities distribution.Component values must be unique and correspond row by row")
    mapping = {}
    for component, value in zip(components, values):
        if value is None or value == 0 or value == "":
            names = []
        elif isinstance(value, str):
            names = [token.strip() for token in value.split(",")]
        else:
            raise ValueError(f"Vulnerabilities distribution: invalid vulnerability references for {component!r}: {value!r}")
        if len(names) != len(set(names)) or any(name not in known for name in names):
            raise ValueError(f"Vulnerabilities distribution: {component!r} contains unknown or duplicate vulnerabilities: {names!r}")
        mapping[component] = names
    return mapping


def parse_patch_config(patch_sheet: dict, vuln_sheet=None) -> dict:
    
    patch_config = {}
    labels = patch_sheet.get("Label", [])
    effects = patch_sheet.get("Patching effectiveness", [])

    for label, eff in zip(labels, effects):
        patch_config[label] = eff
    if vuln_sheet is None:
        return patch_config
    return {
        name: patch_config[label]
        for name, label in vulnerability_patch_labels(vuln_sheet, patch_sheet).items()
        if label is not None
    }
