

from copy import deepcopy
import numpy as np
from pymoo.core.repair import Repair

from common.parsers import (
    LEGACY_FIREWALL_POSITIONS,
    LEGACY_IDS_POSITIONS,
    parse_firewall_config,
    parse_ids_config,
    vulnerability_patch_labels,
    position_by_label,
)


def normalize_decision_vector(x, defense_items):
    
    values = np.asarray(x, dtype=float).reshape(-1).copy()
    defense_count = len(defense_items)
    if values.size not in (defense_count, defense_count + 1):
        raise ValueError("Decision-vector length must equal the number of defense measures, optionally plus p_response")
    if not np.isfinite(values).all():
        raise ValueError("The decision vector must contain finite numbers")
    values[:defense_count] = values[:defense_count] >= 0.5
    rp_indices = [
        i for i, (label, _) in enumerate(defense_items)
        if label in ("RP1", "RP2", "RP3")
    ]
    selected_rps = [i for i in rp_indices if values[i]]
    if len(selected_rps) > 1:
        highest_idx = max(
            selected_rps,
            key=lambda i: int(defense_items[i][0][-1])
        )
        for i in selected_rps:
            if i != highest_idx:
                values[i] = 0.0
    if values.size > defense_count:
        values[-1] = float(np.clip(values[-1], 0.0, 1.0)) if selected_rps else 0.0
    return values


def repair_response_selection(x, defense_items):
    
    x[:] = normalize_decision_vector(x, defense_items)
    return x


class DefenseRepair(Repair):
    

    def __init__(self, defense_items=None):
        super().__init__()
        self.defense_items = defense_items

    def _do(self, problem, X, **kwargs):
        items = (
            self.defense_items
            if self.defense_items is not None
            else getattr(problem, "defense_items", None)
        )
        if items is None:
            return X

        return np.asarray([normalize_decision_vector(row, items) for row in X], dtype=float).reshape(X.shape)


def create_selected_configuration(
    x,
    defense_items,
    all_data
):
    
    if len(x) != len(defense_items):
        raise ValueError(
            "The defense-selection vector length does not match the number of measures in Allocation"
        )

    
    x_repaired = normalize_decision_vector(x, defense_items).astype(bool)

    labels = [item[0] for item in defense_items]
    selected = [
        labels[i]
        for i, enabled in enumerate(x_repaired)
        if bool(enabled)
    ]

    full_ids = parse_ids_config(
        all_data["Set IDS"]
    )
    full_fw = parse_firewall_config(
        all_data["Set firewall"]
    )

    patch_sheet = all_data[
        "Patch vulnerability"
    ]
    patch_labels = patch_sheet.get(
        "Label",
        []
    )
    patch_effectiveness = patch_sheet.get(
        "Patching effectiveness",
        []
    )

    patch_labels_by_name = vulnerability_patch_labels(all_data["Vulnerability"], patch_sheet)
    patch_by_label = dict(zip(patch_labels, patch_effectiveness))
    patch_config = {
        name: patch_by_label[label] for name, label in patch_labels_by_name.items()
        if label in selected
    }

    fw_label_map = position_by_label(
        all_data["Set firewall"], LEGACY_FIREWALL_POSITIONS
    )
    firewall_config = {
        component: (
            full_fw[component]["eff"]
            if label in selected
            else 0.0
        )
        for label, component in (
            fw_label_map.items()
        )
    }

    ids_label_map = position_by_label(
        all_data["Set IDS"], LEGACY_IDS_POSITIONS
    )
    ids_config = {
        component: (
            full_ids[component]
            if label in selected
            else {
                "eff": 0.0,
                "cycle": 100,
                "label": label,
            }
        )
        for label, component in (
            ids_label_map.items()
        )
    }

    rp_level = 0
    for label in (
        "RP1",
        "RP2",
        "RP3"
    ):
        if label in selected:
            rp_level = max(
                rp_level,
                int(label[-1])
            )

    vuln_data_sim = deepcopy(
        all_data["Vulnerability"]
    )
    number_of_vulnerabilities = len(
        vuln_data_sim["Vulnerability"]
    )

    if rp_level == 0:
        defense_success = (
            [0.0]
            * number_of_vulnerabilities
        )
    else:
        column = (
            "Defense Success Probability "
            f"after detecting ({rp_level})"
        )
        if column not in vuln_data_sim:
            raise ValueError(f"Selected RP{rp_level} is missing required column: {column}")
        try:
            probabilities = np.asarray(vuln_data_sim[column], dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{column} must contain finite probabilities in [0, 1]") from exc
        if (
            probabilities.ndim != 1
            or probabilities.size != number_of_vulnerabilities
        ):
            raise ValueError(
                f"The row count of {column} must match Vulnerability: "
                f"expected {number_of_vulnerabilities}, actual shape {probabilities.shape}"
            )
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError(f"{column} must contain only finite probabilities in [0, 1]")
        defense_success = probabilities.tolist()

    vuln_data_sim[
        "Defense Success Probability after detecting (0)"
    ] = defense_success

    return {
        "selected": selected,
        "patch_config": patch_config,
        "firewall_config": firewall_config,
        "ids_config": ids_config,
        "vuln_data_sim": vuln_data_sim,
    }
