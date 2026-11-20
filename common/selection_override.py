

from datetime import datetime
import json
import math
from pathlib import Path
import re


SELECTION_METADATA_COLUMNS = [
    "Selection_Source",
    "Selection_Hash_Override",
    "Selection_Original_Input_SHA256",
    "Selection_Current_Input_SHA256",
    "Selection_Override_Path",
]


def get_selection_override_file(selected_ma_file):
    return Path(selected_ma_file).with_name("selected_ma.override.json")


def selection_metadata(original_hash, current_hash, override_path=None):
    used = override_path is not None
    return dict(zip(SELECTION_METADATA_COLUMNS, [
        "selected_ma.json+override" if used else "selected_ma.json",
        used,
        original_hash,
        current_hash,
        str(Path(override_path).resolve()) if used else "",
    ]))


def read_selection_override(path, current_hash, *, run_budget=None, context_budget=None):
    
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(document, dict):
            raise ValueError("A JSON object is required")
        if type(document.get("override_version")) is not int or document["override_version"] != 1:
            raise ValueError("override_version must be 1")
        original = document.get("original_input_sha256")
        if not isinstance(original, str) or not (
            original == "unknown" or re.fullmatch(r"[0-9a-fA-F]{64}", original)
        ):
            raise ValueError("original_input_sha256 must be a SHA256 hash or unknown")
        current = document.get("current_input_sha256")
        if not isinstance(current, str) or current.lower() != current_hash.lower():
            raise ValueError("current_input_sha256 does not match the current workbook")
        if not isinstance(document.get("reason"), str) or not document["reason"].strip():
            raise ValueError("reason must be nonempty")
        if document.get("operator") != "author":
            raise ValueError("operator must be author")
        timestamp = document.get("timestamp")
        if not isinstance(timestamp, str) or "T" not in timestamp:
            raise ValueError("timestamp must be an ISO 8601 date and time")
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if type(document.get("allow_downstream_run")) is not bool:
            raise ValueError("allow_downstream_run must be a JSON boolean")
        if "budget" in document:
            values = [document["budget"], document.get("context_budget"), run_budget]
            if context_budget is not None:
                values.append(context_budget)
            if any(isinstance(value, bool) for value in values):
                raise ValueError("budget cannot be boolean")
            numbers = [float(value) for value in values]
            if not all(math.isfinite(value) for value in numbers) or not all(
                math.isclose(numbers[0], value, rel_tol=0.0, abs_tol=1e-12)
                for value in numbers[1:]
            ):
                raise ValueError("budget must equal override.context_budget, run.budget and the selected context budget")
    except (OSError, ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid selection-hash override: {path.resolve()}: {error}") from error
    return document
