"""Persist every result before plotting; never launch a search from this layer."""

from hashlib import sha256
import json
from pathlib import Path

from common.config import get_feature_output_dir


def export(result):
    directory = Path(result["output_dir"])
    metadata = result["metadata"]
    manifest_path = directory / "experiment_manifest.json"
    if directory.exists() and any(directory.iterdir()):
        if not manifest_path.is_file():
            raise ValueError("Output directory contains an incomplete or unrelated experiment")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["Experiment_ID"] != metadata["Experiment_ID"] or previous["Input_SHA256"] != metadata["Input_SHA256"]:
            raise ValueError("Refusing to overwrite a different experiment")
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = result["input_snapshot"]
    if sha256(snapshot).hexdigest() != metadata["Input_SHA256"]:
        raise ValueError("Input snapshot hash does not match run metadata")
    snapshot_name = Path(result["input_excel"]).name
    (directory / snapshot_name).write_bytes(snapshot)
    hashes = {snapshot_name: metadata["Input_SHA256"]}
    for name, frame in result["tables"].items():
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")
        hashes[path.name] = sha256(path.read_bytes()).hexdigest()
    manifest = {**metadata, "Files_SHA256": hashes}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    root = get_feature_output_dir("probability_mapping", result["input_excel"])
    pointer = "latest_smoke.json" if metadata["Configuration"]["smoke_only"] else "latest.json"
    root.mkdir(parents=True, exist_ok=True)
    (root / pointer).write_text(json.dumps({"output_dir": str(directory)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Probability mapping CSV results: {directory}", flush=True)
    return directory
