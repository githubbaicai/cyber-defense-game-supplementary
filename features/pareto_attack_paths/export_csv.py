"""Export the saved result tables without simulating or reading upstream inputs."""

import argparse

from common.config import CSV_FILES, get_feature_output_dir


def export(result):
    directory = get_feature_output_dir("pareto_attack_paths", result["input_excel"])
    tables = {name: result[name] for name in CSV_FILES["pareto_attack_paths"]}
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(directory / CSV_FILES["pareto_attack_paths"][name], index=False, encoding="utf-8-sig",
                     float_format="%.17g")
    return tables


if __name__ == "__main__":
    from common.pipeline import run_export

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-excel")
    args = parser.parse_args()
    run_export("pareto_attack_paths", export, input_excel=args.input_excel)
