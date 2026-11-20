

import pickle

from common import config
from common.case_config import get_input_excel


CACHE_FILENAME = "solve_result.pkl"


def get_cache_path(feature_name, input_excel=None):
    
    if feature_name not in config.FEATURE_OUTPUT_DIRS:
        raise ValueError(
            f"Unknown feature: {feature_name}; "
            f"Available features: {', '.join(config.FEATURE_OUTPUT_DIRS)}"
        )
    return config.get_feature_output_dir(feature_name, get_input_excel(input_excel)) / CACHE_FILENAME


def save_result(feature_name, result, input_excel=None):
    
    if input_excel is None and isinstance(result, dict):
        input_excel = result.get("input_excel")
    cache_path = get_cache_path(feature_name, input_excel)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as file:
        pickle.dump(
            result,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )
    return cache_path


def load_result(feature_name, input_excel=None):
    
    cache_path = get_cache_path(feature_name, input_excel)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Solver-result cache not found: {cache_path}\n"
            f"Run python -m features.{feature_name}.solve first"
        )

    
    with cache_path.open("rb") as file:
        return pickle.load(file)


def run_solve(feature_name, solve, export, **solve_kwargs):
    
    explicit_input = solve_kwargs.get("input_excel")
    if explicit_input is not None:
        solve_kwargs["input_excel"] = get_input_excel(explicit_input)
    result = solve(**solve_kwargs)
    save_result(feature_name, result, solve_kwargs.get("input_excel"))
    export(result)
    return result


def run_export(feature_name, export, *, input_excel=None):
    
    result = load_result(feature_name, input_excel)
    export(result)
    return result
