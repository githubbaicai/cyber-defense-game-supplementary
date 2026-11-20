# Supplementary Materials: Cyber-Defense Game-Theoretic Experiments

This archive contains the implementation, input workbooks, configuration files, attack-path tables, random-seed settings, raw experiment tables, serialized solver outputs, and figures used for the study. It is intended to make the computational results independently inspectable and reproducible.

The source tree is in [`codex_game_1/`](codex_game_1/). The supplied case inputs are [`game-theory.xlsx`](codex_game_1/game-theory.xlsx) and [`game-theory-purdue.xlsx`](codex_game_1/game-theory-purdue.xlsx). Experimental outputs are under [`codex_game_1/results/`](codex_game_1/results/). CSV files preserve tabular results and metadata; SVG/PNG/PDF files are figures; PKL files are serialized solve objects; JSON files preserve manifests, configurations, and provenance.

## Reproducing the Python environment

The local Conda environment `game1` was inspected on November 18, 2026. It contains Python 3.9.25 on Windows and the 41 packages listed below. This is the environment snapshot at archive preparation; individual historical experiments also record their dependency versions in their manifests. The same package list is provided in [`requirements-game1.txt`](requirements-game1.txt).

From this `supplementary_materials` directory, activate the existing environment:

```powershell
conda activate game1
python --version
python -m pip list
```

To create an environment on another Windows computer, run the following commands instead. Choose a different environment name if `game1` already exists.

```powershell
conda create -n game1 python=3.9.25 pip=25.0
conda activate game1
python -m pip install -r requirements-game1.txt
```

The snapshot includes development tools and Windows-specific packages such as `wincertstore`; it is a record of the complete environment rather than a minimal cross-platform dependency list.

The recorded `pip list` for `game1` is:

| Package | Version |
| --- | --- |
| `about-time` | `4.2.1` |
| `alive-progress` | `3.1.5` |
| `autograd` | `1.5` |
| `certifi` | `2020.6.20` |
| `cma` | `3.2.2` |
| `contourpy` | `1.2.0` |
| `cycler` | `0.11.0` |
| `Deprecated` | `1.2.15` |
| `dill` | `0.3.9` |
| `et_xmlfile` | `2.0.0` |
| `fonttools` | `4.51.0` |
| `future` | `1.0.0` |
| `grapheme` | `0.6.0` |
| `importlib_resources` | `6.5.2` |
| `joblib` | `1.4.2` |
| `kiwisolver` | `1.4.4` |
| `matplotlib` | `3.9.2` |
| `multiprocess` | `0.70.17` |
| `networkx` | `3.2.1` |
| `numpy` | `1.26.4` |
| `openpyxl` | `3.1.5` |
| `packaging` | `24.1` |
| `pandas` | `2.2.2` |
| `pillow` | `10.4.0` |
| `pip` | `25.0` |
| `pymoo` | `0.6.1.3` |
| `pyparsing` | `3.1.2` |
| `python-dateutil` | `2.9.0.post0` |
| `pytz` | `2024.1` |
| `ruff` | `0.17.1` |
| `SALib` | `1.5.1` |
| `scipy` | `1.12.0` |
| `setuptools` | `58.0.4` |
| `six` | `1.16.0` |
| `tomli` | `2.5.0` |
| `tzdata` | `2023.3` |
| `vulture` | `2.16` |
| `wheel` | `0.37.1` |
| `wincertstore` | `0.2` |
| `wrapt` | `1.14.1` |
| `zipp` | `3.23.1` |

`pip check` completed without broken requirements in the supplied environment.

## Package layout and execution model

```text
supplementary_materials/
├── README.md
├── requirements-game1.txt
└── codex_game_1/
    ├── game-theory.xlsx
    ├── game-theory-purdue.xlsx
    ├── common/                 shared models, simulation, optimization, plotting
    ├── external/               workbook and attack-graph compatibility code
    ├── features/               feature-specific solve/export/plot modules
    └── results/                raw tables, manifests, serialized outputs, figures
```

Where available, feature modules follow a three-stage lifecycle: `solve.py` computes a Python result, `export_csv.py` writes tables and metadata, and `plot.py` reads exported CSV files to render figures. `common/pipeline.py` stores a solve object as `solve_result.pkl`. Most solve module entry points also invoke caching and CSV export; running `export_csv` separately regenerates tables from the cache.

From this `supplementary_materials` directory, enter the source directory before running modules. The following example uses the first case; use `game-theory-purdue.xlsx` as the environment variable value for the second case. Run reproduction commands in a working copy because they can overwrite feature outputs.

```powershell
conda activate game1
Set-Location .\codex_game_1
$env:GAME_THEORY_EXCEL = "game-theory.xlsx"
python -m features.initial_simulation.solve
python -m features.initial_simulation.export_csv
python -m features.initial_simulation.plot
```

For module entry points that provide an argument parser, such as `features.nsga2_ma.solve`, inspect supported options with `python -m features.nsga2_ma.solve --help`. Do not assume every module accepts the same CLI arguments. Configuration precedence is explicit function/CLI overrides, then the workbook `Strategy` sheet, then shared fallback defaults, with additional feature-specific JSON settings where supported.

The archive does not include the repository-level launcher `run.py`. Some feature READMEs refer to that launcher, and the probability-mapping and scalability solvers hash it as part of their source identity. Those workflows require the original launcher before a fresh solve. The independent-reference solver also expects a default `revision_20261102.json` configuration that is absent from this archive; its `solve()` API accepts an explicit configuration path. Existing results remain available for inspection and CSV-based plotting. These packaging limitations were identified during documentation; a complete rerun of every experiment was not performed.

## Inputs, parameters, seeds, and results

The workbook sheets contain the attack graph, labels, vulnerability distributions, allocation measures, patch settings, firewall settings, IDS settings, vulnerability attributes, and strategy parameters. `common/case_config.py`, `common/parsers.py`, and `common/case_validation.py` load and validate these sheets. Feature JSON files record feature-local overrides and protocols.

Seeds are stored in feature configuration JSON files and exported run metadata. `features/sensitivity_analysis/review_settings.json` specifies optimization seeds 42 and 43 and noise seeds 1042 and 1043. `features/probability_mapping/review_settings.json` specifies 42, 43, and 44; the archived `mapping_20261105_081945_ddaa4e58` experiment manifest records the overridden set 42 through 46. The saved multi-algorithm comparison records 50 runs starting at seed 42 and representative seed 42. Use each experiment's saved metadata for its effective seeds and parameters. Available fields include population size, FE or generation budget, game iterations, learning rate, pruning settings, model protocol, and input/source/configuration hashes.

In the following table, paths are relative to `codex_game_1/`, and `<case>` means `game-theory` or `game-theory-purdue`. The cases contain different subsets of the result groups.

| Result directory | Contents |
| --- | --- |
| `results/<case>/initial_simulation/` | Initial compromise probabilities, strategy histories, availability, and global totals. |
| `results/<case>/nsga2_ma/` | NSGA-II+MA fronts, defense selections, cost breakdowns, histories, and run metadata. |
| `results/<case>/macomp/` | MA variant runs, FE histories, rankings, selected policy, and manifests. |
| `results/<case>/multi_algorithm/` | Matched NSGA-II, SPEA2, MOEA/D comparisons, metrics, tests, and fronts. |
| `results/<case>/pareto_attack_paths/` | Attack graph tables, candidate paths, scenarios, decisions, and annotations. |
| `results/<case>/pareto_path_analysis/` | Knee selections, node risk, defense selections, and path-evolution tables. |
| `results/<case>/model_validation/` | Independent loss reconciliation and terminal-trajectory evidence. |
| `results/<case>/equilibrium_validation/` | Equilibrium diagnostics, residual traces, eigenvalues, and initialization sensitivity. |
| `results/<case>/structural_ablation/` | Fixed portfolios, ablation fronts, cross-evaluations, and model checks. |
| `results/<case>/probability_mapping/` | EPSS mapping inputs, search runs, cross-evaluations, decisions, and robustness figures. |
| `results/<case>/sensitivity_analysis/` | Parameter samples, metrics, Sobol indices, fronts, and portfolio evaluations. |
| `results/<case>/state_visualization/` | Attack/state graph tables, risk, availability, and main-path tables. |
| `results/<case>/independent_reference/` | Frozen independent reference fronts and reevaluation manifests. |

The `revision_registry` feature can generate full component-path lists (`component_paths.csv` and `component_paths_purdue.csv`) and vulnerability-expanded lists (`graph_paths.csv` and `graph_paths_purdue.csv`), together with resolved parameters and provenance. Generated registry outputs are absent from this archive. Archived path evidence is available in `pareto_attack_paths/candidate_paths.csv`, `pareto_attack_paths/path_steps.csv`, `structural_ablation/ablation_paths.csv`, and `state_visualization/main_attack_path.csv`, where present. The `scalability` code writes timings to `results/scalability/<experiment>/`; no scalability output directory is included.

Archived manifests identify implementation version `2026-11-02.legacy1` and model protocol `population-average-detection-response-v1` where those fields apply.

## Python file reference

The following reference covers every `.py` file in this archive.

| File | Purpose |
| --- | --- |
| `features/__init__.py` | Marks the package containing the experiment features. |

### `common/`

| File | Purpose |
| --- | --- |
| `common/__init__.py` | Marks the shared core package. |
| `common/attack_graph_data.py` | Builds component and vulnerability attack graph tables. |
| `common/attack_graph_layout.py` | Computes deterministic graph layout geometry for figures. |
| `common/case_config.py` | Loads and validates workbook case parameters and overrides. |
| `common/case_validation.py` | Validates case input semantics before simulation and optimization. |
| `common/config.py` | Defines project paths, defaults, output directories, and CSV contracts. |
| `common/defense_config.py` | Normalizes defense vectors and reconstructs simulation configurations. |
| `common/game_termination.py` | Defines local evolutionary-game stopping rules and diagnostics. |
| `common/game_theory.py` | Implements evolutionary game dynamics and payoff calculations. |
| `common/implementation.py` | Stores implementation and model protocol versions. |
| `common/memetic.py` | Implements Memetic NSGA-II and local-search FE accounting. |
| `common/model_options.py` | Defines immutable threat-model options for each run. |
| `common/models.py` | Defines shared vulnerability and attack-strategy models. |
| `common/optimization_execution.py` | Adapts optimizers, enforces budgets, and records traces. |
| `common/optimization_fronts.py` | Selects Pareto fronts and computes empirical reference metrics. |
| `common/parsers.py` | Parses IDS, firewall, patch, and defense settings. |
| `common/pipeline.py` | Connects solve, cache, export, and reload stages. |
| `common/plotting.py` | Provides shared figure styling and save helpers. |
| `common/problem.py` | Implements the shared CyberDefenseProblem evaluator. |
| `common/problem_factory.py` | Loads a complete optimization problem context. |
| `common/sampling.py` | Provides feasible sampling, crossover, mutation, and response repair. |
| `common/selection_override.py` | Reads and validates optimizer-selection metadata. |
| `common/simulation.py` | Runs attack simulation and computes expected values. |
| `common/termination.py` | Implements Pareto convergence and safe termination. |

### `external/`

| File | Purpose |
| --- | --- |
| `external/__init__.py` | Marks the external compatibility package. |
| `external/attack_graph.py` | Builds attack paths and state-transition graphs. |
| `external/excel_processor.py` | Reads and validates the multi-sheet workbooks. |

### `features/equilibrium_validation/`

| File | Purpose |
| --- | --- |
| `features/equilibrium_validation/__init__.py` | Marks the equilibrium_validation feature package. |
| `features/equilibrium_validation/diagnostics.py` | Computes map residuals, induced payoff matrices, Jacobian eigenvalues, and local stability diagnostics. |
| `features/equilibrium_validation/export_csv.py` | Exports equilibrium game definitions, diagnostics, traces, path sensitivity, and metadata. |
| `features/equilibrium_validation/inputs.py` | Validates E3 settings and loads complete, traceable portfolio decisions. |
| `features/equilibrium_validation/plot.py` | Plots strategy traces, initialization sensitivity, and path-frequency/loss diagnostics. |
| `features/equilibrium_validation/publication.py` | Assembles, validates, and reads independently identified publication evidence tables. |
| `features/equilibrium_validation/solve.py` | Runs bounded local-game diagnostics and independent full-graph portfolio replays. |

### `features/independent_reference/`

| File | Purpose |
| --- | --- |
| `features/independent_reference/__init__.py` | Marks the independent_reference feature package. |
| `features/independent_reference/analysis.py` | Freezes independent search observations, recalculates HV/IGD, and performs statistical comparisons. |
| `features/independent_reference/config.py` | Validates independent-reference search budgets, seeds, source directories, and game stopping settings. |
| `features/independent_reference/export_csv.py` | Freezes reference fronts and normalization data, verifies hashes, and exports reevaluation tables. |
| `features/independent_reference/inputs.py` | Loads immutable comparison CSVs, verifies objective compatibility, and replays saved objectives. |
| `features/independent_reference/plot.py` | Plots independent-reference Pareto comparisons and metric distributions from saved CSVs. |
| `features/independent_reference/solve.py` | Runs independent multi-seed reference searches and reevaluates the saved comparison fronts. |

### `features/initial_simulation/`

| File | Purpose |
| --- | --- |
| `features/initial_simulation/__init__.py` | Marks the initial_simulation feature package. |
| `features/initial_simulation/export_csv.py` | Exports node metrics, availability, global totals, strategy probabilities, and game histories. |
| `features/initial_simulation/plot.py` | Plots compromise probabilities and per-state strategy evolution from the initial simulation CSVs. |
| `features/initial_simulation/solve.py` | Loads the case defenses, builds attack states, and runs the initial evolutionary-game simulation. |

### `features/macomp/`

| File | Purpose |
| --- | --- |
| `features/macomp/__init__.py` | Marks the macomp feature package. |
| `features/macomp/config.py` | Defines MA variants, local-search settings, strict FE budgets, and repeated-run settings. |
| `features/macomp/export_csv.py` | Exports variant runs, final decisions, rankings, manifests, and selected_ma.json. |
| `features/macomp/plot.py` | Plots FE convergence, confidence intervals, variant rankings, and resource-use diagnostics. |
| `features/macomp/solve.py` | Runs repeated Memetic NSGA-II variants, compares them, and selects the local-search configuration. |
| `features/macomp/utils.py` | Provides front traces, empirical metric references, and confidence-interval calculations. |

### `features/model_validation/`

| File | Purpose |
| --- | --- |
| `features/model_validation/__init__.py` | Marks the model_validation feature package. |
| `features/model_validation/diagnostics.py` | Independently reconciles probability-weighted incremental loss and terminal trajectory calculations. |
| `features/model_validation/export_csv.py` | Exports loss reconciliation, action probabilities, prefix contributions, trajectories, and metadata. |
| `features/model_validation/plot.py` | Plots production-versus-independent loss reconciliation and path-prefix loss contributions. |
| `features/model_validation/solve.py` | Builds fixed and case-based checks against independent stopping-trajectory loss calculations. |

### `features/multi_algorithm/`

| File | Purpose |
| --- | --- |
| `features/multi_algorithm/__init__.py` | Marks the multi_algorithm feature package. |
| `features/multi_algorithm/execution.py` | Preserves the historical execution-module import interface by exposing the shared execution implementation. |
| `features/multi_algorithm/export_csv.py` | Exports per-run algorithm metrics, fronts, statistics, histories, and experiment metadata. |
| `features/multi_algorithm/fronts.py` | Preserves the historical front-processing import interface using the shared implementation. |
| `features/multi_algorithm/plot.py` | Plots paired-budget Pareto fronts, HV/IGD distributions, convergence, and measured runtime. |
| `features/multi_algorithm/solve.py` | Compares NSGA-II+MA, NSGA-II, SPEA2, and MOEA/D under matched FE, time, or termination protocols. |

### `features/nsga2_ma/`

| File | Purpose |
| --- | --- |
| `features/nsga2_ma/__init__.py` | Marks the nsga2_ma feature package. |
| `features/nsga2_ma/annotations.py` | Selects representative Pareto points and evaluates fixed portfolios for cost-breakdown annotations. |
| `features/nsga2_ma/export_csv.py` | Exports Pareto solutions, defense selections, cost tables, path details, histories, and metadata. |
| `features/nsga2_ma/plot.py` | Plots the Pareto front, defense-cost breakdown, selected reference portfolios, and FE history. |
| `features/nsga2_ma/solve.py` | Loads the applicable MA settings, runs NSGA-II+MA, and analyzes the final Pareto solutions. |

### `features/pareto_attack_paths/`

| File | Purpose |
| --- | --- |
| `features/pareto_attack_paths/__init__.py` | Marks the pareto_attack_paths feature package. |
| `features/pareto_attack_paths/export_csv.py` | Exports the computed graph, scenario, decision, path-step, and annotation tables. |
| `features/pareto_attack_paths/inputs.py` | Loads a verified upstream Pareto snapshot and resolves options and complete decision vectors. |
| `features/pareto_attack_paths/path_analysis.py` | Finds the most probable complete successful vulnerability sequence for each analyzed scenario. |
| `features/pareto_attack_paths/plot.py` | Draws attack-path panels and defense annotations using saved scenario CSVs. |
| `features/pareto_attack_paths/solve.py` | Resolves three representative portfolios and computes scenario-specific attack paths and annotations. |

### `features/pareto_path_analysis/`

| File | Purpose |
| --- | --- |
| `features/pareto_path_analysis/__init__.py` | Marks the pareto_path_analysis feature package. |
| `features/pareto_path_analysis/export_csv.py` | Exports selected Pareto points, defenses, node risks, strategy histories, and spreadsheet summaries. |
| `features/pareto_path_analysis/plot.py` | Plots selected-point comparisons, node risk, per-path evolution, and manuscript summary panels. |
| `features/pareto_path_analysis/solve.py` | Selects knee and minimum-cost-plus-loss solutions, replays them, and records path/game evolution. |

### `features/probability_mapping/`

| File | Purpose |
| --- | --- |
| `features/probability_mapping/__init__.py` | Marks the probability_mapping feature package. |
| `features/probability_mapping/config.py` | Defines EPSS mapping scenarios and validates case-local seeds, budgets, and optimizer settings. |
| `features/probability_mapping/evaluation.py` | Retains complete decisions, cross-evaluates candidate pools, and ranks choices within each mapping. |
| `features/probability_mapping/export_csv.py` | Exports mapping inputs, runs, fronts, decisions, cross-evaluations, snapshots, and manifests. |
| `features/probability_mapping/inputs.py` | Validates EPSS provenance and dates and builds independently transformed probability scenarios. |
| `features/probability_mapping/plot.py` | Reads mapping result CSVs and plots objective, loss, candidate-gap, and resource-choice sensitivity. |
| `features/probability_mapping/solve.py` | Runs isolated mapping searches and cross-evaluates complete decisions under each target mapping. |

### `features/pruning_validation/`

| File | Purpose |
| --- | --- |
| `features/pruning_validation/__init__.py` | Marks the pruning_validation feature package. |
| `features/pruning_validation/experiments.py` | Provides compatibility helpers for the former duplicated equilibrium-experiment entry point. |
| `features/pruning_validation/export_csv.py` | Exports pruning comparisons, per-case details, and evolutionary history tables. |
| `features/pruning_validation/initial_conditions.py` | Provides local-game compatibility helpers after initial-condition diagnostics moved to E3. |
| `features/pruning_validation/matrices.py` | Builds local payoff and detection matrices for pruning-specific diagnostics. |
| `features/pruning_validation/plot.py` | Plots pruning comparisons and per-step probability histories and generates HTML evolution reports. |
| `features/pruning_validation/solve.py` | Compares pruned and unpruned simulations on deterministic defense vectors and records evolution. |
| `features/pruning_validation/stability.py` | Diagnoses attacker equilibria using an induced matrix with the final defender response frozen. |

### `features/revision_registry/`

| File | Purpose |
| --- | --- |
| `features/revision_registry/__init__.py` | Marks the revision_registry feature package. |
| `features/revision_registry/_shared.py` | Provides workbook names, unknown-value handling, SHA256 hashing, and canonical JSON serialization. |
| `features/revision_registry/export_csv.py` | Validates registry schemas and writes resolved parameters, path lists, and provenance tables. |
| `features/revision_registry/parameters.py` | Records case, defense, vulnerability, and derived parameter values with their source locations. |
| `features/revision_registry/paths.py` | Enumerates simple component paths, expands vulnerability combinations, and summarizes Purdue paths. |
| `features/revision_registry/provenance.py` | Preserves existing artifact records and checks input hashes and selection-override metadata. |
| `features/revision_registry/solve.py` | Registers both cases, derives parameter and path tables, and checks input integrity without optimization. |
| `features/revision_registry/workbook.py` | Reads workbook rows and constructs component/vulnerability mappings while preserving source row numbers. |

### `features/scalability/`

| File | Purpose |
| --- | --- |
| `features/scalability/__init__.py` | Marks the scalability feature package. |
| `features/scalability/config.py` | Validates timing settings and schedules cases, portfolios, game modes, and repeats. |
| `features/scalability/export_csv.py` | Validates raw timing rows and derives summary statistics from their saved CSV. |
| `features/scalability/solve.py` | Runs bounded serial subprocess evaluations and records environment, source, input, and timing manifests. |
| `features/scalability/worker.py` | Builds a fresh uncached problem and measures one complete fixed-portfolio objective evaluation. |

### `features/sensitivity_analysis/`

| File | Purpose |
| --- | --- |
| `features/sensitivity_analysis/__init__.py` | Marks the sensitivity_analysis feature package. |
| `features/sensitivity_analysis/analysis.py` | Constructs Saltelli designs and computes Sobol indices, confidence intervals, and diagnostics. |
| `features/sensitivity_analysis/config.py` | Resolves experiment settings, ranges, sampling levels, bootstrap settings, and random seeds. |
| `features/sensitivity_analysis/export_csv.py` | Exports and validates parameter, sample, metric, Sobol, front, and decision-evaluation CSV tables. |
| `features/sensitivity_analysis/optimization.py` | Creates independent scenario problems, evaluates complete decisions, and merges repeated-search fronts. |
| `features/sensitivity_analysis/parameters.py` | Defines sensitivity factors, transforms case data and model options, and records parameter provenance. |
| `features/sensitivity_analysis/plot.py` | Plots Sobol, decision, budget, and convergence diagnostics and the manuscript sensitivity figures. |
| `features/sensitivity_analysis/progress.py` | Reports parent-process progress, completed stages, and elapsed time during long experiments. |
| `features/sensitivity_analysis/solve.py` | Coordinates scenario sampling, fixed-decision evaluation, repeated optimization, and Sobol statistics. |

### `features/state_visualization/`

| File | Purpose |
| --- | --- |
| `features/state_visualization/__init__.py` | Marks the state_visualization feature package. |
| `features/state_visualization/attack_graph_data.py` | Re-exports shared component/vulnerability attack-graph table builders for compatibility. |
| `features/state_visualization/attack_graph_layout.py` | Re-exports the shared deterministic attack-graph layout for compatibility. |
| `features/state_visualization/export_csv.py` | Exports attack/state graph tables, probabilities, node metrics, main paths, and availability. |
| `features/state_visualization/plot.py` | Plots component/vulnerability attack graphs, state risks, main paths, and state transitions from CSVs. |
| `features/state_visualization/solve.py` | Builds attack-graph tables and computes simulated state risks, transition probabilities, and main paths. |

### `features/structural_ablation/`

| File | Purpose |
| --- | --- |
| `features/structural_ablation/__init__.py` | Marks the structural_ablation feature package. |
| `features/structural_ablation/config.py` | Defines E7 model scenarios and resolves case-local ablation, search, and game settings. |
| `features/structural_ablation/export_csv.py` | Validates and exports fixed portfolios, ablation fronts, cross-evaluations, paths, checks, and manifests. |
| `features/structural_ablation/plot.py` | Plots fixed-portfolio losses, native/common fronts, defense-domain costs, and parameter effects. |
| `features/structural_ablation/solve.py` | Evaluates fixed portfolios, reoptimizes under matched FE budgets, and cross-evaluates model ablations. |

## Reproducibility notes

Run module commands from `codex_game_1/` so imports and relative workbook paths resolve consistently. Preserve the recorded configuration and random seeds, and compare the input workbook SHA256 with the experiment metadata. Several older runs record a workbook hash different from the current top-level workbook. Where an experiment includes its own workbook snapshot, use that snapshot; the current workbook should not be assumed to reproduce all historical runs exactly.

Some saved `latest.json` files and CSV/JSON metadata contain absolute paths from the original computer. For plotting APIs that accept `output_dir`, supply the actual extracted experiment directory explicitly rather than relying on historical pointers. Some archive directory names also contain spaces or non-English characters; quote these paths. Original raw results and manifests have been retained, including their historical hashes. Removing comments changes source-file SHA256 values, so the cleaned source files have different byte hashes from the original experiment manifests.

All Chinese text has been removed from Python comments and docstrings. Operational string literals, including messages and labels, retain their original values. The cleanup was checked by compiling all 126 Python files, verifying zero Chinese comments/docstrings, and comparing abstract syntax trees after excluding docstrings; the executable logic matched the original sources. These checks do not constitute a complete rerun of the experiments.
