# Model validation

This feature validates expected-loss aggregation over shared attack prefixes and stopping-trajectory calculations. It does not run an optimization algorithm.

`solve.py` validates a fixed hand-calculated example, a small example using the production simulation, and the case in the specified workbook. `diagnostics.py` independently enumerates stopping trajectories to reconcile the production aggregation in `common.simulation`. `export_csv.py` saves five evidence tables, and `plot.py` generates plots only from these CSV files. Features do not depend on each other's entry points.

## Mathematical conventions

Let $h$ denote the component-selection history, $i$ the next attack action, $c(i)$ the target component, $p(i\mid h)$ the action-selection probability, and $\ell(h,i)$ the expected loss at the current step. The local loss already includes the current-step success probability and defense effects.

$$
m(c\mid h)=\sum_{i:c(i)=c}p(i\mid h),\qquad
g(h,c)=\sum_{i:c(i)=c}p(i\mid h)\ell(h,i).
$$

$$
\rho(S)=1,\qquad\rho(hc)=\rho(h)m(c\mid h),\qquad
L=\sum_{(h,c)}\rho(h)g(h,c).
$$

Each unique history extension $(h,c)$ contributes the current-step loss once. The independent algorithm enumerates mutually exclusive complete action and stopping trajectories $\omega$ and calculates $\sum_\omega P(\omega)\sum_t\ell(h_t,i_t)$. It does not use arrival probabilities or stored weighted losses from the production aggregation as reference answers. The same physical component may occur in different histories and is weighted separately by each history's probability.

`DoNothing` terminates subsequent contributions; a history with no subsequent action is marked `candidate_history_end`. The cumulative loss of these trajectories is the sum of local expected losses, rather than a loss sample from a realized attack. The current probability propagation does not require previous compromises to succeed and does not include separate absorbing events for failure or blocking. Passing validation therefore does not establish that these events have been implemented. Exact reconciliation explicitly disables pruning without changing the shared defaults.

## Running the feature

From the project root, use a Python environment with the project's existing NumPy, pandas, Matplotlib and NetworkX dependencies installed:

```powershell
python run.py --feature model_validation --mode all --input-excel game-theory.xlsx
python run.py --feature model_validation --mode export --input-excel game-theory.xlsx
python run.py --feature model_validation --mode plot --input-excel game-theory.xlsx
python -m unittest discover -s tests -p test_model_validation.py -v
```

`model_validation` is optional and is not included in the existing `--feature all` set. The commands `python -m features.model_validation.solve`, `python -m features.model_validation.export_csv` and `python -m features.model_validation.plot` are also supported. All accept `--input-excel`; standalone plotting also accepts `--output-dir`.

`workbook_case` reads the graph, vulnerability distribution, IDSs, firewalls, patches, and the Strategy values Game_Iterations/Game_Learning_Rate from the specified workbook, then evaluates the case's initial configuration. Response uses the workbook's RP0 values and the existing production-function defaults; no optimizer is run. Fixed and synthetic integration examples remain in the CSV files and tests as general formula-regression cases. The two plots for each case use only `workbook_case`, identify the case in their titles, and exclude the fixed and synthetic examples.

Each invocation reads only the specified workbook and writes only to the corresponding `results/<case>/model_validation/` directory. It does not automatically run other cases or create their directories. The cache and CSV files record Case_ID, Input_Excel, Input_SHA256 and data provenance. Legacy caches containing only synthetic data require a new solve.

The workbook case groups vulnerabilities with the same target by the next component and uses the conditional loss `sum(p*ell)/sum(p)` to independently enumerate component-selection and stopping trajectories. This exactly preserves the current model's total expectation while avoiding the many duplicate trajectories produced by exporting vulnerability combinations. Fixed and synthetic integration examples still enumerate complete vulnerability-action trajectories; both modes are identified in Trajectory_Mode. The synthetic integration example uses 40 local evolutionary iterations and a learning rate of 0.2, without setting the global random state.

## Outputs and validation criteria

Results are saved in `results/<workbook_filename_without_extension>/model_validation/`:

- `validation_summary.csv`: production aggregation, independent enumeration, hand-calculated references, errors and PASS/FAIL status.
- `action_probabilities.csv`: action probabilities and current-step expected losses for each history, including DoNothing.
- `prefix_contributions.csv`: probabilities and loss contributions of unique history extensions.
- `terminal_trajectories.csv`: mutually exclusive stopping trajectories, their probabilities, cumulative losses and contributions.
- `run_metadata.csv`: case files and hashes, formulas, actual parameters, and probability and stopping conventions.
- `solve_result.pkl`: an intermediate cache generated by this project, used only for standalone export.
- `loss_reconciliation.png/.svg`: reconciliation of the workbook's production aggregation, sum of unique-prefix contributions, and sum over independently enumerated stopping trajectories. The three calculations should agree within tolerance; the plot also reports the maximum difference.
- `prefix_contributions.png/.svg`: unique-prefix contributions for the workbook. If there are more than 12, the 12 largest contributions by absolute value are shown and the remaining prefixes are combined into one entry. The plot preserves the full total, and the CSV retains all details.

After successful plotting, legacy `fixed_prefix_contributions.png/.svg` files are removed from the current case directory to avoid confusing the general hand-calculated example with case results. After updating the plotting code, use `--mode plot` to regenerate plots directly from the existing CSV files without solving again.

The fixed example should have prefix contributions of 4, 24 and 24, with total loss 52; the probabilities of its four stopping trajectories sum to 1. The production integration example covers duplicate input prefixes, multiple vulnerabilities on the same component, and different histories reaching the same component.

The absolute tolerance is $10^{-10}$ and the relative tolerance is $10^{-9}$, with acceptance threshold `atol + rtol * abs(expected)`. Invalid probabilities, missing histories and enumeration beyond 10000 trajectories cause explicit failures rather than silent truncation or renormalization. When numerical validation fails, evidence CSV files are saved before exiting with a nonzero status; plotting rejects failed results.
