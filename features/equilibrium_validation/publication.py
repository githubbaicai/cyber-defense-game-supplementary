"""Optional, independently identified CSV evidence for the paper summary figure.

The publication cohort contains attacker-initialization changes only. Its
full-graph runs retain the normal finite budget and early stopping. Local
comparisons freeze baseline A/L/f and execute the same fixed number of updates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.game_theory import replicator_dynamics

TABLE_NAMES = ('publication_initialization_runs', 'publication_initialization_trace')
IDENTITY = ('Case_ID', 'Experiment_ID', 'Portfolio_ID', 'Run_ID')
RUN_COLUMNS = (*IDENTITY, 'Input_Excel', 'Input_SHA256', 'Initialization', 'Seed',
               'Initialization_Label', 'Initialization_Order', 'Is_Baseline',
               'Baseline_Run_ID', 'Learning_Rate', 'Response_Initial',
               'Full_Graph_Iterations', 'Local_Iterations', 'Trace_Stride',
               'Prune_Tolerance', 'Baseline_Path', 'Path_Weight',
               'Path_Rank_Certified', 'Total_Defender_Loss', 'Local_Node_Paths_JSON',
               'Decision_JSON', 'Provenance_JSON', 'Status', 'Reason')
TRACE_COLUMNS = (*IDENTITY, 'Node_Path', 'Iteration', 'Maximum_Attack_Strategy_Difference')


def path_bounds(graph, results):
    """Choice-path weights or conservative bounds at unexpanded prefixes."""
    rows = []
    for path in graph.attack_paths:
        weight, exact = 1., True
        for index, target in enumerate(path[1:], 1):
            if target == 'T':
                break
            prefix = ' → '.join(path[:index])
            if prefix not in results or not results[prefix].get('strategies'):
                exact = False
                break
            state = results[prefix]
            weight *= sum(float(probability) for strategy, probability in
                          zip(state['strategies'], state['attacker_strategy'])
                          if getattr(strategy, 'target_component', None) == target)
            if weight == 0:
                break
        rows.append({'path': ' → '.join(path), 'weight': weight, 'exact': exact})
    return rows


def assemble_tables(records, baseline_paths, local_differences, portfolio, config, provenance):
    """Assemble traceable standard tables from already computed evidence.

    records contain keys/design/loss/path_bounds. local_differences maps each
    full Run_ID to node -> (actual rounds, max absolute p difference from the
    baseline at those same rounds). No interpolation or terminal padding occurs.
    """
    if not records:
        return {name: [] for name in TABLE_NAMES}
    expected = [('original', None)]
    for kind in config['attacker_initializations']:
        if kind != 'original':
            expected.extend((kind, seed) for seed in (config['dirichlet_seeds'] if kind == 'dirichlet' else [None]))
    observed = [(row['design']['initialization'], row['design']['seed']) for row in records]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError('publication summary: incomplete attacker-initialization cohort')
    baseline = next(row for row in records if row['design']['design_factor'] == 'baseline')
    candidates = sorted((row for row in baseline_paths if row['exact']),
                        key=lambda row: (-row['weight'], row['path']))
    if not candidates or any(row['weight'] > candidates[0]['weight'] for row in baseline_paths):
        raise ValueError('publication summary: baseline dominant path is not certified against omitted bounds')
    path = candidates[0]['path']
    settings = config['publication_summary']
    rows, traces = [], []
    for order, record in enumerate(records):
        keys, design = record['keys'], record['design']
        is_baseline = record is baseline
        label = 'O' if is_baseline else 'U' if design['initialization'] == 'uniform' else f'D{config["dirichlet_seeds"].index(design["seed"]) + 1}'
        bound = next(row for row in record['path_bounds'] if row['path'] == path)
        differences = local_differences[keys['Run_ID']]
        rows.append({**keys, 'Input_Excel': provenance['input_excel'],
                     'Input_SHA256': provenance['input_sha256'],
                     'Initialization': design['initialization'], 'Seed': design['seed'],
                     'Initialization_Label': label, 'Initialization_Order': order,
                     'Is_Baseline': is_baseline, 'Baseline_Run_ID': baseline['keys']['Run_ID'],
                     'Learning_Rate': design['learning_rate'], 'Response_Initial': design['response'],
                     'Full_Graph_Iterations': design['iterations'],
                     'Local_Iterations': settings['local_iterations'], 'Trace_Stride': settings['trace_stride'],
                     'Prune_Tolerance': design['prune_tol'], 'Baseline_Path': path,
                     'Path_Weight': bound['weight'] if bound['exact'] else None,
                     'Path_Rank_Certified': bound['exact'] and all(row['weight'] <= bound['weight'] for row in record['path_bounds']),
                     'Total_Defender_Loss': record['loss'],
                     'Local_Node_Paths_JSON': json.dumps(list(differences), ensure_ascii=False),
                     'Decision_JSON': json.dumps(portfolio['decision']),
                     'Provenance_JSON': json.dumps(provenance, ensure_ascii=False, sort_keys=True),
                     'Status': 'ok' if bound['exact'] else 'path_unknown',
                     'Reason': 'baseline choice path tracked unchanged; T uses evaluated-prefix fallback; local fixed-round comparison, response initialization held fixed'})
        for node, (rounds, values) in differences.items():
            for iteration, value in zip(rounds, values):
                traces.append({**keys, 'Node_Path': node, 'Iteration': int(iteration),
                               'Maximum_Attack_Strategy_Difference': float(value)})
    return {TABLE_NAMES[0]: rows, TABLE_NAMES[1]: traces}


def build_tables(records, graph, portfolio, config, provenance):
    """Compute only the optional fixed-local traces using normal run results."""
    records = [row for row in records if row['design']['design_factor'] in ('baseline', 'attacker_initialization')]
    if len(records) < 2:
        raise ValueError('publication summary requires a successful baseline and alternate attacker initialization')
    baseline = next(row for row in records if row['design']['design_factor'] == 'baseline')
    for row in records:
        row['path_bounds'] = path_bounds(graph, row['results'])
    exact = sorted((row for row in baseline['path_bounds'] if row['exact']), key=lambda row: (-row['weight'], row['path']))
    if not exact:
        raise ValueError('publication summary has no evaluated baseline path')
    parts = exact[0]['path'].split(' → ')
    states = baseline['results']
    nodes = [' → '.join(parts[:i]) for i in range(1, len(parts))]
    nodes = [node for node in nodes if node in states and 'game_result' in states[node]
             and len(states[node]['strategies']) > 1]
    if not nodes:
        raise ValueError('publication summary requires a nondegenerate local game on the baseline path')
    from features.equilibrium_validation.solve import _initializer
    settings = config['publication_summary']
    rounds = np.unique(np.r_[np.arange(0, settings['local_iterations'] + 1, settings['trace_stride']), settings['local_iterations']])
    differences = {row['keys']['Run_ID']: {} for row in records}
    for node in nodes:
        reference = None
        source = states[node]['game_result']
        inputs = source['game_inputs']
        for row in records:
            design = row['design']
            initial = inputs['p_init'].copy()
            initializer = _initializer(design['initialization'], design['seed'])
            if initializer is not None:
                initial = initializer(states[node]['strategies'], initial)
            game = replicator_dynamics(initial, inputs['attacker_gain_matrix'], inputs['defender_loss_matrix'], inputs['f'],
                                       iterations=settings['local_iterations'], learning_rate=design['learning_rate'],
                                       has_response=source['has_response'], p_response=design['response'], game_termination=None)
            history = np.asarray(game['p_history'])[rounds]
            if row is baseline:
                reference = history
            if reference is None:
                raise ValueError('publication baseline must precede alternate initializations')
            differences[row['keys']['Run_ID']][node] = (rounds, np.max(np.abs(history - reference), axis=1))
    return assemble_tables(records, baseline['path_bounds'], differences, portfolio, config, provenance)


def validate_tables(runs, trace):
    """Reject mixed provenance, misaligned rounds and untracked path evidence."""
    for frame, required in ((runs, RUN_COLUMNS), (trace, TRACE_COLUMNS)):
        if not set(required).issubset(frame.columns):
            raise ValueError('publication initialization CSV: missing required columns')
    if runs.empty and trace.empty:
        return
    for frame in (runs, trace):
        for column in IDENTITY:
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq('').any():
                raise ValueError(f'publication initialization CSV: empty identity {column}')
    if runs['Input_SHA256'].isna().any() or not runs['Input_SHA256'].astype(str).str.fullmatch(r'[a-fA-F0-9]{64}').all():
        raise ValueError('publication initialization CSV: invalid input hash')
    if runs.empty or trace.empty or runs.duplicated(list(IDENTITY)).any():
        raise ValueError('publication initialization CSV: missing or duplicate runs')
    keys = list(IDENTITY)
    if trace.duplicated(keys + ['Node_Path', 'Iteration']).any():
        raise ValueError('publication initialization CSV: duplicate trace key')
    joined = trace.merge(runs[keys], on=keys, how='left', indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('publication initialization CSV: trace has unknown run identity')
    for frame, columns in ((runs, ('Initialization_Order', 'Learning_Rate', 'Response_Initial', 'Full_Graph_Iterations', 'Local_Iterations', 'Trace_Stride', 'Path_Weight', 'Total_Defender_Loss')),
                           (trace, ('Iteration', 'Maximum_Attack_Strategy_Difference'))):
        for column in columns:
            numeric = pd.to_numeric(frame[column], errors='coerce')
            if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                raise ValueError(f'publication initialization CSV: nonfinite {column}')
            frame[column] = numeric
    for column in ('Full_Graph_Iterations', 'Local_Iterations', 'Trace_Stride'):
        if ((runs[column] < 1) | (runs[column] % 1 != 0)).any():
            raise ValueError(f'publication initialization CSV: {column} must be a positive integer')
    for frame, column in ((runs, 'Initialization_Order'), (trace, 'Iteration')):
        if ((frame[column] < 0) | (frame[column] % 1 != 0)).any():
            raise ValueError(f'publication initialization CSV: {column} must be a nonnegative integer')
    if (runs['Learning_Rate'] < 0).any() or not runs['Response_Initial'].between(0, 1).all():
        raise ValueError('publication initialization CSV: invalid adaptation settings')
    if not runs['Initialization'].isin(('original', 'uniform', 'dirichlet')).all():
        raise ValueError('publication initialization CSV: unknown initialization')
    if not runs['Status'].eq('ok').all():
        raise ValueError('publication initialization CSV: missing path evidence')
    if not trace['Maximum_Attack_Strategy_Difference'].between(0, 1).all() or not runs['Path_Weight'].between(0, 1).all():
        raise ValueError('publication initialization CSV: probabilities outside [0,1]')
    for _, cohort in runs.groupby(['Case_ID', 'Experiment_ID', 'Portfolio_ID'], sort=False):
        for column in ('Input_Excel', 'Input_SHA256', 'Learning_Rate', 'Response_Initial', 'Full_Graph_Iterations', 'Local_Iterations', 'Trace_Stride', 'Prune_Tolerance', 'Baseline_Path', 'Decision_JSON', 'Local_Node_Paths_JSON'):
            if cohort[column].fillna('none').nunique() != 1:
                raise ValueError(f'publication initialization CSV: mixed {column} within cohort')
        baseline = cohort.loc[cohort['Is_Baseline'].astype(str).str.lower().isin(('true', '1'))]
        if len(baseline) != 1 or not cohort['Baseline_Run_ID'].eq(baseline.iloc[0]['Run_ID']).all():
            raise ValueError('publication initialization CSV: ambiguous baseline')
        if baseline.iloc[0]['Initialization'] != 'original':
            raise ValueError('publication initialization CSV: baseline must use prescribed initialization')
        if cohort[['Initialization', 'Seed']].duplicated().any():
            raise ValueError('publication initialization CSV: duplicate initialization/seed')
        if cohort['Initialization_Order'].duplicated().any() or cohort['Initialization_Label'].duplicated().any():
            raise ValueError('publication initialization CSV: duplicate initialization order/label')
        expected_nodes = json.loads(cohort.iloc[0]['Local_Node_Paths_JSON'])
        if not expected_nodes or len(expected_nodes) != len(set(expected_nodes)):
            raise ValueError('publication initialization CSV: invalid local node selection')
        expected_rounds = np.unique(np.r_[np.arange(0, cohort.iloc[0]['Local_Iterations'] + 1, cohort.iloc[0]['Trace_Stride']), cohort.iloc[0]['Local_Iterations']])
        for row in cohort.to_dict('records'):
            selected = trace
            for key in IDENTITY:
                selected = selected.loc[selected[key].astype(str).eq(str(row[key]))]
            if set(selected['Node_Path']) != set(expected_nodes):
                raise ValueError('publication initialization CSV: missing local node')
            if row['Run_ID'] == baseline.iloc[0]['Run_ID'] and not selected['Maximum_Attack_Strategy_Difference'].eq(0).all():
                raise ValueError('publication initialization CSV: baseline self difference is not zero')
            for _, node in selected.groupby('Node_Path'):
                if not np.array_equal(np.sort(node['Iteration'].to_numpy()), expected_rounds):
                    raise ValueError('publication initialization CSV: unmatched actual rounds')


def export_tables(tables, directory):
    """Export only the independent publication tables; leave legacy E3 files alone."""
    frames = [pd.DataFrame(tables.get(name, []), columns=columns) for name, columns in
              zip(TABLE_NAMES, (RUN_COLUMNS, TRACE_COLUMNS))]
    validate_tables(*frames)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(TABLE_NAMES, frames):
        frame.to_csv(directory / f'{name}.csv', index=False)
    return dict(zip(TABLE_NAMES, frames))


def read_tables(directory):
    """Read an optional complete CSV pair, independent of legacy run_metadata."""
    paths = [Path(directory) / f'{name}.csv' for name in TABLE_NAMES]
    if not any(path.is_file() for path in paths):
        return None
    if not all(path.is_file() for path in paths):
        raise ValueError('publication initialization CSV: incomplete table pair')
    frames = [pd.read_csv(path) for path in paths]
    validate_tables(*frames)
    return frames
