

from common.config import (
    GAME_OPTIMIZATION_ITERATIONS,
    GAME_OPTIMIZATION_LEARNING_RATE,
    PRUNE_TOL_DEFAULT,
)
from common.game_theory import run_evolutionary_game
from common.game_termination import DEFAULT_GAME_TERMINATION
from common.models import AttackStrategy, DoNothingStrategy, Vulnerability
from common.parsers import component_vulnerability_names


# The exponential update keeps a small positive tail. This threshold remains
# a real complexity control while preserving the no-prune result numerically.


def resolve_prune_tolerance(use_pruning):
    """Map a validated case switch to the effective simulation threshold."""
    if type(use_pruning) is not bool:
        raise ValueError("use_pruning must be a boolean")
    return PRUNE_TOL_DEFAULT if use_pruning else None


def build_vulnerability_dict(vuln_data: dict, component_dist: dict) -> dict:
    
    vuln_list = vuln_data['Vulnerability']
    success_probs = vuln_data['Attack Success Probability']
    base_times = vuln_data['Attack time (h)']
    costs = vuln_data['Attack cost (monetary)']
    benefits_m = vuln_data['Attack Benefit (monetary)']
    benefits_d = vuln_data['Attack benefit (destruction)']
    w_m = vuln_data['Weight_ABM']
    w_d = vuln_data['Weight_ABD']
    defend_losses_m = vuln_data['Defend loss (monetary)']
    defend_losses_d = vuln_data['Defend loss (destruction)']
    w_dlm = vuln_data['Weight_DLM']
    w_dld = vuln_data['Weight_DLD']
    after_detect_probs = vuln_data[
        'Defense Success Probability after detecting (0)'
    ]

    final_benefits = [
        bm * wm + bd * wd
        for bm, wm, bd, wd in zip(benefits_m, w_m, benefits_d, w_d)
    ]
    final_defend_loss = [
        abs(lm * wl + ld * wd)
        for lm, wl, ld, wd in zip(
            defend_losses_m,
            w_dlm,
            defend_losses_d,
            w_dld
        )
    ]

    vuln_obj_map = {}
    for i, name in enumerate(vuln_list):
        vuln = Vulnerability(
            name=name,
            base_success=success_probs[i],
            base_attack_time=base_times[i],
            attack_cost=costs[i],
            attack_benefit=final_benefits[i],
            defend_loss=final_defend_loss[i],
            defense_success_prob=after_detect_probs[i],
            defend_loss_destruction=abs(defend_losses_d[i]),
            weight_dld=w_dld[i],
        )
        vuln_obj_map[name] = vuln

    return {
        component: [vuln_obj_map[name] for name in names]
        for component, names in component_vulnerability_names(vuln_data, component_dist).items()
    }


def generate_attack_strategies(
    current_node,
    next_edges,
    vuln_dict,
    prev_time,
    ids_config,
    firewall_config,
    patch_config,
    strategy_class=AttackStrategy,
    *,
    model_options=None,
):
    path_history = (
        current_node.split(' → ')
        if current_node != 'S'
        else ['S']
    )
    next_components = sorted({
        edge[1].split(' → ')[-1]
        for edge in next_edges
    })

    strategies = []
    strategy_options = {} if model_options is None else {"model_options": model_options}
    for comp in next_components:
        for vuln in vuln_dict.get(comp, []):
            strategies.append(
                strategy_class(
                    comp,
                    vuln,
                    path_history,
                    prev_time,
                    ids_config,
                    firewall_config,
                    patch_config,
                    **strategy_options,
                )
            )
    strategies.append(DoNothingStrategy())
    return strategies


def run_simulation(
    state_graph,
    vuln_data,
    component_vuln_dist,
    ids_config,
    firewall_config,
    patch_config,
    iterations=GAME_OPTIMIZATION_ITERATIONS,
    learning_rate=GAME_OPTIMIZATION_LEARNING_RATE,
    has_response=None,
    p_response=0.0,
    prune_tol=PRUNE_TOL_DEFAULT,
    *,
    attacker_initializer=None,
    capture_diagnostics=False,
    model_options=None,
    game_termination=DEFAULT_GAME_TERMINATION,
):
    
    
    
    if has_response is None:
        after_detect_probs = vuln_data.get(
            'Defense Success Probability after detecting (0)',
            []
        )
        has_response = any(p > 0.0 for p in after_detect_probs)

    vuln_dict = build_vulnerability_dict(
        vuln_data,
        component_vuln_dist
    )
    successors = {}
    for from_node, to_node in state_graph:
        successors.setdefault(from_node, []).append(to_node)

    results = {}

    def dfs(current_node):
        if current_node == "S":
            results["S"] = {
                'attack_time': 0.0,
                'attacker_strategy': [],
                'defender_strategy': [],
                'attacker_payoff': 0.0,
                'defender_loss': 0.0,
                'strategies': [],
                'evolution_history': []
            }

        next_nodes = successors.get(current_node, [])
        if not next_nodes:
            return

        strategies = generate_attack_strategies(
            current_node,
            [(current_node, n) for n in next_nodes],
            vuln_dict,
            results[current_node]['attack_time'],
            ids_config,
            firewall_config,
            patch_config,
            **({} if model_options is None else {"model_options": model_options}),
        )

        game_options = {}
        # Omit the shared default so legacy wrappers that do not accept the
        # optional keyword remain compatible.  Explicit None is forwarded to
        # disable early stopping for fixed-iteration experiments.
        if game_termination is not DEFAULT_GAME_TERMINATION:
            game_options["game_termination"] = game_termination
        if model_options is not None:
            game_options["model_options"] = model_options
        if attacker_initializer is not None:
            game_options["attacker_initializer"] = attacker_initializer
        if capture_diagnostics:
            game_options["capture_diagnostics"] = True
        game_result = run_evolutionary_game(
            strategies,
            has_response=has_response,
            iterations=iterations,
            learning_rate=learning_rate,
            p_response=p_response,
            **game_options,
        )

        results[current_node]['attacker_strategy'] = (
            game_result['final_p'].tolist()
        )
        results[current_node]['defender_strategy'] = (
            game_result['final_q'].tolist()
        )
        results[current_node]['strategies'] = strategies
        results[current_node]['evolution_history'] = (
            game_result['p_history']
        )
        results[current_node]['defender_evolution_history'] = (
            game_result['q_history']
        )
        results[current_node]['response_policy_history'] = (
            game_result['response_policy_history']
        )
        results[current_node]['game_result'] = game_result

        component_stats = {}
        for index, (s, p) in enumerate(zip(strategies, game_result['final_p'])):
            if isinstance(s, DoNothingStrategy):
                continue
            comp = s.target_component
            next_path = current_node + " → " + comp
            if next_path not in component_stats:
                component_stats[next_path] = {
                    'time': 0.0,
                    'payoff': 0.0,
                    'loss': 0.0,
                    'probability': 0.0,
                }
            component_stats[next_path]['time'] += (
                p * s.vulnerability.base_attack_time
            )
            component_stats[next_path]['payoff'] += (
                p * game_result['final_attacker_payoffs'][index]
            )
            component_stats[next_path]['loss'] += (
                p * game_result['final_defender_losses'][index]
            )
            component_stats[next_path]['probability'] += p

        for next_path, stat in component_stats.items():
            if prune_tol is not None and stat['probability'] <= prune_tol:
                continue
            results[next_path] = {
                'attack_time': (
                    results[current_node]['attack_time']
                    + stat['time']
                ),
                'attacker_payoff': stat['payoff'],
                'defender_loss': stat['loss'],
                'attacker_strategy': [],
                'defender_strategy': [],
                'strategies': [],
                'evolution_history': []
            }
            dfs(next_path)

    dfs("S")
    return results


def compute_total_expected_values(results: dict) -> tuple:
    
    availability = {"S": 1.0}
    total_attacker_payoff = 0.0
    total_defender_loss = 0.0

    sorted_nodes = sorted(
        results.keys(),
        key=lambda x: x.count("→")
    )

    for node in sorted_nodes:
        if node == "S":
            continue

        parts = node.split(" → ")
        if len(parts) < 2:
            continue
        parent_node = " → ".join(parts[:-1])

        if parent_node not in results:
            continue

        parent_prob = availability.get(parent_node, 0.0)
        if parent_prob == 0.0:
            availability[node] = 0.0
            continue

        current_comp = parts[-1]
        strategies = results[parent_node].get("strategies", [])
        probs = results[parent_node].get(
            "attacker_strategy",
            []
        )


        
        component_prob = sum(
            p
            for s, p in zip(strategies, probs)
            if hasattr(s, 'target_component') and s.target_component == current_comp
        )

        
        availability[node] = parent_prob * component_prob

        payoff = results[node].get("attacker_payoff", 0.0)
        loss = results[node].get("defender_loss", 0.0)

        
        
        
        total_attacker_payoff += parent_prob * payoff
        total_defender_loss += parent_prob * loss

    return total_attacker_payoff, total_defender_loss, availability
