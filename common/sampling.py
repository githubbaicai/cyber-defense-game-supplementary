

import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

from common.defense_config import normalize_decision_vector
from common.parsers import defense_categories as defense_categories


_COST_EPS = 1e-12


class ResponseAwareCrossover(TwoPointCrossover):
    """Two-point crossover for binary defenses and arithmetic p_response."""

    def _do(self, problem, X, **kwargs):
        offspring = np.asarray(X, dtype=float).copy()
        defense_count = getattr(problem, "defense_count", len(problem.defense_items) if hasattr(problem, "defense_items") else problem.n_var - 1)
        for mating in range(X.shape[1]):
            if defense_count >= 2:
                points = np.sort(np.random.choice(
                    np.arange(1, defense_count),
                    size=min(2, defense_count - 1),
                    replace=False,
                ))
                start = int(points[0])
                stop = int(points[-1]) if len(points) > 1 else defense_count
                offspring[0, mating, start:stop] = X[1, mating, start:stop]
                offspring[1, mating, start:stop] = X[0, mating, start:stop]
        if defense_count < problem.n_var:
            weights = np.random.random(X.shape[1])
            parent_response = X[:, :, defense_count]
            offspring[0, :, defense_count] = weights * parent_response[0] + (1.0 - weights) * parent_response[1]
            offspring[1, :, defense_count] = (1.0 - weights) * parent_response[0] + weights * parent_response[1]
        return offspring


class ResponseAwareMutation(BitflipMutation):
    """Bit flip defenses and apply bounded Gaussian mutation to p_response."""

    def _do(self, problem, X, **kwargs):
        offspring = np.asarray(X, dtype=float).copy()
        defense_count = getattr(problem, "defense_count", len(problem.defense_items) if hasattr(problem, "defense_items") else problem.n_var - 1)
        probability = self.get_prob_var(problem, size=(len(X), 1))
        mask = np.random.random((len(X), defense_count)) < probability
        rows, columns = np.where(mask)
        offspring[rows, columns] = 1.0 - offspring[rows, columns]
        if defense_count < problem.n_var:
            response_mask = np.random.random((len(X), 1)) < probability
            offspring[:, defense_count] = np.where(
                response_mask[:, 0],
                np.clip(offspring[:, defense_count] + np.random.normal(0.0, 0.1, len(X)), 0.0, 1.0),
                offspring[:, defense_count],
            )
        return offspring


def response_probability(prefix, defense_items):
    """Use a neutral response propensity for injected RP configurations."""
    return 0.5 if any(
        float(value) >= 0.5 and label in ("RP1", "RP2", "RP3")
        for value, (label, _) in zip(prefix, defense_items)
    ) else 0.0


def _repair_response_probability(values, defense_items):
    values[:] = normalize_decision_vector(values, defense_items)
    return values


class FeasibleBinarySampling(Sampling):
    def __init__(self, defense_items, budget, extra_injected=None, all_data=None):
        super().__init__()
        self.defense_items = defense_items
        self.budget = budget
        self.extra_injected = (
            list(extra_injected) if extra_injected else []
        )
        self.injected_solutions = []
        self.categories = defense_categories(defense_items, all_data)
        self._value_per_cost = self._compute_value_per_cost(defense_items, all_data)

    @staticmethod
    def _compute_value_per_cost(defense_items, all_data):
        
        lookup = {}
        if isinstance(all_data, dict):
            for sheet_name, column in (
                ("Patch vulnerability", "Patching effectiveness"),
                ("Set firewall", "Firewall effectiveness"),
                ("Set IDS", "Detection effectiveness"),
            ):
                sheet = all_data.get(sheet_name)
                if not isinstance(sheet, dict):
                    continue
                for label, value in zip(
                    sheet.get("Label", []),
                    sheet.get(column, []),
                ):
                    try:
                        number = float(value)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if np.isfinite(number):
                        lookup[str(label)] = max(0.0, number)

        value_per_cost = np.zeros(len(defense_items), dtype=float)
        for index, (label, cost) in enumerate(defense_items):
            effectiveness = (
                int(label[-1])
                if label in ("RP1", "RP2", "RP3")
                else lookup.get(str(label), 0.0)
            )
            value_per_cost[index] = effectiveness / max(float(cost), _COST_EPS)
        return value_per_cost

    def _do(self, problem, n_samples, **kwargs):
        n_var = len(self.defense_items)
        total_var = problem.n_var
        if total_var not in (n_var, n_var + 1):
            raise ValueError("The sampling dimension does not match the number of defense measures")
        X = np.zeros((n_samples, total_var), dtype=float)

        
        n_extra = min(len(self.extra_injected), n_samples)
        for i in range(n_extra):
            injected = np.asarray(
                self.extra_injected[i],
                dtype=float,
            ).reshape(-1)
            X[i, :min(total_var, injected.size)] = injected[:total_var]
            if total_var > n_var and injected.size <= n_var:
                X[i, -1] = response_probability(X[i, :n_var], self.defense_items)

        injected_count = min(3, n_samples - n_extra)
        if injected_count >= 1:
            X[n_extra, :n_var] = self._generate_greedy_solution(
                n_var
            ).astype(float)
            if total_var > n_var:
                X[n_extra, -1] = response_probability(X[n_extra, :n_var], self.defense_items)
            self.injected_solutions.append(X[n_extra].copy())
        if injected_count >= 2:
            X[n_extra + 1, :n_var] = self._generate_balanced_solution(
                n_var
            ).astype(float)
            if total_var > n_var:
                X[n_extra + 1, -1] = response_probability(X[n_extra + 1, :n_var], self.defense_items)
            self.injected_solutions.append(X[n_extra + 1].copy())
        if injected_count >= 3:
            X[n_extra + 2, :n_var] = self._generate_minimal_cost_solution(
                n_var
            ).astype(float)
            if total_var > n_var:
                X[n_extra + 2, -1] = response_probability(X[n_extra + 2, :n_var], self.defense_items)
            self.injected_solutions.append(X[n_extra + 2].copy())

        for i in range(n_extra + injected_count, n_samples):
            X[i, :n_var] = self._generate_random_feasible_solution(
                n_var
            ).astype(float)
            if total_var > n_var:
                X[i, -1] = (
                    float(np.random.random())
                    if response_probability(X[i, :n_var], self.defense_items) > 0.0
                    else 0.0
                )

        
        for i in range(n_samples):
            _repair_response_probability(X[i], self.defense_items)

        print(f"[INFO] Injected {injected_count} initial solutions")
        return X

    def _generate_greedy_solution(self, n_var):
        
        other = [
            index
            for index, (label, _) in enumerate(self.defense_items[:n_var])
            if label not in ("RP1", "RP2", "RP3")
        ]
        responses = [
            index
            for index, (label, _) in enumerate(self.defense_items[:n_var])
            if label in ("RP1", "RP2", "RP3")
        ]
        other.sort(key=lambda index: (-self._value_per_cost[index], index))
        responses.sort(
            key=lambda index: (-int(self.defense_items[index][0][-1]), index)
        )

        solution = np.zeros(n_var, dtype=bool)
        total_cost = 0
        rp_selected = False

        for idx in other + responses:
            label, cost = self.defense_items[idx]
            is_rp = label in ("RP1", "RP2", "RP3")
            if is_rp and rp_selected:
                continue

            if total_cost + cost <= self.budget:
                solution[idx] = True
                total_cost += cost
                if is_rp:
                    rp_selected = True

        print(f"  Greedy solution cost: {total_cost}")
        return solution

    def _generate_balanced_solution(self, n_var):
        
        solution = np.zeros(n_var, dtype=bool)
        total_cost = 0

        prevention = []
        detection = []
        response = []

        for idx in range(n_var):
            cost = self.defense_items[idx][1]
            if self.categories[idx] == "Prevention":
                prevention.append((idx, cost))
            elif self.categories[idx] == "Detection":
                detection.append((idx, cost))
            elif self.categories[idx] == "Response":
                response.append((idx, cost))

        prevention.sort(key=lambda item: (-self._value_per_cost[item[0]], item[0]))
        detection.sort(key=lambda item: (-self._value_per_cost[item[0]], item[0]))
        response.sort(
            key=lambda item: (-int(self.defense_items[item[0]][0][-1]), item[0])
        )

        categories = [prevention, detection, response]
        cat_idx = 0
        cat_pos = [0, 0, 0]
        rp_selected = False

        while total_cost < self.budget:
            added = False
            for _ in range(3):
                if cat_idx == 2 and rp_selected:
                    cat_idx = (cat_idx + 1) % 3
                    continue

                while cat_pos[cat_idx] < len(categories[cat_idx]):
                    idx, cost = categories[cat_idx][cat_pos[cat_idx]]
                    cat_pos[cat_idx] += 1
                    if total_cost + cost <= self.budget:
                        solution[idx] = True
                        total_cost += cost
                        if cat_idx == 2:
                            rp_selected = True
                        added = True
                        break
                cat_idx = (cat_idx + 1) % 3
                if added:
                    break

            if not added:
                break

        print(f"  Balanced solution cost: {total_cost}")
        return solution

    def _generate_minimal_cost_solution(self, n_var):
        
        solution = np.zeros(n_var, dtype=bool)
        total_cost = 0

        best_prevention = None
        best_detection = None
        best_response = None

        for idx in range(n_var):
            if self.categories[idx] == "Prevention":
                if (
                    best_prevention is None
                    or self._value_per_cost[idx] > self._value_per_cost[best_prevention]
                ):
                    best_prevention = idx
            elif self.categories[idx] == "Detection":
                if (
                    best_detection is None
                    or self._value_per_cost[idx] > self._value_per_cost[best_detection]
                ):
                    best_detection = idx
            elif self.categories[idx] == "Response":
                if (
                    best_response is None
                    or int(self.defense_items[idx][0][-1])
                    > int(self.defense_items[best_response][0][-1])
                ):
                    best_response = idx

        rp_selected = False
        for idx in (best_prevention, best_detection, best_response):
            if idx is None:
                continue
            label, cost = self.defense_items[idx]
            is_rp = label in ("RP1", "RP2", "RP3")
            if is_rp and rp_selected:
                continue
            if total_cost + cost <= self.budget:
                solution[idx] = True
                total_cost += cost
                if is_rp:
                    rp_selected = True

        print(f"  Minimal solution cost: {total_cost}")
        return solution

    def _generate_random_feasible_solution(self, n_var):
        
        solution = np.zeros(n_var, dtype=bool)
        total_cost = 0
        indices = list(range(n_var))
        np.random.shuffle(indices)
        rp_selected = False

        for idx in indices:
            label, cost = self.defense_items[idx]
            is_rp = label in ("RP1", "RP2", "RP3")
            if is_rp and rp_selected:
                continue

            if total_cost + cost <= self.budget:
                if np.random.random() < 0.5:
                    solution[idx] = True
                    total_cost += cost
                    if is_rp:
                        rp_selected = True

        return solution
