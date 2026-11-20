

from __future__ import annotations

import json
from multiprocessing import current_process
import os
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.core.callback import Callback
from pymoo.core.population import Population
from pymoo.core.termination import Termination
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from common.config import NSGA2_MA_CROSSOVER_PROBABILITY, NSGA2_MA_MUTATION_PROBABILITY
from common.defense_config import DefenseRepair
from common.game_termination import termination_metadata
from common.memetic import MemeticNSGA2, get_local_search_statistics
from common.problem import CyberDefenseProblem
from common.sampling import FeasibleBinarySampling, ResponseAwareCrossover, ResponseAwareMutation
from common.simulation import PRUNE_TOL_DEFAULT
from common.termination import FrontWindowTermination, MyTerminationSelector
from common.optimization_fronts import select_front


class InitializationBudgetExceeded(RuntimeError):
    pass


class ComparisonProblem(CyberDefenseProblem):
    

    def __init__(self, *args, penalized=False, **kwargs):
        super().__init__(*args, use_constraint=True, use_cache=True, **kwargs)
        self.penalized = bool(penalized)
        self.evaluation_deliveries = 0
        if self.penalized:
            self.n_ieq_constr = 0

    def _evaluate(self, decision, out, *args, **kwargs):
        self.evaluation_deliveries += 1
        super()._evaluate(decision, out, *args, **kwargs)
        out["F_True"] = out["F"].copy()
        if self.penalized:
            violation = max(0.0, float(out["G"][0]))
            out["F"] = out["F"].copy()
            out["F"][1] += violation * 10000.0
            out["G"] = np.empty(0, dtype=float)
            out["CV"] = np.zeros(1, dtype=float)


class _ExecutionTiming:
    def _deadline_reached(self):
        return self._comparison_deadline is not None and self._comparison_clock() >= self._comparison_deadline

    def _initialize_infill(self):
        if self._deadline_reached():
            raise InitializationBudgetExceeded("wall-clock budget exhausted before initialization")
        return super()._initialize_infill()

    def _post_advance(self):
        result = super()._post_advance()
        trace = self.callback
        ended = self._comparison_clock()
        if trace.data["rows"]:
            tail = ended - trace.last_completed
            row = trace.data["rows"][-1]
            if row["Generation"] == 1:
                trace.initialization_seconds += tail
            else:
                row["Generation_Seconds"] += tail
            trace.last_completed = ended
        return result

    def next(self):
        if self.is_initialized and self._deadline_reached():
            generation = self.n_gen
            self.n_gen = self.callback.data["rows"][-1]["Generation"]
            try:
                self.termination.update(self)
            finally:
                self.n_gen = generation
            if self.termination.has_terminated():
                self.finalize()
                return
        return super().next()

    def finalize(self):
        started = self._comparison_clock()
        self._comparison_finalizing = True
        self._comparison_main_fe = int(self.evaluator.n_eval)
        self._comparison_main_front = select_front(
            self.pop, self.problem.budget, len(self.problem.defense_items),
        )
        try:
            return super().finalize()
        finally:
            self._comparison_finalize_seconds = self._comparison_clock() - started


class ComparisonMemeticNSGA2(_ExecutionTiming, MemeticNSGA2):
    @property
    def early_termination_ready(self):
        return True if self._comparison_equal_rules else super().early_termination_ready

    def _local_fe_available(self):
        if self._deadline_reached() or (
            self._comparison_protocol == "equal_wall_clock" and self._comparison_finalizing
        ):
            return False
        return super()._local_fe_available()

    def _should_local_search(self):
        return not self._deadline_reached() and super()._should_local_search()

    def _run_local_stage(self, merged, stage_cap=None):
        before = int(self.evaluator.n_eval)
        try:
            return super()._run_local_stage(merged, stage_cap=stage_cap)
        finally:
            self._comparison_local_deltas.append(int(self.evaluator.n_eval) - before)

    def _update_archive(self, merged):
        before = int(self.evaluator.n_eval)
        super()._update_archive(merged)
        if self.use_archive:
            self._comparison_archive_updates += 1
        if int(self.evaluator.n_eval) != before:
            raise RuntimeError("archive update unexpectedly evaluated a decision")

    def _fill_final_genetic_fe(self, merged):
        if self._comparison_protocol != "equal_wall_clock":
            return super()._fill_final_genetic_fe(merged)
        while self.evaluator.n_eval < self.max_fe and not self._deadline_reached():
            remaining = self.max_fe - int(self.evaluator.n_eval)
            offspring = self.mating.do(
                self.problem, self.pop, min(self.n_offsprings, remaining),
                algorithm=self, **self._operator_kwargs(),
            )
            if offspring is None or not len(offspring):
                raise RuntimeError("Unable to generate actual genetic offspring")
            offspring = offspring[:remaining]
            offspring.set("n_gen", self.n_gen)
            offspring.set("n_iter", self.n_gen)
            before = int(self.evaluator.n_eval)
            self.evaluator.eval(self.problem, offspring, algorithm=self, skip_already_evaluated=False)
            self.n_final_refill_genetic_evals += int(self.evaluator.n_eval) - before
            merged = Population.merge(merged, offspring)
        return merged


class BudgetedSPEA2(_ExecutionTiming, SPEA2):
    def __init__(self, *, max_fe=None, **kwargs):
        super().__init__(**kwargs)
        self.max_fe = max_fe

    def _infill(self):
        if self.max_fe is None:
            return super()._infill()
        remaining = self.max_fe - int(self.evaluator.n_eval)
        if remaining <= 0:
            raise RuntimeError("FE termination failed to stop before an empty generation")
        original = self.n_offsprings
        try:
            self.n_offsprings = min(original, remaining)
            return super()._infill()
        finally:
            self.n_offsprings = original


class BudgetedMOEAD(_ExecutionTiming, MOEAD):
    def __init__(self, *, max_fe=None, **kwargs):
        super().__init__(**kwargs)
        self.max_fe = max_fe

    def next(self):
        if not self.is_initialized:
            return super().next()
        if self._deadline_reached():
            return self._finish_deadline_iteration()
        infills = self.infill()
        if self._deadline_reached():
            return self._finish_deadline_iteration()
        if infills is not None:
            self.evaluator.eval(self.problem, infills, algorithm=self)
            self.advance(infills=infills)
        else:
            self.advance()

    def _finish_deadline_iteration(self):
        
        if self.generator is not None:
            self.generator.close()
            self.generator = None
            self.state = None
        completed = self.callback.data["rows"][-1]["Total_FE"]
        if self.evaluator.n_eval > completed:
            self._post_advance()
            if self.termination.has_terminated():
                return self.finalize()
        else:
            self._comparison_interrupted_seconds += self._comparison_clock() - self.callback.last_completed
            return super().next()

    def _next(self):
        population = self.pop
        for position in np.random.permutation(len(population)):
            if (self.max_fe is not None and self.evaluator.n_eval >= self.max_fe) or self._deadline_reached():
                break
            parents = self.selection.do(
                self.problem, population, 1, self.mating.crossover.n_parents,
                neighbors=[self.neighbors[position]],
            )
            offspring = np.random.choice(self.mating.do(
                self.problem, population, 1, parents=parents, n_max_iterations=1,
            ))
            evaluated = yield offspring
            self.ideal = np.min(np.vstack([self.ideal, evaluated.F]), axis=0)
            self._replace(position, evaluated)


def make_algorithm(task, items, all_data, clock, started):
    options = {
        "sampling": FeasibleBinarySampling(items, task["budget"], all_data=all_data),
        "crossover": ResponseAwareCrossover(prob=NSGA2_MA_CROSSOVER_PROBABILITY),
        "mutation": ResponseAwareMutation(prob=NSGA2_MA_MUTATION_PROBABILITY),
        "repair": DefenseRepair(items),
    }
    name = task["name"]
    if name == "MOEA/D":
        directions = get_reference_directions("das-dennis", 2, n_partitions=task["pop_size"] - 1)
        if len(directions) != task["pop_size"]:
            raise RuntimeError("MOEA/D reference directions changed population size")
        algorithm = BudgetedMOEAD(
            max_fe=task["max_fe"], ref_dirs=directions,
            n_neighbors=min(15, task["pop_size"]), prob_neighbor_mating=0.9, **options,
        )
    elif name == "SPEA2":
        algorithm = BudgetedSPEA2(
            max_fe=task["max_fe"], pop_size=task["pop_size"], eliminate_duplicates=False, **options,
        )
    elif name in ("NSGA2", "NSGA2_MA"):
        parameters = task["ma_parameters"] if name == "NSGA2_MA" else {"local_search_prob": 0.0}
        algorithm = ComparisonMemeticNSGA2(
            max_fe=task["max_fe"], local_fe_reference=task["local_fe_reference"],
            pop_size=task["pop_size"], eliminate_duplicates=False, **parameters, **options,
        )
    else:
        raise ValueError(f"Unknown algorithm: {name}")
    algorithm._comparison_clock = clock
    algorithm._comparison_deadline = (
        started + task["target_seconds"] if task["protocol"] == "equal_wall_clock" else None
    )
    algorithm._comparison_protocol = task["protocol"]
    algorithm._comparison_equal_rules = not task.get("legacy", False)
    algorithm._comparison_finalizing = False
    algorithm._comparison_finalize_seconds = 0.0
    algorithm._comparison_interrupted_seconds = 0.0
    algorithm._comparison_local_deltas = []
    algorithm._comparison_archive_updates = 0
    return algorithm


class _StrictFrontView(SimpleNamespace):
    @property
    def pop(self):
        snapshot = select_front(self.algorithm.pop, self.budget, len(self.algorithm.problem.defense_items))
        return Population.create(*snapshot.individuals)


class ComparisonFrontTermination(FrontWindowTermination):
    def _update(self, algorithm):
        proxy = _StrictFrontView(
            algorithm=algorithm, budget=self.budget, n_gen=algorithm.n_gen,
            evaluator=algorithm.evaluator, early_termination_ready=True,
        )
        return super()._update(proxy)


class DeadlineTermination(Termination):
    def __init__(self, *, deadline, clock, max_fe, max_gen):
        super().__init__()
        self.deadline, self.clock = deadline, clock
        self.max_fe, self.max_gen = max_fe, max_gen
        self.stop_reason = None
        self.history = []

    def _update(self, algorithm):
        count = int(algorithm.evaluator.n_eval)
        generation = int(algorithm.n_gen)
        self.stop_reason = (
            "max_fe" if self.max_fe is not None and count >= self.max_fe
            else "max_generations" if generation >= self.max_gen
            else "wall_clock_deadline" if self.clock() >= self.deadline
            else None
        )
        record = {"generation": generation, "total_fe": count, "stop_reason": self.stop_reason}
        if self.history and self.history[-1]["generation"] == generation:
            self.history[-1] = record
        else:
            self.history.append(record)
        return float(self.stop_reason is not None)


class ComparisonTrace(Callback):
    def __init__(self, budget, defense_count, clock=perf_counter, started=None):
        super().__init__()
        self.budget, self.defense_count, self.clock = budget, defense_count, clock
        self.last_completed = clock() if started is None else started
        self.initialization_seconds = 0.0
        self.initialized_population_size = 0
        self.data["rows"] = []

    def notify(self, algorithm):
        generation = int(algorithm.n_gen)
        rows = self.data["rows"]
        replacing = bool(rows and rows[-1]["Generation"] == generation)
        previous = rows[-2] if replacing and len(rows) > 1 else rows[-1] if rows and not replacing else None
        total = int(algorithm.evaluator.n_eval)
        local = get_local_search_statistics(algorithm)["Local_FE"]
        strict = select_front(algorithm.pop, self.budget, self.defense_count)
        display = select_front(algorithm.pop, self.budget, self.defense_count, "display_compatible")
        genetic_delta = total - local - (previous["Genetic_FE"] if previous else 0)
        row = {
            "Generation": generation, "Total_FE": total, "Genetic_FE": total - local, "Local_FE": local,
            "Population_Size": len(algorithm.pop),
            "FE_Delta": total - (previous["Total_FE"] if previous else 0),
            "Genetic_FE_Delta": genetic_delta,
            "Local_FE_Delta": local - (previous["Local_FE"] if previous else 0),
            "Pareto_Size": len(strict), "Feasible_Pareto_Size": len(strict),
            "Resource_Configuration_Count": strict.resource_configuration_count,
            "Display_Compatible_Pareto_Size": len(display), "Front_Kind": "strict",
            "Minimum_Cost": float(strict.F[:, 0].min()) if len(strict) else np.nan,
            "Minimum_Loss": float(strict.F[:, 1].min()) if len(strict) else np.nan,
            "Front": strict.F.copy(), "Decisions": strict.decisions,
            "Display_Front": display.F.copy(), "Display_Decisions": display.decisions,
            "Generation_Kind": "initialization" if generation == 1 else "partial" if genetic_delta < algorithm.pop_size else "complete",
            "Is_Partial_Generation": generation > 1 and genetic_delta < algorithm.pop_size,
        }
        ended = self.clock()
        if replacing:
            row["Generation_Seconds"] = rows[-1]["Generation_Seconds"]
            row["Main_Loop_FE_Delta"] = rows[-1].get("Main_Loop_FE_Delta", rows[-1]["FE_Delta"])
            rows[-1] = row
        else:
            duration = ended - self.last_completed
            if generation == 1:
                self.initialization_seconds = duration
                self.initialized_population_size = len(algorithm.pop)
            row["Generation_Seconds"] = np.nan if generation == 1 else duration
            row["Main_Loop_FE_Delta"] = row["FE_Delta"]
            rows.append(row)
            self.last_completed = ended


def _termination(task, algorithm, clock):
    if task["protocol"] == "equal_fe":
        return ("n_eval", task["max_fe"])
    if task.get("legacy"):
        return MyTerminationSelector.get_termination(
            strategy=task["strategy"], max_gen=task["max_gen"], tol=task["tol"],
            n_stable=task["n_stable"], period=task["period"], max_fe=task["max_fe"],
        )
    if task["protocol"] == "equal_wall_clock":
        return DeadlineTermination(
            deadline=algorithm._comparison_deadline, clock=clock,
            max_fe=task["max_fe"], max_gen=task["max_gen"],
        )
    return ComparisonFrontTermination(
        max_gen=task["max_gen"], max_fe=task["max_fe"], budget=task["budget"],
        loss_scale=task["loss_scale"], tol=task["tol"], n_stable=task["n_stable"],
        min_gen=task["min_gen"], min_front_size=task["min_front_size"], ref_point=task["ref_point"],
    )


def run_task(task, context, minimize_fn=minimize, *, clock=perf_counter):
    
    task = {"target_seconds": None, "loss_scale": None, "min_gen": 20,
            "min_front_size": 2, "ref_point": (1.1, 1.1), **task}
    items, all_data, graph = context
    np.random.seed(task["seed"])
    game_options = {"game_termination": task["game_termination"]} if "game_termination" in task else {}
    problem = ComparisonProblem(
        items, all_data, graph, budget=task["budget"], penalized=task["name"] == "MOEA/D",
        game_iterations=task["game_iterations"], game_learning_rate=task["game_learning_rate"],
        prune_tol=task.get("prune_tol", PRUNE_TOL_DEFAULT),
        **game_options,
    )
    started = clock()
    algorithm = make_algorithm(task, items, all_data, clock, started)
    monitor = _termination(task, algorithm, clock)
    trace = ComparisonTrace(task["budget"], len(items), clock, started)
    exhausted_setup = False
    try:
        result = minimize_fn(
            problem, algorithm, termination=monitor, seed=task["seed"], callback=trace,
            copy_algorithm=False, copy_termination=False, save_history=False, verbose=False,
        )
        algorithm = result.algorithm
    except InitializationBudgetExceeded:
        exhausted_setup = True
    elapsed = clock() - started
    actual = int(algorithm.evaluator.n_eval)
    if actual != problem.total_evals or actual != problem.evaluation_deliveries:
        raise RuntimeError("Independent evaluation submissions, problem.total_evals and evaluator FE disagree")
    if task["max_fe"] is not None and actual > task["max_fe"]:
        raise RuntimeError("The global FE safety cap was exceeded")
    if task["protocol"] == "equal_fe" and actual != task["max_fe"]:
        raise RuntimeError("equal_fe did not consume exactly the target budget")
    statistics = get_local_search_statistics(algorithm)
    front = select_front(algorithm.pop, task["budget"], len(items))
    display = select_front(algorithm.pop, task["budget"], len(items), "display_compatible")
    main_front = getattr(algorithm, "_comparison_main_front", front)
    main_fe = getattr(algorithm, "_comparison_main_fe", actual)
    changed = not np.array_equal(main_front.F, front.F)
    stop_reason = "max_fe" if task["protocol"] == "equal_fe" else getattr(algorithm.termination, "stop_reason", None)
    if exhausted_setup:
        stop_reason = "initialization_budget_insufficient"
    initialization = trace.initialization_seconds if trace.data["rows"] else elapsed
    status = "budget_insufficient" if task["protocol"] == "equal_wall_clock" and (
        exhausted_setup or initialization > task["target_seconds"]
    ) else "ok" if len(front) else "no_feasible_front"
    protocol_completed = (
        actual == task["max_fe"] if task["protocol"] == "equal_fe"
        else stop_reason == "wall_clock_deadline" and status != "budget_insufficient"
        if task["protocol"] == "equal_wall_clock"
        else stop_reason in ("max_fe", "max_generations", "front_stable", "stability")
    )
    parameters = task["ma_parameters"] if task["name"] == "NSGA2_MA" else {}
    return {
        "Algorithm": task["name"], "Run": task["run"], "Seed": task["seed"],
        "Protocol": task["protocol"], "Termination_Strategy": task["strategy"],
        "Use_Pruning": task.get("prune_tol", PRUNE_TOL_DEFAULT) is not None,
        "Prune_Tolerance": task.get("prune_tol", PRUNE_TOL_DEFAULT),
        "Game_Termination_JSON": json.dumps(termination_metadata(problem.game_termination), sort_keys=True),
        "Status": status, "Protocol_Completed": protocol_completed,
        "Comparison_Valid": status == "ok" and protocol_completed,
        "Stop_Reason": stop_reason, "Main_Loop_Stop_Reason": stop_reason,
        "Main_Loop_Total_FE": main_fe, "Postprocessing_FE": actual - main_fe,
        "Postprocessing_Front_Changed": changed,
        "Final_Front_Window_Verified": stop_reason == "front_stable" and not changed,
        "Main_Loop_Front": main_front.F.copy(),
        "Main_Loop_Decisions": main_front.decisions,
        "Total_FE": actual, "Genetic_FE": actual - statistics["Local_FE"], **statistics,
        "Problem_Total_Evals": int(problem.total_evals),
        "Observed_Evaluation_Deliveries": int(problem.evaluation_deliveries),
        "Cache_Hits": int(problem.cache_hits), "Unique_Simulations": len(problem.eval_cache),
        "Actual_Population_Size": trace.initialized_population_size,
        "Initialized_Population_Size": trace.initialized_population_size,
        "Final_Population_Size": len(algorithm.pop) if algorithm.pop is not None else 0,
        "Population_Size": task["pop_size"], "Generations": len(trace.data["rows"]),
        "Runtime_Seconds": float(elapsed), "Run_Wall_Seconds": float(elapsed),
        "Actual_Wall_Seconds": float(elapsed), "Target_Wall_Seconds": task["target_seconds"],
        "Deadline_Overrun_Seconds": max(0.0, elapsed - task["target_seconds"]) if task["target_seconds"] is not None else 0.0,
        "Wall_Budget_Completed": task["protocol"] == "equal_wall_clock" and protocol_completed,
        "Initialization_Seconds": float(initialization),
        "Finalize_Seconds": float(algorithm._comparison_finalize_seconds),
        "Interrupted_Iteration_Seconds": float(algorithm._comparison_interrupted_seconds),
        "Stop_Check_Granularity": (
            "initialization_batch;moead_child" if task["name"] == "MOEA/D"
            else "initialization_batch;genetic_batch;local_candidate" if task["name"] == "NSGA2_MA"
            else "initialization_batch;genetic_batch"
        ),
        "Local_Stage_FE_Deltas": algorithm._comparison_local_deltas,
        "Archive_Update_Calls": algorithm._comparison_archive_updates,
        "MA_Parameters_JSON": json.dumps(parameters, sort_keys=True) if parameters else "",
        "Algorithm_Parameters_JSON": json.dumps({
            "population_size": task["pop_size"], "eliminate_duplicates": False,
            "crossover": "ResponseAwareCrossover", "crossover_probability": NSGA2_MA_CROSSOVER_PROBABILITY,
            "mutation": "ResponseAwareMutation", "mutation_probability": NSGA2_MA_MUTATION_PROBABILITY,
            "repair": "DefenseRepair", "sampling": "FeasibleBinarySampling",
            **({"reference_directions": task["pop_size"], "n_neighbors": min(15, task["pop_size"]), "prob_neighbor_mating": 0.9} if task["name"] == "MOEA/D" else {}),
        }, sort_keys=True),
        "Max_Local_FE_Ratio": parameters.get("max_local_fe_ratio"),
        "Local_Search_Trigger": parameters.get("local_search_trigger", ""),
        "Acceptance_Mode": parameters.get("acceptance_mode", ""),
        "Local_FE_Reference_Source": task.get(
            "local_fe_reference_source",
            "max_fe" if task["max_fe"] is not None else "pop_size_times_workbook_max_gen",
        ),
        "Local_FE_Reference": task["local_fe_reference"] if task["name"] == "NSGA2_MA" else None,
        "Termination_Parameters": {key: task[key] for key in (
            "max_gen", "max_fe", "budget", "loss_scale", "tol", "n_stable", "min_gen", "min_front_size", "ref_point",
        )},
        "Termination_History": list(getattr(algorithm.termination, "history", [])),
        "History": trace.data["rows"], "Front": front.F.copy(), "Decisions": front.decisions,
        "Display_Front": display.F.copy(), "Display_Decisions": display.decisions,
        "Resource_Configuration_Count": front.resource_configuration_count,
        "Display_Compatible_Pareto_Size": len(display), "Front_Kind": "strict",
        "Worker_PID": os.getpid(), "Worker_Process_Name": current_process().name,
    }
