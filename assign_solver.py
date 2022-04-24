import inspyred
import multiprocessing
import numpy as np
from random import Random
import cProfile
import pstats
from assign_problems import (
    AssignmentProblem,
    assign_mutator,
    assign_crossover,
    hungarian_solve,
    assign_matrix,
    mc_full_mutator,
    mc_mutator_revised,
    mc_mutator_dynamic,
    mc_mutator_fix_length,
    proj_assign_matrix,
    best_observer,
)
from cp_mip_solver import run_linear_assignment_solver
import time

def run_assign(n_ppl, n_proj, max_weight, beta, prng=None, display=False, seed=1234,
               mutator_choice="mc", cost_type="int", mode=1):
    if prng is None:
        prng = Random()
        prng.seed(time.time())

    if cost_type == "int":
        cost = assign_matrix(n_ppl, n_proj, max_weight, seed=seed)
    else:
        # todo:
        cost = - proj_assign_matrix(n_ppl, n_proj, 4, seed=seed, mode=mode)
    hungarian_solve(cost, False)
    run_linear_assignment_solver(cost, "cp")
    problem = AssignmentProblem(n_ppl, n_proj, cost, beta)
    ea = inspyred.ec.EvolutionaryComputation(prng)
    ea.selector = inspyred.ec.selectors.tournament_selection
    if mutator_choice == "mc":
        ea.variator = [assign_crossover, mc_mutator_fix_length]
    elif mutator_choice == "swap":
        ea.variator = [assign_crossover, assign_mutator]
    elif mutator_choice == "both":
        ea.variator = [assign_crossover, mc_full_mutator, assign_mutator]
    elif mutator_choice == "None":
        ea.variator = [assign_crossover]
    else:
        raise ValueError("Wrong mutator ")
    ea.replacer = inspyred.ec.replacers.generational_replacement
    ea.terminator = inspyred.ec.terminators.generation_termination
    ea.observer = best_observer
    final_pop = ea.evolve(
        generator=problem.generator,
        evaluator=problem.evaluator,
        bounder=problem.bounder,
        maximize=False,
        pop_size=1000,
        max_generations=20,
        tournament_size=5,
        num_selected=1000,
        num_elites=2,
        n_ppl=problem.dimensions,
        n_proj=problem.n_proj,
        proj_prob=problem.proj_prob,
        am=problem.assign_matrix_val,
        mutator=mutator_choice,
    )

    if display:
        best = max(ea.population)
        print("Best Solution: {0}: {1}".format(str(best.candidate), best.fitness))
    return ea


def assign_island(problem, island_number, mp_migrator, prng=None):
    if prng is None:
        prng = Random()
        prng.seed(time.time())

    ea = inspyred.ec.EvolutionaryComputation(prng)
    ea.selector = inspyred.ec.selectors.tournament_selection
    ea.variator = [assign_crossover, mc_full_mutator]
    ea.migrator = mp_migrator
    ea.replacer = inspyred.ec.replacers.generational_replacement
    ea.terminator = inspyred.ec.terminators.generation_termination
    ea.observer = inspyred.ec.observers.best_observer
    final_pop = ea.evolve(
        generator=problem.generator,
        evaluator=problem.evaluator,
        bounder=problem.bounder,
        maximize=False,
        pop_size=4000,
        max_generations=100,
        tournament_size=100,
        num_selected=4000,
        num_elites=2,
        statistics_file=open(f"data_archive/stats_{island_number}.csv", "w"),
        individuals_file=open(f"data_archive/inds_{island_number}.csv", "w"),
        n_ppl=problem.dimensions,
        n_proj=problem.n_proj,
        proj_prob=problem.proj_prob,
        am=problem.assign_matrix_val,
    )
    return ea


def run_island(n_ppl, n_proj, max_weight, beta, prng=None, seed=1234):
    cpus = 4
    mp_migrator = inspyred.ec.migrators.MultiprocessingMigrator(20)
    cost = assign_matrix(n_ppl, n_proj, max_weight, seed=seed)
    hungarian_solve(cost, False)
    problem = AssignmentProblem(n_ppl, n_proj, cost, beta)
    jobs = []
    for i in range(cpus):
        p = multiprocessing.Process(
            target=assign_island, args=(problem, i, mp_migrator, prng)
        )
        p.start()
        jobs.append(p)
    for j in jobs:
        j.join()


if __name__ == "__main__":
    n_ppl = 100
    n_proj = 100
    max_weight = 50
    beta = 0.5
    mutator = "swap"
    cost_type = "int"
    # mutator: mc, swap, or both, cost: int or whatever,
    t0 = time.perf_counter()
    with cProfile.Profile() as pr:
        run_assign(n_ppl, n_proj, max_weight, beta, None, False, mutator_choice=mutator, cost_type=cost_type, mode=1)
    t1 = time.perf_counter()
    print(F"Time used: {t1 - t0}")
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()

    # run_island(n_ppl, n_proj, max_weight, beta)



