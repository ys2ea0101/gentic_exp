import numpy as np
import inspyred
import multiprocessing
from random import Random
from time import time
from utility import create_tag_inputs
from tag_problems import TagMatchingProblemSingle, TagMatchingProblemMulti, tag_mutator


def run_single(Nppl, Nproj, D, prng=None, display=False, manipulate_proj=False):
    if prng is None:
        prng = Random()
        prng.seed(time())

    params = create_tag_inputs(Nppl, Nproj, D, manipulate_proj=manipulate_proj)
    problem = TagMatchingProblemSingle(Nppl, params["people_tag"], params["project_tag"][0])
    ea = inspyred.ec.EvolutionaryComputation(prng)
    ea.selector = inspyred.ec.selectors.tournament_selection
    ea.variator = [inspyred.ec.variators.uniform_crossover,
                   inspyred.ec.variators.bit_flip_mutation]
    ea.replacer = inspyred.ec.replacers.generational_replacement
    ea.terminator = inspyred.ec.terminators.generation_termination
    ea.observer = inspyred.ec.observers.best_observer
    final_pop = ea.evolve(generator=problem.generator,
                          evaluator=problem.evaluator,
                          bounder=problem.bounder,
                          maximize=False,
                          pop_size=100,
                          max_generations=50,
                          tournament_size=5,
                          num_selected=100,
                          num_elites=1,
                          )

    if display:
        best = max(ea.population)
        print('Best Solution: {0}: {1}'.format(str(best.candidate), best.fitness))
    return ea


def run_multi(Nppl, Nproj, D, prng=None, display=False, seed=1234, manipulate_proj=False):
    if prng is None:
        prng = Random()
        prng.seed(time())

    params = create_tag_inputs(Nppl, Nproj, D, seed=seed, manipulate_proj=manipulate_proj)
    problem = TagMatchingProblemMulti(Nppl, Nproj, params["people_tag"], params["project_tag"])
    pre_filter = problem.pre_filter_result
    ea = inspyred.ec.EvolutionaryComputation(prng)
    ea.selector = inspyred.ec.selectors.tournament_selection
    ea.variator = [inspyred.ec.variators.uniform_crossover,
                   tag_mutator]
    ea.replacer = inspyred.ec.replacers.generational_replacement
    ea.terminator = inspyred.ec.terminators.generation_termination
    ea.observer = inspyred.ec.observers.best_observer
    final_pop = ea.evolve(generator=problem.generator,
                          evaluator=problem.evaluator,
                          bounder=problem.bounder,
                          maximize=False,
                          pop_size=4000,
                          max_generations=100,
                          tournament_size=100,
                          num_selected=4000,
                          num_elites=2,
                          pre_filter=pre_filter,
                          )

    if display:
        best = max(ea.population)
        print('Best Solution: {0}: {1}'.format(str(best.candidate), best.fitness))
    print("Solution: {}".format(params["solution"]))
    return ea


def run_multi_island(problem, island_number, mp_migrator, prng=None):
    if prng is None:
        prng = Random()
        prng.seed(time())

    pre_filter = problem.pre_filter_result
    ea = inspyred.ec.EvolutionaryComputation(prng)
    ea.selector = inspyred.ec.selectors.tournament_selection
    ea.variator = [inspyred.ec.variators.uniform_crossover,
                   tag_mutator]
    ea.replacer = inspyred.ec.replacers.generational_replacement
    ea.terminator = inspyred.ec.terminators.generation_termination
    ea.observer = inspyred.ec.observers.best_observer
    ea.migrator = mp_migrator
    final_pop = ea.evolve(generator=problem.generator,
                          evaluator=problem.evaluator,
                          bounder=problem.bounder,
                          maximize=False,
                          pop_size=4000,
                          max_generations=100,
                          tournament_size=40,
                          num_selected=4000,
                          num_elites=1,
                          pre_filter=pre_filter,
                          statistics_file=open(f"data_archive/stats_{island_number}.csv", "w"),
                          individuals_file=open(f"data_archive/inds_{island_number}.csv", "w"),
                          max_evaluations=100,
                          )


def run_island(Nppl, Nproj, D, prng=None, seed=1234, manipulate_proj=False):
    cpus = 4
    mp_migrator = inspyred.ec.migrators.MultiprocessingMigrator(50)
    params = create_tag_inputs(Nppl, Nproj, D, seed=seed, manipulate_proj=manipulate_proj)
    problem = TagMatchingProblemMulti(Nppl, Nproj, params["people_tag"], params["project_tag"])
    jobs = []
    for i in range(cpus):
        p = multiprocessing.Process(target=run_multi_island, args=(problem, i, mp_migrator, prng))
        p.start()
        jobs.append(p)
    for j in jobs:
        j.join()


if __name__ == "__main__":

    Nppl = 300
    Nproj = 40
    D = 20
    # run_single(300, 50, 10, display=True, manipulate_proj=False)
    # params = create_tag_inputs(Nppl, Nproj, D, manipulate_proj=True)
    # problem = TagMatchingProblemMulti(Nppl, Nproj, params["people_tag"], params["project_tag"])
    # g = problem.generator(np.random, {})
    # v = problem.evaluator([params["solution"]], {})
    # run_multi(Nppl, Nproj, D, display=True, manipulate_proj=True)
    run_island(Nppl, Nproj, D, None, manipulate_proj=True)
    pass
