from random import Random
from time import time
import inspyred
import csv

display=True

prng = Random()
prng.seed(time())

problem = inspyred.benchmarks.Kursawe(3)
ea = inspyred.ec.emo.NSGA2(prng)

ea.variator = [inspyred.ec.variators.blend_crossover,
               inspyred.ec.variators.gaussian_mutation]

ea.terminator = inspyred.ec.terminators.generation_termination

final_pop = ea.evolve(generator=problem.generator,
                      evaluator=problem.evaluator,
                      pop_size=100,
                      maximize=problem.maximize,
                      bounder=problem.bounder,
                      max_generations=100,
                      crossover_rate = 1,
                      mutation_rate=0.1)

if display:
    final_arc = ea.archive
    #print('Best Solutions: \n')
    #for f in final_arc:
        #print(f)
    import matplotlib.pyplot as plt
    x = []
    y = []
    for f in final_arc:
        x.append(f.fitness[0])
        y.append(f.fitness[1])
    plt.scatter(x, y, color='b')
    plt.xlabel("Function 1")
    plt.ylabel("Function 2")
    #Plot and save in pdf format
    plt.savefig('{0} Example ({1}).pdf'.format(ea.__class__.__name__,
                                               problem.__class__.__name__),
                format='pdf')
    plt.show()
