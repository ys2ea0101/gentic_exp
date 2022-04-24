import numpy as np
import copy
import os
from os.path import isfile
from scipy.optimize import linear_sum_assignment
from inspyred.benchmarks import Benchmark
from inspyred import ec
from inspyred.ec.variators.mutators import mutator
from inspyred.ec.variators.crossovers import crossover
from numba import njit

def assign_matrix(n_ppl, n_proj, max_weight, seed=4321):
    """

    :param n_ppl: ideally not smaller than n_proj
    :param n_proj:
    :param max_weight:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    return np.random.randint(max_weight, size=(n_ppl, n_proj))


def proj_assign_matrix(n_ppl, n_proj, max_weight=4, seed=4321, mode=1):
    np.random.seed(seed)
    if mode == 0:
        m1 = np.random.uniform(0, max_weight, size=(n_ppl, 4))
        m2 = np.random.uniform(0, max_weight, size=(4, n_proj))

        return np.dot(m1, m2)
    elif mode == 1:
        cut = False
        m1 = np.random.randint(0, 9, size=(n_ppl, 4))
        m2 = np.random.randint(0, 9, size=(4, n_proj))

        ret = np.dot(m1, m2)

        if cut:
            max0 = np.max(ret, axis=0)
            max1 = np.max(ret, axis=1)

            cut_rate = 0.8
            cut_l0 = int(cut_rate * len(max0))
            cut_l1 = int(cut_rate * len(max1))
            cut0 = sorted(max0, reverse=True)[-cut_l0]
            cut1 = sorted(max1, reverse=True)[-cut_l1]

            id0 = np.argwhere(max0 > cut0)
            id1 = np.argwhere(max1 > cut1)

            ret = ret[id0, id1.T]
        return ret

    elif mode == 2:
        m1 = np.random.randint(0, 2, size=(n_ppl, 40)) * 2 - 1
        m2 = np.random.randint(0, 2, size=(40, n_proj)) * 2 - 1

        return np.dot(m1, m2)
    else:
        raise ValueError("Mode not known")



def hungarian_solve(rev_matrix, maximize):
    row_ind, col_ind = linear_sum_assignment(rev_matrix, maximize=maximize)
    print(f"row ids: {row_ind}")
    print(f"col ids: {col_ind}")
    cost = np.sum(rev_matrix[row_ind, col_ind])
    print(f"Best bound: {cost}")


class AssignmentProblem(Benchmark):
    def __init__(self, dimensions, n_proj, assign_matrix_val, beta):
        """
        data structure:
            dim: (n_ppl, )
            meaning: for the first n_proj elements, data[i]
                represents that project i is assigned to data[i]
                the rest in the data array is not assigned to a project

        :param dimensions: same as n_ppl
        :param n_proj:
        :param assign_matrix_val: (n_ppl, n_proj)
        :param beta:
        """
        Benchmark.__init__(self, assign_matrix_val.shape[0])
        self.n_proj = assign_matrix_val.shape[1]
        self.assign_matrix_val = assign_matrix_val
        self.bounder = ec.DiscreteBounder([i for i in range(self.n_proj)])
        self.beta = beta
        self.proj_prob = self.prepare_proj_prob()

    def prepare_proj_prob(self):
        minproj = np.min(self.assign_matrix_val, axis=0, keepdims=True)
        # todo:
        maxproj = np.max(self.assign_matrix_val, axis=0, keepdims=True)
        beta_1 = 10 / (maxproj - minproj + 0.001)
        prob_1 = np.exp((minproj - self.assign_matrix_val) * beta_1)

        minppl = np.min(self.assign_matrix_val, axis=1, keepdims=True)
        maxppl = np.max(self.assign_matrix_val, axis=1, keepdims=True)
        beta_2 = 10 / (maxppl - minppl + 0.001)
        prob_2 = np.exp((minppl - self.assign_matrix_val) * beta_2)

        prob = prob_1 + prob_2 * 0.5
        norm = np.sum(prob, axis=0)
        prob = prob * (1.0 / norm)
        return prob

    def generator(self, random, args):
        guess = np.arange(self.dimensions)
        np.random.shuffle(guess)
        return guess

    def evaluator(self, candidates, args):
        return self._evaluator(np.array(candidates), self.assign_matrix_val, self.n_proj)

    @staticmethod
    def _evaluator(candidates, assign_matrix_val, n_proj):
        fitness = []
        for candi in candidates:
            fitness.append(
                np.sum(
                    assign_matrix_val[
                        candi[: n_proj], list(range(n_proj))
                    ]
                )
            )

        return fitness


@mutator
def assign_mutator(random, candidate, args):
    loop_rate = args.setdefault("loop_rate", 0.2)
    worm_rate = args.setdefault("worm_rate", 0.2)
    rdm = random.random()
    l_worm = 2
    l_worm = min(l_worm, args["n_proj"])
    if rdm < loop_rate:
        idx = random.sample(range(args["n_proj"]), l_worm)

        return loop_swap(candidate, idx)
    elif rdm < loop_rate + worm_rate:
        idx = random.sample(range(args["n_proj"]), l_worm)
        if args["n_ppl"] > args["n_proj"]:
            ends = random.sample(range(args["n_proj"], args["n_ppl"]), 1)
            idx = idx + ends

        return loop_swap(candidate, idx)
    else:
        return candidate


def loop_swap(array, idx_pool):
    ret = array.copy()
    s = array[idx_pool[0]]
    for i in range(len(idx_pool) - 1):
        ret[idx_pool[i]] = ret[idx_pool[i + 1]]
    ret[idx_pool[-1]] = s
    return ret


@mutator
def mc_full_mutator(random, candidate, args):
    """
    :param random:
    :param candidate:
    :param args:
    :return:
    """
    rate = args.setdefault("mc_mutate_rate", 1)
    if random.random() > rate:
        return candidate
    start_proj_id = np.random.randint(0, args["n_proj"])
    no_proj_ppl = candidate[args["n_proj"] :]
    start_ppl_id = candidate[start_proj_id]
    path = [start_ppl_id]
    proj_path = [start_proj_id]
    current_proj = start_proj_id

    while True:
        next_ppl = np.random.choice(
            np.arange(args["n_ppl"]), p=args["proj_prob"][:, current_proj]
        )
        if next_ppl in path:
            idx = path.index(next_ppl)
            cut_path = proj_path[idx:]
            break
        elif next_ppl in no_proj_ppl:
            proj_path.append(candidate.index(next_ppl))
            cut_path = proj_path
            break
        else:
            path.append(next_ppl)
            current_proj = candidate.index(next_ppl)
            proj_path.append(current_proj)

    eb = np.sum(args["am"][candidate[:args["n_proj"]], list(range(args["n_proj"]))])
    if len(cut_path) > 1:
        ret = loop_swap(candidate, cut_path)
        ea = np.sum(args["am"][ret[:args["n_proj"]], list(range(args["n_proj"]))])
        diff = ea - eb
        # print("change: ", ea - eb)
        # todo:
        if diff < 50:
            return ret
        else:
            return candidate
    else:
        return candidate


@mutator
def mc_mutator_revised(random, candidate, args):
    """
    Try to get better termination condition
    :param random:
    :param candidate:
    :param args:
    :return:
    """

    start_proj_id = np.random.randint(0, args["n_proj"])
    no_proj_ppl = candidate[args["n_proj"] :]
    start_ppl_id = candidate[start_proj_id]
    path = [start_ppl_id]
    proj_path = [start_proj_id]
    current_proj = start_proj_id
    pool = np.ones(shape=(args["n_ppl"], ), dtype=bool)
    delta_e = - args["am"][start_ppl_id, start_proj_id]
    ppl_id = np.arange(args["n_ppl"])

    while True:
        norm = np.sum(args["proj_prob"][pool, current_proj])
        next_ppl = np.random.choice(
            ppl_id[pool], p=args["proj_prob"][pool, current_proj] * (1/norm)
        )
        delta_e += (args["am"][next_ppl, current_proj])
        if next_ppl == start_ppl_id:
            delta_e += args["am"][next_ppl, current_proj] - args["am"][start_ppl_id, current_proj]
            break
        elif next_ppl in no_proj_ppl:
            proj_path.append(candidate.index(next_ppl))
            break
        else:
            path.append(next_ppl)
            current_proj = candidate.index(next_ppl)
            proj_path.append(current_proj)
            pool[next_ppl] = False
            delta_e -= args["am"][next_ppl, current_proj]

    if len(proj_path) > 1 and delta_e < 5:
        return loop_swap(candidate, proj_path)
    else:
        return candidate


@mutator
def mc_mutator_fix_length(random, candidate, args):
    """
    Try to get better termination condition
    :param random:
    :param candidate:
    :param args:
    :return:
    """

    # max_length = max(int(100 / (args["num_generations"] + 1)), 2)
    max_length = 2
    max_delta = 50

    start_proj_id = np.random.randint(0, args["n_proj"])
    no_proj_ppl = candidate[args["n_proj"] :]
    start_ppl_id = candidate[start_proj_id]
    path = [start_ppl_id]
    proj_path = [start_proj_id]
    current_proj = start_proj_id
    pool = np.ones(shape=(args["n_ppl"], ), dtype=bool)
    delta_e = - args["am"][start_ppl_id, start_proj_id]
    ppl_id = np.arange(args["n_ppl"])

    while True:
        norm = np.sum(args["proj_prob"][pool, current_proj])
        next_ppl = np.random.choice(
            ppl_id[pool], p=args["proj_prob"][pool, current_proj] * (1/norm)
        )
        delta_e += (args["am"][next_ppl, current_proj])
        if len(proj_path) == max_length:
            break
        if next_ppl == start_ppl_id:
            delta_e += args["am"][next_ppl, current_proj] - args["am"][start_ppl_id, current_proj]
            break
        elif next_ppl in no_proj_ppl:
            # proj_path.append(candidate.index(next_ppl))
            proj_path.append(np.argwhere(candidate == next_ppl)[0, 0])
            break
        else:
            path.append(next_ppl)
            # current_proj = candidate.index(next_ppl)
            current_proj = np.argwhere(candidate == next_ppl)[0, 0]
            proj_path.append(current_proj)
            pool[next_ppl] = False
            delta_e -= args["am"][next_ppl, current_proj]

    if len(proj_path) > 1 and delta_e < max_delta:
        return loop_swap(candidate, proj_path)
    else:
        return candidate


@mutator
def mc_mutator_dynamic(random, candidate, args):
    """
    calculate prob at each point
    now only works for square case
    :param random:
    :param candidate:
    :param args:
    :return:
    """
    if args["mutator"] == "both":
        if random.random() > 0.05:
            return candidate
    # convert to index: people, val: assigned proj
    beta = max(-0.2 * args["num_generations"] + 4, 0.1)
    beta = 0.2
    ppl_assignment = sorted(zip(candidate, list(range(args["n_proj"]))))
    ppl_assignment = [x[0] for x in ppl_assignment]
    ppl_weight = args["am"][np.arange(args["n_proj"]), ppl_assignment]
    # todo:
    ppl_weight = 0

    start_proj_id = np.random.randint(0, args["n_proj"])
    no_proj_ppl = candidate[args["n_proj"] :]
    start_ppl_id = candidate[start_proj_id]
    path = [start_ppl_id]
    proj_path = [start_proj_id]
    current_proj = start_proj_id
    pool = np.ones(shape=(args["n_ppl"],), dtype=bool)
    ppl_id = np.arange(args["n_ppl"])

    while True:
        ppl_de = args["am"][:, current_proj] - ppl_weight
        ppl_prob = np.exp(- ppl_de * beta)[pool]
        norm = np.sum(ppl_prob)
        ppl_prob = ppl_prob / norm
        next_ppl = np.random.choice(
            ppl_id[pool], p=ppl_prob
        )
        if next_ppl in path:
            idx = path.index(next_ppl)
            cut_path = proj_path[idx:]
            break
        elif next_ppl in no_proj_ppl:
            proj_path.append(candidate.index(next_ppl))
            cut_path = proj_path
            break
        else:
            path.append(next_ppl)
            current_proj = candidate.index(next_ppl)
            proj_path.append(current_proj)
            # todo:
            # pool[next_ppl] = False

    eb = np.sum(args["am"][candidate[:args["n_proj"]], list(range(args["n_proj"]))])
    if len(cut_path) > 1:
        ret = loop_swap(candidate, cut_path)
        ea = np.sum(args["am"][ret[:args["n_proj"]], list(range(args["n_proj"]))])
        diff = ea - eb
        # print("change: ", ea - eb)
        # todo:
        if diff < 20:
            return ret
        else:
            return candidate
    else:
        return candidate


@crossover
def assign_crossover(random, mom, dad, args):
    crossover_rate = args.setdefault("crossover_rate", 1.0)
    if random.random() < crossover_rate:
        size = args["n_proj"]
        points = random.sample(range(size), 2)
        x, y = min(points), max(points)
        bro = copy.copy(dad)
        bro[x : y + 1] = mom[x : y + 1]
        sis = copy.copy(mom)
        sis[x : y + 1] = dad[x : y + 1]
        for parent, child in zip([dad, mom], [bro, sis]):
            for i in range(x, y + 1):
                if parent[i] not in child[x : y + 1]:
                    spot = i
                    while x <= spot <= y:
                        # spot = parent.index(child[spot])
                        spot = np.argwhere(parent == child[spot])[0, 0]
                    child[spot] = parent[i]
        return [bro, sis]
    else:
        return [mom, dad]


def best_observer(population, num_generations, num_evaluations, args):
    """Print the best individual in the population to the screen.

    This function displays the best individual in the population to
    the screen.

    .. Arguments:
       population -- the population of Individuals
       num_generations -- the number of elapsed generations
       num_evaluations -- the number of candidate solution evaluations
       args -- a dictionary of keyword arguments

    """
    f_name = "data_archive/" + args["mutator"] + "_" + str(args["n_ppl"]) + "_" + str(args["n_proj"]) + ".dat"
    if num_generations == 1 and isfile(f_name):
        os.remove(f_name)
    with open(f_name, "a") as f:
        f.write(f"{num_generations}, {max(population).fitness}\n")
    print("Best Individual: {0}\n".format(str(max(population))))


if __name__ == "__main__":
    # n_ppl = 300
    # n_proj = 80
    # max_weight = 400
    # # cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    # cost = assign_matrix(n_ppl, n_proj, max_weight)
    # hungarian_solve(cost, False)

    ar = np.arange(20)
    idx_pool = [3, 8, 2, 4]

    print(loop_swap(ar, idx_pool))
