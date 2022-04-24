import numpy as np
import inspyred
from inspyred.benchmarks import Benchmark
from time import time
from inspyred.ec.variators.mutators import mutator
import copy


def objective_function(ppl_tag, proj_tag, ppl_selection, filter_dummy=False):
    """
    从人员和项目标签，推导匹配好坏的程度。
    由于数据和业务都还没有到位，现在只能靠脑补
    :param ppl_tag:
    :param proj_tag:
    :param ppl_selection:
    :param filter_dummy:
    :return:
    """
    tag_sum = np.dot(ppl_selection.transpose(), ppl_tag)
    if filter_dummy and proj_tag.ndim != 2:
        raise ValueError("Can't filter dummy ")
    if not filter_dummy:
        obj = np.sum(np.abs(tag_sum - proj_tag))
    else:
        obj = np.sum(np.abs(tag_sum[1:, :] - proj_tag[1:, :]))

    return obj


class TagMatchingProblemSingle(Benchmark):
    def __init__(self, dimensions, people_tags, project_tag):
        Benchmark.__init__(self, dimensions)
        self.bounder = inspyred.ec.DiscreteBounder([0, 1])
        if dimensions < 5:
            raise ValueError("Dimension too small ")
        self.people_tags = people_tags
        self.project_tag = project_tag

    def generator(self, random, args):
        """
        :param random:
        :param args:
        :return:
        """
        guess = np.zeros((self.dimensions, ), dtype=int)
        flip = np.random.choice(np.arange(self.dimensions), size=(3, ), replace=False)
        for i in flip:
            guess[i] = 1
        return list(guess)

    def evaluator(self, candidates, args):
        if self.project_tag is None:
            raise ValueError("Project tag is not set")
        fitness = []
        for condi in candidates:
            fitness.append(objective_function(self.people_tags, self.project_tag, np.array(condi)))

        return fitness


class TagMatchingProblemMulti(Benchmark):
    def __init__(self, dimensions, Nproj, people_tags, project_tag):
        Benchmark.__init__(self, dimensions)
        self.bounder = inspyred.ec.DiscreteBounder([i for i in range(Nproj + 1)])
        self.Nproj = Nproj
        d = project_tag.shape[1]
        # 第一列对于没有分配项目的
        pad = np.zeros((1, d))
        self.people_tag = people_tags
        self.project_tag = np.concatenate([pad, project_tag], axis=0)
        self.pre_filter_result = [[] for _ in range(self.dimensions)]
        self.pre_filter()

    def pre_filter(self):
        for i in range(self.dimensions):
            for j in range(self.Nproj):
                if np.sum(np.abs(self.project_tag[j])) > np.sum(np.abs(self.project_tag[j] - self.people_tag[i])):
                    self.pre_filter_result[i].append(j)
            self.pre_filter_result[i] = np.array(self.pre_filter_result[i])

    def generator(self, random, args, pre_filter=True):
        """
        :param random:
        :param args:
        :param pre_filter:
        :return:
        """
        guess = np.zeros((self.dimensions, ), dtype=int)
        if not pre_filter:
            assign = np.random.uniform(0, 1, (self.dimensions,))
            assign = (assign < (3 * self.Nproj / self.dimensions)).astype(int)
            aproj = np.random.randint(0, self.Nproj, (self.dimensions,)) + 1
            assign = assign * aproj
            guess = guess + assign

        else:
            for i in range(self.dimensions):
                if random.uniform(0.0, 1.0) < (3 * self.Nproj / self.dimensions):
                    if len(self.pre_filter_result[i]) > 0:
                        guess[i] = np.random.choice(self.pre_filter_result[i], 1)

        return list(guess)

    def evaluator(self, candidates, args):
        fitness = []
        for condi in candidates:
            onehot = np.eye(self.Nproj+1)[condi]
            fitness.append(objective_function(self.people_tag, self.project_tag, onehot, filter_dummy=True))

        return fitness


@mutator
def tag_mutator(random, candidate, args):
    flip_rate = args.setdefault('flip_rate', 0.004)
    switch_rate = args.setdefault('switch_rate', 0.004)
    pre_filter = args['pre_filter']
    mutant = copy.copy(candidate)
    for i, m in enumerate(mutant):
        if random.random() < flip_rate:
            if m == 0 and len(pre_filter[i]) > 0:
                mutant[i] = np.random.choice(pre_filter[i], 1)[0]
            elif m != 0:
                mutant[i] = 0
        elif random.random() < flip_rate + switch_rate:
            if m != 0:
                mutant[i] = np.random.choice(pre_filter[i], 1)[0]
    return mutant
