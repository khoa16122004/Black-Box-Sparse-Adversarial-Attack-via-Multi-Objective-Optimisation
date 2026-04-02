import numpy as np
from copy import deepcopy
from operator import attrgetter
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class Solution:
    def __init__(self, pixels, values, x, p_size):
        self.pixels = pixels  # list of Integers
        self.values = values  # list of Binary tuples, i.e. [0, 1, 1]
        self.x = x  # (w x w x 3)
        self.fitnesses = []
        self.is_adversarial = None
        self.w = x.shape[0]
        self.delta = len(self.pixels)
        self.domination_count = None
        self.dominated_solutions = None
        self.rank = None
        self.crowding_distance = None

        self.loss = None
        self.pred_label = -1
        self.p_size = p_size

    def copy(self):
        return deepcopy(self)

    def euc_distance(self, img):
        return np.sum((img - self.x.copy()) ** 2)

    def l0_distance(self, img):
        return np.sum(np.any(img != self.x.copy(), axis=-1))

    def generate_image(self):
        x_adv = self.x.copy()
        for i in range(self.delta):
            x_adv[self.pixels[i] // self.w, self.pixels[i] % self.w] += (self.values[i] * self.p_size)

        return np.clip(x_adv, 0, 1)

    def evaluate(self, loss_function, include_dist, objective2_fn=None):
        img_adv = self.generate_image()
        fs = loss_function(img_adv)
        self.is_adversarial = fs[0]  # Assume first element is boolean always
        if len(fs) > 2:
            self.pred_label = int(fs[2])
        elif hasattr(loss_function, "get_label"):
            self.pred_label = int(loss_function.get_label(img_adv))
        else:
            self.pred_label = -1

        self.fitnesses = [float(fs[1])]
        if include_dist:
            if objective2_fn is None:
                raise ValueError("objective2_fn is required: L2 objective has been removed")
            obj2 = float(objective2_fn(img_adv))
            self.fitnesses.append(obj2)
        else:
            self.fitnesses.append(0)

        self.fitnesses = np.array(self.fitnesses)
        self.loss = fs[1]
        self.l0 = self.l0_distance(img_adv)

    def dominates(self, soln):
        # Standard Pareto dominance (minimization): no worse on all objectives
        # and strictly better on at least one objective.
        return bool(np.all(self.fitnesses <= soln.fitnesses) and np.any(self.fitnesses < soln.fitnesses))


def fast_nondominated_sort(population):
    if len(population) == 0:
        return []

    F = np.array([individual.fitnesses for individual in population], dtype=float)
    front_indices = NonDominatedSorting().do(F)

    fronts = []
    for rank, idxs in enumerate(front_indices):
        front = [population[i] for i in idxs]
        for individual in front:
            individual.rank = rank
        fronts.append(front)

    return fronts


def calculate_crowding_distance(front):
    if len(front) > 0:
        solutions_num = len(front)
        for individual in front:
            individual.crowding_distance = 0

        for m in range(len(front[0].fitnesses)):
            front.sort(key=lambda individual: individual.fitnesses[m])
            front[0].crowding_distance = 10 ** 9
            front[solutions_num - 1].crowding_distance = 10 ** 9
            m_values = [individual.fitnesses[m] for individual in front]
            scale = max(m_values) - min(m_values)
            if scale == 0: scale = 1
            for i in range(1, solutions_num - 1):
                front[i].crowding_distance += (front[i + 1].fitnesses[m] - front[i - 1].fitnesses[m]) / scale


def crowding_operator(individual, other_individual):
    if (individual.rank < other_individual.rank) or ((individual.rank == other_individual.rank) and (
            individual.crowding_distance > other_individual.crowding_distance)):
        return 1
    else:
        return -1


def __tournament(population, tournament_size):
    participants = np.random.choice(population, size=(tournament_size,), replace=False)
    best = None
    for participant in participants:
        if best is None or (
                crowding_operator(participant, best) == 1):  # and self.__choose_with_prob(self.tournament_prob)):
            best = participant

    return best


def tournament_selection(population, tournament_size):
    parents = []
    while len(parents) < len(population) // 2:
        parent1 = __tournament(population, tournament_size)
        parent2 = __tournament(population, tournament_size)

        parents.append([parent1, parent2])
    return parents
