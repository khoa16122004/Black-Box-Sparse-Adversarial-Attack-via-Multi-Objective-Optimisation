from MOAA.Solutions import Solution
import numpy as np

def mutation_flexible(soln: Solution, pm: float, all_pixels: np.array, zero_prob: float, eps_max: int):
    """Mutation with flexible pixel count: offspring pixel count random in [1, eps_max]."""
    pixels = soln.pixels.copy()
    rgbs = soln.values.copy()
    eps_old = len(pixels)
    # Decide new pixel count
    eps_new = np.random.randint(1, eps_max + 1)
    # Keep a random subset of old pixels
    keep_count = min(eps_old, int(eps_new * (1 - pm)))
    if keep_count > 0:
        keep_idx = np.random.choice(eps_old, size=keep_count, replace=False)
        new_pixels = pixels[keep_idx]
        new_rgbs = rgbs[keep_idx]
    else:
        new_pixels = np.array([], dtype=pixels.dtype)
        new_rgbs = np.array([], dtype=rgbs.dtype)
    # Add new pixels
    u_m = np.setdiff1d(all_pixels, new_pixels)
    add_count = eps_new - len(new_pixels)
    if add_count > 0:
        add_pixels = np.random.choice(u_m, size=add_count, replace=False)
        ones_prob = (1 - zero_prob) / 2
        add_rgbs = np.random.choice([-1, 1, 0], size=(add_count, 3), p=(ones_prob, ones_prob, zero_prob))
        new_pixels = np.concatenate([new_pixels, add_pixels], axis=0)
        new_rgbs = np.concatenate([new_rgbs, add_rgbs], axis=0)
    soln.pixels = new_pixels
    soln.values = new_rgbs

def crossover_flexible(soln1: Solution, soln2: Solution, pc: float, eps_max: int):
    """Crossover with flexible pixel count: offspring pixel count random between parents, up to eps_max."""
    # Decide offspring pixel count
    eps1 = len(soln1.pixels)
    eps2 = len(soln2.pixels)
    eps_new = np.random.randint(1, min(max(eps1, eps2), eps_max) + 1)
    # Merge pixel sets
    merged_pixels = np.concatenate([soln1.pixels, soln2.pixels])
    merged_pixels = np.unique(merged_pixels)
    # Randomly select eps_new pixels
    offspring_pixels = np.random.choice(merged_pixels, size=eps_new, replace=False)
    # For each pixel, randomly pick value from one of parents
    values = []
    for px in offspring_pixels:
        if px in soln1.pixels and px in soln2.pixels:
            if np.random.rand() < 0.5:
                idx = np.where(soln1.pixels == px)[0][0]
                values.append(soln1.values[idx])
            else:
                idx = np.where(soln2.pixels == px)[0][0]
                values.append(soln2.values[idx])
        elif px in soln1.pixels:
            idx = np.where(soln1.pixels == px)[0][0]
            values.append(soln1.values[idx])
        else:
            idx = np.where(soln2.pixels == px)[0][0]
            values.append(soln2.values[idx])
    offspring_values = np.array(values)
    offspring = soln1.copy()
    offspring.pixels = offspring_pixels
    offspring.values = offspring_values
    return offspring



def mutation(soln: Solution, pm: float, all_pixels: np.array, zero_prob: float):
    all_pixels = all_pixels.copy()
    pixels = soln.pixels.copy()
    rgbs = soln.values.copy()

    eps_it = max([int(len(soln.pixels) * pm), 1])
    eps = len(soln.pixels)

    # select pixels to keep
    A_ = np.random.choice(eps, size=(eps - eps_it,), replace=False)
    new_pixels = pixels[A_]
    new_rgbs = rgbs[A_]

    # select new pixels to replace
    u_m = np.delete(all_pixels, pixels)
    B = np.random.choice(u_m, size=(eps_it,), replace=False)

    ones_prob = (1 - zero_prob) / 2
    rgbs_ = np.random.choice([-1, 1, 0], size=(eps_it, 3), p=(ones_prob, ones_prob, zero_prob))
    pixels_ = all_pixels[B]

    new_pixels = np.concatenate([new_pixels, pixels_], axis=0)
    new_rgbs = np.concatenate([new_rgbs, rgbs_], axis=0)

    soln.pixels = new_pixels
    soln.values = new_rgbs


def crossover(soln1: Solution, soln2: Solution, pc: float):
    l = max([int(len(soln1.pixels) * pc), 1])
    k = len(soln1.pixels)
    # S1 crossover with S2
    # 1. Generate set of different pixels in S2
    delta = np.asarray([pi for pi in range(k) if soln2.pixels[pi] not in soln1.pixels])

    offspring1 = soln1.copy()
    if len(delta)>0:
        l = l if l <= len(delta) else len(delta)
        switched_pixels = np.random.choice(delta, size=(l,))
        offspring1.pixels[switched_pixels] = soln2.pixels[switched_pixels].copy()
        offspring1.values[switched_pixels] = soln2.values[switched_pixels].copy()

    # S2 crossover with S1
    # 1. Generate set of different pixels in S2
    delta = np.asarray([pi for pi in range(k) if soln1.pixels[pi] not in soln2.pixels])
    offspring2 = soln1.copy()
    if len(delta)>0:
        l = l if l <= len(delta) else len(delta)
        switched_pixels = np.random.choice(delta, size=(l,))
        offspring2.pixels[switched_pixels] = soln1.pixels[switched_pixels].copy()
        offspring2.values[switched_pixels] = soln1.values[switched_pixels].copy()

    return offspring1, offspring2


def generate_offspring(parents, pc, pm, all_pixels, zero_prob):
    children = []
    for pi in parents:
        offspring1, offspring2 = crossover(pi[0], pi[1], pc)
        mutation(offspring1, pm, all_pixels, zero_prob)
        mutation(offspring2, pm, all_pixels, zero_prob)

        assert len(np.unique(offspring1.pixels)) == len(offspring1.pixels)
        assert len(np.unique(offspring2.pixels)) == len(offspring2.pixels)
        children.extend([offspring1, offspring2])

    return children

def generate_offspring_flexible(parents, pc, pm, all_pixels, zero_prob):
    # eps_max should be provided, e.g. as max allowed pixel count
    # For compatibility, infer from parents if not passed
    eps_max = max([len(pi[0].pixels) for pi in parents] + [len(pi[1].pixels) for pi in parents])
    children = []
    for pi in parents:
        offspring1 = crossover_flexible(pi[0], pi[1], pc, eps_max)
        offspring2 = crossover_flexible(pi[1], pi[0], pc, eps_max)
        mutation_flexible(offspring1, pm, all_pixels, zero_prob, eps_max)
        mutation_flexible(offspring2, pm, all_pixels, zero_prob, eps_max)

        assert len(np.unique(offspring1.pixels)) == len(offspring1.pixels)
        assert len(np.unique(offspring2.pixels)) == len(offspring2.pixels)
        children.extend([offspring1, offspring2])

    return children




