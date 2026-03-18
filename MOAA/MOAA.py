from MOAA.operators import *
from MOAA.Solutions import *
import numpy as np
import time
from tqdm import tqdm

def p_selection(it, p_init, n_queries):
    it = int(it / n_queries * 10000)
    if 0 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 5
    elif 500 < it <= 1000:
        p = p_init / 6
    elif 1000 < it <= 2000:
        p = p_init / 8
    elif 2000 < it <= 4000:
        p = p_init / 10
    elif 4000 < it <= 6000:
        p = p_init / 12
    elif 6000 < it <= 8000:
        p = p_init / 15
    elif 8000 < it:
        p = p_init / 20
    else:
        p = p_init

    return p


class Population:
    def __init__(self, solutions: list, loss_function, include_dist, objective2_fn=None):
        self.population = solutions
        self.fronts = None
        self.loss_function = loss_function
        self.include_dist = include_dist
        self.objective2_fn = objective2_fn

    def evaluate(self):
        if len(self.population) == 0:
            return

        imgs_adv = np.array([pi.generate_image() for pi in self.population], dtype=np.float32)

        if hasattr(self.loss_function, "batch"):
            fs_batch = self.loss_function.batch(imgs_adv)
        else:
            fs_batch = [self.loss_function(img) for img in imgs_adv]

        
        if self.include_dist:
            if self.objective2_fn is None:
                raise ValueError("objective2_fn is required when include_dist is True")

            if hasattr(self.objective2_fn, "batch"):
                obj2_batch = self.objective2_fn.batch(imgs_adv)
            else:
                obj2_batch = [self.objective2_fn(img) for img in imgs_adv]
        else:
            obj2_batch = [0.0 for _ in self.population]

        for idx, pi in enumerate(self.population):
            fs = fs_batch[idx]
            pi.is_adversarial = bool(fs[0])
            pi.fitnesses = np.array([float(fs[1]), float(obj2_batch[idx])], dtype=float)
            pi.loss = float(fs[1])
            pi.l0 = pi.l0_distance(pi.generate_image())

    def find_adv_solns(self, max_dist):
        adv_solns = []
        for pi in self.population:
            if pi.is_adversarial and pi.fitnesses[1] <= max_dist:
                adv_solns.append(pi)

        return adv_solns


class Attack:
    def __init__(self, params):
        self.params = params
        self.fitness = []
        self.verbose = bool(self.params.get("verbose", False))
        self.print_every = int(self.params.get("print_every", 10))

        self.data = []

    def objective_mins(self, population_solutions):
        fitnesses = np.array([soln.fitnesses for soln in population_solutions], dtype=float)
        if fitnesses.ndim == 1:
            fitnesses = fitnesses[None, :]
        return np.min(fitnesses, axis=0)

    # def update_data(self, front):

    def completion_procedure(self, population: Population, loss_function, fe, success):

        #print(success, fe)
        #print(1/0)

        adversarial_labels = []
        for soln in population.fronts[0]:
            adversarial_labels.append(loss_function.get_label(soln.generate_image()))

        d = {"front0_imgs": [soln.generate_image() for soln in population.fronts[0]],
             "queries": fe,
             "true_label": loss_function.true,
             "adversarial_labels": adversarial_labels,
             "front0_fitness": [soln.fitnesses for soln in population.fronts[0]],
             "fitness_process": self.fitness,
             "success": success
             }
        d["init_front0_fitness"] = self.init_front0_fitness
        # print(d["true_label"], d["adversarial_labels"])
        np.save(self.params["save_directory"], d, allow_pickle=True)

    def attack(self, loss_function):
        start = time.time()
        # print(loss_function(self.params["x"]))
        # print(self.params["n_pixels"])
        # Minimizes
        h, w, c = self.params["x"].shape[0:]
        pm = self.params["pm"]
        n_pixels = h * w
        all_pixels = np.arange(n_pixels)
        ones_prob = (1 - self.params["zero_probability"]) / 2
        init_solutions = [Solution(np.random.choice(all_pixels,
                                                    size=(self.params["eps"]), replace=False),
                                   np.random.choice([-1, 1, 0], size=(self.params["eps"], 3),
                                                    p=(ones_prob, ones_prob, self.params["zero_probability"])),
                                   self.params["x"].copy(), self.params["p_size"]) for _ in
                          range(self.params["pop_size"])]




        objective2_fn = self.params.get("objective2_fn", None)
        population = Population(init_solutions, loss_function, self.params["include_dist"], objective2_fn)
        population.evaluate()
        # for soln in init_solutions:
        #     print("Init l0: ", soln.l0)
        # raise

        fe = len(population.population)
        population.fronts = fast_nondominated_sort(population.population)
        self.init_front0_fitness = [soln.fitnesses.copy() for soln in population.fronts[0]]
        if self.verbose:
            print(
                f"[MOAA] start | pop_size={self.params['pop_size']} | iterations={self.params['iterations']} "
                f"| eps={self.params['eps']} | include_dist={self.params['include_dist']}"
            )

        for it in tqdm(range(1, self.params["iterations"])):
            #pm = p_selection(it, self.params["pm"], self.params["iterations"])
            pm = self.params["pm"]
            population.fronts = fast_nondominated_sort(population.population)
            obj_mins = self.objective_mins(population.population)

            adv_solns = population.find_adv_solns(self.params["max_dist"])
            if self.verbose and (it == 1 or it % max(self.print_every, 1) == 0):
                min_obj0 = float(obj_mins[0]) if len(obj_mins) > 0 else float("nan")
                min_obj1 = float(obj_mins[1]) if len(obj_mins) > 1 else float("nan")
                print(
                    f"[MOAA] iter={it}/{self.params['iterations'] - 1} | queries={fe} "
                    f"| min_obj0={min_obj0:.6f} | min_obj1={min_obj1:.6f} "
                    f"| adv_found={len(adv_solns)}"
                )

            if len(adv_solns) > 0:
                self.fitness.append(obj_mins)
                self.completion_procedure(population, loss_function, fe, True)
                if self.verbose:
                    print(
                        f"[MOAA] success | iter={it} | queries={fe} "
                        f"| elapsed={time.time() - start:.2f}s"
                    )
                return

            self.fitness.append(obj_mins)

            #print(fe, self.fitness[-1])

            for front in population.fronts:
                calculate_crowding_distance(front)
            parents = tournament_selection(population.population, self.params["tournament_size"])
            children = generate_offspring_flexible(parents,
                                          self.params["pc"],
                                          pm,
                                          all_pixels,
                                          self.params["zero_probability"])

            offsprings = Population(children, loss_function, self.params["include_dist"], objective2_fn)
            fe += len(offsprings.population)
            offsprings.evaluate()
            population.population.extend(offsprings.population)
            population.fronts = fast_nondominated_sort(population.population)
            front_num = 0
            new_solutions = []
            while len(new_solutions) + len(population.fronts[front_num]) <= self.params["pop_size"]:
                calculate_crowding_distance(population.fronts[front_num])
                new_solutions.extend(population.fronts[front_num])
                front_num += 1

            calculate_crowding_distance(population.fronts[front_num])
            population.fronts[front_num].sort(key=attrgetter("crowding_distance"), reverse=True)
            new_solutions.extend(population.fronts[front_num][0:self.params["pop_size"] - len(new_solutions)])

            population = Population(new_solutions, loss_function, self.params["include_dist"], objective2_fn)

        population.fronts = fast_nondominated_sort(population.population)
        self.fitness.append(self.objective_mins(population.population))
        self.completion_procedure(population, loss_function, fe, False)
        if self.verbose:
            print(f"[MOAA] finished | success=False | queries={fe} | elapsed={time.time() - start:.2f}s")
        #print(time.time() - start)ff
        return



class Attack_Flexible_L0(Attack): # the attack that having different l0 of each solution
    def __init__(self, params):
        super().__init__(params)

    def attack(self, loss_function):
        start = time.time()
        # print(loss_function(self.params["x"]))
        # print(self.params["n_pixels"])
        # Minimizes
        h, w, c = self.params["x"].shape[0:]
        pm = self.params["pm"]
        n_pixels = h * w
        all_pixels = np.arange(n_pixels)
        ones_prob = (1 - self.params["zero_probability"]) / 2


        # overide
        init_solutions = []
        for _ in range(self.params["pop_size"]):
            eps = np.random.randint(1, self.params["eps"] + 1)
            pixels = np.random.choice(all_pixels, size=(eps,), replace=False)
            perturbations = np.random.choice([-1, 1, 0], size=(eps, 3),
                                             p=(ones_prob, ones_prob, self.params["zero_probability"]))
            init_solutions.append(Solution(pixels, perturbations, self.params["x"].copy(), self.params["p_size"]))

        objective2_fn = self.params.get("objective2_fn", None)
        population = Population(init_solutions, loss_function, self.params["include_dist"], objective2_fn)
        population.evaluate()
        fe = len(population.population)
        population.fronts = fast_nondominated_sort(population.population)
        self.init_front0_fitness = [soln.fitnesses.copy() for soln in population.fronts[0]]
        if self.verbose:
            print(
                f"[MOAA] start | pop_size={self.params['pop_size']} | iterations={self.params['iterations']} "
                f"| eps={self.params['eps']} | include_dist={self.params['include_dist']}"
            )

        for it in tqdm(range(1, self.params["iterations"])):
            #pm = p_selection(it, self.params["pm"], self.params["iterations"])
            pm = self.params["pm"]
            population.fronts = fast_nondominated_sort(population.population)
            obj_mins = self.objective_mins(population.population)

            adv_solns = population.find_adv_solns(self.params["max_dist"])
            if self.verbose and (it == 1 or it % max(self.print_every, 1) == 0):
                min_obj0 = float(obj_mins[0]) if len(obj_mins) > 0 else float("nan")
                min_obj1 = float(obj_mins[1]) if len(obj_mins) > 1 else float("nan")
                print(
                    f"[MOAA] iter={it}/{self.params['iterations'] - 1} | queries={fe} "
                    f"| min_obj0={min_obj0:.6f} | min_obj1={min_obj1:.6f} "
                    f"| adv_found={len(adv_solns)}"
                )

            if len(adv_solns) > 0:
                self.fitness.append(obj_mins)
                self.completion_procedure(population, loss_function, fe, True)
                if self.verbose:
                    print(
                        f"[MOAA] success | iter={it} | queries={fe} "
                        f"| elapsed={time.time() - start:.2f}s"
                    )
                return

            self.fitness.append(obj_mins)

            #print(fe, self.fitness[-1])

            for front in population.fronts:
                calculate_crowding_distance(front)
            parents = tournament_selection(population.population, self.params["tournament_size"])
            children = generate_offspring(parents,
                                          self.params["pc"],
                                          pm,
                                          all_pixels,
                                          self.params["zero_probability"])

            offsprings = Population(children, loss_function, self.params["include_dist"], objective2_fn)
            fe += len(offsprings.population)
            offsprings.evaluate()
            population.population.extend(offsprings.population)
            population.fronts = fast_nondominated_sort(population.population)
            front_num = 0
            new_solutions = []
            while len(new_solutions) + len(population.fronts[front_num]) <= self.params["pop_size"]:
                calculate_crowding_distance(population.fronts[front_num])
                new_solutions.extend(population.fronts[front_num])
                front_num += 1

            calculate_crowding_distance(population.fronts[front_num])
            population.fronts[front_num].sort(key=attrgetter("crowding_distance"), reverse=True)
            new_solutions.extend(population.fronts[front_num][0:self.params["pop_size"] - len(new_solutions)])

            population = Population(new_solutions, loss_function, self.params["include_dist"], objective2_fn)

        population.fronts = fast_nondominated_sort(population.population)
        self.fitness.append(self.objective_mins(population.population))
        self.completion_procedure(population, loss_function, fe, False)
        if self.verbose:
            print(f"[MOAA] finished | success=False | queries={fe} | elapsed={time.time() - start:.2f}s")
        #print(time.time() - start)ff
        return        
