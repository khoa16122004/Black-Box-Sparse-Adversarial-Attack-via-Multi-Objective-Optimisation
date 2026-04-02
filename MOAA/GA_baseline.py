from MOAA.operators import *
from MOAA.Solutions import *
from MOAA.MOAA import Attack, Population
import numpy as np
import time
from tqdm import tqdm

class AttackGA(Attack):
    def __init__(self, params):
        super().__init__(params)
        self.lambda_1 = params.get('lambda_1', 0.5)
        self.lambda_2 = params.get('lambda_2', 0.5)
        self.tournament_size = params.get('tournament_size', 2)

    def attack(self, loss_function):
        h, w, _ = self.params["x"].shape[0:]
        n_pixels = h * w
        all_pixels = np.arange(n_pixels)
        ones_prob = (1 - self.params["zero_probability"]) / 2

        init_solutions = [
            Solution(
                np.random.choice(all_pixels, size=self.params["eps"], replace=False),
                np.random.choice(
                    [-1, 1, 0],
                    size=(self.params["eps"], 3),
                    p=(ones_prob, ones_prob, self.params["zero_probability"]),
                ),
                self.params["x"].copy(),
                self.params["p_size"],
            )
            for _ in range(self.params["pop_size"])
        ]

        objective2_fn = self.params.get("objective2_fn", None)
        population = Population(init_solutions, loss_function, self.params["include_dist"], objective2_fn)
        population.evaluate()
        for soln in population.population:
            soln.fitness_score = self._compute_weighted_score(soln)

        fe = len(population.population)
        fronts_gen1 = fast_nondominated_sort(population.population)
        if len(fronts_gen1) > 0 and len(fronts_gen1[0]) > 0:
            self.init_front0_fitness = [soln.fitnesses.copy() for soln in fronts_gen1[0]]
            arkiv = list(fronts_gen1[0])
        else:
            self.init_front0_fitness = [soln.fitnesses.copy() for soln in population.population]
            arkiv = list(population.population)

        if len(arkiv) > 0:
            self.fitness = [self.objective_mins(arkiv)]
        else:
            n_obj = len(population.population[0].fitnesses) if len(population.population) > 0 else 2
            self.fitness = [np.full(n_obj, np.nan)]

        best_weighted = self._find_best_weighted_candidate(population.population)
        self.weighted_best_process = [self._candidate_summary(best_weighted)]
        self.weighted_best_process_records = [self._candidate_record(best_weighted, include_image=False)]
        population.fronts_history = [arkiv.copy()]

        for it in tqdm(range(1, self.params["iterations"])):
            pm = self.params["pm"]

            for soln in population.population:
                soln.fitness_score = self._compute_weighted_score(soln)

            parents = self.tournament_selection_single_objective(population.population, self.tournament_size)
            children = generate_offspring(
                parents,
                self.params["pc"],
                pm,
                all_pixels,
                self.params["zero_probability"],
            )

            offsprings = Population(children, loss_function, self.params["include_dist"], objective2_fn)
            fe += len(offsprings.population)
            offsprings.evaluate()
            for soln in offsprings.population:
                soln.fitness_score = self._compute_weighted_score(soln)

            pool = population.population + offsprings.population
            population.population = self.tournament_selection_from_pool(pool, self.params["pop_size"], self.tournament_size)

            for s in offsprings.population:
                dominated = False
                remove_idx = []
                for idx, a in enumerate(arkiv):
                    if np.all(a.fitnesses <= s.fitnesses) and np.any(a.fitnesses < s.fitnesses):
                        dominated = True
                        break
                    if np.all(s.fitnesses <= a.fitnesses) and np.any(s.fitnesses < a.fitnesses):
                        remove_idx.append(idx)
                if not dominated:
                    for idx in sorted(remove_idx, reverse=True):
                        del arkiv[idx]
                    arkiv.append(s)

            best_weighted = self._find_best_weighted_candidate(population.population)
            self.weighted_best_process.append(self._candidate_summary(best_weighted))
            self.weighted_best_process_records.append(self._candidate_record(best_weighted, include_image=False))
            if len(arkiv) > 0:
                self.fitness.append(self.objective_mins(arkiv))
            else:
                n_obj = len(population.population[0].fitnesses) if len(population.population) > 0 else 2
                self.fitness.append(np.full(n_obj, np.nan))
            population.fronts_history.append(arkiv.copy())

            if len([s for s in offsprings.population if s.is_adversarial and s.fitnesses[1] <= self.params["max_dist"]]) > 0:
                population.fronts = fast_nondominated_sort(population.population)
                self._save_result(population, loss_function, fe, success=True)
                return

        population.fronts = fast_nondominated_sort(population.population)
        self._save_result(population, loss_function, fe, success=False)
        return

    def _save_result(self, population, loss_function, fe, success):
        final_front0 = []
        if hasattr(population, "fronts") and population.fronts is not None and len(population.fronts) > 0:
            if isinstance(population.fronts[0], list):
                final_front0 = population.fronts[0]
            else:
                final_front0 = population.fronts

        fallback = final_front0 if len(final_front0) > 0 else population.population
        weighted_best_final = self._find_best_weighted_candidate(fallback)

        d = {
            "front0_imgs": [getattr(soln, "generate_image", lambda: None)() for soln in final_front0],
            "queries": fe,
            "true_label": getattr(loss_function, "true", None) if hasattr(loss_function, "true") else getattr(loss_function, "target", None),
            "adversarial_labels": [getattr(soln, "pred_label", -1) for soln in final_front0],
            "front0_fitness": [getattr(soln, "fitnesses", None) for soln in final_front0],
            "fitness_process": self.fitness,
            "weighted_best_process": self.weighted_best_process,
            "weighted_best_process_records": self.weighted_best_process_records,
            "weighted_best_final": self._candidate_record(weighted_best_final),
            "success": success,
            "init_front0_fitness": getattr(self, "init_front0_fitness", None),
        }
        save_path = self.params.get("save_directory", None)
        if save_path is not None:
            np.save(save_path, d)

    def _compute_weighted_score(self, soln):
        return self.lambda_1 * soln.fitnesses[0] + self.lambda_2 * soln.fitnesses[1]

    def _find_best_weighted_candidate(self, candidates):
        if candidates is None or len(candidates) == 0:
            return None
        for s in candidates:
            if not hasattr(s, "fitness_score"):
                s.fitness_score = self._compute_weighted_score(s)
        return min(candidates, key=lambda s: s.fitness_score)

    def _candidate_summary(self, soln):
        if soln is None:
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        return np.array([soln.fitnesses[0], soln.fitnesses[1], soln.fitness_score], dtype=float)

    def _candidate_record(self, soln, include_image=True):
        if soln is None:
            return None
        record = {
            "fitnesses": np.array(soln.fitnesses, dtype=float),
            "fitness_score": float(soln.fitness_score),
            "pred_label": int(getattr(soln, "pred_label", -1)),
            "is_adversarial": bool(getattr(soln, "is_adversarial", False)),
        }
        if include_image:
            record["image"] = getattr(soln, "generate_image", lambda: None)()
        return record

    def tournament_selection_from_pool(self, pool, n_select, tournament_size):
        # Tournament selection trên pool (2N) để chọn ra N cá thể
        selected = []
        pool = pool.copy()
        while len(selected) < n_select:
            participants = np.random.choice(pool, size=(tournament_size,), replace=False)
            best = None
            for p in participants:
                if best is None or (p.fitness_score < best.fitness_score):
                    best = p
            selected.append(best)
            pool.remove(best)
        return selected

    def tournament_selection_single_objective(self, population, tournament_size):
        parents = []
        while len(parents) < len(population) // 2:
            parent1 = self.__tournament_single_objective(population, tournament_size)
            parent2 = self.__tournament_single_objective(population, tournament_size)
            parents.append([parent1, parent2])
        return parents

    def __tournament_single_objective(self, population, tournament_size):
        participants = np.random.choice(population, size=(tournament_size,), replace=False)
        best = None
        for participant in participants:
            if best is None or (participant.fitness_score < best.fitness_score):
                best = participant
        return best
