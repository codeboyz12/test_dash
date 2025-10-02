from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os, numpy as np, pandas as pd
from typing import Callable, List, Tuple, Optional, Dict
from venv import logger

class GeneticAlgorithm:
    def __init__(
        self,
        cont_bounds: Optional[List[Tuple[float, float]]],
        n_binary: int,
        predict_fn: Callable[[np.ndarray], float],
        cost_fn: Optional[Callable[[np.ndarray], float]] = None,
        safety_constraints: Optional[List[Callable[[np.ndarray], float]]] = None,
        pop_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.2,
        elite_frac: float = 0.1,
        maximize: bool = True,
        random_seed: int = 42,

        n_jobs: int = -1,                        
        parallel_backend: str = "thread",        
        batch_predictor: Optional[Callable[[np.ndarray], np.ndarray]] = None, 
    ):
        self.cont_bounds = cont_bounds or []
        self.n_cont = len(self.cont_bounds)
        self.n_bin = int(n_binary)
        self.predict_fn = predict_fn
        self.cost_fn = cost_fn
        self.safety_constraints = safety_constraints or []
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_frac = elite_frac
        self.maximize = maximize

        np.random.seed(random_seed)
        self.population: List[np.ndarray] = []
        self.scores: List[float] = []
        self.best_history: List[float] = []
        self.best_individual: Optional[np.ndarray] = None
        self.best_score: Optional[float] = None

        # ---------- NEW ----------
        self.n_jobs = (os.cpu_count() or 1) if n_jobs in (-1, 0, None) else max(1, int(n_jobs))
        self.parallel_backend = "process" if parallel_backend.lower().startswith("process") else "thread"
        self.batch_predictor = batch_predictor

    # ---------------- utils ----------------
    def _concat(self, cont: np.ndarray, binv: np.ndarray) -> np.ndarray:
        if cont.size == 0:  return binv.astype(float)
        if binv.size == 0:  return cont.astype(float)
        return np.concatenate([cont.astype(float), binv.astype(float)])

    def _split(self, individual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.n_cont == 0:           return np.array([]), individual.astype(int)
        elif self.n_bin == 0:          return individual[:self.n_cont], np.array([], dtype=int)
        else:                          return individual[:self.n_cont], individual[self.n_cont:].astype(int)

    # ---------------- init/eval ----------------
    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            cont = np.array([np.random.uniform(low, high)
                             for (low, high) in self.cont_bounds]) if self.n_cont > 0 else np.array([])
            binv = np.random.randint(0, 2, self.n_bin).astype(int) if self.n_bin > 0 else np.array([], dtype=int)
            self.population.append(self._concat(cont, binv))
        self.scores = [None] * self.pop_size

    def _fitness_from_pred(self, individual: np.ndarray, pred: float) -> float:
        penalty = 0.0
        if self.cost_fn is not None:
            try: penalty += float(self.cost_fn(individual.copy()))
            except: pass
        for constr in self.safety_constraints:
            try: penalty += float(constr(individual.copy()))
            except: pass
        fit = pred - penalty
        return fit if self.maximize else -fit

    def evaluate_individual(self, individual: np.ndarray) -> float:
        try:
            pred = float(self.predict_fn(individual.copy()))
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            pred = -1e9 if self.maximize else 1e9
        return self._fitness_from_pred(individual, pred)

    def _evaluate_population_parallel(self, pop: List[np.ndarray]) -> List[float]:
        # Fast path: batch predictor (ถ้ามี)
        if callable(self.batch_predictor):
            try:
                preds = np.asarray(self.batch_predictor(np.vstack(pop)), dtype=float).reshape(-1)
                fits = [self._fitness_from_pred(ind, preds[i]) for i, ind in enumerate(pop)]
                return fits
            except Exception as e:
                logger.warning(f"batch_predictor failed, fallback to per-individual: {e}")

        # Per-individual parallel
        if self.n_jobs == 1:
            return [self.evaluate_individual(ind) for ind in pop]

        Executor = ProcessPoolExecutor if self.parallel_backend == "process" else ThreadPoolExecutor
        with Executor(max_workers=self.n_jobs) as ex:
            return list(ex.map(self.evaluate_individual, pop, chunksize=max(1, len(pop)//(self.n_jobs*4))))

    def evaluate_population(self):
        self.scores = self._evaluate_population_parallel(self.population)
        best_idx = int(np.argmax(self.scores))
        self.best_individual = self.population[best_idx].copy()
        self.best_score = float(self.scores[best_idx])
        self.best_history.append(self.best_score)

    # ---------------- GA ops ----------------
    def _tournament_select(self, k=3) -> np.ndarray:
        idxs = np.random.choice(range(self.pop_size), size=k, replace=False)
        best = max(idxs, key=lambda i: self.scores[i])
        return self.population[best].copy()

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cont_a, bin_a = self._split(a); cont_b, bin_b = self._split(b)
        if self.n_cont > 0:
            alpha = np.random.uniform(0, 1, size=self.n_cont)
            c1 = alpha * cont_a + (1 - alpha) * cont_b
            c2 = alpha * cont_b + (1 - alpha) * cont_a
            for i in range(self.n_cont):
                low, high = self.cont_bounds[i]
                c1[i] = np.clip(c1[i], low, high); c2[i] = np.clip(c2[i], low, high)
        else:
            c1 = c2 = np.array([])
        if self.n_bin > 0:
            mask = np.random.rand(self.n_bin) < 0.5
            b1 = np.where(mask, bin_a, bin_b)
            b2 = np.where(mask, bin_b, bin_a)
        else:
            b1 = b2 = np.array([], dtype=int)
        return self._concat(c1, b1), self._concat(c2, b2)

    def _mutate(self, ind: np.ndarray):
        cont, binv = self._split(ind)
        if self.n_cont > 0:
            for i in range(self.n_cont):
                if np.random.rand() < self.mutation_rate:
                    low, high = self.cont_bounds[i]
                    sigma = (high - low) * 0.05
                    cont[i] = np.clip(cont[i] + np.random.normal(0, sigma), low, high)
        if self.n_bin > 0:
            flip = np.random.rand(self.n_bin) < self.mutation_rate
            binv[flip] = 1 - binv[flip]
        return self._concat(cont, binv)

    # ---------------- evolve ----------------
    def evolve(self, verbose: bool = True, feature_names: Optional[List[str]] = None,
               base_row: Optional[pd.DataFrame] = None, model=None):
        self.initialize_population()
        self.evaluate_population()
        if verbose: logger.info(f"Initial best score: {self.best_score:.4f}")

        n_elite = max(1, int(self.elite_frac * self.pop_size))

        for gen in range(self.generations):
            # keep elites
            elite_idxs = np.argsort(self.scores)[::-1][:n_elite]
            new_pop = [self.population[i].copy() for i in elite_idxs]

            # offspring
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                if np.random.rand() < self.crossover_rate:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))

            self.population = new_pop[:self.pop_size]
            self.evaluate_population()

            if verbose and ((gen + 1) % max(1, self.generations // 10) == 0):
                logger.info(f"Gen {gen+1}/{self.generations} best_score={self.best_score:.4f}")

        result = {
            "best_individual": self.best_individual,
            "best_score": self.best_score,
            "history": self.best_history
        }

        # Optional summary (ก่อน/หลังปรับ)
        if feature_names is not None and base_row is not None and model is not None and self.best_individual is not None:
            base_pred = float(model.predict(base_row)[0])
            row_after = base_row.copy()
            for i, f in enumerate(feature_names):
                row_after.at[row_after.index[0], f] = self.best_individual[i]
            new_pred = float(model.predict(row_after)[0])

            changes = {}
            for i, f in enumerate(feature_names):
                old_val = float(base_row[f].iloc[0]); new_val = float(self.best_individual[i])
                if abs(new_val - old_val) > 1e-6:
                    changes[f] = {"before": old_val, "after": new_val}

            result.update({
                "pred_before": base_pred,
                "pred_after": new_pred,
                "improvement": new_pred - base_pred,
                "adjusted_features": changes,
            })
        return result

    def decode_best(self):
        if self.best_individual is None: return None, None
        return self._split(self.best_individual)


