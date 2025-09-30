import random, time
from venv import logger
import numpy as np
import pandas as pd
from typing import Callable, List, Tuple, Optional, Dict


class GeneticAlgorithm:
    """Enhanced GA supporting continuous and binary features"""
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
        random_seed: int = 42
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
        self.population = []
        self.scores = []
        self.best_history = []
        self.best_individual = None
        self.best_score = None
    
    def _concat(self, cont: np.ndarray, binv: np.ndarray) -> np.ndarray:
        """Concatenate continuous and binary genes"""
        if cont.size == 0:
            return binv.astype(float)
        if binv.size == 0:
            return cont.astype(float)
        return np.concatenate([cont.astype(float), binv.astype(float)])
    
    def _split(self, individual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split individual into continuous and binary parts"""
        if self.n_cont == 0:
            return np.array([]), individual.astype(int)
        elif self.n_bin == 0:
            return individual[:self.n_cont], np.array([], dtype=int)
        else:
            return individual[:self.n_cont], individual[self.n_cont:].astype(int)
    
    def initialize_population(self):
        """Initialize random population"""
        self.population = []
        for _ in range(self.pop_size):
            # Continuous genes
            cont = np.array([np.random.uniform(low, high) 
                           for (low, high) in self.cont_bounds]) if self.n_cont > 0 else np.array([])
            # Binary genes
            binv = np.random.randint(0, 2, self.n_bin).astype(int) if self.n_bin > 0 else np.array([], dtype=int)
            ind = self._concat(cont, binv)
            self.population.append(ind)
        self.scores = [None] * self.pop_size
    
    def evaluate_individual(self, individual: np.ndarray) -> float:
        """Evaluate fitness of individual"""
        try:
            pred = float(self.predict_fn(individual.copy()))
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            pred = -1e9 if self.maximize else 1e9
        
        penalty = 0.0
        if self.cost_fn is not None:
            try:
                penalty += float(self.cost_fn(individual.copy()))
            except:
                penalty += 0.0
        
        for constr in self.safety_constraints:
            try:
                penalty += float(constr(individual.copy()))
            except:
                penalty += 0.0
        
        fitness = pred - penalty
        return fitness if self.maximize else -fitness
    
    def evaluate_population(self):
        """Evaluate all individuals"""
        self.scores = [self.evaluate_individual(ind) for ind in self.population]
        best_idx = int(np.argmax(self.scores))
        self.best_individual = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]
        self.best_history.append(self.best_score)
    
    def _tournament_select(self, k=3) -> np.ndarray:
        """Tournament selection"""
        import random
        idxs = random.sample(range(self.pop_size), k)
        best = max(idxs, key=lambda i: self.scores[i])
        return self.population[best].copy()
    
    def _crossover(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mixed crossover for continuous and binary genes"""
        cont_a, bin_a = self._split(a)
        cont_b, bin_b = self._split(b)
        
        # Blend crossover for continuous
        if self.n_cont > 0:
            alpha = np.random.uniform(0, 1, size=self.n_cont)
            c1 = alpha * cont_a + (1 - alpha) * cont_b
            c2 = alpha * cont_b + (1 - alpha) * cont_a
            
            for i in range(self.n_cont):
                low, high = self.cont_bounds[i]
                c1[i] = np.clip(c1[i], low, high)
                c2[i] = np.clip(c2[i], low, high)
        else:
            c1 = c2 = np.array([])
        
        # Uniform crossover for binary
        if self.n_bin > 0:
            mask = np.random.rand(self.n_bin) < 0.5
            b1 = np.where(mask, bin_a, bin_b)
            b2 = np.where(mask, bin_b, bin_a)
        else:
            b1 = b2 = np.array([], dtype=int)
        
        return self._concat(c1, b1), self._concat(c2, b2)
    
    def _mutate(self, ind: np.ndarray):
        """Mutation for mixed genes"""
        import random
        cont, binv = self._split(ind)
        
        # Gaussian mutation for continuous
        if self.n_cont > 0:
            for i in range(self.n_cont):
                if random.random() < self.mutation_rate:
                    low, high = self.cont_bounds[i]
                    sigma = (high - low) * 0.05
                    cont[i] += np.random.normal(0, sigma)
                    cont[i] = np.clip(cont[i], low, high)
        
        # Bit flip for binary
        if self.n_bin > 0:
            for j in range(self.n_bin):
                if random.random() < self.mutation_rate:
                    binv[j] = 1 - int(binv[j])
        
        return self._concat(cont, binv)
    
    def evolve(self, verbose: bool = True, feature_names: Optional[List[str]] = None,
           base_row: Optional[pd.DataFrame] = None, model=None):   # >>> Added
        """Main GA evolution loop"""
        import random
        self.initialize_population()
        self.evaluate_population()
        
        if verbose:
            logger.info(f"Initial best score: {self.best_score:.4f}")
        
        n_elite = max(1, int(self.elite_frac * self.pop_size))
        
        for gen in range(self.generations):
            new_pop = []
            
            # Elite selection
            elite_idxs = sorted(range(len(self.scores)), key=lambda i: self.scores[i], reverse=True)[:n_elite]
            for i in elite_idxs:
                new_pop.append(self.population[i].copy())
            
            # Generate offspring
            while len(new_pop) < self.pop_size:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                child1 = self._mutate(child1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(child1)
                if len(new_pop) < self.pop_size:
                    child2 = self._mutate(child2)
                    new_pop.append(child2)
            
            self.population = new_pop[:self.pop_size]
            self.evaluate_population()
            
            if verbose and ((gen + 1) % max(1, self.generations // 10) == 0):
                logger.info(f"Gen {gen+1}/{self.generations} best_score={self.best_score:.4f}")
        
        result = {
            "best_individual": self.best_individual,
            "best_score": self.best_score,
            "history": self.best_history
        }

        # >>> Added: สรุปผล prediction ก่อนและหลังปรับ + feature changes
        if feature_names is not None and base_row is not None and model is not None:
            base_pred = float(model.predict(base_row)[0])
            row_after = base_row.copy()
            for i, f in enumerate(feature_names):
                row_after.at[row_after.index[0], f] = self.best_individual[i]
            new_pred = float(model.predict(row_after)[0])

            changes = {}
            for i, f in enumerate(feature_names):
                old_val = float(base_row[f].iloc[0])
                new_val = float(self.best_individual[i])
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
        """Decode best solution into continuous and binary parts"""
        if self.best_individual is None:
            return None, None
        return self._split(self.best_individual)







