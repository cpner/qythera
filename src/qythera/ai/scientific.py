"""Scientific computing and discovery modules: SymbolicRegression, SINDy, CausalDiscovery, PINN, BayesianOptimization."""

import random
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class SymbolicRegression:
    """Genetic programming for symbolic regression with expression trees."""

    def __init__(self, population_size: int = 100, generations: int = 50,
                 tournament_size: int = 5, crossover_rate: float = 0.7,
                 mutation_rate: float = 0.2, max_depth: int = 6):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_depth = max_depth
        self.operators = ['+', '-', '*', '/']
        self.terminals: List[str] = []
        self.population: List[Dict] = []
        self.best_tree: Optional[Dict] = None
        self.best_fitness: float = float('inf')

    def _random_tree(self, max_depth: int) -> Dict:
        if max_depth <= 1 or (max_depth <= 3 and random.random() < 0.3):
            return {'type': 'var', 'index': random.randint(0, len(self.terminals)-1)}
        op = random.choice(self.operators)
        return {
            'type': 'op',
            'op': op,
            'left': self._random_tree(max_depth - 1),
            'right': self._random_tree(max_depth - 1),
        }

    def _evaluate_tree(self, node: Dict, X: np.ndarray) -> np.ndarray:
        if node['type'] == 'var':
            return X[:, node['index']]
        left = self._evaluate_tree(node['left'], X)
        right = self._evaluate_tree(node['right'], X)
        ops = {'+': lambda a, b: a + b, '-': lambda a, b: a - b,
               '*': lambda a, b: a * b, '/': lambda a, b: np.where(np.abs(b) > 1e-10, a / b, 0.0)}
        return ops[node['op']](left, right)

    def _tournament_select(self) -> Dict:
        candidates = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return min(candidates, key=lambda t: self._fitness(t))

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        child = dict(parent1)
        return child

    def _mutate(self, node: Dict) -> Dict:
        if random.random() < 0.3:
            return self._random_tree(self.max_depth)
        node = dict(node)
        if node['type'] == 'op':
            node['left'] = self._mutate(node['left'])
            node['right'] = self._mutate(node['right'])
        return node

    def _fitness(self, tree: Dict) -> float:
        if not hasattr(self, '_X') or not hasattr(self, '_y'):
            return float('inf')
        pred = self._evaluate_tree(tree, self._X)
        return float(np.mean((pred - self._y) ** 2))

    def _tree_depth(self, node: Dict) -> int:
        if node['type'] == 'var':
            return 1
        return 1 + max(self._tree_depth(node['left']), self._tree_depth(node['right']))

    def fit(self, X: np.ndarray, y: np.ndarray, n_features: int) -> Dict:
        self._X = X
        self._y = y
        self.terminals = [f'x{i}' for i in range(n_features)]
        self.population = [self._random_tree(self.max_depth) for _ in range(self.population_size)]

        for gen in range(self.generations):
            for tree in self.population:
                if self._tree_depth(tree) > self.max_depth:
                    tree = self._random_tree(self.max_depth)
                fit = self._fitness(tree)
                if fit < self.best_fitness:
                    self.best_fitness = fit
                    self.best_tree = tree

            new_pop = [self.best_tree]
            while len(new_pop) < self.population_size:
                if random.random() < self.crossover_rate:
                    p1, p2 = self._tournament_select(), self._tournament_select()
                    child = self._crossover(p1, p2)
                else:
                    child = self._tournament_select()
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                new_pop.append(child)
            self.population = new_pop

        return self.best_tree

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_tree is None:
            raise ValueError("Model not fitted")
        self._X = X
        return self._evaluate_tree(self.best_tree, X)

    def _tree_to_string(self, node: Dict) -> str:
        if node['type'] == 'var':
            return self.terminals[node['index']]
        left = self._tree_to_string(node['left'])
        right = self._tree_to_string(node['right'])
        return f"({left} {node['op']} {right})"

    def get_expression(self) -> str:
        if self.best_tree is None:
            return ""
        return self._tree_to_string(self.best_tree)


class SINDy:
    """Sparse Identification of Nonlinear Dynamics."""

    def __init__(self, threshold: float = 0.1, max_iter: int = 10,
                 feature_names: Optional[List[str]] = None):
        self.threshold = threshold
        self.max_iter = max_iter
        self.feature_names = feature_names
        self.coefficients: Optional[np.ndarray] = None
        self.library_functions: List[Callable] = []

    def _default_library(self) -> List[Callable]:
        return [
            lambda x: np.ones_like(x),
            lambda x: x,
            lambda x: x ** 2,
            lambda x: x ** 3,
            lambda x: np.sin(x),
            lambda x: np.cos(x),
        ]

    def _build_library(self, X: np.ndarray) -> np.ndarray:
        if not self.library_functions:
            self.library_functions = self._default_library()
        lib = []
        for func in self.library_functions:
            lib.append(func(X))
        return np.column_stack(lib)

    def fit(self, X: np.ndarray, dt: float = 0.01) -> np.ndarray:
        n_samples, n_features = X.shape
        dX = np.gradient(X, dt, axis=0)

        Theta = np.ones((n_samples, 1))
        for col in range(n_features):
            lib_col = self._build_library(X[:, col:col+1])
            Theta = np.column_stack([Theta, lib_col])

        Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]

        for _ in range(self.max_iter):
            small = np.abs(Xi) < self.threshold
            Xi[small] = 0
            for j in range(n_features):
                big = ~small[:, j]
                if np.any(big):
                    Xi[:, j] = np.zeros(n_features)
                    Xi[big, j] = np.linalg.lstsq(Theta[:, big], dX[:, j], rcond=None)[0]

        self.coefficients = Xi
        return Xi

    def predict(self, X: np.ndarray, dt: float = 0.01) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("Model not fitted")
        Theta = np.ones((X.shape[0], 1))
        for col in range(X.shape[1]):
            lib_col = self._build_library(X[:, col:col+1])
            Theta = np.column_stack([Theta, lib_col])
        return Theta @ self.coefficients

    def get_equations(self) -> List[str]:
        if self.coefficients is None:
            return []
        equations = []
        lib_names = ['1'] + [f'lib{i}' for i in range(self.coefficients.shape[0]-1)]
        for j in range(self.coefficients.shape[1]):
            terms = []
            for i, coeff in enumerate(self.coefficients[:, j]):
                if abs(coeff) > self.threshold:
                    terms.append(f"{coeff:.4f}*{lib_names[i]}")
            equations.append(' + '.join(terms) if terms else '0')
        return equations


class CausalDiscovery:
    """PC algorithm for causal discovery with v-structures and Meek rules."""

    def __init__(self, alpha: float = 0.05, max_conditioning: int = 3):
        self.alpha = alpha
        self.max_conditioning = max_conditioning
        self.graph: Optional[np.ndarray] = None
        self.data: Optional[np.ndarray] = None

    def _partial_correlation(self, i: int, j: int, conditioning: List[int]) -> float:
        if not conditioning:
            if self.data is None:
                return 0.0
            return np.corrcoef(self.data[:, i], self.data[:, j])[0, 1]

        data = self.data
        n = data.shape[0]
        X = data[:, [i] + conditioning]
        y = data[:, j]
        ones = np.ones((n, 1))
        X_aug = np.hstack([ones, X])
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        resid = y - X_aug @ beta
        Xi = data[:, i]
        Xi_aug = np.hstack([ones, data[:, conditioning]])
        beta_i = np.linalg.lstsq(Xi_aug, Xi, rcond=None)[0]
        resid_i = Xi - Xi_aug @ beta_i
        if np.std(resid) < 1e-10 or np.std(resid_i) < 1e-10:
            return 0.0
        return np.corrcoef(resid, resid_i)[0, 1]

    def _fisher_z_test(self, i: int, j: int, conditioning: List[int]) -> bool:
        r = self._partial_correlation(i, j, conditioning)
        n = self.data.shape[0]
        k = len(conditioning)
        z = 0.5 * np.log(np.abs((1 + r) / (1 - r) + 1e-10))
        se = 1.0 / np.sqrt(n - k - 3)
        return abs(z / se) > 1.96

    def _skeleton_learning(self, n_vars: int) -> np.ndarray:
        graph = np.ones((n_vars, n_vars), dtype=int)
        np.fill_diagonal(graph, 0)
        sep_set: Dict[Tuple[int, int], List[int]] = {}

        for cond_size in range(0, self.max_conditioning + 1):
            edges = list(zip(*np.where(graph == 1)))
            for i, j in edges:
                if i >= j:
                    continue
                neighbors = [k for k in range(n_vars) if graph[i, k] == 1 and k != j]
                for subset in self._subsets(neighbors, cond_size):
                    if not self._fisher_z_test(i, j, list(subset)):
                        graph[i, j] = 0
                        graph[j, i] = 0
                        sep_set[(i, j)] = list(subset)
                        sep_set[(j, i)] = list(subset)
                        break
        return graph

    def _subsets(self, lst: List[int], size: int) -> List[Tuple]:
        if size == 0:
            return [()]
        if size > len(lst):
            return []
        result = []
        for i, val in enumerate(lst):
            rest = self._subsets(lst[i+1:], size - 1)
            for sub in rest:
                result.append((val,) + sub)
        return result

    def _orient_v_structures(self, graph: np.ndarray, sep_set: Dict) -> np.ndarray:
        directed = graph.copy()
        n_vars = graph.shape[0]
        for j in range(n_vars):
            neighbors = [i for i in range(n_vars) if graph[i, j] == 1]
            for i_idx in range(len(neighbors)):
                for k_idx in range(i_idx + 1, len(neighbors)):
                    i, k = neighbors[i_idx], neighbors[k_idx]
                    if graph[i, k] == 0 and (i, k) not in sep_set:
                        directed[i, j] = -1
                        directed[j, i] = 1
                        directed[k, j] = -1
                        directed[j, k] = 1
        return directed

    def _apply_meek_rules(self, directed: np.ndarray) -> np.ndarray:
        changed = True
        while changed:
            changed = False
            n = directed.shape[0]
            for i in range(n):
                for j in range(n):
                    if directed[i, j] == 1:
                        for k in range(n):
                            if directed[j, k] == 0 and directed[i, k] == 1:
                                directed[j, k] = -1
                                directed[k, j] = 1
                                changed = True
        return directed

    def fit(self, data: np.ndarray) -> np.ndarray:
        self.data = data
        n_vars = data.shape[1]
        skeleton = self._skeleton_learning(n_vars)
        sep_set: Dict = {}
        directed = self._orient_v_structures(skeleton, sep_set)
        self.graph = self._apply_meek_rules(directed)
        return self.graph

    def get_edges(self) -> List[Tuple[int, int, str]]:
        if self.graph is None:
            return []
        edges = []
        for i in range(self.graph.shape[0]):
            for j in range(i + 1, self.graph.shape[1]):
                if self.graph[i, j] == 1:
                    edges.append((i, j, 'undirected'))
                elif self.graph[i, j] == -1:
                    edges.append((j, i, 'directed'))
                elif self.graph[j, i] == -1:
                    edges.append((i, j, 'directed'))
        return edges


class PINN:
    """Physics-Informed Neural Network with PDE residual in loss."""

    def __init__(self, hidden_layers: List[int] = None, activation: str = 'tanh',
                 pde_residual_fn: Optional[Callable] = None):
        self.hidden_layers = hidden_layers or [20, 20, 20]
        self.activation = activation
        self.pde_residual_fn = pde_residual_fn
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._initialized = False

    def _init_weights(self, input_dim: int, output_dim: int):
        self.weights = []
        self.biases = []
        layers = [input_dim] + self.hidden_layers + [output_dim]
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros(layers[i+1])
            self.weights.append(w)
            self.biases.append(b)
        self._initialized = True

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        return np.sin(x)

    def _activate_deriv(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'tanh':
            return 1.0 - np.tanh(x) ** 2
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        return np.cos(x)

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w + b
            if i < len(self.weights) - 1:
                h = self._activate(z)
            else:
                h = z
        return h

    def _compute_pde_residual(self, X: np.ndarray) -> np.ndarray:
        if self.pde_residual_fn is not None:
            return self.pde_residual_fn(self, X)
        return np.zeros(X.shape[0])

    def loss(self, X_data: np.ndarray, y_data: np.ndarray,
             X_pde: Optional[np.ndarray] = None, lambda_pde: float = 1.0) -> float:
        y_pred = self.forward(X_data)
        data_loss = float(np.mean((y_pred - y_data) ** 2))

        pde_loss = 0.0
        if X_pde is not None and self.pde_residual_fn is not None:
            residual = self._compute_pde_residual(X_pde)
            pde_loss = float(np.mean(residual ** 2))

        return data_loss + lambda_pde * pde_loss

    def fit(self, X_data: np.ndarray, y_data: np.ndarray,
            X_pde: Optional[np.ndarray] = None, epochs: int = 1000,
            lr: float = 0.001, lambda_pde: float = 1.0) -> List[float]:
        input_dim = X_data.shape[1]
        output_dim = y_data.shape[1] if y_data.ndim > 1 else 1
        if y_data.ndim == 1:
            y_data = y_data.reshape(-1, 1)
        self._init_weights(input_dim, output_dim)

        losses = []
        for epoch in range(epochs):
            l = self.loss(X_data, y_data, X_pde, lambda_pde)
            losses.append(l)

            h = X_data
            activations = [h]
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                z = h @ w + b
                if i < len(self.weights) - 1:
                    h = self._activate(z)
                else:
                    h = z
                activations.append(h)

            error = 2.0 * (activations[-1] - y_data) / X_data.shape[0]
            for i in range(len(self.weights) - 1, -1, -1):
                if i < len(self.weights) - 1:
                    error = error * self._activate_deriv(activations[i+1])
                dw = activations[i].T @ error
                db = np.mean(error, axis=0)
                self.weights[i] -= lr * dw
                self.biases[i] -= lr * db
                error = error @ self.weights[i].T

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


class BayesianOptimization:
    """Bayesian optimization with surrogate model for parameter search."""

    def __init__(self, bounds: List[Tuple[float, float]], n_initial: int = 5,
                 n_iterations: int = 25, acquisition: str = 'ei'):
        self.bounds = bounds
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.acquisition = acquisition
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        self.best_x: Optional[np.ndarray] = None
        self.best_y: float = float('inf')

    def _sample_prior(self) -> np.ndarray:
        return np.array([np.random.uniform(lo, hi) for lo, hi in self.bounds])

    def _fit_gpr(self, X: np.ndarray, y: np.ndarray, length_scale: float = 1.0):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                diff = (X[i] - X[j]) / length_scale
                K[i, j] = np.exp(-0.5 * np.sum(diff ** 2))
        K += 1e-6 * np.eye(n)
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return K, alpha, L

    def _predict_gpr(self, K: np.ndarray, alpha: np.ndarray, X: np.ndarray,
                     X_new: np.ndarray, length_scale: float = 1.0) -> Tuple[float, float]:
        k_star = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            diff = (X[i] - X_new) / length_scale
            k_star[i] = np.exp(-0.5 * np.sum(diff ** 2))
        k_star_star = 1.0
        mu = k_star @ alpha
        v = np.linalg.solve(K, k_star)
        sigma = np.sqrt(max(k_star_star - k_star @ v, 1e-10))
        return mu, sigma

    def _expected_improvement(self, mu: float, sigma: float) -> float:
        if sigma < 1e-10:
            return 0.0
        z = (self.best_y - mu) / sigma
        ei = (self.best_y - mu) * (1.0 - 0.5 * (1.0 + math.erf(z / np.sqrt(2)))) + \
             sigma * (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z ** 2)
        return ei

    def _upper_confidence_bound(self, mu: float, sigma: float, kappa: float = 2.0) -> float:
        return mu - kappa * sigma

    def _select_next(self, K: np.ndarray, alpha: np.ndarray, X_obs: np.ndarray) -> np.ndarray:
        best_x = None
        best_acq = float('-inf')
        for _ in range(1000):
            x_cand = self._sample_prior()
            mu, sigma = self._predict_gpr(K, alpha, X_obs, x_cand)
            if self.acquisition == 'ei':
                acq = self._expected_improvement(mu, sigma)
            else:
                acq = self._upper_confidence_bound(mu, sigma)
            if acq > best_acq:
                best_acq = acq
                best_x = x_cand
        return best_x

    def maximize(self, objective_fn: Callable) -> Tuple[np.ndarray, float]:
        for _ in range(self.n_initial):
            x = self._sample_prior()
            y = objective_fn(x)
            self.X_observed.append(x)
            self.y_observed.append(y)
            if y < self.best_y:
                self.best_y = y
                self.best_x = x.copy()

        for _ in range(self.n_iterations):
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            K, alpha, L = self._fit_gpr(X, y)
            x_next = self._select_next(K, alpha, X)
            y_next = objective_fn(x_next)
            self.X_observed.append(x_next)
            self.y_observed.append(y_next)
            if y_next < self.best_y:
                self.best_y = y_next
                self.best_x = x_next.copy()

        return self.best_x, self.best_y

    def minimize(self, objective_fn: Callable) -> Tuple[np.ndarray, float]:
        return self.maximize(lambda x: -objective_fn(x))
