"""Probabilistic models: NaiveBayes, HiddenMarkovModel, KalmanFilter, VariationalAutoencoder, DiffusionModel, ConditionalRandomField."""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class NaiveBayes:
    """Naive Bayes classifier with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_priors: Dict[Any, float] = {}
        self.class_likelihoods: Dict[Any, Dict[int, Dict[Any, float]]] = {}
        self.classes: List[Any] = []

    def fit(self, X: List[List[Any]], y: List[Any]) -> None:
        self.classes = list(set(y))
        n_samples = len(y)

        for c in self.classes:
            self.class_priors[c] = sum(1 for yi in y if yi == c) / n_samples
            self.class_likelihoods[c] = {}
            indices = [i for i, yi in enumerate(y) if yi == c]
            n_features = len(X[0]) if X else 0

            for j in range(n_features):
                values = set(X[i][j] for i in indices)
                value_counts = {}
                for v in values:
                    value_counts[v] = sum(1 for i in indices if X[i][j] == v)
                total = len(indices) + self.alpha * len(values)
                self.class_likelihoods[c][j] = {}
                for v in values:
                    self.class_likelihoods[c][j][v] = (value_counts[v] + self.alpha) / total

    def predict_proba(self, x: List[Any]) -> Dict[Any, float]:
        log_probs = {}
        for c in self.classes:
            log_p = math.log(self.class_priors[c])
            for j, val in enumerate(x):
                if val in self.class_likelihoods.get(c, {}).get(j, {}):
                    log_p += math.log(self.class_likelihoods[c][j][val])
            log_probs[c] = log_p

        max_log = max(log_probs.values())
        exp_probs = {c: math.exp(p - max_log) for c, p in log_probs.items()}
        total = sum(exp_probs.values())
        return {c: p / total for c, p in exp_probs.items()}

    def predict(self, X: List[List[Any]]) -> List[Any]:
        return [max(self.predict_proba(x), key=self.predict_proba(x).get) for x in X]


class HiddenMarkovModel:
    """Hidden Markov Model with Viterbi decoding and Baum-Welch training."""

    def __init__(self, n_states: int, n_observations: int):
        self.n_states = n_states
        self.n_observations = n_observations
        self.pi: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None
        self.log_A: Optional[np.ndarray] = None
        self.log_B: Optional[np.ndarray] = None
        self.log_pi: Optional[np.ndarray] = None

    def _initialize(self) -> None:
        self.pi = np.ones(self.n_states) / self.n_states
        self.A = np.ones((self.n_states, self.n_states)) / self.n_states
        self.B = np.ones((self.n_states, self.n_observations)) / self.n_observations
        eps = 1e-30
        self.log_pi = np.log(self.pi + eps)
        self.log_A = np.log(self.A + eps)
        self.log_B = np.log(self.B + eps)

    def _log_normalize(self, log_vec: np.ndarray) -> np.ndarray:
        max_val = np.max(log_vec)
        log_sum = max_val + np.log(np.sum(np.exp(log_vec - max_val)))
        return log_vec - log_sum

    def _forward(self, obs: List[int]) -> np.ndarray:
        T = len(obs)
        log_alpha = np.zeros((T, self.n_states))
        log_alpha[0] = self.log_pi + self.log_B[:, obs[0]]

        for t in range(1, T):
            for j in range(self.n_states):
                log_alpha[t, j] = self._log_sum_exp(log_alpha[t-1] + self.log_A[:, j]) + self.log_B[j, obs[t]]

        return log_alpha

    def _backward(self, obs: List[int]) -> np.ndarray:
        T = len(obs)
        log_beta = np.zeros((T, self.n_states))
        log_beta[T-1] = 0.0

        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = self._log_sum_exp(self.log_A[i, :] + self.log_B[:, obs[t+1]] + log_beta[t+1])

        return log_beta

    def _log_sum_exp(self, log_vec: np.ndarray) -> float:
        max_val = np.max(log_vec)
        return max_val + np.log(np.sum(np.exp(log_vec - max_val)))

    def viterbi(self, obs: List[int]) -> List[int]:
        if self.pi is None:
            self._initialize()
        T = len(obs)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        delta[0] = self.log_pi + self.log_B[:, obs[0]]

        for t in range(1, T):
            for j in range(self.n_states):
                candidates = delta[t-1] + self.log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + self.log_B[j, obs[t]]

        states = [int(np.argmax(delta[T-1]))]
        for t in range(T-2, -1, -1):
            states.insert(0, int(psi[t+1, states[0]]))
        return states

    def baum_welch(self, sequences: List[List[int]], max_iter: int = 100) -> float:
        self._initialize()
        log_likelihoods = []

        for iteration in range(max_iter):
            log_pi_num = np.full(self.n_states, -np.inf)
            log_A_num = np.full((self.n_states, self.n_states), -np.inf)
            log_B_num = np.full((self.n_states, self.n_observations), -np.inf)
            total_log_prob = 0.0

            for obs in sequences:
                T = len(obs)
                log_alpha = self._forward(obs)
                log_beta = self._backward(obs)
                log_prob = self._log_sum_exp(log_alpha[T-1])
                total_log_prob += log_prob

                log_gamma = log_alpha + log_beta - log_prob
                for t in range(T):
                    gamma = np.exp(log_gamma[t])
                    log_pi_num = np.logaddexp(log_pi_num, log_gamma[0] if t == 0 else log_pi_num)
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            log_xi = (log_alpha[t, i] + self.log_A[i, j] +
                                     self.log_B[j, obs[t+1]] + log_beta[t+1, j] - log_prob if t < T-1 else -np.inf)
                            log_A_num[i, j] = np.logaddexp(log_A_num[i, j], log_xi)
                        log_B_num[i, obs[t]] = np.logaddexp(log_B_num[i, obs[t]], log_gamma[t])

            self.log_pi = self._log_normalize(log_pi_num)
            self.log_A = np.zeros_like(log_A_num)
            for i in range(self.n_states):
                self.log_A[i, :] = self._log_normalize(log_A_num[i, :])
            self.log_B = np.zeros_like(log_B_num)
            for i in range(self.n_states):
                self.log_B[i, :] = self._log_normalize(log_B_num[i, :])

            avg_ll = total_log_prob / len(sequences)
            log_likelihoods.append(avg_ll)

        return log_likelihoods[-1] if log_likelihoods else 0.0


class KalmanFilter:
    """Kalman Filter with predict/update cycle."""

    def __init__(self, dim_state: int, dim_obs: int):
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        self.F: Optional[np.ndarray] = None
        self.H: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None

    def initialize(self, x0: np.ndarray, P0: np.ndarray,
                   F: np.ndarray, H: np.ndarray,
                   Q: np.ndarray, R: np.ndarray) -> None:
        self.x = x0.copy()
        self.P = P0.copy()
        self.F = F.copy()
        self.H = H.copy()
        self.Q = Q.copy()
        self.R = R.copy()

    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.F is None or self.x is None or self.P is None:
            raise ValueError("Filter not initialized")
        self.x = self.F @ self.x
        if u is not None:
            self.x += u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.H is None or self.x is None or self.P is None:
            raise ValueError("Filter not initialized")
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.dim_state)
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy(), self.P.copy()

    def step(self, z: np.ndarray, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        self.predict(u)
        return self.update(z)

    def filter_sequence(self, observations: List[np.ndarray],
                        controls: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        estimates = []
        for t, z in enumerate(observations):
            u = controls[t] if controls is not None else None
            x_est, _ = self.step(z, u)
            estimates.append(x_est)
        return estimates


class VariationalAutoencoder:
    """Variational Autoencoder with reparameterization trick."""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 latent_dim: int = 32, lr: float = 0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self._init_weights()

    def _init_weights(self) -> None:
        scale = 0.01
        self.W_enc1 = np.random.randn(self.input_dim, self.hidden_dim) * scale
        self.b_enc1 = np.zeros(self.hidden_dim)
        self.W_enc_mu = np.random.randn(self.hidden_dim, self.latent_dim) * scale
        self.b_enc_mu = np.zeros(self.latent_dim)
        self.W_enc_logvar = np.random.randn(self.hidden_dim, self.latent_dim) * scale
        self.b_enc_logvar = np.zeros(self.latent_dim)

        self.W_dec1 = np.random.randn(self.latent_dim, self.hidden_dim) * scale
        self.b_dec1 = np.zeros(self.hidden_dim)
        self.W_dec2 = np.random.randn(self.hidden_dim, self.input_dim) * scale
        self.b_dec2 = np.zeros(self.input_dim)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = self._relu(x @ self.W_enc1 + self.b_enc1)
        mu = h @ self.W_enc_mu + self.b_enc_mu
        logvar = h @ self.W_enc_logvar + self.b_enc_logvar
        return mu, logvar

    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps

    def decode(self, z: np.ndarray) -> np.ndarray:
        h = self._relu(z @ self.W_dec1 + self.b_dec1)
        x_recon = self._sigmoid(h @ self.W_dec2 + self.b_dec2)
        return x_recon

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss(self, x: np.ndarray) -> Tuple[float, float, float]:
        x_recon, mu, logvar = self.forward(x)
        recon = -np.mean(np.sum(x * np.log(x_recon + 1e-10) + (1-x) * np.log(1-x_recon + 1e-10), axis=1))
        kl = -0.5 * np.mean(np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=1))
        return recon + kl, recon, kl

    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32) -> List[float]:
        n_samples = X.shape[0]
        losses = []
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                batch = X[indices[start:start+batch_size]]
                total_loss, _, _ = self.loss(batch)
                epoch_loss += total_loss
                n_batches += 1
                self._update_weights(batch)
            losses.append(epoch_loss / n_batches)
        return losses

    def _update_weights(self, x: np.ndarray) -> None:
        eps = 1e-7
        x_recon, mu, logvar = self.forward(x)
        z = self.reparameterize(mu, logvar)

        grad_x_recon = (x_recon - x) / x.shape[0]
        grad_h2 = grad_x_recon @ self.W_dec2.T
        grad_W_dec2 = self._relu(z @ self.W_dec1 + self.b_dec1).T @ grad_x_recon
        grad_b_dec2 = np.mean(grad_x_recon, axis=0)

        grad_h1 = grad_h2 * (z @ self.W_dec1 + self.b_dec1 > 0).astype(float)
        grad_W_dec1 = z.T @ grad_h1
        grad_b_dec1 = np.mean(grad_h1, axis=0)

        grad_z = grad_h1 @ self.W_dec1.T
        grad_mu = grad_z + mu
        grad_logvar = grad_z * np.exp(0.5 * logvar) * 0.5 - 0.5 * np.exp(logvar)

        h_enc = self._relu(x @ self.W_enc1 + self.b_enc1)
        grad_h_enc = (grad_mu @ self.W_enc_mu.T + grad_logvar @ self.W_enc_logvar.T) * (h_enc > 0).astype(float)
        grad_W_enc1 = x.T @ grad_h_enc
        grad_b_enc1 = np.mean(grad_h_enc, axis=0)
        grad_W_enc_mu = h_enc.T @ grad_mu
        grad_b_enc_mu = np.mean(grad_mu, axis=0)
        grad_W_enc_logvar = h_enc.T @ grad_logvar
        grad_b_enc_logvar = np.mean(grad_logvar, axis=0)

        self.W_dec2 -= self.lr * grad_W_dec2
        self.b_dec2 -= self.lr * grad_b_dec2
        self.W_dec1 -= self.lr * grad_W_dec1
        self.b_dec1 -= self.lr * grad_b_dec1
        self.W_enc1 -= self.lr * grad_W_enc1
        self.b_enc1 -= self.lr * grad_b_enc1
        self.W_enc_mu -= self.lr * grad_W_enc_mu
        self.b_enc_mu -= self.lr * grad_b_enc_mu
        self.W_enc_logvar -= self.lr * grad_W_enc_logvar
        self.b_enc_logvar -= self.lr * grad_b_enc_logvar

    def generate(self, n_samples: int = 1) -> np.ndarray:
        z = np.random.randn(n_samples, self.latent_dim)
        return self.decode(z)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        x_recon, _, _ = self.forward(x)
        return x_recon


class DiffusionModel:
    """Diffusion model with forward noise and reverse denoising."""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 n_timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_timesteps = n_timesteps
        self.betas = np.linspace(beta_start, beta_end, n_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)
        self._init_weights()

    def _init_weights(self) -> None:
        scale = 0.01
        self.W1 = np.random.randn(self.input_dim + 1, self.hidden_dim) * scale
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.hidden_dim) * scale
        self.b2 = np.zeros(self.hidden_dim)
        self.W3 = np.random.randn(self.hidden_dim, self.input_dim) * scale
        self.b3 = np.zeros(self.input_dim)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _noise_predict(self, x_noisy: np.ndarray, t: int) -> np.ndarray:
        t_emb = np.full((x_noisy.shape[0], 1), t / self.n_timesteps)
        h = np.hstack([x_noisy, t_emb])
        h = self._relu(h @ self.W1 + self.b1)
        h = self._relu(h @ self.W2 + self.b2)
        return h @ self.W3 + self.b3

    def forward_diffusion(self, x0: np.ndarray, t: int,
                          noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if noise is None:
            noise = np.random.randn(*x0.shape)
        alpha_bar_t = self.alpha_bars[t]
        xt = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1 - alpha_bar_t) * noise
        return xt, noise

    def reverse_step(self, xt: np.ndarray, t: int) -> np.ndarray:
        pred_noise = self._noise_predict(xt, t)
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        mean = (1.0 / np.sqrt(alpha_t)) * (xt - (1 - alpha_t) / np.sqrt(1 - alpha_bar_t) * pred_noise)
        if t > 0:
            noise = np.random.randn(*xt.shape)
            sigma = np.sqrt(self.betas[t])
            mean = mean + sigma * noise
        return mean

    def denoise(self, xt: np.ndarray, t_start: Optional[int] = None) -> np.ndarray:
        if t_start is None:
            t_start = self.n_timesteps - 1
        x = xt.copy()
        for t in range(t_start, -1, -1):
            x = self.reverse_step(x, t)
        return x

    def sample(self, n_samples: int = 1) -> np.ndarray:
        x = np.random.randn(n_samples, self.input_dim)
        return self.denoise(x)

    def training_loss(self, x0: np.ndarray) -> float:
        batch_size = x0.shape[0]
        t = np.random.randint(0, self.n_timesteps, size=batch_size)
        noise = np.random.randn(*x0.shape)
        losses = []
        for i in range(batch_size):
            alpha_bar_t = self.alpha_bars[t[i]]
            xt = np.sqrt(alpha_bar_t) * x0[i] + np.sqrt(1 - alpha_bar_t) * noise[i]
            pred_noise = self._noise_predict(xt[i:i+1], t[i])
            loss = np.mean((noise[i] - pred_noise) ** 2)
            losses.append(loss)
        return float(np.mean(losses))

    def fit(self, X: np.ndarray, epochs: int = 100, lr: float = 0.001,
            batch_size: int = 32) -> List[float]:
        n_samples = X.shape[0]
        losses = []
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                batch = X[indices[start:start+batch_size]]
                loss = self.training_loss(batch)
                epoch_loss += loss
                n_batches += 1
            losses.append(epoch_loss / n_batches)
        return losses


class ConditionalRandomField:
    """Conditional Random Field for sequence labeling with pairwise potentials."""

    def __init__(self, n_states: int, n_features: int, l2_reg: float = 0.01):
        self.n_states = n_states
        self.n_features = n_features
        self.l2_reg = l2_reg
        self.W: Optional[np.ndarray] = None
        self.T: Optional[np.ndarray] = None

    def _initialize(self) -> None:
        self.W = np.random.randn(self.n_features, self.n_states) * 0.01
        self.T = np.random.randn(self.n_states, self.n_states) * 0.01

    def _log_normalize(self, log_vec: np.ndarray) -> np.ndarray:
        max_val = np.max(log_vec)
        return log_vec - max_val - np.log(np.sum(np.exp(log_vec - max_val)))

    def _log_sum_exp(self, log_vec: np.ndarray) -> float:
        max_val = np.max(log_vec)
        return max_val + np.log(np.sum(np.exp(log_vec - max_val)))

    def _compute_potentials(self, X: List[np.ndarray]) -> np.ndarray:
        T = len(X)
        pot = np.zeros((T, self.n_states))
        for t in range(T):
            pot[t] = X[t] @ self.W
        return pot

    def _forward(self, X: List[np.ndarray]) -> np.ndarray:
        T = len(X)
        pot = self._compute_potentials(X)
        alpha = np.zeros((T, self.n_states))
        alpha[0] = pot[0]
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = self._log_sum_exp(alpha[t-1] + self.T[:, j]) + pot[t, j]
        return alpha

    def _backward(self, X: List[np.ndarray]) -> np.ndarray:
        T = len(X)
        pot = self._compute_potentials(X)
        beta = np.zeros((T, self.n_states))
        beta[T-1] = 0.0
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = self._log_sum_exp(self.T[i, :] + pot[t+1, :] + beta[t+1])
        return beta

    def _log_partition(self, X: List[np.ndarray]) -> float:
        alpha = self._forward(X)
        return self._log_sum_exp(alpha[-1])

    def _compute_marginals(self, X: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        alpha = self._forward(X)
        beta = self._backward(X)
        T = len(X)
        log_Z = self._log_sum_exp(alpha[-1])

        unary = np.zeros((T, self.n_states))
        pairwise = np.zeros((T-1, self.n_states, self.n_states))
        pot = self._compute_potentials(X)

        for t in range(T):
            log_gamma = alpha[t] + beta[t] - log_Z
            unary[t] = np.exp(log_gamma)

        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_xi = (alpha[t, i] + self.T[i, j] +
                             pot[t+1, j] + beta[t+1, j] - log_Z)
                    pairwise[t, i, j] = np.exp(log_xi)

        return unary, pairwise

    def loss(self, X: List[np.ndarray], y: List[int]) -> float:
        log_Z = self._log_partition(X)
        pot = self._compute_potentials(X)
        score = sum(pot[t, y[t]] for t in range(len(y)))
        for t in range(len(y)-1):
            score += self.T[y[t], y[t+1]]
        l2 = self.l2_reg * (np.sum(self.W**2) + np.sum(self.T**2))
        return -score + log_Z + l2

    def fit(self, sequences: List[Tuple[List[np.ndarray], List[int]]],
            epochs: int = 100, lr: float = 0.01) -> List[float]:
        self._initialize()
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            grad_W = np.zeros_like(self.W)
            grad_T = np.zeros_like(self.T)

            for X, y in sequences:
                loss = self.loss(X, y)
                epoch_loss += loss

                unary, pairwise = self._compute_marginals(X)
                pot = self._compute_potentials(X)

                for t in range(len(y)):
                    grad_W += X[t][:, np.newaxis] * (unary[t] - np.eye(self.n_states)[y[t]])
                    for i in range(self.n_states):
                        grad_W[:, i] += X[t] * unary[t, i]

                for t in range(len(y)-1):
                    grad_T[y[t], y[t+1]] -= 1
                    grad_T += pairwise[t]

            grad_W += 2 * self.l2_reg * self.W
            grad_T += 2 * self.l2_reg * self.T
            self.W -= lr * grad_W / len(sequences)
            self.T -= lr * grad_T / len(sequences)
            losses.append(epoch_loss / len(sequences))

        return losses

    def viterbi(self, X: List[np.ndarray]) -> List[int]:
        T = len(X)
        pot = self._compute_potentials(X)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        delta[0] = pot[0]

        for t in range(1, T):
            for j in range(self.n_states):
                candidates = delta[t-1] + self.T[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + pot[t, j]

        states = [int(np.argmax(delta[T-1]))]
        for t in range(T-2, -1, -1):
            states.insert(0, int(psi[t+1, states[0]]))
        return states

    def predict(self, X: List[np.ndarray]) -> List[int]:
        return self.viterbi(X)
