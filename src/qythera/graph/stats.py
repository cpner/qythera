import numpy as np
from typing import Tuple, Optional, Union


def shannon_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    p = p / p.sum()
    return -np.sum(p * np.log2(p))


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    q = np.clip(q, 1e-15, None)
    return -np.sum(p * np.log(q))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def mutual_information(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.int64)
    Y = np.asarray(Y, dtype=np.int64)
    n = len(X)
    xy = np.column_stack([X, Y])
    xy_unique, xy_counts = np.unique(xy, axis=0, return_counts=True)
    x_unique, x_counts = np.unique(X, return_counts=True)
    y_unique, y_counts = np.unique(Y, return_counts=True)
    p_xy = xy_counts / n
    p_x = x_counts / n
    p_y = y_counts / n
    mi = 0.0
    for i, (xi, yi) in enumerate(xy_unique):
        px = p_x[np.searchsorted(x_unique, xi)]
        py = p_y[np.searchsorted(y_unique, yi)]
        mi += p_xy[i] * np.log(p_xy[i] / (px * py))
    return mi


def bayesian_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    prior = np.asarray(prior, dtype=np.float64)
    likelihood = np.asarray(likelihood, dtype=np.float64)
    posterior = prior * likelihood
    total = posterior.sum()
    if total > 0:
        posterior = posterior / total
    return posterior


def em_algorithm(data: np.ndarray, num_clusters: int, max_iter: int = 100, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=np.float64)
    n, d = data.shape
    rng = np.random.default_rng(42)
    indices = rng.choice(n, num_clusters, replace=False)
    means = data[indices].copy()
    covariances = np.array([np.eye(d) for _ in range(num_clusters)])
    weights = np.ones(num_clusters) / num_clusters

    for iteration in range(max_iter):
        resp = np.zeros((n, num_clusters))
        for k in range(num_clusters):
            diff = data - means[k]
            inv_cov = np.linalg.inv(covariances[k])
            det_cov = np.linalg.det(covariances[k])
            exp_term = np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=1))
            resp[:, k] = weights[k] * exp_term / (np.sqrt((2 * np.pi) ** d * det_cov) + 1e-15)
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum = np.clip(resp_sum, 1e-15, None)
        resp = resp / resp_sum

        Nk = resp.sum(axis=0)
        Nk = np.clip(Nk, 1e-15, None)
        new_means = np.array([resp[:, k] @ data / Nk[k] for k in range(num_clusters)])
        new_covs = np.zeros_like(covariances)
        for k in range(num_clusters):
            diff = data - new_means[k]
            new_covs[k] = (resp[:, k, np.newaxis] * diff).T @ diff / Nk[k]
            new_covs[k] += 1e-6 * np.eye(d)
        new_weights = Nk / n

        if np.max(np.abs(new_means - means)) < tol:
            break

        means = new_means
        covariances = new_covs
        weights = new_weights

    assignments = np.argmax(resp, axis=1)
    return means, covariances, assignments


def bootstrap_ci(data: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 1000, stat_fn: callable = None) -> Tuple[float, float, float]:
    data = np.asarray(data, dtype=np.float64)
    if stat_fn is None:
        stat_fn = np.mean
    rng = np.random.default_rng(42)
    stats = np.array([stat_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)])
    alpha = 1 - confidence
    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return stat_fn(data), lower, upper


def hypothesis_test(sample1: np.ndarray, sample2: np.ndarray, test_type: str = 't_test') -> dict:
    sample1 = np.asarray(sample1, dtype=np.float64)
    sample2 = np.asarray(sample2, dtype=np.float64)

    if test_type == 't_test':
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        pooled_se = np.sqrt(var1 / n1 + var2 / n2)
        if pooled_se == 0:
            t_stat = 0.0
        else:
            t_stat = (mean1 - mean2) / pooled_se
        df = n1 + n2 - 2
        p_value = 2 * (1 - _t_cdf(np.abs(t_stat), df))
        return {
            'test_type': 't_test',
            'statistic': t_stat,
            'p_value': p_value,
            'df': df,
            'means': (mean1, mean2),
            'reject_null': p_value < 0.05
        }

    elif test_type == 'chi_squared':
        observed = np.asarray(sample1, dtype=np.float64)
        expected = np.asarray(sample2, dtype=np.float64)
        expected = expected * observed.sum() / expected.sum()
        chi2 = np.sum((observed - expected) ** 2 / (expected + 1e-15))
        df = len(observed) - 1
        p_value = 1 - _chi2_cdf(chi2, df)
        return {
            'test_type': 'chi_squared',
            'statistic': chi2,
            'p_value': p_value,
            'df': df,
            'reject_null': p_value < 0.05
        }

    else:
        raise ValueError(f"Unknown test type: {test_type}")


def _t_cdf(t: float, df: int) -> float:
    x = df / (df + t ** 2)
    return 1 - 0.5 * _beta_inc(df / 2, 0.5, x)


def _chi2_cdf(x: float, k: int) -> float:
    if x <= 0:
        return 0.0
    return _gamma_inc(k / 2, x / 2) / _gamma(k / 2)


def _gamma_inc(a: float, x: float) -> float:
    if x == 0:
        return 0.0
    total = 0.0
    term = 1.0 / a
    total = term
    for n in range(1, 200):
        term *= x / (a + n)
        total += term
        if abs(term) < 1e-12:
            break
    return total * np.exp(-x + a * np.log(x) - _log_gamma(a))


def _gamma(z: float) -> float:
    return np.exp(_log_gamma(z))


def _log_gamma(z: float) -> float:
    if z < 0.5:
        return np.log(np.pi / np.sin(np.pi * z)) - _log_gamma(1 - z)
    z -= 1
    g = 7
    c = np.array([
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    ])
    x = c[0]
    for i in range(1, g + 2):
        x += c[i] / (z + i)
    t = z + g + 0.5
    return 0.5 * np.log(2 * np.pi) + (z + 0.5) * np.log(t) - t + np.log(x)


def _beta_inc(a: float, b: float, x: float) -> float:
    if x < 0 or x > 1:
        return 0.0
    if x == 0 or x == 1:
        return x
    lbeta = _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)
    front = np.exp(np.log(x) * a + np.log(1 - x) * b - lbeta) / a
    f = 1.0
    c = 1.0
    d = 1 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1 / d
    f = d
    for m in range(1, 200):
        m2 = 2 * m
        even = m * (b - m) * x / ((a + m2 - 2) * (a + m2 - 1))
        d = 1 + even * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1 + even / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        f *= d * c
        odd = -(a + m) * (a + b + m) * x / ((a + m2 - 1) * (a + m2))
        d = 1 + odd * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1 + odd / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        delta = d * c
        f *= delta
        if abs(delta - 1) < 1e-8:
            break
    return front * (f - 1)
