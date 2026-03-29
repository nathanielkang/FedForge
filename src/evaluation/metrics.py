"""
Evaluation metrics for FedSynth-Engine.

Provides marginal TV distance, workload error, ML utility (TSTR),
throughput measurement, and communication efficiency metrics.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def _safe_probs(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s < 1e-15:
        return np.ones_like(p) / len(p)
    p = p / s
    p = np.clip(p, 0.0, None)
    return p / p.sum()


def compute_histogram(
    data: np.ndarray,
    columns: List[int],
    domain_sizes: List[int],
) -> np.ndarray:
    """Compute a normalized joint histogram over specified columns."""
    shape = tuple(domain_sizes)
    hist = np.zeros(shape, dtype=np.float64)
    n = data.shape[0]

    for row in data:
        idx = tuple(int(row[c]) for c in columns)
        valid = all(0 <= idx[i] < shape[i] for i in range(len(shape)))
        if valid:
            hist[idx] += 1.0

    if n > 0:
        hist = hist / n
    return hist.ravel()


def marginal_tv_distance(
    synth_data: np.ndarray,
    real_data: np.ndarray,
    columns: List[int],
    domain_sizes: List[int],
) -> float:
    """Compute total variation distance between marginals of two datasets."""
    synth_hist = compute_histogram(synth_data, columns, domain_sizes)
    real_hist = compute_histogram(real_data, columns, domain_sizes)

    p = _safe_probs(synth_hist)
    q = _safe_probs(real_hist)
    min_len = min(len(p), len(q))
    return 0.5 * np.sum(np.abs(p[:min_len] - q[:min_len]))


def average_marginal_tv(
    synth_data: np.ndarray,
    real_data: np.ndarray,
    column_names: List[str],
    domain_sizes: Dict[str, int],
    max_way: int = 2,
    num_marginals: int = 50,
    random_state: int = 42,
) -> Tuple[float, Dict[str, float]]:
    """Compute average TV distance across random 2-way marginals."""
    rng = np.random.default_rng(random_state)
    num_cols = len(column_names)
    per_marginal = {}

    for _ in range(num_marginals):
        cols = sorted(rng.choice(num_cols, size=min(max_way, num_cols), replace=False).tolist())
        key = "_".join(str(c) for c in cols)
        if key in per_marginal:
            continue

        ds_list = [domain_sizes[column_names[c]] for c in cols]
        tv = marginal_tv_distance(synth_data, real_data, cols, ds_list)
        per_marginal[key] = tv

    avg = np.mean(list(per_marginal.values())) if per_marginal else 1.0
    return avg, per_marginal


def workload_error(
    synth_data: np.ndarray,
    target_marginals: Dict[str, np.ndarray],
    queries: List,
    weights: Dict[str, float],
    column_names: List[str],
    domain_sizes: Dict[str, int],
) -> float:
    """Compute weighted workload error (L1)."""
    total_err = 0.0
    total_w = 0.0

    for q in queries:
        if q.key not in target_marginals:
            continue
        col_indices = q.columns
        ds_list = [domain_sizes[column_names[c]] for c in col_indices]

        synth_hist = compute_histogram(synth_data, col_indices, ds_list)
        target_hist = target_marginals[q.key]

        min_len = min(len(synth_hist), len(target_hist))
        l1 = np.sum(np.abs(synth_hist[:min_len] - target_hist[:min_len]))

        w = weights.get(q.key, 1.0)
        total_err += w * l1
        total_w += w

    return total_err / max(total_w, 1e-10)


def ml_utility(
    real_data: np.ndarray,
    synth_data: np.ndarray,
    target_col: int = -1,
    classifiers: Optional[List[str]] = None,
    max_rows: int = 10000,
) -> Tuple[float, Dict[str, float]]:
    """Train-on-synthetic, test-on-real ML utility (F1 score)."""
    if classifiers is None:
        classifiers = [
            "logistic_regression", "random_forest", "gradient_boosting",
            "mlp", "decision_tree",
        ]

    if target_col < 0:
        target_col = real_data.shape[1] + target_col

    rng = np.random.default_rng(42)
    n_real = min(len(real_data), max_rows)
    n_synth = min(len(synth_data), max_rows)
    real_sample = real_data[rng.choice(len(real_data), n_real, replace=False)]
    synth_sample = synth_data[rng.choice(len(synth_data), n_synth, replace=False)]

    feat_cols = [i for i in range(real_data.shape[1]) if i != target_col]
    X_train, y_train = synth_sample[:, feat_cols], synth_sample[:, target_col]
    X_test, y_test = real_sample[:, feat_cols], real_sample[:, target_col]

    n_classes = len(np.unique(y_test))
    if n_classes < 2:
        return 0.5, {c: 0.5 for c in classifiers}

    clf_map = {
        "logistic_regression": LogisticRegression(max_iter=500, solver="lbfgs"),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    }

    scores = {}
    for name in classifiers:
        clf = clf_map.get(name)
        if clf is None:
            continue
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            avg = "macro" if n_classes > 2 else "binary"
            scores[name] = f1_score(y_test, y_pred, average=avg, zero_division=0.0)
        except Exception:
            scores[name] = 0.0

    return np.mean(list(scores.values())) if scores else 0.0, scores


def measure_throughput(
    generate_fn,
    num_samples: int = 10000,
) -> Tuple[float, float]:
    """Measure synthesis throughput in records/sec."""
    t0 = time.time()
    generate_fn(num_samples)
    elapsed = time.time() - t0
    throughput = num_samples / max(elapsed, 1e-6)
    return throughput, elapsed


def compute_all_metrics(
    synth_data: np.ndarray,
    real_data: np.ndarray,
    target_marginals: Dict[str, np.ndarray],
    queries: List,
    weights: Dict[str, float],
    column_names: List[str],
    domain_sizes: Dict[str, int],
    evaluate_ml: bool = True,
) -> Dict[str, float]:
    """Compute all evaluation metrics and return as a flat dictionary."""
    avg_tv, _ = average_marginal_tv(
        synth_data, real_data, column_names, domain_sizes
    )

    wl_err = workload_error(
        synth_data, target_marginals, queries, weights, column_names, domain_sizes
    )

    result = {
        "marginal_tv_avg": avg_tv,
        "workload_error": wl_err,
    }

    if evaluate_ml:
        ml_f1, ml_per = ml_utility(real_data, synth_data)
        result["ml_utility_f1"] = ml_f1
        for k, v in ml_per.items():
            result[f"ml_{k}_f1"] = v

    return result
