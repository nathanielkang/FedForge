"""
Quality Monitor for FedSynth-Engine.

Continuously validates synthetic data fidelity against the workload queries.
Reports marginal TV distance, workload error, and ML utility metrics.
Supports quality gates for pipeline retry decisions.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


def _safe_probs(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s < 1e-15:
        return np.ones_like(p) / len(p)
    p = p / s
    p = np.clip(p, 0.0, None)
    return p / p.sum()


@dataclass
class QualityReport:
    """Structured report from quality validation."""
    marginal_tv_avg: float
    marginal_tv_per_query: Dict[str, float]
    workload_error: float
    ml_utility: Optional[float] = None
    ml_utility_per_classifier: Optional[Dict[str, float]] = None
    passed: bool = True
    details: str = ""


class QualityMonitor:
    """
    Validates synthetic data quality against target marginals and workload.
    Supports configurable quality thresholds for pipeline gate decisions.
    """

    def __init__(
        self,
        tv_threshold: float = 0.1,
        workload_error_threshold: float = 0.15,
        ml_utility_threshold: float = 0.5,
    ):
        self.tv_threshold = tv_threshold
        self.workload_error_threshold = workload_error_threshold
        self.ml_utility_threshold = ml_utility_threshold

    def compute_marginal_histogram(
        self,
        data: np.ndarray,
        columns: List[int],
        domain_sizes: List[int],
    ) -> np.ndarray:
        """Compute a normalized joint histogram."""
        shape = tuple(domain_sizes)
        hist = np.zeros(shape, dtype=np.float64)
        n = data.shape[0]

        for row in data:
            idx = tuple(int(row[c]) for c in columns)
            valid = True
            for i, s in enumerate(shape):
                if idx[i] < 0 or idx[i] >= s:
                    valid = False
                    break
            if valid:
                hist[idx] += 1.0

        if n > 0:
            hist = hist / n

        return hist.ravel()

    def marginal_tv_distance(
        self,
        synthetic_hist: np.ndarray,
        target_hist: np.ndarray,
    ) -> float:
        """Total variation distance between two marginal distributions."""
        p = _safe_probs(synthetic_hist)
        q = _safe_probs(target_hist)

        if p.shape != q.shape:
            min_len = min(len(p), len(q))
            p = p[:min_len]
            q = q[:min_len]
            p = _safe_probs(p)
            q = _safe_probs(q)

        return 0.5 * np.sum(np.abs(p - q))

    def evaluate_marginals(
        self,
        synthetic_data: np.ndarray,
        target_marginals: Dict[str, np.ndarray],
        queries: List,
        column_names: List[str],
        domain_sizes: Dict[str, int],
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate marginal TV distance for all workload queries."""
        tv_per_query = {}

        for q in queries:
            if q.key not in target_marginals:
                continue

            col_indices = q.columns
            ds_list = [domain_sizes[column_names[c]] for c in col_indices]

            synth_hist = self.compute_marginal_histogram(
                synthetic_data, col_indices, ds_list
            )
            target_hist = target_marginals[q.key]

            tv = self.marginal_tv_distance(synth_hist, target_hist)
            tv_per_query[q.key] = tv

        avg_tv = np.mean(list(tv_per_query.values())) if tv_per_query else 1.0
        return avg_tv, tv_per_query

    def evaluate_workload_error(
        self,
        synthetic_data: np.ndarray,
        target_marginals: Dict[str, np.ndarray],
        queries: List,
        weights: Dict[str, float],
        column_names: List[str],
        domain_sizes: Dict[str, int],
    ) -> float:
        """Compute weighted workload error (L1)."""
        total_error = 0.0
        total_weight = 0.0

        for q in queries:
            if q.key not in target_marginals:
                continue

            col_indices = q.columns
            ds_list = [domain_sizes[column_names[c]] for c in col_indices]

            synth_hist = self.compute_marginal_histogram(
                synthetic_data, col_indices, ds_list
            )
            target_hist = target_marginals[q.key]

            min_len = min(len(synth_hist), len(target_hist))
            l1 = np.sum(np.abs(synth_hist[:min_len] - target_hist[:min_len]))

            w = weights.get(q.key, 1.0)
            total_error += w * l1
            total_weight += w

        return total_error / max(total_weight, 1e-10)

    def evaluate_ml_utility(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        target_col: int = -1,
        classifiers: Optional[List[str]] = None,
        max_rows: int = 10000,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Train-on-synthetic, test-on-real ML utility evaluation.
        Reports average F1 across multiple classifiers.
        """
        if classifiers is None:
            classifiers = [
                "logistic_regression", "random_forest", "gradient_boosting",
                "mlp", "decision_tree",
            ]

        if target_col < 0:
            target_col = real_data.shape[1] + target_col

        n_real = min(len(real_data), max_rows)
        n_synth = min(len(synthetic_data), max_rows)

        rng = np.random.default_rng(42)
        real_sample = real_data[rng.choice(len(real_data), n_real, replace=False)]
        synth_sample = synthetic_data[rng.choice(len(synthetic_data), n_synth, replace=False)]

        feature_cols = [i for i in range(real_data.shape[1]) if i != target_col]

        X_train = synth_sample[:, feature_cols]
        y_train = synth_sample[:, target_col]
        X_test = real_sample[:, feature_cols]
        y_test = real_sample[:, target_col]

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

        f1_scores = {}
        for name in classifiers:
            clf = clf_map.get(name)
            if clf is None:
                continue
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                avg = "macro" if n_classes > 2 else "binary"
                f1 = f1_score(y_test, y_pred, average=avg, zero_division=0.0)
                f1_scores[name] = f1
            except Exception as e:
                logger.warning(f"Classifier {name} failed: {e}")
                f1_scores[name] = 0.0

        avg_f1 = np.mean(list(f1_scores.values())) if f1_scores else 0.0
        return avg_f1, f1_scores

    def full_evaluation(
        self,
        synthetic_data: np.ndarray,
        real_data: np.ndarray,
        target_marginals: Dict[str, np.ndarray],
        queries: List,
        weights: Dict[str, float],
        column_names: List[str],
        domain_sizes: Dict[str, int],
        evaluate_ml: bool = True,
    ) -> QualityReport:
        """Run complete quality evaluation and produce a report."""
        avg_tv, tv_per_query = self.evaluate_marginals(
            synthetic_data, target_marginals, queries, column_names, domain_sizes
        )

        wl_error = self.evaluate_workload_error(
            synthetic_data, target_marginals, queries, weights, column_names, domain_sizes
        )

        ml_f1 = None
        ml_per_clf = None
        if evaluate_ml:
            ml_f1, ml_per_clf = self.evaluate_ml_utility(real_data, synthetic_data)

        passed = (
            avg_tv <= self.tv_threshold
            and wl_error <= self.workload_error_threshold
        )
        if ml_f1 is not None:
            passed = passed and (ml_f1 >= self.ml_utility_threshold)

        report = QualityReport(
            marginal_tv_avg=avg_tv,
            marginal_tv_per_query=tv_per_query,
            workload_error=wl_error,
            ml_utility=ml_f1,
            ml_utility_per_classifier=ml_per_clf,
            passed=passed,
            details=f"TV={avg_tv:.4f}, WL_Error={wl_error:.4f}, ML_F1={ml_f1 if ml_f1 else 'N/A'}",
        )

        logger.info(f"Quality report: {report.details}, passed={passed}")
        return report
