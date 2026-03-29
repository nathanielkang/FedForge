"""
Party Agent for FedSynth-Engine.

Each party agent runs locally at a data holder's site. It computes noisy
marginals from local data and participates in the aggregation protocol.
Raw data never leaves the local environment.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .communication import AdaptiveCommunicationProtocol, CompressedPayload


class PartyStatus(Enum):
    REGISTERED = "registered"
    ACTIVE = "active"
    STALE = "stale"
    DROPPED = "dropped"


@dataclass
class MarginalQuery:
    """Specifies a marginal to compute: subset of column indices."""
    key: str
    columns: List[int]
    budget_rho: float


def _safe_probs(p: np.ndarray) -> np.ndarray:
    """Safely normalize a probability vector to sum to 1.0 exactly."""
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total < 1e-15:
        return np.ones_like(p) / len(p)
    p = p / total
    p = np.clip(p, 0.0, None)
    p = p / p.sum()
    return p


class PartyAgent:
    """
    Local agent running at each data holder.

    Responsibilities:
    - Store local data (never transmitted)
    - Compute local histograms for requested marginals
    - Add calibrated DP noise
    - Compress and transmit noisy marginals
    """

    def __init__(
        self,
        party_id: int,
        local_data: np.ndarray,
        domain_sizes: Dict[str, int],
        column_names: List[str],
        comm_protocol: Optional[AdaptiveCommunicationProtocol] = None,
        random_state: int = 42,
    ):
        self.party_id = party_id
        self.local_data = local_data
        self.domain_sizes = domain_sizes
        self.column_names = column_names
        self.num_records = local_data.shape[0]
        self.status = PartyStatus.REGISTERED

        self.comm = comm_protocol or AdaptiveCommunicationProtocol()
        self._rng = np.random.default_rng(random_state + party_id)
        self._marginal_cache: Dict[str, np.ndarray] = {}
        self._last_heartbeat = time.time()

    def activate(self):
        self.status = PartyStatus.ACTIVE
        self._last_heartbeat = time.time()

    def heartbeat(self) -> float:
        self._last_heartbeat = time.time()
        return self._last_heartbeat

    @property
    def time_since_heartbeat(self) -> float:
        return time.time() - self._last_heartbeat

    def get_schema(self) -> Dict[str, int]:
        return dict(self.domain_sizes)

    def compute_histogram(
        self, columns: List[int], domain_sizes_list: List[int]
    ) -> np.ndarray:
        """Compute a joint histogram over the specified columns."""
        if len(columns) == 0:
            return np.array([1.0])

        shape = tuple(domain_sizes_list)
        hist = np.zeros(shape, dtype=np.float64)

        for row in self.local_data:
            idx = tuple(int(row[c]) for c in columns)
            valid = True
            for i, s in enumerate(shape):
                if idx[i] < 0 or idx[i] >= s:
                    valid = False
                    break
            if valid:
                hist[idx] += 1.0

        hist_flat = hist.ravel()
        if self.num_records > 0:
            hist_flat = hist_flat / self.num_records

        return hist_flat

    def compute_noisy_marginal(
        self, query: MarginalQuery
    ) -> Tuple[str, np.ndarray]:
        """Compute a noisy marginal with calibrated Gaussian noise."""
        col_indices = query.columns
        ds_list = []
        for c in col_indices:
            col_name = self.column_names[c]
            ds_list.append(self.domain_sizes[col_name])

        hist = self.compute_histogram(col_indices, ds_list)

        sigma_base = 1.0 / np.sqrt(2.0 * query.budget_rho) if query.budget_rho > 0 else 1e6
        sensitivity = 1.0 / max(self.num_records, 1)
        noise_sigma = sigma_base * sensitivity

        noise = self._rng.normal(0.0, noise_sigma, size=hist.shape)
        noisy_hist = hist + noise

        noisy_hist = _safe_probs(noisy_hist)

        self._marginal_cache[query.key] = noisy_hist.copy()

        return query.key, noisy_hist

    def compute_all_marginals(
        self, queries: List[MarginalQuery]
    ) -> Dict[str, np.ndarray]:
        """Compute noisy marginals for all requested queries."""
        results = {}
        for q in queries:
            key, marginal = self.compute_noisy_marginal(q)
            results[key] = marginal
        return results

    def compress_and_send(
        self, marginals: Dict[str, np.ndarray]
    ) -> List[Optional[CompressedPayload]]:
        """Compress marginals for transmission."""
        payloads = []
        for key, marginal in marginals.items():
            payload = self.comm.compress_marginal(self.party_id, key, marginal)
            payloads.append(payload)
        return payloads

    def get_cached_marginal(self, key: str) -> Optional[np.ndarray]:
        return self._marginal_cache.get(key)
