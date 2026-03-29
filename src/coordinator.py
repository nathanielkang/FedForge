"""
Coordinator Node for FedSynth-Engine.

Central control plane that orchestrates the entire federated synthesis pipeline:
schema discovery, budget allocation, marginal aggregation, synthesis dispatch,
and quality validation. Supports checkpoint-resume and fault tolerance.
"""

import json
import logging
import math
import os
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .communication import AdaptiveCommunicationProtocol
from .party_agent import MarginalQuery, PartyAgent, PartyStatus

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    SCHEMA_DISCOVERY = auto()
    BUDGET_ALLOCATION = auto()
    MARGINAL_ESTIMATION = auto()
    SECURE_AGGREGATION = auto()
    DIFFUSION_SYNTHESIS = auto()
    QUALITY_VALIDATION = auto()


@dataclass
class Checkpoint:
    """Serializable system state at a barrier point."""
    stage: PipelineStage
    aggregated_marginals: Dict[str, np.ndarray]
    party_statuses: Dict[int, str]
    budgets_consumed: Dict[int, float]
    model_state: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadSpec:
    """Defines the set of marginal queries and their weights."""
    queries: List[MarginalQuery]
    weights: Dict[str, float]

    @property
    def marginal_keys(self) -> List[str]:
        return [q.key for q in self.queries]


class Coordinator:
    """
    Central coordinator for the FedSynth-Engine pipeline.

    Manages party lifecycle, orchestrates pipeline stages, handles
    checkpoint/resume, and coordinates fault recovery.
    """

    def __init__(
        self,
        num_parties: int,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        staleness_decay: float = 0.95,
        stage_timeout: float = 300.0,
        max_stale_rounds: int = 10,
        checkpoint_dir: str = "checkpoints",
        random_state: int = 42,
    ):
        self.num_parties = num_parties
        self.epsilon = epsilon
        self.delta = delta
        self.staleness_decay = staleness_decay
        self.stage_timeout = stage_timeout
        self.max_stale_rounds = max_stale_rounds
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._rng = np.random.default_rng(random_state)
        self._parties: Dict[int, PartyAgent] = {}
        self._party_weights: Dict[int, float] = {}
        self._active_set: set = set()
        self._cached_marginals: Dict[str, Dict[int, np.ndarray]] = {}
        self._cache_staleness: Dict[str, Dict[int, int]] = {}
        self._aggregated_marginals: Dict[str, np.ndarray] = {}
        self._budgets_consumed: Dict[int, float] = {}
        self._current_stage = PipelineStage.SCHEMA_DISCOVERY
        self._round: int = 0
        self._unified_schema: Dict[str, int] = {}
        self._workload: Optional[WorkloadSpec] = None

    def register_party(self, party: PartyAgent):
        """Register a party agent with the coordinator."""
        pid = party.party_id
        self._parties[pid] = party
        self._active_set.add(pid)
        self._budgets_consumed[pid] = 0.0
        party.activate()
        logger.info(f"Party {pid} registered ({party.num_records} records)")

    def register_all_parties(self, parties: List[PartyAgent]):
        for p in parties:
            self.register_party(p)
        total_records = sum(p.num_records for p in parties)
        for pid, p in self._parties.items():
            self._party_weights[pid] = p.num_records / max(total_records, 1)

    def discover_schema(self) -> Dict[str, int]:
        """Stage 1: Collect and unify schemas from all active parties."""
        self._current_stage = PipelineStage.SCHEMA_DISCOVERY
        schemas = {}
        for pid in self._active_set:
            schemas[pid] = self._parties[pid].get_schema()

        if not schemas:
            raise RuntimeError("No active parties for schema discovery")

        ref_pid = next(iter(schemas))
        unified = dict(schemas[ref_pid])

        for pid, schema in schemas.items():
            if pid == ref_pid:
                continue
            common_keys = set(unified.keys()) & set(schema.keys())
            for k in list(unified.keys()):
                if k not in common_keys:
                    del unified[k]
                else:
                    unified[k] = max(unified[k], schema[k])

        self._unified_schema = unified
        logger.info(f"Unified schema: {len(unified)} columns")
        return unified

    def allocate_budget(
        self, workload: WorkloadSpec
    ) -> Dict[str, float]:
        """Stage 2: Allocate privacy budget across marginals."""
        self._current_stage = PipelineStage.BUDGET_ALLOCATION
        self._workload = workload

        eps = self.epsilon
        rho_total = eps ** 2 / (2.0 * math.log(1.0 / self.delta + 1e-15))
        rho_total = max(rho_total, eps * 0.1)

        weights = workload.weights
        keys = list(weights.keys())
        w_arr = np.array([weights[k] for k in keys], dtype=np.float64)
        w_arr = np.clip(w_arr, 1e-8, None)

        w_scaled = np.power(w_arr, 2.0 / 3.0)
        w_scaled = w_scaled / w_scaled.sum()

        budget_map = {}
        for i, key in enumerate(keys):
            rho_j = rho_total * w_scaled[i]
            rho_j = max(rho_j, 1e-10)
            budget_map[key] = rho_j

        for q in workload.queries:
            if q.key in budget_map:
                q.budget_rho = budget_map[q.key]

        logger.info(f"Budget allocated across {len(budget_map)} marginals, rho_total={rho_total:.6f}")
        return budget_map

    def estimate_marginals(self) -> Dict[str, Dict[int, np.ndarray]]:
        """Stage 3: Request noisy marginals from all active parties."""
        self._current_stage = PipelineStage.MARGINAL_ESTIMATION

        if self._workload is None:
            raise RuntimeError("Workload not set. Call allocate_budget first.")

        all_marginals: Dict[str, Dict[int, np.ndarray]] = {}
        failed_parties = set()

        for pid in list(self._active_set):
            party = self._parties[pid]
            try:
                party.heartbeat()
                marginals = party.compute_all_marginals(self._workload.queries)
                for key, m in marginals.items():
                    if key not in all_marginals:
                        all_marginals[key] = {}
                    all_marginals[key][pid] = m
                    if key not in self._cached_marginals:
                        self._cached_marginals[key] = {}
                        self._cache_staleness[key] = {}
                    self._cached_marginals[key][pid] = m.copy()
                    self._cache_staleness[key][pid] = 0
            except Exception as e:
                logger.warning(f"Party {pid} failed during marginal estimation: {e}")
                failed_parties.add(pid)

        self._handle_failures(failed_parties, all_marginals)
        self._round += 1
        self._save_checkpoint(PipelineStage.MARGINAL_ESTIMATION)

        return all_marginals

    def aggregate_marginals(
        self, per_party_marginals: Dict[str, Dict[int, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Stage 4: Securely aggregate marginals across parties."""
        self._current_stage = PipelineStage.SECURE_AGGREGATION
        aggregated = {}

        for key, party_dict in per_party_marginals.items():
            weighted_sum = None
            total_weight = 0.0

            for pid, marginal in party_dict.items():
                w = self._party_weights.get(pid, 1.0 / self.num_parties)

                staleness = self._cache_staleness.get(key, {}).get(pid, 0)
                if staleness > 0:
                    w *= self.staleness_decay ** staleness

                if weighted_sum is None:
                    weighted_sum = w * marginal
                else:
                    if marginal.shape == weighted_sum.shape:
                        weighted_sum += w * marginal
                total_weight += w

            if weighted_sum is not None and total_weight > 0:
                aggregated[key] = weighted_sum / total_weight
            elif weighted_sum is not None:
                aggregated[key] = weighted_sum

        self._aggregated_marginals = aggregated
        self._save_checkpoint(PipelineStage.SECURE_AGGREGATION)

        logger.info(f"Aggregated {len(aggregated)} marginals from {len(self._active_set)} parties")
        return aggregated

    def run_pipeline(
        self, workload: WorkloadSpec
    ) -> Dict[str, np.ndarray]:
        """Execute the full pipeline: stages 1-4 (schema through aggregation)."""
        self.discover_schema()
        self.allocate_budget(workload)
        per_party = self.estimate_marginals()
        aggregated = self.aggregate_marginals(per_party)
        return aggregated

    def _handle_failures(
        self,
        failed_pids: set,
        all_marginals: Dict[str, Dict[int, np.ndarray]],
    ):
        """Handle party failures using cached marginals with staleness decay."""
        for pid in failed_pids:
            has_cache = False
            for key in self._cached_marginals:
                if pid in self._cached_marginals[key]:
                    has_cache = True
                    staleness = self._cache_staleness[key].get(pid, 0) + 1
                    self._cache_staleness[key][pid] = staleness

                    if staleness <= self.max_stale_rounds:
                        if key not in all_marginals:
                            all_marginals[key] = {}
                        all_marginals[key][pid] = self._cached_marginals[key][pid]
                        logger.info(f"Using cached marginal for party {pid}, key={key}, staleness={staleness}")
                    else:
                        logger.warning(f"Party {pid} cache too stale for key={key}")

            if not has_cache:
                self._parties[pid].status = PartyStatus.DROPPED
                self._active_set.discard(pid)
                logger.warning(f"Party {pid} dropped (no cache available)")
            else:
                self._parties[pid].status = PartyStatus.STALE

    def drop_party(self, pid: int):
        """Permanently remove a party from the active set."""
        self._active_set.discard(pid)
        if pid in self._parties:
            self._parties[pid].status = PartyStatus.DROPPED
        total_records = sum(
            self._parties[p].num_records
            for p in self._active_set
            if p in self._parties
        )
        for p in self._active_set:
            if p in self._parties:
                self._party_weights[p] = self._parties[p].num_records / max(total_records, 1)
        logger.info(f"Party {pid} dropped; {len(self._active_set)} parties remain")

    def rejoin_party(self, pid: int):
        """Allow a stale party to rejoin as active."""
        if pid in self._parties:
            self._parties[pid].status = PartyStatus.ACTIVE
            self._active_set.add(pid)
            total_records = sum(
                self._parties[p].num_records for p in self._active_set
            )
            for p in self._active_set:
                self._party_weights[p] = self._parties[p].num_records / max(total_records, 1)
            logger.info(f"Party {pid} rejoined as active")

    def _save_checkpoint(self, stage: PipelineStage):
        """Persist system state to stable storage."""
        ckpt = Checkpoint(
            stage=stage,
            aggregated_marginals=dict(self._aggregated_marginals),
            party_statuses={
                pid: p.status.value for pid, p in self._parties.items()
            },
            budgets_consumed=dict(self._budgets_consumed),
            metadata={
                "round": self._round,
                "active_parties": list(self._active_set),
                "schema": dict(self._unified_schema),
            },
        )
        path = self.checkpoint_dir / f"checkpoint_stage{stage.value}_r{self._round}.pkl"
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> Checkpoint:
        """Restore system state from a checkpoint."""
        with open(path, "rb") as f:
            ckpt: Checkpoint = pickle.load(f)
        self._aggregated_marginals = ckpt.aggregated_marginals
        self._current_stage = ckpt.stage
        self._round = ckpt.metadata.get("round", 0)
        logger.info(f"Checkpoint loaded from {path}, stage={ckpt.stage}, round={self._round}")
        return ckpt

    @property
    def active_party_count(self) -> int:
        return len(self._active_set)

    @property
    def aggregated_marginals(self) -> Dict[str, np.ndarray]:
        return self._aggregated_marginals


def generate_workload(
    column_names: List[str],
    domain_sizes: Dict[str, int],
    num_queries: int = 50,
    min_way: int = 2,
    max_way: int = 3,
    random_state: int = 42,
) -> WorkloadSpec:
    """Generate a random workload of marginal queries."""
    rng = np.random.default_rng(random_state)
    num_cols = len(column_names)
    queries = []
    weights = {}

    for i in range(num_queries):
        way = rng.integers(min_way, max_way + 1)
        way = min(way, num_cols)
        cols = sorted(rng.choice(num_cols, size=way, replace=False).tolist())
        key = "_".join(str(c) for c in cols)

        if key in weights:
            weights[key] += 1.0
            continue

        w = float(rng.uniform(0.5, 2.0))
        weights[key] = w
        queries.append(MarginalQuery(key=key, columns=cols, budget_rho=0.0))

    total_w = sum(weights.values())
    for k in weights:
        weights[k] /= total_w

    return WorkloadSpec(queries=queries, weights=weights)


class HierarchicalAggregator:
    """
    Two-level hierarchical aggregation to reduce coordinator communication burden.
    Parties are grouped into clusters; each cluster has a leader that pre-aggregates.
    """

    def __init__(
        self,
        num_clusters: Optional[int] = None,
        random_state: int = 42,
    ):
        self.num_clusters = num_clusters
        self._rng = np.random.default_rng(random_state)
        self.clusters: Dict[int, List[int]] = {}
        self.leaders: Dict[int, int] = {}

    def form_clusters(
        self,
        party_ids: List[int],
        latencies: Optional[Dict[int, float]] = None,
    ):
        """Assign parties to clusters. If latencies provided, use them for grouping."""
        K = len(party_ids)
        if self.num_clusters is None or self.num_clusters <= 0:
            G = max(1, math.ceil(math.sqrt(K)))
        else:
            G = min(self.num_clusters, K)

        shuffled = list(party_ids)
        self._rng.shuffle(shuffled)

        self.clusters = {}
        for g in range(G):
            self.clusters[g] = []

        for i, pid in enumerate(shuffled):
            self.clusters[i % G].append(pid)

        self.leaders = {}
        for g, members in self.clusters.items():
            if members:
                self.leaders[g] = members[0]

    def aggregate_hierarchical(
        self,
        per_party_marginals: Dict[str, Dict[int, np.ndarray]],
        party_weights: Dict[int, float],
    ) -> Dict[str, np.ndarray]:
        """Two-phase aggregation: within-cluster then across-cluster."""
        cluster_aggregates: Dict[str, Dict[int, np.ndarray]] = {}
        cluster_weights: Dict[int, float] = {}

        for g, members in self.clusters.items():
            cw = sum(party_weights.get(pid, 0) for pid in members)
            cluster_weights[g] = cw

            for key in per_party_marginals:
                if key not in cluster_aggregates:
                    cluster_aggregates[key] = {}

                agg = None
                local_total_w = 0.0
                for pid in members:
                    if pid in per_party_marginals[key]:
                        w = party_weights.get(pid, 0)
                        m = per_party_marginals[key][pid]
                        if agg is None:
                            agg = w * m
                        else:
                            agg += w * m
                        local_total_w += w

                if agg is not None and local_total_w > 0:
                    cluster_aggregates[key][g] = agg / local_total_w

        global_aggregated = {}
        for key in cluster_aggregates:
            weighted_sum = None
            total_w = 0.0
            for g, ca in cluster_aggregates[key].items():
                w = cluster_weights.get(g, 0)
                if weighted_sum is None:
                    weighted_sum = w * ca
                else:
                    weighted_sum += w * ca
                total_w += w
            if weighted_sum is not None and total_w > 0:
                global_aggregated[key] = weighted_sum / total_w

        return global_aggregated
