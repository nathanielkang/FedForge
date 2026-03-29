#!/usr/bin/env python3
"""
Smoke test for FedSynth-Engine.

Runs the full pipeline on tiny synthetic data with 2 epochs to verify
correctness before deploying to remote VM. Catches crashes early.
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.communication import AdaptiveCommunicationProtocol
from src.coordinator import Coordinator, HierarchicalAggregator, generate_workload
from src.data.datasets import DatasetInfo, generate_synthetic_dataset, partition_dataset
from src.party_agent import PartyAgent
from src.quality_monitor import QualityMonitor
from src.synthesis_engine import SynthesisEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("FedSynth-Engine Smoke Test")
    logger.info("=" * 60)

    NUM_PARTIES = 3
    NUM_EPOCHS = 2
    NUM_ROWS = 500
    NUM_COLS = 6
    NUM_BINS = 4
    NUM_QUERIES = 5
    BATCH_SIZE = 128
    SEED = 42
    t_start = time.time()

    logger.info("Step 1: Generate tiny synthetic dataset")
    dataset = generate_synthetic_dataset(
        num_rows=NUM_ROWS,
        num_cols=NUM_COLS,
        num_bins=NUM_BINS,
        random_state=SEED,
    )
    logger.info(f"  Dataset: {dataset.num_rows} rows x {dataset.num_cols} cols")

    logger.info("Step 2: Partition data across parties")
    partitions = partition_dataset(dataset, NUM_PARTIES, random_state=SEED)
    for i, p in enumerate(partitions):
        logger.info(f"  Party {i}: {p.shape[0]} records")

    logger.info("Step 3: Create party agents with adaptive communication")
    comm = AdaptiveCommunicationProtocol(quant_bits=8, delta_threshold=1e-4)
    agents = []
    for i, partition in enumerate(partitions):
        agent = PartyAgent(
            party_id=i,
            local_data=partition,
            domain_sizes=dataset.domain_sizes,
            column_names=dataset.column_names,
            comm_protocol=comm,
            random_state=SEED,
        )
        agents.append(agent)

    logger.info("Step 4: Initialize coordinator and register parties")
    coordinator = Coordinator(
        num_parties=NUM_PARTIES,
        epsilon=1.0,
        delta=1e-5,
        checkpoint_dir="checkpoints_smoke",
        random_state=SEED,
    )
    coordinator.register_all_parties(agents)

    logger.info("Step 5: Generate workload")
    workload = generate_workload(
        column_names=dataset.column_names,
        domain_sizes=dataset.domain_sizes,
        num_queries=NUM_QUERIES,
        min_way=2,
        max_way=2,
        random_state=SEED,
    )
    logger.info(f"  Workload: {len(workload.queries)} queries")

    logger.info("Step 6: Run pipeline (schema → budget → marginals → aggregation)")
    aggregated = coordinator.run_pipeline(workload)
    logger.info(f"  Aggregated {len(aggregated)} marginals")

    logger.info("Step 7: Test hierarchical aggregation")
    hier = HierarchicalAggregator(num_clusters=2, random_state=SEED)
    hier.form_clusters(list(range(NUM_PARTIES)))
    per_party = {}
    for key, agg_m in aggregated.items():
        per_party[key] = {}
        for pid in range(NUM_PARTIES):
            cached = agents[pid].get_cached_marginal(key)
            if cached is not None:
                per_party[key][pid] = cached
    party_weights = {i: 1.0 / NUM_PARTIES for i in range(NUM_PARTIES)}
    hier_agg = hier.aggregate_hierarchical(per_party, party_weights)
    logger.info(f"  Hierarchical aggregation: {len(hier_agg)} marginals")

    logger.info("Step 8: Train diffusion model (2 epochs)")
    engine = SynthesisEngine(
        data_dim=dataset.num_cols,
        domain_sizes=dataset.domain_sizes,
        column_names=dataset.column_names,
        diffusion_steps=10,
        beta_start=1e-4,
        beta_end=0.2,
        hidden_dim=64,
        num_layers=2,
        learning_rate=1e-3,
        workload_weight=0.5,
        device="cpu",
    )
    history = engine.train(
        data=dataset.data,
        target_marginals=aggregated,
        queries=workload.queries,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        log_interval=1,
    )
    logger.info(f"  Training done. Final denoise loss: {history[-1]['denoise_loss']:.4f}")

    logger.info("Step 9: Generate synthetic data")
    synth_data = engine.generate(num_samples=NUM_ROWS, batch_size=BATCH_SIZE)
    logger.info(f"  Generated {synth_data.shape[0]} records")

    logger.info("Step 10: Quality evaluation")
    monitor = QualityMonitor(tv_threshold=0.5, workload_error_threshold=0.5)
    report = monitor.full_evaluation(
        synthetic_data=synth_data,
        real_data=dataset.data,
        target_marginals=aggregated,
        queries=workload.queries,
        weights=workload.weights,
        column_names=dataset.column_names,
        domain_sizes=dataset.domain_sizes,
        evaluate_ml=True,
    )
    logger.info(f"  Marginal TV: {report.marginal_tv_avg:.4f}")
    logger.info(f"  Workload Error: {report.workload_error:.4f}")
    logger.info(f"  ML Utility (F1): {report.ml_utility:.4f}")

    logger.info("Step 11: Test communication compression")
    comm.reset_stats()
    for agent in agents:
        marginals = agent.compute_all_marginals(workload.queries)
        agent.compress_and_send(marginals)
    stats = comm.stats
    logger.info(f"  Uncompressed: {stats.total_uncompressed_bytes} bytes")
    logger.info(f"  Compressed: {stats.total_compressed_bytes} bytes")
    logger.info(f"  Savings: {stats.savings_pct:.1f}%")
    logger.info(f"  Skipped: {stats.num_skipped}")

    logger.info("Step 12: Test fault tolerance (simulate party drop)")
    coordinator.drop_party(0)
    logger.info(f"  Active parties after drop: {coordinator.active_party_count}")
    coordinator.rejoin_party(0)
    logger.info(f"  Active parties after rejoin: {coordinator.active_party_count}")

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"SMOKE TEST PASSED in {elapsed:.1f}s")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
