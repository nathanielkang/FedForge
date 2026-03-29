# FedSynth-Engine

A scalable, fault-tolerant system for federated tabular synthetic data generation with cross-silo orchestration and differential privacy.

## Overview

FedSynth-Engine transforms federated synthetic data generation from a research prototype into a production-grade data engineering pipeline. The system orchestrates the full lifecycle: schema discovery, privacy budget allocation, noisy marginal estimation, secure aggregation, diffusion-based synthesis, and quality validation.

### Key Features

- **Adaptive Communication**: Quantization + delta encoding + sparse transmission reduces bandwidth 60–80%
- **Fault Tolerance**: Checkpoint-resume with graceful degradation when parties disconnect
- **Horizontal Scalability**: Hierarchical aggregation scales linearly up to 100 parties
- **Streaming Workload Adaptation**: Incrementally refine synthetic data for new queries without full re-synthesis
- **Differential Privacy**: Per-party ε-DP guarantees via calibrated Gaussian noise and secure aggregation

## Project Structure

```
2_Code_a/
├── configs/
│   └── default.yaml            # Default configuration
├── scripts/
│   ├── smoke_test.py           # Quick local test (tiny data, 2 epochs)
│   └── run_experiments.py      # Full experiment runner (8 datasets)
├── src/
│   ├── __init__.py
│   ├── coordinator.py          # Coordinator node (orchestration, checkpoints)
│   ├── party_agent.py          # Party agent (local marginals, DP noise)
│   ├── synthesis_engine.py     # Diffusion model with workload-guided loss
│   ├── communication.py        # Adaptive communication protocol
│   ├── quality_monitor.py      # Quality validation module
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py         # Dataset loading for 8 benchmarks
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py          # Evaluation metrics (TV, workload error, ML utility)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

### Smoke Test (local, tiny data, 2 epochs)

```bash
python scripts/smoke_test.py
```

### Full Experiment Run

```bash
python scripts/run_experiments.py --config configs/default.yaml
```

### Custom Configuration

```bash
python scripts/run_experiments.py --config configs/default.yaml --datasets adult bank --num_parties 5 --epsilon 1.0
```

## Datasets

| Dataset   | Records  | Columns | Domain     |
|-----------|----------|---------|------------|
| adult     | 48,842   | 15      | Census     |
| bank      | 45,211   | 17      | Marketing  |
| acs_pums  | 100,000  | 20      | Census     |
| credit    | 30,000   | 24      | Financial  |
| mushroom  | 8,124    | 23      | Biology    |
| shopping  | 12,330   | 18      | E-commerce |
| diabetes  | 101,766  | 50      | Medical    |
| covertype | 581,012  | 55      | Ecology    |

## Configuration

See `configs/default.yaml` for all available options including:
- Privacy parameters (epsilon, delta, mechanism)
- Communication compression settings (bits, delta threshold)
- Diffusion model hyperparameters (steps, beta schedule, hidden dim)
- Workload generation parameters
- Hierarchical aggregation settings

## Author

Nathaniel Kang  
School of Computer Science and Engineering, IT College  
Kyungpook National University  
natekang@knu.ac.kr
