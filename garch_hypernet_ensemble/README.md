# GARCH-HyperNetwork Adaptive Stacking Ensemble

Production-ready adaptive ensemble for quantitative trading built around a GARCH-driven HyperNetwork regulator. The architecture combines 60+ asynchronous base learners, a stacking blender, and an adaptive hypernetwork that dynamically re-weights the ensemble according to market regimes and GARCH volatility forecasts.

## Key Features
- **GARCH-driven meta signals** with online/offline separation and EWMA smoothing
- **HyperNetwork meta-controller** that enforces entropy, correlation penalties, and online adaptation
- **Kelly + CVaR risk sizing** with regime-aware calibration and circuit breakers
- **Asynchronous training pipeline** with stress scenario augmentation and pruning
- **Production observability** via Prometheus metrics, versioned artifacts, and structured logging

## Project Layout
```
garch_hypernet_ensemble/
├── configs/              # YAML configs for production, tests, experiments
├── notebooks/            # Research and monitoring notebooks
├── scripts/              # Entry points (training, serving, maintenance)
├── src/                  # Core library
│   ├── core/             # Orchestrator, trainer, predictor
│   ├── garch/            # GARCH(1,1) implementation and trackers
│   ├── hypernetwork/     # TensorFlow hypernetwork and trainer
│   ├── monitoring/       # Performance tracking & stress testing
│   ├── utils/            # Caching & version-control utilities
│   └── validation/       # Data validation & walk-forward CV
├── tests/                # Pytest suite
├── Dockerfile            # CUDA-enabled runtime image
├── docker-compose.yml    # Local orchestration with Redis cache
├── requirements.txt      # Python dependencies
└── README.md             # Project overview
```

## Quick Start
```bash
# Install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest -q

# Train with sample parquet dataset
aiohttp  # ensure CUDA drivers are available if using GPU
python scripts/train.py --data-path data/market_data.parquet
```

## Docker Workflow
```bash
# Build image
sudo docker build -t garch-ensemble:v2.0.0 .

# Launch stack (includes Redis cache)
sudo docker-compose up -d

# Stream logs
sudo docker logs -f garch-ensemble
```

## Prometheus Metrics
- `garch_ensemble_predictions_total`
- `garch_ensemble_accuracy`
- `garch_ensemble_diversity_score`
- `garch_ensemble_garch_volatility`
- `garch_ensemble_effective_bets`
- `garch_ensemble_emergency_mode_activations`

## Testing & Quality Gates
- Full async end-to-end pipeline test (`tests/test_ensemble.py`)
- Data validation layer applied at every interface (`[FIX-12]`)
- HyperNetwork online trainer with gradient clipping (`[FIX-4]`)

## License
Proprietary © AlgoTrading Team
