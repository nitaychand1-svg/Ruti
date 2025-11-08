# ACTS v6.0 — Adaptive Causal Trading System

> **Complete Integration**: v5.0 (Advanced Multi-Agent Intelligence) + v5.5 (ELBO + Sequential Interventions)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

ACTS v6.0 is a state-of-the-art autonomous trading system that combines:

- **🧠 Multi-Modal Perception**: NLP + Vision + Audio fusion
- **🤖 Multi-Agent Debate**: 6 LLM agents for strategic consensus
- **⚡ Hierarchical MARL**: 5 specialized execution agents
- **🔗 Causal World Model**: Sequential interventions with temporal chains
- **💾 Episodic Memory**: Vector database for historical learning
- **🎯 Risk Management**: Black swan simulation with importance sampling
- **🔄 Self-Evolution**: Autonomous improvement via NAS + Meta-RL
- **🔒 Privacy-Preserving**: Federated learning with differential privacy
- **💬 Human-AI Interface**: Conversational explanations

## 📊 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Latency (p95) | < 500ms | ✅ Optimized |
| Out-of-Sample Sharpe | > 2.4 | 🎯 Achievable |
| Max Drawdown | < 7% | ✅ Risk-managed |
| Regime Accuracy | > 97% | ✅ Bayesian |
| AMI Score | > 0.90 | ✅ Multi-agent |

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  ACTS v6.0 — FULL STACK                      │
├──────────────────────────────────────────────────────────────┤
│  LAYER 1: Multi-Modal Perception                             │
│    • NLP (RoBERTa) + Vision (CLIP) + Audio (Whisper)        │
│    • Bayesian Regime Predictor with ELBO Training            │
│                                                               │
│  LAYER 2: Strategic Intelligence                             │
│    • 6 LLM Agents (Bull, Bear, Risk, Ethical, etc.)         │
│    • Weighted Consensus + RLHF Alignment                     │
│                                                               │
│  LAYER 3: Execution Control                                  │
│    • 5 MARL Agents (Liquidity, Slippage, Timing, etc.)     │
│    • Adversarial HFT Defense                                 │
│                                                               │
│  CORE: Adaptive Causal Kernel                                │
│    • World Model (Assets + Countries + Central Banks)        │
│    • Sequential Interventions (do(X₁, t₁) → do(X₂, t₂))    │
│    • Counterfactual Engine + Episodic Memory                 │
│                                                               │
│  LAYER 4: Risk Management                                    │
│    • 5 Existential Scenarios (Solar Flare, Cyber, etc.)     │
│    • Importance Sampling for Tail Events                     │
│                                                               │
│  LAYER 5: Self-Improvement                                   │
│    • NAS + Meta-RL for Architecture Search                   │
│    • ELBO Optimizer + Adaptive MC Sampler                    │
│                                                               │
│  LAYER 6: Human Interface                                    │
│    • Conversational Explanations + DAG Visualization         │
└──────────────────────────────────────────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd acts_v6

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Advanced Installation (with all features)

```bash
# Install with transformers for multi-modal support
pip install transformers tokenizers

# Install FAISS for fast vector search
pip install faiss-cpu  # or faiss-gpu for GPU

# Install CLIP for vision
pip install git+https://github.com/openai/CLIP.git

# Optional: for production deployment
pip install fastapi uvicorn prometheus-client
```

## 🚀 Quick Start

### 1. Basic Usage

```python
import numpy as np
import asyncio
from acts_v6_complete import ACTSv6Complete

# Initialize system
system = ACTSv6Complete(
    input_dim=100,
    n_assets=10,
    device='cuda',  # or 'cpu'
    use_pretrained=False
)

# Prepare data
market_data = np.random.randn(100, 50)
news_articles = ["Fed signals rate hike", "Tech stocks rally"]
portfolio = {'BTC': 100000, 'SPY': 50000}
constraints = {'max_position_size': 0.25}

# Run full trading cycle
async def main():
    result = await system.full_trading_cycle(
        market_data=market_data,
        news_articles=news_articles,
        portfolio=portfolio,
        constraints=constraints
    )
    
    print(f"Detected Regime: {result['regime']}")
    print(f"Strategy: {result['strategy']}")
    print(f"Execution: {result['execution']}")

asyncio.run(main())
```

### 2. Training Regime Predictor

```python
# Prepare training data
training_data = [
    (features_1, label_1),  # features: [4096], label: 0/1/2
    (features_2, label_2),
    # ...
]

# Train with ELBO
training_result = await system.train_regime_predictor(
    training_data=training_data,
    n_epochs=100
)

print(f"Final loss: {training_result['final_loss']:.4f}")
print(f"Converged: {training_result['converged']}")
```

### 3. Causal Interventions

```python
from acts_v6_complete import TemporalIntervention

# Define interventions
interventions = [
    TemporalIntervention(
        variable='FED',
        value=0.06,  # 6% interest rate
        timestep=5,
        metadata={'description': 'Fed rate hike'}
    ),
    TemporalIntervention(
        variable='BTC',
        value=-0.20,  # -20% shock
        timestep=10,
        metadata={'description': 'BTC crash'}
    )
]

# Run intervention chain
result = system.run_causal_intervention(
    interventions=interventions,
    horizon=30
)

print(f"Final regime: {result['final_state']['regime_probs']}")
print(f"Convergence R̂: {result['convergence_metrics']['rhat']:.4f}")
```

### 4. Risk Analysis

```python
# Simulate existential risks
portfolio = {'BTC': 100000, 'SPY': 50000, 'GLD': 25000}

risk_result = system.existential_risk.simulate_scenario(
    scenario_name='cyber_attack',
    portfolio=portfolio,
    n_samples=10000
)

print(f"Expected Loss: ${risk_result['expected_loss']:,.0f}")
print(f"VaR (95%): ${risk_result['var_95']:,.0f}")
print(f"Survival Probability: {risk_result['survival_probability']:.2%}")
```

### 5. Human Explanations

```python
# Get conversational explanations
explanation = await system.explain_to_human(
    "Why did you choose this strategy?"
)

print(explanation)
```

## 📂 Project Structure

```
acts_v6/
├── src/
│   └── acts_v6_complete.py      # Main system implementation
├── config/
│   └── default_config.yaml      # Configuration file
├── examples/
│   ├── basic_usage.py           # Basic usage example
│   └── advanced_interventions.py # Advanced causal analysis
├── tests/
│   └── test_acts_v6.py          # Unit tests
├── data/
│   └── .gitkeep                 # Data directory
├── models/
│   └── .gitkeep                 # Model checkpoints
├── logs/
│   └── .gitkeep                 # Log files
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🧪 Running Examples

### Basic Usage

```bash
python examples/basic_usage.py
```

This example demonstrates:
- System initialization
- Full trading cycle
- Regime prediction
- Strategy generation
- Execution
- State persistence

### Advanced Interventions

```bash
python examples/advanced_interventions.py
```

This example demonstrates:
- Complex temporal intervention chains
- Multiple scenario comparison
- Counterfactual analysis
- Risk attribution

## 🔧 Configuration

Edit `config/default_config.yaml` to customize:

```yaml
system:
  device: "cuda"  # or "cpu"
  use_pretrained: false

model:
  input_dim: 100
  n_assets: 10
  hidden_dim: 128
  n_regimes: 3

# ... see config file for all options
```

## 📊 Components Deep Dive

### Multi-Modal Fusion Engine

Combines multiple data modalities:

```python
holistic_features, uncertainty = await system.multi_modal_fusion.perceive_world(
    market_data=ohlcv_data,
    news_articles=news_list,
    social_posts=tweets,
    geo_images=satellite_images,  # optional
    audio_speeches=fed_speeches    # optional
)
```

### Multi-Agent Debate System

6 LLM agents debate strategy:

```python
debate_result = await system.multi_agent_debate.orchestrate_debate(
    features=features,
    regime_probs={'bull': 0.6, 'normal': 0.3, 'crisis': 0.1},
    constraints=constraints
)

consensus = debate_result['consensus_strategy']
```

### Hierarchical MARL Swarm

5 specialized execution agents:

```python
execution_result = await system.marl_swarm.swarm_execute(
    strategy=strategy,
    market_state=current_state,
    urgency='high'
)
```

### Sequential Intervention Engine

Temporal causal analysis:

```python
# Build world model first
causal_graph = await system.world_model_builder.build_world_graph(
    market_data=data,
    news_corpus=news,
    knowledge_base=kb
)

# Run interventions
result = system.run_causal_intervention(
    interventions=[...],
    horizon=30
)
```

### Episodic Memory

Store and recall trading episodes:

```python
# Store episode
system.episodic_memory.store_episode(
    state={'regime': 'crisis'},
    action={'weights': [0.2, 0.3, 0.5]},
    outcome={'pnl': 5000},
    metadata={'timestamp': datetime.now()}
)

# Recall similar episodes
similar = system.episodic_memory.recall_similar_episodes(
    current_state={'regime': 'crisis'},
    top_k=10
)
```

## 🎯 Performance Optimization

### GPU Acceleration

```python
system = ACTSv6Complete(
    input_dim=100,
    n_assets=10,
    device='cuda',  # Enable GPU
    use_pretrained=True
)
```

### Adaptive Sampling

The system automatically adjusts MC sample count based on entropy:

```python
# Low entropy (high confidence) → fewer samples
# High entropy (uncertainty) → more samples
n_samples = system.adaptive_mc_sampler.determine_n_samples(
    entropy=0.12,
    budget_ms=500
)
```

### Model Checkpointing

```python
# Save state
system.save_state('models/acts_v6_checkpoint.pkl')

# Load state
system.load_state('models/acts_v6_checkpoint.pkl')
```

## 🔒 Security & Privacy

### Federated Learning

Privacy-preserving distributed training:

```python
result = await system.federated_coordinator.federated_training_round()
print(f"Privacy budget used: {result['privacy_budget_used']}")
```

### Compliance

- Zero-Knowledge Proofs (ZKP) for transaction privacy
- Differential Privacy for data protection
- GDPR-compliant data handling

## 📈 Monitoring & Observability

### Performance Metrics

```python
metrics = system.performance_metrics
print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
print(f"Max Drawdown: {metrics['drawdown']:.2%}")
print(f"Total PnL: ${metrics['pnl']:,.0f}")
```

### Self-Evolution

The system monitors itself and triggers improvements:

```python
improvement_result = await system.self_evolution.monitor_and_improve()

if improvement_result['improvements_made']:
    print("System improvements triggered:")
    for improvement in improvement_result['improvements_made']:
        print(f"  - {improvement['type']}: {improvement['reason']}")
```

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```python
# Use CPU or reduce batch size
system = ACTSv6Complete(device='cpu')
```

### Transformers Import Error

```bash
# Install transformers
pip install transformers tokenizers
```

### FAISS Not Available

```bash
# System falls back to simple storage automatically
# Or install FAISS:
pip install faiss-cpu
```

## 📚 Advanced Topics

### Custom Causal Graphs

```python
from acts_v6_complete import SimpleCausalGraph

graph = SimpleCausalGraph()
graph.add_node('GDP', prior_mean=0.02, prior_std=0.01)
graph.add_node('Stocks', prior_mean=0.08, prior_std=0.15)
graph.add_edge('GDP', 'Stocks', weight=3.5)
```

### Custom Risk Scenarios

```python
system.existential_risk.scenarios['custom_event'] = {
    'description': 'Custom black swan event',
    'probability': 0.001,
    'impact': {'BTC': -0.50, 'GLD': +0.20}
}
```

### Custom LLM Backends

```python
# Override debate system with your LLM
system.multi_agent_debate.llm_backend = "your-llm-endpoint"
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Pearl (2009)**: Causal inference framework
- **Kingma & Welling (2013)**: Variational inference
- **OpenAI**: CLIP, Whisper, GPT
- **Hugging Face**: Transformers library
- **Facebook Research**: FAISS vector search

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: [Full Docs](https://docs.your-site.com)
- **Email**: support@your-site.com

## 🗺️ Roadmap

### v6.1 (Q1 2026)
- [ ] Real-time streaming data ingestion
- [ ] Advanced NAS with reinforcement learning
- [ ] Multi-exchange connectivity

### v6.2 (Q2 2026)
- [ ] Quantum-resistant cryptography
- [ ] Advanced explainability (SHAP, LIME)
- [ ] Mobile app interface

### v7.0 (Q4 2026)
- [ ] Full AGI integration
- [ ] Neuromorphic computing support
- [ ] Decentralized autonomous organization (DAO)

---

**Version**: 6.0.0 (Complete)  
**Date**: 2025-11-07  
**Author**: ACTS Development Team  

⭐ **Star this repo if you find it useful!**
