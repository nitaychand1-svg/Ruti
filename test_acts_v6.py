"""
Example usage and tests for ACTS v6.0
"""

import pytest
import asyncio
import numpy as np
import torch
from acts_v6_complete import (
    ACTSv6Complete,
    MultiModalFusionEngine,
    BayesianRegimePredictor,
    ELBOOptimizer,
    MultiAgentDebateSystem,
    SequentialInterventionEngine,
    TemporalIntervention,
    SimpleCausalGraph,
    EpisodicMemory,
    ExistentialRiskSimulator,
    AdaptiveMCSampler
)


@pytest.mark.asyncio
async def test_multi_modal_fusion():
    """Test multi-modal fusion engine"""
    engine = MultiModalFusionEngine()
    
    market_data = np.random.randn(10, 50)
    news = ["Test news article"]
    posts = ["Test social post"]
    
    features, uncertainty = await engine.perceive_world(
        market_data=market_data,
        news_articles=news,
        social_posts=posts
    )
    
    assert features.shape == (4096,)
    assert 'epistemic' in uncertainty
    assert 'aleatoric' in uncertainty
    assert 'total' in uncertainty
    print("✅ Multi-modal fusion test passed")


@pytest.mark.asyncio
async def test_regime_predictor():
    """Test Bayesian regime predictor"""
    predictor = BayesianRegimePredictor(input_dim=4096, hidden_dim=128, n_regimes=3)
    
    # Create dummy features
    features = torch.randn(1, 4096)
    
    # Predict with uncertainty
    probs, epistemic_unc, entropy = predictor.predict_with_uncertainty(features, n_samples=10)
    
    assert probs.shape == (1, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)
    assert epistemic_unc.shape == (1,)
    assert entropy.shape == (1,)
    print("✅ Regime predictor test passed")


@pytest.mark.asyncio
async def test_multi_agent_debate():
    """Test multi-agent debate system"""
    debate_system = MultiAgentDebateSystem()
    
    features = np.random.randn(4096)
    regime_probs = {'bull': 0.6, 'bear': 0.3, 'crisis': 0.1}
    constraints = {'max_leverage': 2.0}
    
    result = await debate_system.orchestrate_debate(
        features=features,
        regime_probs=regime_probs,
        constraints=constraints
    )
    
    assert 'consensus_strategy' in result
    assert 'agent_proposals' in result
    assert 'alignment_score' in result
    assert len(result['agent_proposals']) == 6  # 6 agents
    print("✅ Multi-agent debate test passed")


@pytest.mark.asyncio
async def test_sequential_interventions():
    """Test sequential intervention engine"""
    # Create causal graph
    graph = SimpleCausalGraph()
    graph.add_node('FED', prior_mean=0.05, prior_std=0.01)
    graph.add_node('BTC', prior_mean=50000, prior_std=1000)
    graph.add_node('SPY', prior_mean=400, prior_std=10)
    graph.add_edge('FED', 'BTC', weight=-2.0)
    graph.add_edge('BTC', 'SPY', weight=0.5)
    
    engine = SequentialInterventionEngine(graph, n_mc_samples=1000)
    
    interventions = [
        TemporalIntervention(variable='FED', value=0.06, timestep=5),
        TemporalIntervention(variable='BTC', value=55000, timestep=10)
    ]
    
    result = engine.apply_intervention_chain(interventions, horizon=30)
    
    assert 'final_state' in result
    assert 'trajectory' in result
    assert 'causal_attribution' in result
    assert 'convergence_metrics' in result
    assert len(result['trajectory']) == 30
    print("✅ Sequential interventions test passed")


@pytest.mark.asyncio
async def test_episodic_memory():
    """Test episodic memory"""
    memory = EpisodicMemory()
    
    # Store episodes
    for i in range(5):
        memory.store_episode(
            state={'features': np.random.randn(100)},
            action={'weights': np.random.randn(5)},
            outcome={'pnl': np.random.randn()},
            metadata={'regime': 'bull' if i % 2 == 0 else 'bear'}
        )
    
    # Recall similar episodes
    similar = memory.recall_similar_episodes({'features': np.random.randn(100)}, top_k=3)
    assert len(similar) <= 3
    
    # Extract crisis playbook
    playbook = memory.extract_crisis_playbook()
    assert 'n_crises' in playbook
    assert 'success_rate' in playbook
    print("✅ Episodic memory test passed")


def test_existential_risk():
    """Test existential risk simulator"""
    simulator = ExistentialRiskSimulator()
    
    portfolio = {'BTC': 100000, 'SPY': 50000, 'GLD': 30000}
    
    result = simulator.simulate_scenario('solar_flare', portfolio, n_samples=1000)
    
    assert 'scenario' in result
    assert 'expected_loss' in result
    assert 'var_95' in result
    assert 'survival_probability' in result
    assert 'recommended_hedges' in result
    print("✅ Existential risk test passed")


def test_adaptive_mc_sampler():
    """Test adaptive MC sampler"""
    sampler = AdaptiveMCSampler(min_samples=1000, max_samples=50000)
    
    # Low entropy -> fewer samples
    n_low = sampler.determine_n_samples(entropy=0.02, budget_ms=500.0)
    assert n_low == 1000
    
    # High entropy -> more samples
    n_high = sampler.determine_n_samples(entropy=0.20, budget_ms=500.0)
    assert n_high > n_low
    
    # Budget constraint
    n_budget = sampler.determine_n_samples(entropy=0.20, budget_ms=10.0)
    assert n_budget <= int(10.0 / 0.01)
    print("✅ Adaptive MC sampler test passed")


@pytest.mark.asyncio
async def test_complete_system():
    """Test complete ACTS v6.0 system"""
    system = ACTSv6Complete(input_dim=100, n_assets=5)
    
    # Check system status
    status = system.get_system_status()
    assert status['version'] == '6.0.0'
    assert 'components' in status
    
    # Run trading cycle
    market_data = np.random.randn(50, 50)
    news = ["Test news"]
    posts = ["Test post"]
    
    result = await system.process_trading_cycle(
        market_data=market_data,
        news_articles=news,
        social_posts=posts
    )
    
    assert 'regime_prediction' in result
    assert 'consensus_strategy' in result
    assert 'risk_assessment' in result
    assert 'execution_result' in result
    print("✅ Complete system test passed")


@pytest.mark.asyncio
async def test_training():
    """Test regime predictor training"""
    system = ACTSv6Complete(input_dim=100, n_assets=5)
    
    # Generate dummy training data
    features = np.random.randn(100, 4096)
    targets = np.random.randint(0, 3, size=100)
    
    # Train for a few epochs
    history = await system.train_regime_predictor(features, targets, epochs=5)
    
    assert 'elbo' in history
    assert 'loss' in history
    assert 'kl' in history
    assert len(history['elbo']) == 5
    print("✅ Training test passed")


if __name__ == "__main__":
    # Run all tests
    print("="*70)
    print("Running ACTS v6.0 Tests")
    print("="*70)
    
    # Run async tests
    asyncio.run(test_multi_modal_fusion())
    asyncio.run(test_regime_predictor())
    asyncio.run(test_multi_agent_debate())
    asyncio.run(test_sequential_interventions())
    asyncio.run(test_episodic_memory())
    test_existential_risk()
    test_adaptive_mc_sampler()
    asyncio.run(test_complete_system())
    asyncio.run(test_training())
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
