#!/usr/bin/env python3
"""
Unit tests for ACTS v6.0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import numpy as np
import torch
from acts_v6_complete import (
    ACTSv6Complete,
    BayesianRegimePredictor,
    TemporalIntervention,
    SimpleCausalGraph,
    EpisodicMemory,
    ExistentialRiskSimulator,
    AdaptiveMCSampler
)


class TestBayesianRegimePredictor:
    """Test Bayesian regime predictor"""
    
    def test_initialization(self):
        model = BayesianRegimePredictor(input_dim=100, hidden_dim=64, n_regimes=3)
        assert model is not None
    
    def test_forward_pass(self):
        model = BayesianRegimePredictor(input_dim=100, hidden_dim=64, n_regimes=3)
        x = torch.randn(5, 100)
        output = model(x)
        
        assert output.shape == (5, 3)
        assert torch.allclose(output.sum(dim=1), torch.ones(5), atol=1e-5)
    
    def test_uncertainty_prediction(self):
        model = BayesianRegimePredictor(input_dim=100, hidden_dim=64, n_regimes=3)
        x = torch.randn(5, 100)
        
        mean_probs, epistemic_unc, entropy = model.predict_with_uncertainty(x, n_samples=10)
        
        assert mean_probs.shape == (5, 3)
        assert epistemic_unc.shape == (5,)
        assert entropy.shape == (5,)


class TestSimpleCausalGraph:
    """Test causal graph implementation"""
    
    def test_add_node(self):
        graph = SimpleCausalGraph()
        graph.add_node('A', prior_mean=0.5, prior_std=0.1)
        
        assert 'A' in graph.nodes
        assert graph.nodes['A']['prior_mean'] == 0.5
    
    def test_add_edge(self):
        graph = SimpleCausalGraph()
        graph.add_node('A', prior_mean=0.0, prior_std=1.0)
        graph.add_node('B', prior_mean=0.0, prior_std=1.0)
        graph.add_edge('A', 'B', weight=2.0)
        
        assert 'A' in graph.nodes['B']['parents']
        assert graph.nodes['B']['weights']['A'] == 2.0
    
    def test_topological_sort(self):
        graph = SimpleCausalGraph()
        graph.add_node('A')
        graph.add_node('B')
        graph.add_node('C')
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        
        order = graph.topological_sort()
        
        assert order.index('A') < order.index('B')
        assert order.index('B') < order.index('C')


class TestEpisodicMemory:
    """Test episodic memory"""
    
    def test_store_episode(self):
        memory = EpisodicMemory(db_backend='simple', embedding_dim=128)
        
        memory.store_episode(
            state={'regime': 'bull'},
            action={'weights': [0.5, 0.5]},
            outcome={'pnl': 1000},
            metadata={'timestamp': 'now'}
        )
        
        assert len(memory.episodes) == 1
    
    def test_recall_episodes(self):
        memory = EpisodicMemory(db_backend='simple', embedding_dim=128)
        
        for i in range(10):
            memory.store_episode(
                state={'regime': 'bull'},
                action={'weights': [0.5, 0.5]},
                outcome={'pnl': i * 100},
                metadata={'id': i}
            )
        
        similar = memory.recall_similar_episodes({'regime': 'bull'}, top_k=5)
        
        assert len(similar) <= 5


class TestExistentialRiskSimulator:
    """Test risk simulator"""
    
    def test_scenarios_exist(self):
        simulator = ExistentialRiskSimulator()
        
        assert 'solar_flare' in simulator.scenarios
        assert 'cyber_attack' in simulator.scenarios
        assert len(simulator.scenarios) == 5
    
    def test_simulate_scenario(self):
        simulator = ExistentialRiskSimulator()
        portfolio = {'BTC': 100000, 'SPY': 50000}
        
        result = simulator.simulate_scenario(
            scenario_name='cyber_attack',
            portfolio=portfolio,
            n_samples=1000
        )
        
        assert 'expected_loss' in result
        assert 'var_95' in result
        assert 'survival_probability' in result


class TestAdaptiveMCSampler:
    """Test adaptive MC sampler"""
    
    def test_determine_samples_low_entropy(self):
        sampler = AdaptiveMCSampler(min_samples=1000, max_samples=50000)
        n_samples = sampler.determine_n_samples(entropy=0.02, budget_ms=500)
        
        assert n_samples == 1000
    
    def test_determine_samples_high_entropy(self):
        sampler = AdaptiveMCSampler(min_samples=1000, max_samples=50000)
        n_samples = sampler.determine_n_samples(entropy=0.20, budget_ms=500)
        
        assert n_samples > 1000


@pytest.mark.asyncio
class TestACTSv6Complete:
    """Test complete ACTS v6.0 system"""
    
    @pytest.fixture
    def system(self):
        return ACTSv6Complete(
            input_dim=100,
            n_assets=10,
            device='cpu',
            use_pretrained=False
        )
    
    def test_initialization(self, system):
        assert system is not None
        assert system.device == 'cpu'
        assert system.n_assets == 10
    
    async def test_full_trading_cycle(self, system):
        market_data = np.random.randn(100, 50)
        news_articles = ["Test news"]
        portfolio = {'BTC': 100000}
        constraints = {'max_position_size': 0.25}
        
        result = await system.full_trading_cycle(
            market_data=market_data,
            news_articles=news_articles,
            portfolio=portfolio,
            constraints=constraints
        )
        
        assert 'regime' in result
        assert 'strategy' in result
        assert 'execution' in result
    
    async def test_train_regime_predictor(self, system):
        training_data = [(np.random.randn(4096), np.random.randint(0, 3)) for _ in range(10)]
        
        result = await system.train_regime_predictor(
            training_data=training_data,
            n_epochs=2
        )
        
        assert 'final_loss' in result
        assert 'converged' in result
    
    def test_save_and_load_state(self, system, tmp_path):
        save_path = tmp_path / "test_state.pkl"
        
        system.save_state(str(save_path))
        assert save_path.exists()
        
        system.load_state(str(save_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
