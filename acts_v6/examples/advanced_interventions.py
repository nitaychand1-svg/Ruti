#!/usr/bin/env python3
"""
ACTS v6.0 — Advanced Causal Interventions Example

Demonstrates:
1. Complex temporal intervention chains
2. Counterfactual analysis
3. Scenario comparison
4. Risk attribution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import asyncio
from acts_v6_complete import (
    ACTSv6Complete,
    TemporalIntervention,
    logger
)


async def main():
    logger.info("=" * 70)
    logger.info("ACTS v6.0 — Advanced Causal Interventions")
    logger.info("=" * 70)
    
    # Initialize system
    system = ACTSv6Complete(
        input_dim=100,
        n_assets=10,
        device='cpu',
        use_pretrained=False
    )
    
    # Prepare data
    market_data = np.random.randn(100, 50)
    news_articles = ["Market update", "Economic report"]
    portfolio = {'BTC': 100000, 'SPY': 50000, 'GLD': 25000}
    constraints = {'max_position_size': 0.25}
    
    # Build world model first
    logger.info("\n[1] Building world model...")
    await system.full_trading_cycle(
        market_data=market_data,
        news_articles=news_articles,
        portfolio=portfolio,
        constraints=constraints
    )
    
    # ========================================================================
    # SCENARIO 1: Fed Rate Hike Chain
    # ========================================================================
    
    logger.info("\n[2] Scenario 1: Fed Rate Hike Chain")
    logger.info("-" * 70)
    
    scenario1_interventions = [
        TemporalIntervention(
            variable='FED',
            value=0.055,
            timestep=0,
            metadata={'description': 'Initial rate 5.5%'}
        ),
        TemporalIntervention(
            variable='FED',
            value=0.060,
            timestep=5,
            metadata={'description': 'Rate hike to 6.0%'}
        ),
        TemporalIntervention(
            variable='FED',
            value=0.065,
            timestep=10,
            metadata={'description': 'Rate hike to 6.5%'}
        ),
    ]
    
    result1 = system.run_causal_intervention(
        interventions=scenario1_interventions,
        horizon=30
    )
    
    logger.info(f"Final regime: {result1['final_state']['regime_probs']}")
    logger.info(f"Convergence R̂: {result1['convergence_metrics']['rhat']:.4f}")
    logger.info(f"Converged: {result1['convergence_metrics']['converged']}")
    
    # ========================================================================
    # SCENARIO 2: Crypto Crash + Recovery
    # ========================================================================
    
    logger.info("\n[3] Scenario 2: Crypto Crash + Recovery")
    logger.info("-" * 70)
    
    scenario2_interventions = [
        TemporalIntervention(
            variable='BTC',
            value=-0.50,  # -50% crash
            timestep=5,
            metadata={'description': 'Major crypto crash'}
        ),
        TemporalIntervention(
            variable='BTC',
            value=0.30,  # +30% recovery
            timestep=15,
            metadata={'description': 'Partial recovery'}
        ),
    ]
    
    result2 = system.run_causal_intervention(
        interventions=scenario2_interventions,
        horizon=30
    )
    
    logger.info(f"Final regime: {result2['final_state']['regime_probs']}")
    logger.info(f"Causal attribution: {result2['causal_attribution']}")
    
    # ========================================================================
    # SCENARIO 3: Global Crisis
    # ========================================================================
    
    logger.info("\n[4] Scenario 3: Global Economic Crisis")
    logger.info("-" * 70)
    
    scenario3_interventions = [
        TemporalIntervention(
            variable='China',
            value=-0.10,  # GDP shock
            timestep=2,
            metadata={'description': 'China GDP decline'}
        ),
        TemporalIntervention(
            variable='FED',
            value=0.03,  # Emergency cut
            timestep=5,
            metadata={'description': 'Emergency Fed cut to 3%'}
        ),
        TemporalIntervention(
            variable='SPY',
            value=-0.25,
            timestep=8,
            metadata={'description': 'Stock market crash'}
        ),
    ]
    
    result3 = system.run_causal_intervention(
        interventions=scenario3_interventions,
        horizon=30
    )
    
    logger.info(f"Final regime: {result3['final_state']['regime_probs']}")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    
    logger.info("\n[5] Scenario Comparison")
    logger.info("-" * 70)
    
    scenarios = {
        'Fed Rate Hike': result1,
        'Crypto Crash': result2,
        'Global Crisis': result3
    }
    
    for name, result in scenarios.items():
        regime_probs = result['final_state']['regime_probs']
        crisis_prob = regime_probs[2] if len(regime_probs) > 2 else 0
        logger.info(f"{name:20s} → Crisis Probability: {crisis_prob:.2%}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Advanced Interventions Example Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
