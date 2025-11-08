#!/usr/bin/env python3
"""
ACTS v6.0 — Basic Usage Example

Demonstrates:
1. System initialization
2. Data preparation
3. Full trading cycle
4. Training regime predictor
5. Causal interventions
6. Human explanations
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import asyncio
from acts_v6_complete import (
    ACTSv6Complete,
    TemporalIntervention,
    logger
)


async def main():
    """Main example workflow"""
    
    logger.info("=" * 70)
    logger.info("ACTS v6.0 — Basic Usage Example")
    logger.info("=" * 70)
    
    # ========================================================================
    # 1. INITIALIZE SYSTEM
    # ========================================================================
    
    logger.info("\n[1] Initializing ACTS v6.0 system...")
    
    system = ACTSv6Complete(
        input_dim=100,
        n_assets=10,
        device='cpu',  # Use 'cuda' if GPU available
        use_pretrained=False  # Set to True to use RoBERTa, CLIP, etc.
    )
    
    # ========================================================================
    # 2. PREPARE DATA
    # ========================================================================
    
    logger.info("\n[2] Preparing synthetic market data...")
    
    # Synthetic market data (replace with real data)
    market_data = np.random.randn(100, 50)  # [timesteps, features]
    
    # News articles (replace with real news)
    news_articles = [
        "Fed signals potential rate hike next quarter",
        "Tech stocks rally on strong earnings",
        "Bitcoin breaks $50k resistance level"
    ]
    
    # Portfolio
    portfolio = {
        'BTC': 100000,
        'SPY': 50000,
        'GLD': 25000
    }
    
    # Constraints
    constraints = {
        'max_position_size': 0.25,
        'max_leverage': 2.0,
        'min_sharpe': 1.5
    }
    
    # ========================================================================
    # 3. RUN FULL TRADING CYCLE
    # ========================================================================
    
    logger.info("\n[3] Running full trading cycle...")
    
    result = await system.full_trading_cycle(
        market_data=market_data,
        news_articles=news_articles,
        portfolio=portfolio,
        constraints=constraints
    )
    
    logger.info("\n[Trading Cycle Results]")
    logger.info(f"Timestamp: {result['timestamp']}")
    logger.info(f"Detected Regime: {result['regime']}")
    logger.info(f"Regime Probabilities: {result['regime_probs']}")
    logger.info(f"Epistemic Uncertainty: {result['uncertainty']['epistemic']:.4f}")
    logger.info(f"Strategy: {result['strategy']}")
    logger.info(f"Execution: {result['execution']['aggregate_metrics']}")
    
    # ========================================================================
    # 4. TRAIN REGIME PREDICTOR (Optional)
    # ========================================================================
    
    logger.info("\n[4] Training regime predictor with synthetic data...")
    
    # Generate synthetic training data
    training_data = []
    for _ in range(100):
        features = np.random.randn(4096)
        label = np.random.randint(0, 3)  # 0=bull, 1=normal, 2=crisis
        training_data.append((features, label))
    
    training_result = await system.train_regime_predictor(
        training_data=training_data,
        n_epochs=10  # Use more epochs for real training
    )
    
    logger.info(f"Training complete. Final loss: {training_result['final_loss']:.4f}")
    logger.info(f"Converged: {training_result['converged']}")
    
    # ========================================================================
    # 5. RUN CAUSAL INTERVENTIONS
    # ========================================================================
    
    logger.info("\n[5] Running causal interventions...")
    
    # Define temporal interventions
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
    
    intervention_result = system.run_causal_intervention(
        interventions=interventions,
        horizon=30
    )
    
    logger.info(f"Intervention complete. Convergence R̂: {intervention_result['convergence_metrics']['rhat']:.4f}")
    logger.info(f"Final state regime: {intervention_result['final_state']['regime_probs']}")
    
    # ========================================================================
    # 6. GET HUMAN EXPLANATIONS
    # ========================================================================
    
    logger.info("\n[6] Getting human-readable explanations...")
    
    queries = [
        "Why did you choose this strategy?",
        "What if the Fed raises rates to 7%?"
    ]
    
    for query in queries:
        explanation = await system.explain_to_human(query)
        logger.info(f"\nQ: {query}")
        logger.info(f"A: {explanation}")
    
    # ========================================================================
    # 7. SAVE SYSTEM STATE
    # ========================================================================
    
    logger.info("\n[7] Saving system state...")
    
    save_path = "models/acts_v6_state.pkl"
    Path(save_path).parent.mkdir(exist_ok=True)
    system.save_state(save_path)
    
    logger.info(f"State saved to {save_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("ACTS v6.0 Example Complete!")
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("1. Replace synthetic data with real market data")
    logger.info("2. Train regime predictor on historical data")
    logger.info("3. Connect to real trading infrastructure")
    logger.info("4. Enable LLM backends for debate system")
    logger.info("5. Deploy with monitoring & alerts")


if __name__ == "__main__":
    asyncio.run(main())
