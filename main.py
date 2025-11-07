#!/usr/bin/env python3
"""
Main entry point for ACTS v6.0
Simplified interface for running the trading system
"""

import asyncio
import logging
import sys
import numpy as np
from acts_v6_complete import ACTSv6Complete

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


async def run_trading_system():
    """Run a complete trading cycle"""
    logger.info("Initializing ACTS v6.0...")
    
    # Initialize system
    system = ACTSv6Complete(input_dim=100, n_assets=5)
    
    # Display system status
    status = system.get_system_status()
    logger.info("\n" + "="*70)
    logger.info("System Status:")
    logger.info("="*70)
    for key, value in status.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for k, v in value.items():
                logger.info(f"  {k}: {v}")
        else:
            logger.info(f"{key}: {value}")
    
    # Simulate market data
    logger.info("\n" + "="*70)
    logger.info("Processing Trading Cycle...")
    logger.info("="*70)
    
    market_data = np.random.randn(100, 50)
    news_articles = [
        "Federal Reserve raises interest rates by 0.25%",
        "Bitcoin adoption increases among institutional investors",
        "China announces new economic stimulus package"
    ]
    social_posts = [
        "BTC to the moon! 🚀",
        "Market crash incoming, be careful",
        "Gold is the safe haven"
    ]
    
    try:
        result = await system.process_trading_cycle(
            market_data=market_data,
            news_articles=news_articles,
            social_posts=social_posts,
            constraints={
                'max_leverage': 2.0,
                'max_position': 0.1,
                'risk_budget': 0.05
            }
        )
        
        # Display results
        logger.info("\n" + "="*70)
        logger.info("Trading Cycle Results:")
        logger.info("="*70)
        logger.info(f"Regime Prediction: {result['regime_prediction']}")
        logger.info(f"Uncertainty (Epistemic): {result['uncertainty']['epistemic']:.4f}")
        logger.info(f"Uncertainty (Entropy): {result['uncertainty']['entropy']:.4f}")
        logger.info(f"Consensus Strategy Sharpe: {result['consensus_strategy']['sharpe']:.2f}")
        logger.info(f"Risk Scenario: {result['risk_assessment']['scenario']}")
        logger.info(f"Expected Loss: {result['risk_assessment']['expected_loss']:.2f}")
        logger.info("="*70)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in trading cycle: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        result = asyncio.run(run_trading_system())
        logger.info("\n✅ Trading cycle completed successfully!")
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)
