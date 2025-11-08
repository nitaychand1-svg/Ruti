#!/usr/bin/env python3
"""Production prediction script."""
import asyncio
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.orchestrator import GARCHHyperAdaptiveEnsemble
from src.core.config import load_config

async def main():
    parser = argparse.ArgumentParser(description='Run prediction with trained ensemble')
    parser.add_argument('--config', default='configs/production.yaml')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--regime', default='normal', 
                       choices=['normal', 'low_volatility_bull', 'high_volatility', 'chaotic'])
    args = parser.parse_args()
    
    config = load_config(args.config)
    ensemble = GARCHHyperAdaptiveEnsemble(config)
    
    # Load data
    import pandas as pd
    df = pd.read_parquet(args.data_path)
    X = df.values
    
    print(f"Making predictions on {len(X)} samples...")
    
    # Predict
    result = await ensemble.predict(X, market_regime=args.regime)
    
    print(f"\n✅ Prediction Results:")
    print(f"   Prediction: {result['prediction']}")
    print(f"   Confidence: {result['confidence']:.4f}")
    print(f"   Diversity: {result['diversity']:.4f}")
    print(f"   Position Size: {result['position_size']:.4f}")
    print(f"   Effective Bets: {result['effective_bets']:.2f}")
    print(f"   GARCH Volatility: {result['garch_volatility']:.6f}")

if __name__ == '__main__':
    asyncio.run(main())
