#!/usr/bin/env python3
"""Production training script."""
import asyncio
import argparse
import yaml
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.orchestrator import GARCHHyperAdaptiveEnsemble
from src.core.config import load_config

async def main():
    parser = argparse.ArgumentParser(description='Train GARCH-HyperNetwork Ensemble')
    parser.add_argument('--config', default='configs/production.yaml', 
                       help='Path to config file')
    parser.add_argument('--data-path', required=True,
                       help='Path to training data (parquet format)')
    parser.add_argument('--target-column', default='target',
                       help='Name of target column')
    args = parser.parse_args()
    
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    print("Initializing ensemble...")
    ensemble = GARCHHyperAdaptiveEnsemble(config)
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    import pandas as pd
    df = pd.read_parquet(args.data_path)
    
    X = df.drop(args.target_column, axis=1).values
    y = df[args.target_column].values
    features = df.drop(args.target_column, axis=1).columns.tolist()
    
    print(f"Training on {len(X)} samples with {len(features)} features...")
    
    # Train
    result = await ensemble.train(X, y, features)
    
    print(f"✅ Training completed successfully!")
    print(f"   Version: {result['version']}")
    print(f"   OOS Score: {result.get('oos_score', 'N/A'):.4f}")

if __name__ == '__main__':
    asyncio.run(main())
