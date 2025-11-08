#!/usr/bin/env python3
"""Production training script."""
import asyncio
import argparse
import yaml
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.orchestrator import GARCHHyperAdaptiveEnsemble
from src.core.config import load_config

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/production.yaml')
    parser.add_argument('--data-path', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    ensemble = GARCHHyperAdaptiveEnsemble(config)
    
    # Load data
    import pandas as pd
    df = pd.read_parquet(args.data_path)
    
    X = df.drop('target', axis=1).values
    y = df['target'].values
    features = df.drop('target', axis=1).columns.tolist()
    
    # Train
    result = await ensemble.train(X, y, features)
    print(f"✅ Training completed: {result}")

if __name__ == '__main__':
    asyncio.run(main())
