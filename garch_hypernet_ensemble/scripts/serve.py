#!/usr/bin/env python3
"""Production serving script."""
import asyncio
import logging
from aiohttp import web
import json
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.orchestrator import GARCHHyperAdaptiveEnsemble
from src.core.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global ensemble instance
ensemble = None

async def predict_handler(request):
    """Handle prediction requests."""
    try:
        data = await request.json()
        X = np.array(data['features'])
        regime = data.get('regime', 'normal')
        ground_truth = data.get('ground_truth')
        
        result = await ensemble.predict(X, regime, ground_truth)
        
        return web.json_response({
            'prediction': result['prediction'].tolist(),
            'confidence': float(result['confidence']),
            'diversity': float(result['diversity']),
            'position_size': float(result['position_size'])
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return web.json_response({'error': str(e)}, status=500)

async def health_handler(request):
    """Health check endpoint."""
    return web.json_response({'status': 'healthy'})

async def init_app():
    """Initialize application."""
    global ensemble
    config = load_config('configs/production.yaml')
    ensemble = GARCHHyperAdaptiveEnsemble(config)
    
    app = web.Application()
    app.router.add_post('/predict', predict_handler)
    app.router.add_get('/health', health_handler)
    
    return app

def main():
    app = asyncio.run(init_app())
    web.run_app(app, host='0.0.0.0', port=8080)

if __name__ == '__main__':
    main()
