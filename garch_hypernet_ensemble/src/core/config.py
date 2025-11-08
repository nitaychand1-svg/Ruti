"""Centralized configuration management."""
import os
from dataclasses import dataclass, field
from typing import List, Optional
import yaml

@dataclass
class GARCHConfig:
    """GARCH model configuration."""
    window: int = 252
    min_obs: int = 100
    enable_forecast: bool = True
    fit_interval_seconds: int = 3600

@dataclass
class HyperNetConfig:
    """HyperNetwork configuration."""
    meta_dim: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.2
    learning_rate: float = 0.001
    adaptation_rate: float = 0.05
    entropy_regularization: float = 0.01
    correlation_penalty: float = 0.05

@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size: float = 0.25
    max_risk_per_trade: float = 0.02
    transaction_cost: float = 0.001
    slippage_factor: float = 0.01
    cvar_percentile: int = 5

@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    window_size: int = 100
    max_correlation_threshold: float = 0.85
    min_confidence_threshold: float = 0.55
    min_diversity_threshold: float = 0.25
    enable_shap: bool = True
    adaptation_rate: float = 0.05

@dataclass
class CacheConfig:
    """Caching configuration."""
    ttl_seconds: int = 300
    maxsize: int = 1000
    redis_url: Optional[str] = None

@dataclass
class EnsembleConfig:
    """Main ensemble configuration."""
    garch: GARCHConfig = field(default_factory=GARCHConfig)
    hypernet: HyperNetConfig = field(default_factory=HyperNetConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    n_base_models: int = 64
    prune_threshold: float = 0.52
    model_version: str = "2.0.0"

def load_config(path: str = "configs/production.yaml") -> EnsembleConfig:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    return EnsembleConfig(
        garch=GARCHConfig(**data.get("garch", {})),
        hypernet=HyperNetConfig(**data.get("hypernet", {})),
        risk=RiskConfig(**data.get("risk_management", {})),
        monitoring=MonitoringConfig(**data.get("monitoring", {})),
        cache=CacheConfig(**data.get("cache", {})),
        n_base_models=data.get("ensemble", {}).get("n_base_models", 64),
        prune_threshold=data.get("pruning", {}).get("prune_threshold", 0.52),
        model_version="2.0.0"
    )
