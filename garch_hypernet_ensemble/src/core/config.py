"""Centralized configuration management."""
from __future__ import annotations

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
    enable_prometheus: bool = False
    alert_webhook: Optional[str] = None
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


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.getcwd(), path)


def load_config(path: str = "configs/production.yaml") -> EnsembleConfig:
    """Load configuration from YAML file."""

    resolved_path = _resolve_path(path)
    with open(resolved_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    ensemble_section = data.get("ensemble", {}) or {}
    garch_section = data.get("garch", {}) or {}
    hypernet_section = data.get("hypernet", {}) or {}
    monitoring_section = data.get("monitoring", {}) or {}
    risk_section = (
        data.get("risk", {})
        or data.get("risk_management", {})
        or {}
    )
    cache_section = data.get("cache", {}) or {}
    pruning_section = data.get("pruning", {}) or {}

    # Map legacy keys from ensemble section into garch config
    if "garch_window" in ensemble_section:
        garch_section.setdefault("window", ensemble_section.pop("garch_window"))
    if "garch_min_obs" in ensemble_section:
        garch_section.setdefault("min_obs", ensemble_section.pop("garch_min_obs"))
    if "enable_garch_forecast" in ensemble_section:
        garch_section.setdefault(
            "enable_forecast", ensemble_section.pop("enable_garch_forecast")
        )

    # Pruning thresholds feed into monitoring defaults when not explicitly supplied
    if "initial_max_correlation" in pruning_section:
        monitoring_section.setdefault(
            "max_correlation_threshold",
            pruning_section["initial_max_correlation"],
        )
    if "initial_min_confidence" in pruning_section:
        monitoring_section.setdefault(
            "min_confidence_threshold",
            pruning_section["initial_min_confidence"],
        )
    if "prune_threshold" in pruning_section:
        ensemble_section.setdefault("prune_threshold", pruning_section["prune_threshold"])

    garch_config = GARCHConfig(**garch_section)
    hypernet_config = HyperNetConfig(**hypernet_section)
    monitoring_config = MonitoringConfig(**monitoring_section)
    risk_config = RiskConfig(**risk_section)
    cache_config = CacheConfig(**cache_section)

    return EnsembleConfig(
        garch=garch_config,
        hypernet=hypernet_config,
        monitoring=monitoring_config,
        risk=risk_config,
        cache=cache_config,
        **ensemble_section,
    )
