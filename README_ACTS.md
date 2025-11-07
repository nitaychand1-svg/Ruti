# ACTS v6.0 — Adaptive Causal Trading System

## Полная интеграция v5.0 + v5.5

**ACTS v6.0** — это комплексная система алгоритмической торговли, объединяющая все компоненты из версий 5.0 (Advanced Multi-Agent Intelligence) и 5.5 (Production Improvements).

## Архитектура

### Layer 1: Multi-Modal Perception
- **Multi-Modal Fusion Engine**: Объединение NLP + Vision + Audio
- **Bayesian Regime Predictor**: Предсказание режимов рынка с ELBO обучением
- **Uncertainty Quantification**: Эпистемическая и алеаторная неопределенность

### Layer 2: Strategic Intelligence
- **Multi-Agent Debate System**: 6 LLM агентов для генерации стратегий
- **Consensus via Weighted Voting**: Взвешенное голосование
- **RLHF Alignment**: Выравнивание с человеческими ценностями

### Layer 3: Execution Control
- **Hierarchical MARL Swarm**: 5 специализированных агентов
- **Adversarial HFT Defense**: Защита от высокочастотной торговли
- **Smart Order Routing**: Маршрутизация ордеров

### Core: Adaptive Causal Kernel
- **World Model Builder**: Построение графа причинно-следственных связей
- **Sequential Interventions**: Временные цепочки интервенций (v5.5)
- **Counterfactual Engine**: Контрфактический анализ
- **Episodic Memory**: Векторная база данных для памяти

### Layer 4: Risk Management
- **Existential Risk Simulator**: Симуляция черных лебедей
- **Importance Sampling**: Важностное сэмплирование для редких событий (v5.5)
- **Adaptive VaR**: Адаптивный Value at Risk

### Layer 5: Self-Improvement
- **Self-Evolution Oracle**: Автономное улучшение через NAS + Meta-RL
- **ELBO Optimizer**: Оптимизация Evidence Lower Bound (v5.5)
- **Adaptive MC Sampler**: Адаптивное Монте-Карло сэмплирование (v5.5)
- **Convergence Diagnostics**: Диагностика сходимости (Gelman-Rubin R̂)

### Layer 6: Human Interface
- **Conversational Explanations**: Объяснения на естественном языке
- **Interactive DAG Visualization**: Интерактивная визуализация графов
- **SHAP + Counterfactual Explanations**: Интерпретируемость

## Установка

```bash
# Установка зависимостей
pip install -r requirements.txt

# Для полной функциональности (опционально):
pip install transformers clip-by-openai faiss-cpu networkx
```

## Использование

### Базовый пример

```python
import asyncio
from acts_v6_complete import ACTSv6Complete
import numpy as np

async def main():
    # Инициализация системы
    system = ACTSv6Complete(input_dim=100, n_assets=5)
    
    # Данные рынка
    market_data = np.random.randn(100, 50)
    news_articles = ["Fed raises rates", "Bitcoin adoption increases"]
    social_posts = ["BTC to the moon!", "Market crash incoming"]
    
    # Обработка торгового цикла
    result = await system.process_trading_cycle(
        market_data=market_data,
        news_articles=news_articles,
        social_posts=social_posts,
        constraints={'max_leverage': 2.0}
    )
    
    print(f"Regime: {result['regime_prediction']}")
    print(f"Strategy Sharpe: {result['consensus_strategy']['sharpe']:.2f}")

asyncio.run(main())
```

### Запуск через main.py

```bash
python main.py
```

### Запуск тестов

```bash
python test_acts_v6.py
# или
pytest test_acts_v6.py -v
```

## Компоненты

### 1. Multi-Modal Fusion Engine
Объединяет текстовые, визуальные и аудио данные в единое представление.

```python
engine = MultiModalFusionEngine()
features, uncertainty = await engine.perceive_world(
    market_data=market_data,
    news_articles=news,
    social_posts=posts
)
```

### 2. Bayesian Regime Predictor
Предсказывает режимы рынка с байесовской неопределенностью.

```python
predictor = BayesianRegimePredictor(input_dim=4096)
probs, epistemic_unc, entropy = predictor.predict_with_uncertainty(features)
```

### 3. Sequential Intervention Engine
Применяет временные цепочки причинных интервенций.

```python
interventions = [
    TemporalIntervention(variable='FED', value=0.06, timestep=5),
    TemporalIntervention(variable='BTC', value=55000, timestep=10)
]
result = engine.apply_intervention_chain(interventions, horizon=30)
```

### 4. Existential Risk Simulator
Симулирует экстремальные сценарии с важностным сэмплированием.

```python
simulator = ExistentialRiskSimulator()
risk = simulator.simulate_scenario('solar_flare', portfolio)
```

## Целевые показатели производительности

- **Latency**: <500ms (p95)
- **OOS Sharpe**: >2.4
- **Drawdown**: <7%
- **Regime Accuracy**: >97%
- **AMI Score**: >0.90

## Структура проекта

```
workspace/
├── acts_v6_complete.py    # Основной файл с полной реализацией
├── main.py                 # Точка входа для запуска
├── test_acts_v6.py        # Тесты
├── requirements.txt        # Зависимости
└── README.md              # Документация
```

## Особенности v6.0

### Из v5.0 (AMI Components):
- ✓ Multi-Modal Fusion
- ✓ Multi-Agent Debate
- ✓ Hierarchical MARL
- ✓ World Model Builder
- ✓ Episodic Memory
- ✓ Existential Risk Simulator
- ✓ Self-Evolution Oracle
- ✓ Federated Learning
- ✓ Human-AI Interface

### Из v5.5 (Production Improvements):
- ✓ ELBO-based Training
- ✓ Sequential Interventions
- ✓ KL Annealing
- ✓ Importance Sampling
- ✓ Adaptive MC Sampling
- ✓ Convergence Diagnostics

## Лицензия

Проект разработан командой ACTS Development Team.

## Версия

**v6.0.0** — Complete Integration (2025-11-07)

## Поддержка

Для вопросов и предложений создайте issue в репозитории проекта.
