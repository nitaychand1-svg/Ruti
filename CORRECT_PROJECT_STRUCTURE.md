# Правильная структура ACTS v6.0 - Модульная версия

## Основной файл: `acts_v6/system.py`

Это правильная версия, которая использует модульные компоненты:

```python
from .perception import ELBOOptimizer, MultiModalFusionEngine, BayesianRegimePredictor
from .debate import MultiAgentDebateSystem
from .marl import HierarchicalMARLSwarm
from .world_model import SequentialInterventionEngine, TemporalIntervention, WorldModelBuilder
from .memory import EpisodicMemory
from .risk import ExistentialRiskSimulator, ImportanceSampler
# ... и другие модули
```

## Модульные компоненты (полные реализации):

### 1. `perception.py` (422 строки)
- **MultiModalFusionEngine** - Полная реализация с fallback механизмами
- **BayesianRegimePredictor** - Байесовский предсказатель режимов
- **ELBOOptimizer** - Оптимизатор с KL annealing
- Поддержка RoBERTa, CLIP, Whisper с graceful fallbacks

### 2. `debate.py` (224 строки)
- **MultiAgentDebateSystem** - Полная система дебатов
- **StrategyProposal** - Структурированные предложения
- **DebateRound** - Раунды дебатов
- Heuristic и LLM-based генераторы

### 3. `marl.py` (173 строки)
- **HierarchicalMARLSwarm** - Полная реализация MARL роя
- **ExecutionTask** и **ExecutionResult** - Структурированные задачи
- Transaction Cost Analysis
- Replay buffer для обучения

### 4. `world_model.py` (296 строк)
- **WorldModelBuilder** - Построитель модели мира
- **SequentialInterventionEngine** - Двигатель вмешательств
- **SimpleCausalGraph** - Причинный граф
- Gelman-Rubin диагностика сходимости

### 5. Другие модули:
- `memory.py` - EpisodicMemory
- `risk.py` - ExistentialRiskSimulator
- `self_evolution.py` - SelfEvolutionOracle
- `federated.py` - FederatedTrainingCoordinator
- `interface.py` - HumanAIInterface
- `adaptive_sampling.py` - AdaptiveMCSampler

## Использование правильной версии:

```python
# Правильный импорт (из модульной версии)
from acts_v6 import ACTSv6Complete

# Или напрямую
from acts_v6.system import ACTSv6Complete

system = ACTSv6Complete(input_dim=100, n_assets=10)
result = await system.full_cycle(...)
```

## Разница между версиями:

| Аспект | Модульная (`system.py`) | Монолитная (`acts_v6_complete.py`) |
|--------|-------------------------|-------------------------------------|
| Структура | Разделена на модули | Все в одном файле |
| Код | Полные реализации | Упрощенные версии |
| Комментарии | Production-ready | "Simplified", "Placeholder" |
| Размер | ~2000+ строк (разделено) | 1724 строки (все вместе) |
| Использование | ✅ Рекомендуется | Для быстрого прототипирования |

## Вывод:

**Правильная версия проекта - это модульная структура в `acts_v6/system.py` и отдельных модулях.**

Монолитная версия `acts_v6/src/acts_v6_complete.py` - это упрощенная версия для быстрого старта или демонстрации, но не production-ready код.
