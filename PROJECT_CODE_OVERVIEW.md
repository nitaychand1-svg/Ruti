# Обзор Python кода проектов

## Проект 1: ACTS v6.0 (Adaptive Causal Trading System)

### Основной файл: `acts_v6/src/acts_v6_complete.py`

**Описание**: Полная интеграция системы торговли с мультимодальным восприятием, мультиагентным дебатом, иерархическим MARL, причинным моделированием и самоэволюцией.

**Ключевые компоненты**:

1. **MultiModalFusionEngine** - Мультимодальное восприятие (NLP + Vision + Audio)
2. **BayesianRegimePredictor** - Байесовский предсказатель режимов с ELBO обучением
3. **MultiAgentDebateSystem** - Система дебатов между 6 LLM агентами
4. **HierarchicalMARLSwarm** - Иерархический мультиагентный RL для исполнения
5. **WorldModelBuilder** - Построитель причинной модели мира
6. **SequentialInterventionEngine** - Двигатель последовательных причинных вмешательств
7. **EpisodicMemory** - Эпизодическая память на векторной БД
8. **ExistentialRiskSimulator** - Симулятор экзистенциальных рисков
9. **SelfEvolutionOracle** - Оракул самоэволюции (NAS + Meta-RL)
10. **ACTSv6Complete** - Главный класс, интегрирующий все компоненты

**Пример использования**:
```python
from acts_v6_complete import ACTSv6Complete
import asyncio

system = ACTSv6Complete(input_dim=100, n_assets=10, device='cpu')
result = await system.full_trading_cycle(
    market_data=data,
    news_articles=news,
    portfolio=portfolio,
    constraints=constraints
)
```

---

## Проект 2: GARCH-HyperNetwork Ensemble

### Структура проекта:

#### 1. Основной оркестратор: `src/core/orchestrator.py`
- **GARCHHyperAdaptiveEnsemble** - Главный класс ансамбля
- Методы: `train()`, `predict()`, `online_update()`, `run_stress_tests()`

#### 2. Тренировка: `src/core/trainer.py`
- **EnsembleTrainer** - Асинхронная тренировка базовых моделей
- Реализует 20+ production паттернов (FIX-1 до FIX-19)
- Walk-forward валидация
- Стресс-тестирование без утечек данных

#### 3. Предсказания: `src/core/predictor.py`
- **EnsemblePredictor** - Production движок предсказаний
- Кэширование предсказаний (TTL)
- Circuit breakers и fallback механизмы
- Управление рисками (Kelly criterion + CVaR)

#### 4. GARCH моделирование: `src/garch/model.py`
- **GARCH11** - Кастомная реализация GARCH(1,1)
- Принудительная стационарность
- Прогнозирование волатильности

#### 5. HyperNetwork: `src/hypernetwork/model.py`
- **AdaptiveHyperNetwork** - Адаптивная гиперсеть на TensorFlow
- Динамическое распределение весов моделей
- Регуляризация для предотвращения коллапса

### Примеры использования:

**Тренировка**:
```python
from src.core.orchestrator import GARCHHyperAdaptiveEnsemble
from src.core.config import load_config

config = load_config('configs/production.yaml')
ensemble = GARCHHyperAdaptiveEnsemble(config)
result = await ensemble.train(X, y, feature_names)
```

**Предсказания**:
```python
prediction = await ensemble.predict(
    X, 
    market_regime='normal',
    ground_truth=y_true
)
```

---

## Основные Python файлы

### ACTS v6.0:
- `acts_v6/src/acts_v6_complete.py` - Полная реализация системы (1724 строки)
- `acts_v6/examples/basic_usage.py` - Пример использования
- `acts_v6/examples/advanced_interventions.py` - Продвинутые примеры

### GARCH-HyperNetwork:
- `garch_hypernet_ensemble/src/core/orchestrator.py` - Оркестратор
- `garch_hypernet_ensemble/src/core/trainer.py` - Тренировка (337 строк)
- `garch_hypernet_ensemble/src/core/predictor.py` - Предсказания (457 строк)
- `garch_hypernet_ensemble/src/garch/model.py` - GARCH модель (118 строк)
- `garch_hypernet_ensemble/src/hypernetwork/model.py` - HyperNetwork (73 строки)
- `garch_hypernet_ensemble/scripts/train.py` - Скрипт тренировки
- `garch_hypernet_ensemble/scripts/predict.py` - Скрипт предсказаний

---

## Зависимости

### ACTS v6.0:
- PyTorch 2.0+
- NumPy
- Transformers (опционально, для RoBERTa)
- FAISS (опционально, для векторного поиска)

### GARCH-HyperNetwork:
- NumPy, Pandas
- Scikit-learn
- TensorFlow
- LightGBM, CatBoost
- Scipy

---

## Запуск

### ACTS v6.0:
```bash
cd acts_v6
python examples/basic_usage.py
```

### GARCH-HyperNetwork:
```bash
cd garch_hypernet_ensemble
python scripts/train.py --data-path data/market_data.parquet
python scripts/predict.py --data-path data/test_data.parquet --regime normal
```
