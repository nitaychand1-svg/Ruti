# Полный список Python файлов проектов

## Проект 1: ACTS v6.0

### Основные модули:
1. **`src/acts_v6_complete.py`** (1724 строки) - Главный файл с полной реализацией системы
   - MultiModalFusionEngine
   - BayesianRegimePredictor
   - MultiAgentDebateSystem
   - HierarchicalMARLSwarm
   - WorldModelBuilder
   - SequentialInterventionEngine
   - EpisodicMemory
   - ExistentialRiskSimulator
   - ACTSv6Complete (главный класс)

2. **`examples/basic_usage.py`** - Базовый пример использования
3. **`examples/advanced_interventions.py`** - Продвинутые примеры причинных вмешательств
4. **`examples/risk_analysis.py`** - Анализ рисков

### Модульные компоненты:
5. **`perception.py`** - Модуль восприятия
6. **`debate.py`** - Система дебатов
7. **`marl.py`** - Мультиагентное обучение с подкреплением
8. **`world_model.py`** - Модель мира
9. **`memory.py`** - Эпизодическая память
10. **`risk.py`** - Управление рисками
11. **`self_evolution.py`** - Самоэволюция
12. **`federated.py`** - Федеративное обучение
13. **`interface.py`** - Интерфейс человек-ИИ
14. **`adaptive_sampling.py`** - Адаптивная выборка
15. **`system.py`** - Системные утилиты

### Тесты и конфигурация:
16. **`tests/test_acts_v6.py`** - Юнит-тесты
17. **`setup.py`** - Установка пакета
18. **`__init__.py`** - Инициализация пакета

---

## Проект 2: GARCH-HyperNetwork Ensemble

### Ядро системы (`src/core/`):
1. **`orchestrator.py`** - Главный оркестратор ансамбля
   - GARCHHyperAdaptiveEnsemble класс
   - Методы train(), predict(), online_update()

2. **`trainer.py`** (337 строк) - Асинхронная тренировка
   - EnsembleTrainer класс
   - Реализация 20+ production паттернов
   - Walk-forward валидация

3. **`predictor.py`** (457 строк) - Движок предсказаний
   - EnsemblePredictor класс
   - Кэширование, circuit breakers
   - Управление рисками

4. **`config.py`** - Управление конфигурацией
5. **`components.py`** - Data classes для компонентов

### GARCH моделирование (`src/garch/`):
6. **`model.py`** (118 строк) - GARCH(1,1) модель
   - GARCH11 класс
   - Принудительная стационарность

7. **`tracker.py`** - Трекер GARCH для множественных признаков

### HyperNetwork (`src/hypernetwork/`):
8. **`model.py`** (73 строки) - Адаптивная гиперсеть
   - AdaptiveHyperNetwork класс (TensorFlow)

9. **`trainer.py`** - Тренировка гиперсети

### Мониторинг (`src/monitoring/`):
10. **`performance_tracker.py`** - Трекинг производительности
11. **`stress_testing.py`** - Стресс-тестирование

### Валидация (`src/validation/`):
12. **`data_validation.py`** - Валидация данных
13. **`cross_validation.py`** - Кросс-валидация

### Утилиты (`src/utils/`):
14. **`caching.py`** - Кэширование предсказаний
15. **`version_control.py`** - Версионирование моделей

### Скрипты (`scripts/`):
16. **`train.py`** - Скрипт тренировки
17. **`predict.py`** - Скрипт предсказаний
18. **`serve.py`** - API сервер

### Тесты:
19. **`tests/test_ensemble.py`** - Тесты ансамбля

### Конфигурация:
20. **`setup.py`** - Установка пакета
21. **`__init__.py`** файлы в каждом модуле

---

## Статистика кода

### ACTS v6.0:
- Главный файл: ~1724 строки
- Примеры: ~200 строк каждый
- Модульные компоненты: ~100-300 строк каждый

### GARCH-HyperNetwork:
- Orchestrator: ~100 строк
- Trainer: ~337 строк
- Predictor: ~457 строк
- GARCH model: ~118 строк
- HyperNetwork: ~73 строки
- Всего: ~2000+ строк production кода

---

## Как использовать код

### Для ACTS v6.0:
```python
# Импорт главного класса
from acts_v6.src.acts_v6_complete import ACTSv6Complete

# Инициализация
system = ACTSv6Complete(input_dim=100, n_assets=10)

# Полный торговый цикл
result = await system.full_trading_cycle(...)
```

### Для GARCH-HyperNetwork:
```python
# Импорт оркестратора
from src.core.orchestrator import GARCHHyperAdaptiveEnsemble
from src.core.config import load_config

# Загрузка конфигурации
config = load_config('configs/production.yaml')

# Инициализация
ensemble = GARCHHyperAdaptiveEnsemble(config)

# Тренировка
await ensemble.train(X, y, feature_names)

# Предсказания
prediction = await ensemble.predict(X, market_regime='normal')
```

---

## Ключевые особенности кода

### ACTS v6.0:
- ✅ Асинхронная архитектура (asyncio)
- ✅ Байесовское обучение с ELBO
- ✅ Мультимодальное восприятие
- ✅ Причинное моделирование
- ✅ Самоэволюция системы

### GARCH-HyperNetwork:
- ✅ 20+ production паттернов (FIX-1 до FIX-19)
- ✅ Асинхронная тренировка
- ✅ Circuit breakers и fallback
- ✅ Кэширование предсказаний
- ✅ Управление рисками (Kelly + CVaR)
- ✅ Стресс-тестирование
