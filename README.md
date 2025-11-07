# Trading System Project Setup

Проект для создания торговой системы с использованием FastAPI, OpenTelemetry, Prometheus и других современных технологий.

## Структура проекта

- `setup_trading_system_project.py` - **Рекомендуемый скрипт** для создания проекта (включает совместимость с Python 3.13)
- `setup_trading_project.py` - Альтернативный скрипт установки
- `PYDROID_INSTRUCTIONS.md` - Инструкции по установке в Pydroid 3

## Быстрый старт

### Для Pydroid 3 (Android)

1. Скачайте `setup_trading_system_project.py` на устройство
2. Откройте файл в Pydroid 3
3. Запустите скрипт
4. Проект будет создан в `/storage/emulated/0/trading_system/`

Подробные инструкции см. в `PYDROID_INSTRUCTIONS.md`

### Для Linux/Mac/Windows

```bash
python3 setup_trading_system_project.py
```

Затем:

```bash
cd /storage/emulated/0/trading_system  # или измените BASE_DIR в скрипте
pip install -r requirements.txt
pytest app/tests/
uvicorn app.main:app --reload
```

## Особенности

- ✅ FastAPI backend
- ✅ OpenTelemetry tracing
- ✅ Prometheus metrics
- ✅ Task graph execution
- ✅ LLM integration
- ✅ RL agent (PPO)
- ✅ Kubernetes deployment configs
- ✅ Vault secrets management
- ✅ Совместимость с Python 3.13

## Исправления

- Исправлена проблема с `lru_cache` для async функций
- Добавлена поддержка Python 3.13 через `app/compat.py`
- Улучшена обработка ошибок
- Исправлена кодировка в инструкциях

## API Endpoints

- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /example` - Example endpoint with tracing
- `GET /debate/{ticker}` - Trading decision for ticker

## Docker

```bash
docker build -t trading-system .
docker run -p 8000:8000 trading-system
```

## Kubernetes

```bash
kubectl apply -f k8s/
```
