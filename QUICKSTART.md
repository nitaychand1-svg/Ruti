# 🚀 Быстрый старт

## Проект готов к использованию!

### ✅ Что уже сделано:

1. ✅ Создана полная структура проекта
2. ✅ Установлены все зависимости Python
3. ✅ Все тесты успешно проходят (6/6)
4. ✅ Настроена документация (README.md, PYDROID_INSTRUCTIONS.md)
5. ✅ Добавлен .gitignore

### 📂 Структура проекта:

```
/workspace/
├── app/                          # Основное приложение
│   ├── main.py                   # FastAPI приложение
│   ├── config.yaml               # Конфигурация
│   ├── modules/                  # Модули
│   │   ├── api_routes.py         # API endpoints
│   │   ├── taskgraph.py          # Граф задач
│   │   ├── llm_wrapper.py        # LLM интеграция
│   │   └── rl_agent.py           # RL агент (PPO)
│   ├── tasks/                    # Торговые задачи
│   │   └── debate_tasks.py       # Граф торговых решений
│   └── tests/                    # Тесты
├── k8s/                          # Kubernetes манифесты
├── .github/workflows/            # CI/CD
├── Dockerfile                    # Docker образ
├── requirements.txt              # Зависимости
└── README.md                     # Полная документация
```

### 🏃 Запуск проекта

#### 1. Запустить сервер:

```bash
cd /workspace
export PATH=$PATH:/home/ubuntu/.local/bin
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 2. Запустить тесты:

```bash
cd /workspace
pytest app/tests/ -v
```

#### 3. Проверить API:

```bash
# Health check
curl http://localhost:8000/health

# Торговое решение для AAPL
curl http://localhost:8000/debate/AAPL

# Режим отладки
curl "http://localhost:8000/debate/AAPL?debug=true"

# Метрики Prometheus
curl http://localhost:8000/metrics
```

### 🐳 Docker запуск:

```bash
# Собрать образ
docker build -t trading-system:latest .

# Запустить контейнер
docker run -p 8000:8000 trading-system:latest
```

### 📊 API Endpoints:

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/health` | GET | Проверка здоровья сервиса |
| `/metrics` | GET | Prometheus метрики |
| `/example` | GET | Пример endpoint с трассировкой |
| `/debate/{ticker}` | GET | Торговое решение для тикера |

### 🧪 Тесты:

Все тесты проходят успешно:

- ✅ `test_debate_graph` - тест графа торговых решений
- ✅ `test_debate_graph_error` - обработка ошибок
- ✅ `test_taskgraph_simple` - простой граф задач
- ✅ `test_taskgraph_deps` - граф с зависимостями
- ✅ `test_taskgraph_error` - обработка ошибок в графе
- ✅ `test_hypothesis_dummy` - property-based тестирование

### 📝 Примеры использования:

#### Python API:

```python
from app.tasks.debate_tasks import create_debate_graph

# Создать граф для анализа AAPL
tg, context = create_debate_graph("AAPL")

# Выполнить граф
results = await tg.run(start_nodes=["fetch_news"], context=context)

# Получить решение
decision = results["rl_decision"]["decision"]
print(f"Action: {decision['action']}")
print(f"Reason: {decision['reason']}")
```

#### HTTP API:

```bash
# Получить торговое решение
curl http://localhost:8000/debate/TSLA

# Ответ:
# {
#   "decision": {
#     "action": 0.73,
#     "reason": "LLM analysis: Positive outlook..."
#   },
#   "corr_id": "abc123..."
# }
```

### 🔧 Настройка:

Отредактируйте `app/config.yaml` для настройки:

```yaml
vault:
  address: "https://vault.example.com"
  role_id: "your-role-id"
  secret_id: "your-secret-id"

kubernetes:
  namespace: "trading-system"

otel:
  endpoint: "http://otel-collector:4317"
```

### 📚 Дополнительные ресурсы:

- **README.md** - полная документация проекта
- **PYDROID_INSTRUCTIONS.md** - инструкции для Android/Pydroid 3
- **app/tests/** - примеры использования в тестах

### 🎯 Следующие шаги:

1. **Интеграция с реальным LLM** - замените mock в `app/modules/llm_wrapper.py`
2. **Подключение реальных источников данных** - обновите `app/modules/data_sources.py`
3. **Обучение RL агента** - реализуйте настоящий PPO в `app/modules/rl_agent.py`
4. **Настройка мониторинга** - подключите Prometheus и Grafana
5. **Деплой в Kubernetes** - используйте манифесты в `k8s/`

### 💡 Полезные команды:

```bash
# Форматирование кода
black app/

# Линтинг
flake8 app/

# Покрытие тестами
pytest app/tests/ --cov=app --cov-report=html

# Запуск с reload для разработки
uvicorn app.main:app --reload

# Логи в JSON формате
uvicorn app.main:app | jq .
```

---

**Проект готов к использованию! 🎉**

Смотрите `README.md` для более подробной информации.
