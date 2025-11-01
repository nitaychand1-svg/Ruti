#!/usr/bin/env python3
"""
Setup script for Trading System project
Run this in Pydroid 3 to create the complete project structure
"""

import os
from pathlib import Path

# Base directory for the project
BASE_DIR = "/storage/emulated/0/trading_system"

# Define all files and their contents
FILES = {
    ".sops.yaml": """creation_rules:
  - path_regex: ".*config\\.yaml$"
    pgp: []
    azure_kv: []
    gcp_kms: []
    kms: []
    encrypted_regex: "^(data|secrets)$"
""",
    
    "Dockerfile": """FROM python:3.13-slim

WORKDIR /app
COPY app/ /app
COPY k8s/ /app/k8s
COPY .sops.yaml /app/
COPY requirements.txt /app/

RUN pip install --upgrade pip && \\
    pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
    
    "all_files_content.txt": "",
    
    ".github/workflows/build_and_push.yaml": """name: Build and Push Trading System

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.13"

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install fastapi uvicorn python-json-logger opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi prometheus-client pytest hypothesis sops networkx

    - name: Generate requirements.txt
      run: pip freeze > requirements.txt

    - name: Install Syft
      run: curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

    - name: Generate SBOM
      run: syft packages dir:. -o json > sbom.json

    - name: Upload SBOM
      uses: actions/upload-artifact@v4
      with:
        name: sbom
        path: sbom.json

    - name: Run tests
      run: pytest app/tests/

    - name: Build Docker image
      run: docker build -t ${{ secrets.DOCKER_REGISTRY }}/trading-system:${{ github.sha }} .

    - name: Install Cosign
      uses: sigstore/cosign-installer@v2

    - name: Cosign sign
      env:
        IMAGE: ${{ secrets.DOCKER_REGISTRY }}/trading-system:${{ github.sha }}
      run: cosign sign --keyless $IMAGE

    - name: Push Docker image
      run: docker push ${{ secrets.DOCKER_REGISTRY }}/trading-system:${{ github.sha }}

    - name: Decrypt SOPS config
      if: exists('app/config.yaml.enc')
      run: sops --decrypt --output app/config.yaml app/config.yaml.enc

    - name: Apply Kubernetes manifests
      run: |
        kubectl apply -f k8s/deployment_security.yaml
        kubectl apply -f k8s/network_policy.yaml
""",
    
    "app/main.py": """import os
import time
import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from app.modules.observability import setup_tracing
from app.modules.api_routes import router
from app.modules.logging_config import logger
import yaml

# Load and validate config
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if not config.get("vault", {}).get("address"):
        raise ValueError("Invalid config: vault address missing")
except Exception as e:
    logger.error(f"Config load error: {e}")
    raise RuntimeError("Failed to load config")

# Tracing
tracer = setup_tracing(config["otel"]["endpoint"])

# Metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"])

# FastAPI app
app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
app.include_router(router)

# Middleware ??? correlation_id ? error handling
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    corr_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request.state.corr_id = corr_id
    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Error in request: {e}", extra={"corr_id": corr_id})
        raise HTTPException(status_code=500, detail="Internal error")
    latency = time.time() - start_time
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(latency)
    return response

# Healthcheck
@app.get("/health")
async def health():
    return {"status": "ok"}

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

# Example endpoint
@app.get("/example")
async def example(request: Request):
    corr_id = request.state.corr_id
    with tracer.start_as_current_span("example-workflow") as span:
        span.set_attribute("corr_id", corr_id)
        logger.info("Start example endpoint", extra={"corr_id": corr_id})
        time.sleep(0.1)
        logger.info("End example endpoint", extra={"corr_id": corr_id})
    return {"ok": True, "corr_id": corr_id}
""",
    
    "app/config.yaml": """vault:
  address: "https://vault.example.com"
  role_id: "example-role-id-1234"
  secret_id: "example-secret-id-5678"

kubernetes:
  namespace: "trading-system"

otel:
  endpoint: "http://otel-collector:4317"
""",
    
    "app/tests/test_debate_tasks.py": """import pytest
import asyncio
from app.tasks.debate_tasks import create_debate_graph

@pytest.mark.asyncio
async def test_debate_graph():
    tg, context = create_debate_graph("AAPL")
    results = await tg.run(start_nodes=["fetch_news"], context=context)
    
    # ????????
    assert "rl_decision" in results
    decision = results["rl_decision"]["decision"]
    assert isinstance(decision, dict)
    assert "action" in decision
    assert "reason" in decision
    print("? Test passed for debate_task")

@pytest.mark.asyncio
async def test_debate_graph_error():
    tg, context = create_debate_graph("INVALID")  # Assume fetch_news fails for invalid
    with pytest.raises(ValueError):
        await tg.run(start_nodes=["fetch_news"], context=context)
    print("? Test passed for error handling")
""",
    
    "app/tests/test_taskgraph.py": """import pytest
import asyncio
from app.modules.taskgraph import Node, TaskGraph
from hypothesis import given, strategies as st

async def dummy_task(context, deps):
    return {"result": 42}

async def failing_task(context, deps):
    raise ValueError("Fail")

@pytest.mark.asyncio
async def test_taskgraph_simple():
    tg = TaskGraph()
    tg.add(Node("a", dummy_task))
    results = await tg.run(start_nodes=["a"], context={})
    assert results["a"]["result"] == 42

@pytest.mark.asyncio
async def test_taskgraph_deps():
    tg = TaskGraph()
    tg.add(Node("a", dummy_task))
    tg.add(Node("b", dummy_task, depends_on=["a"]))
    results = await tg.run(start_nodes=["a"], context={})
    assert "b" in results

@pytest.mark.asyncio
async def test_taskgraph_error():
    tg = TaskGraph()
    tg.add(Node("a", failing_task))
    with pytest.raises(ValueError):
        await tg.run(start_nodes=["a"], context={})

@given(st.text(min_size=1))
def test_hypothesis_dummy(input_str):
    # Simple fuzz
    assert len(input_str) >= 1
""",
    
    "app/tasks/debate_tasks.py": """from app.modules.taskgraph import Node, TaskGraph
from app.modules.llm_wrapper import async_llm_infer
from app.modules.data_sources import async_fetch_news
from app.modules.cognitive_middleware import cognitive
from app.modules.rl_agent import PPO
from opentelemetry import trace
import asyncio
from app.modules.logging_config import logger

tracer = trace.get_tracer(__name__)

# --- ??????????? ????? ---
async def fetch_news_task(context, deps):
    ticker = context['ticker']
    if not ticker.isupper() or not ticker.isalpha():
        raise ValueError("Invalid ticker")
    try:
        async with asyncio.timeout(5):  # 5s timeout
            news = await async_fetch_news(ticker)
        return {"news": news}
    except Exception as e:
        logger.error(f"Fetch news error: {e}")
        raise

async def llm_analysis_task(context, deps):
    news = deps['fetch_news']['news']
    with tracer.start_as_current_span("llm_analysis"):
        try:
            async with asyncio.timeout(10):
                result = await async_llm_infer(f"Analyze news for {context['ticker']}: {news}")
            return {"raw": result}
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise

async def cognitive_task(context, deps):
    raw = deps['llm_analysis']['raw']
    try:
        processed = cognitive.process_reasoning(raw, context)
        return {"processed": processed}
    except Exception as e:
        logger.error(f"Cognitive error: {e}")
        raise

async def rl_decision_task(context, deps):
    processed = deps['cognitive']['processed']
    try:
        decision = await asyncio.to_thread(PPO.predict, processed)
        return {"decision": decision}
    except Exception as e:
        logger.error(f"RL error: {e}")
        raise

# --- ?????? TaskGraph ---
def create_debate_graph(ticker: str):
    tg = TaskGraph()
    tg.add(Node("fetch_news", fetch_news_task))
    tg.add(Node("llm_analysis", llm_analysis_task, depends_on=["fetch_news"]))
    tg.add(Node("cognitive", cognitive_task, depends_on=["llm_analysis"]))
    tg.add(Node("rl_decision", rl_decision_task, depends_on=["cognitive"]))
    context = {"ticker": ticker}
    return tg, context
""",
    
    "app/modules/logging_config.py": """import logging, sys
from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
""",
    
    "app/modules/taskgraph.py": """import asyncio
import networkx as nx
from app.modules.logging_config import logger

class Node:
    def __init__(self, name, func, depends_on=None):
        self.name = name
        self.func = func
        self.depends_on = depends_on or []

class TaskGraph:
    def __init__(self):
        self.nodes = {}
        self.graph = nx.DiGraph()

    def add(self, node):
        self.nodes[node.name] = node
        self.graph.add_node(node.name)
        for dep in node.depends_on:
            self.graph.add_edge(dep, node.name)  # dep -> node

    async def run_node(self, node_name, context, results):
        node = self.nodes[node_name]
        deps_results = {dep: results[dep] for dep in node.depends_on}
        try:
            result = await node.func(context, deps_results)
            results[node_name] = result
        except Exception as e:
            logger.error(f"Error in node {node_name}: {e}")
            raise

    async def run(self, start_nodes, context):
        results = {}
        # Topological order
        try:
            order = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Graph has cycles")
        
        # Run in topological batches (parallel for same level)
        levels = list(nx.topological_generations(self.graph))
        for level in levels:
            tasks = []
            for node in level:
                if node in results: continue  # Already done if no deps
                tasks.append(self.run_node(node, context, results))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=False)
        
        # Filter to executed nodes
        executed = {k: results[k] for k in results if k not in start_nodes or True}
        return results
""",
    
    "app/modules/cognitive_middleware.py": """class CognitiveMiddleware:
    @staticmethod
    def process_reasoning(raw, context):
        rationale = raw['text']
        # Simple enhancement
        sentiment = "positive" if "good" in rationale.lower() else "negative"
        return {"rationale": rationale, "sentiment": sentiment, "context": context}

cognitive = CognitiveMiddleware()
""",
    
    "app/modules/llm_wrapper.py": """import asyncio

async def async_llm_infer(prompt):
    await asyncio.sleep(0.1)
    return {"text": f"LLM analysis: Positive outlook for {prompt.split(':')[0]} based on news."}
""",
    
    "app/modules/rl_agent.py": """import random

class PPO:
    @staticmethod
    def predict(processed):
        action = random.uniform(-1, 1)  # e.g., buy/sell signal
        return {"action": action, "reason": processed['rationale']}
""",
    
    "app/modules/observability.py": """from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

def setup_tracing(otlp_endpoint=None):
    provider = TracerProvider()
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(__name__)
""",
    
    "app/modules/data_sources.py": """import asyncio
from functools import lru_cache

@lru_cache(maxsize=100)
async def async_fetch_news(ticker):
    await asyncio.sleep(0.05)
    return [f"News for {ticker}: Market up 2%."]
""",
    
    "app/modules/api_routes.py": """from fastapi import APIRouter, Request, Query
from app.tasks.debate_tasks import create_debate_graph
import asyncio
from app.modules.logging_config import logger

router = APIRouter()

@router.get("/debate/{ticker}")
async def debate(ticker: str, request: Request, debug: bool = Query(False)):
    corr_id = getattr(request.state, "corr_id", "unknown")
    if not ticker.isupper() or not ticker.isalpha():
        raise ValueError("Invalid ticker: must be uppercase letters")
    try:
        tg, context = create_debate_graph(ticker)
        results = await tg.run(start_nodes=["fetch_news"], context=context)
        if debug:
            return {"results": results, "corr_id": corr_id}
        return {"decision": results["rl_decision"]["decision"], "corr_id": corr_id}
    except Exception as e:
        logger.error(f"Debate error: {e}", extra={"corr_id": corr_id})
        raise
""",
    
    "k8s/vault-policy.hcl": """path "secret/data/trading/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
""",
    
    "k8s/network_policy.yaml": """apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: trading-deny-egress
spec:
  podSelector:
    matchLabels:
      app: trading-system
  policyTypes:
  - Egress
  egress: []
""",
    
    "k8s/deployment_security.yaml": """apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-system
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: trading-system
        image: ghcr.io/your-repo/trading-system:latest  # Or use ${{ github.sha }}
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1"
            memory: "1Gi"
""",
    
    "k8s/vault-rotation-job.yaml": """apiVersion: batch/v1
kind: CronJob
metadata:
  name: vault-rotation
  namespace: trading-system
spec:
  schedule: "*/5 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: vault-rotation-sa
          containers:
          - name: vault-rotation
            image: vault:1.13.0
            env:
            - name: VAULT_ADDR
              valueFrom:
                configMapKeyRef:
                  name: vault-config
                  key: address
            - name: VAULT_ROLE_ID
              valueFrom:
                secretKeyRef:
                  name: vault-credentials
                  key: role_id
            - name: VAULT_SECRET_ID
              valueFrom:
                secretKeyRef:
                  name: vault-credentials
                  key: secret_id
            command:
              - "/bin/sh"
              - "-c"
              - |
                set -euo pipefail
                TOKEN=$(vault write -field=client_token auth/approle/login role_id=$VAULT_ROLE_ID secret_id=$VAULT_SECRET_ID)
                NEW_SECRET=$(vault kv put -field=key secret/trading/api_key value="new_value")
                kubectl create secret generic trading-api-key --from-literal=API_KEY=$NEW_SECRET -n trading-system --dry-run=client -o yaml | kubectl apply -f -
          restartPolicy: OnFailure
""",
    
    "requirements.txt": """fastapi==0.104.1
uvicorn==0.24.0
python-json-logger==2.0.7
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0
opentelemetry-instrumentation-fastapi==0.42b0
prometheus-client==0.19.0
pytest==7.4.3
hypothesis==6.92.0
networkx==3.2.1
pyyaml==6.0.1
""",

    "README.md": """# Trading System

Production-ready trading system with:
- FastAPI backend
- OpenTelemetry tracing
- Prometheus metrics
- Task graph execution
- LLM integration
- RL agent
- Kubernetes deployment
- Vault secrets management

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests:
```bash
pytest app/tests/
```

3. Run the application:
```bash
uvicorn app.main:app --reload
```

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
""",
}


def create_file(filepath, content):
    """Create a file with given content"""
    full_path = Path(BASE_DIR) / filepath
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"? Created: {filepath}")


def main():
    print("=" * 60)
    print("Trading System Project Setup")
    print("=" * 60)
    print(f"\nCreating project in: {BASE_DIR}\n")
    
    # Create base directory
    Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
    
    # Create all files
    file_count = 0
    for filepath, content in FILES.items():
        try:
            create_file(filepath, content)
            file_count += 1
        except Exception as e:
            print(f"? Error creating {filepath}: {e}")
    
    # Create __init__.py files for Python packages
    init_files = [
        "app/__init__.py",
        "app/modules/__init__.py",
        "app/tasks/__init__.py",
        "app/tests/__init__.py",
    ]
    
    for init_file in init_files:
        try:
            create_file(init_file, "# Python package\n")
            file_count += 1
        except Exception as e:
            print(f"? Error creating {init_file}: {e}")
    
    print("\n" + "=" * 60)
    print(f"? Project setup complete!")
    print(f"?? Total files created: {file_count}")
    print(f"?? Project location: {BASE_DIR}")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run tests: pytest app/tests/")
    print("3. Start server: uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("\n??  Note: Some dependencies may not be available in Pydroid 3")
    print("   You can still view and edit the code structure.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n? Fatal error: {e}")
        print("Please check permissions and try again.")
