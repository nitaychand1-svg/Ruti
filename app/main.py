from app import compat  # noqa: F401
import os
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
