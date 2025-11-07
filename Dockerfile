FROM python:3.13-slim

WORKDIR /app
COPY app/ /app
COPY k8s/ /app/k8s
COPY .sops.yaml /app/
COPY requirements.txt /app/

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
