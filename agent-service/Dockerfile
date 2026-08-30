FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY agent-service/requirements.txt /tmp/requirements.txt
RUN pip install --requirement /tmp/requirements.txt

COPY agent-service/app /app/agent-service/app
RUN mkdir -p /app/agent-service/data/user-memory /app/rag-data/vector

WORKDIR /app/agent-service

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
