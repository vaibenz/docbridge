FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md LICENSE ./
COPY docbridge ./docbridge
RUN pip install --no-cache-dir ".[serve]"
COPY corpus ./corpus
RUN useradd --create-home app
USER app
ENV DOCBRIDGE_CORPUS=/app/corpus PORT=8080
CMD ["sh", "-c", "uvicorn docbridge.api:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]
