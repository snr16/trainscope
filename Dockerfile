# TrainScope server image: ingestion API (default) or MCP server (streamable HTTP).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[server,postgres,mcp]"

ENV TRAINSCOPE_DB=/data/trainscope.duckdb TRAINSCOPE_HOST=0.0.0.0 TRAINSCOPE_PORT=8765
VOLUME /data
EXPOSE 8765 8766
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz')" || exit 1
CMD ["trainscope-server"]
