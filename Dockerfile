FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    TZ=Asia/Taipei \
    GCS_BUCKET=point-api-512405-stock-history

COPY PY_TradeD-0.1.17.10-py3-none-any.whl /app/
RUN pip install --no-cache-dir \
    streamlit \
    polars \
    plotly \
    pandas \
    numpy \
    requests \
    pydantic \
    solace-pubsubplus \
    google-cloud-storage \
    /app/PY_TradeD-0.1.17.10-py3-none-any.whl

COPY app.py /app/
COPY config /app/config
COPY strategies /app/strategies
COPY src /app/src
COPY data /app/data

# 以非 root 使用者執行；data/ 需可寫入（K 線快取與模擬倉狀態）
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

# 保留 Streamlit 預設的 XSRF 防護（網站公開部署，不可關閉）
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
