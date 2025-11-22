FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANALYSIS_DB_PATH=/tmp/analysis.db \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# 系統需求：git 用來 clone 外部研究庫
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# 先安裝依賴
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY . /app

# 內建外部研究庫（固定路徑 /app/EarningsCallAgenticRag）
RUN git clone https://github.com/la9806958/EarningsCallAgenticRag.git /app/EarningsCallAgenticRag

# 啟動命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
