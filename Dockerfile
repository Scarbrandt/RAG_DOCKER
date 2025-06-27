FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

WORKDIR /app

# 1) System deps
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential python3-dev git cmake libpq-dev poppler-utils libgl1 \
    --fix-missing && rm -rf /var/lib/apt/lists/*

# 2) Copy requirements and install *everything except* flash-attn
COPY requirements.txt .
# Split out flash-attn into its own install later
RUN grep -v "flash-attn" requirements.txt > reqs_no_flash.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r reqs_no_flash.txt

# 3) Now install Flash-Attention *after* torch is in place
RUN pip install --no-cache-dir --no-deps flash-attn

# 4) Copy your code
COPY app/ ./app/
COPY db.py migrations.py sberrag.py app/main_http.py ./

EXPOSE 8000

CMD ["uvicorn", "app.main_http:app", "--host", "0.0.0.0", "--port", "8000"]
