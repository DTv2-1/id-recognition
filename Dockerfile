FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias (cachear layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pesos de modelos (solo si existen, para fallback local)
COPY weights/ weights/ 2>/dev/null || true

# Código
COPY app/ app/
COPY handler.py .

EXPOSE 8000

# Servidor FastAPI (Gemini no necesita GPU)
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
