# Используем официальный Python образ (slim версия для уменьшения размера)
FROM python:3.12-slim-bookworm

# Установка переменных окружения
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Установка рабочей директории
WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Установка и оптимизация Python зависимостей
# 1. Обновление pip
RUN pip install --upgrade pip

# 2. Установка CPU-версии Torch (Легче, ~200MB против 1GB+ у CUDA)
# Это предотвращает скачивание огромной версии с CUDA, что часто вызывает таймауты
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Установка остальных зависимостей
COPY requirements.txt .
# Увеличенный таймаут для медленных соединений
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Копирование файлов проекта
COPY . .

# Открытие порта
EXPOSE 8550

# Запуск приложения
# CMD ["python", "src/ui/main.py"] 
# Запуск сервера FastAPI
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8550"]
