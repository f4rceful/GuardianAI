# Use official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies optimization
# 1. Upgrade pip
RUN pip install --upgrade pip

# 2. Install CPU-only Torch (Lighter, ~200MB vs 1GB+)
# This prevents downloading the massive CUDA version which causes timeouts
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install other dependencies
COPY requirements.txt .
# Increase timeout for slow connections
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8550

# Run the application
# CMD ["python", "src/ui/main.py"] 
# Run the FastAPI server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8550"]
