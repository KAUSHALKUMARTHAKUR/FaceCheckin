FROM python:3.11-slim

WORKDIR /app

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy app
COPY . .
RUN mkdir -p app/static app/templates

EXPOSE 8000
CMD gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --timeout 120 app:app
