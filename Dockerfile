FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install Python packages first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other code
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Expose port 8080
EXPOSE 8080

# Start the app using Gunicorn, binding to 0.0.0.0:8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
