# Use a lightweight Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for OCR, OpenCV, FinRL, Huggingface, RL libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    libmagic-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    gcc \
    g++ \
    make \
    git \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for Cloud Run
EXPOSE 8080

# Run the app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
