# Use official Python image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory in container
WORKDIR /app

# Copy requirement files and install dependencies
COPY requirements.txt .

# Install system dependencies (for pandas, matplotlib, XGBoost, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy entire project into container
COPY . .

# Expose port (optional: useful if serving MLflow or a web UI later)
EXPOSE 5000

# Default command to run main script
CMD ["bash"]
