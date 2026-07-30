# Use a lightweight Python runtime base
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies and install using layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference runtime code and stats artifact
COPY inference/ /app/inference/
COPY data_pipeline/training_stats.npy /app/data_pipeline/training_stats.npy
COPY monitoring/ /app/monitoring/
# Copy the data_pipeline directory into /app/data_pipeline/
COPY data_pipeline/ /app/data_pipeline/

# Set default runtime environment variables
ENV MODEL_PATH="inference/model.tflite"
ENV TRUCK_ID="01"

# Run the inference engine (drift-monitor overrides this via `command:` in compose)
CMD ["python","-u", "inference/inference_service.py"]