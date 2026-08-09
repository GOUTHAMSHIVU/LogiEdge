FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference/inference_service.py /app/inference/inference_service.py
COPY monitoring/ /app/monitoring/
COPY data_pipeline/ /app/data_pipeline/

# Frequently updated model layer
COPY inference/model.tflite /app/inference/model.tflite

ENV MODEL_PATH="/app/inference/model.tflite"
ENV TRUCK_ID="01"

CMD ["python", "-u", "inference/inference_service.py"]