# Multi-stage build for efficient inference engine deployment
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN groupadd -r inference && useradd -r -g inference inference
RUN mkdir -p /app/models /app/logs && \
    chown -R inference:inference /app

# Copy application code
COPY . .
RUN chown -R inference:inference /app

# Switch to non-root user
USER inference

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import grpc; import inference_pb2; import inference_pb2_grpc; \
    channel = grpc.insecure_channel('localhost:50051'); \
    stub = inference_pb2_grpc.InferenceServiceStub(channel); \
    health = stub.Health(inference_pb2.HealthRequest()); \
    exit(0 if health.ok else 1)"

# Default command
CMD ["python", "main.py"] 