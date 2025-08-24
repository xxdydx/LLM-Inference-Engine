# ONNX Inference Engine

High-performance inference engine for ONNX models with dynamic batching, KV caching, and beam search.

## Quick Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add your ONNX model
cp your_model.onnx models/gpt2.onnx

# Start server
python main.py
```

## Features

- **Dynamic Batching** - Automatic request aggregation for optimal throughput
- **KV Caching** - Memory-efficient caching for autoregressive models  
- **Beam Search** - Configurable beam search with length penalty
- **gRPC API** - Production-ready API with health checks
- **Thread-Safe** - Singleton batcher with concurrent request handling

## Usage

### Python Client
```python
import grpc
import inference_pb2, inference_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = inference_pb2_grpc.InferenceServiceStub(channel)

# Generate tokens
response = stub.Predict(inference_pb2.PredictRequest(
    input_ids=[1, 2, 3],
    max_tokens=50,
    beam_size=3
))
print(response.output_ids)
```

### Docker
```bash
docker build -t inference-engine .
docker run -p 50051:50051 -v ./models:/app/models inference-engine
```

## Architecture

```
gRPC Service → Batcher → ONNX Engine → Token Generator → Beam Search
                 ↓           ↓
            Request Queue  KV Cache
```

## Configuration

Edit `src/core/config.py` or pass parameters:

```python
from src.core.config import create_config

config = create_config(
    model_path="models/my-model.onnx",
    batch_timeout_ms=500,
    max_batch_size=32
)
```

## Development

```bash
# Run tests
pytest tests/ -v

# Format code  
black src/ main.py tests/

# Lint
flake8 src/ main.py tests/
```