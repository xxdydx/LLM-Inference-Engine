# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
make setup                    # Install dev dependencies and create models directory
make install                  # Install dependencies only
pip install -r requirements.txt
```

### Development Workflow
```bash
make run                      # Run the inference engine server
make test                     # Run all tests
pytest tests/test_batcher.py  # Run specific test file
make lint                     # Run linting (flake8 + mypy)
make format                   # Format code with black
```

### gRPC Protocol Buffer Generation
When modifying `inference.proto`, regenerate Python bindings:
```bash
python -m grpc_tools.protoc --python_out=. --grpc_python_out=. inference.proto
```

## Architecture Overview

This is a Python LLM inference engine built around a **singleton batcher pattern** with ONNX Runtime backend, gRPC service interface, and advanced optimization features.

### Core Design Principles

**Singleton Batcher**: The `Batcher` class (src/core/batcher.py) uses singleton pattern to ensure only one instance handles all inference requests across the application. It implements dynamic batching with configurable timeout and batch size limits.

**ONNX Engine with Quantization**: The `ONNXInfer` class (src/core/onnx_infer.py) supports multiple quantization types (NONE, DYNAMIC, STATIC) and implements KV caching for autoregressive token generation.

**Async Request Processing**: Requests are submitted to the batcher via `Future` objects, allowing the gRPC service to handle multiple concurrent requests while the batcher processes them in optimized batches.

### Component Interaction Flow

1. **gRPC Request** → `InferenceService.Predict()` validates input
2. **Submit to Batcher** → Request queued with `Future` for async result
3. **Dynamic Batching** → Background thread collects requests until timeout/batch size
4. **ONNX Inference** → Batched requests processed with KV caching and quantization
5. **Result Distribution** → Each request's `Future` receives its individual result

### Key Implementation Details

**KV Caching Strategy**: The engine maintains a cache of key-value pairs indexed by `seq_{sequence_id}_pos_{position}` with LRU eviction after 1000 entries. Cache is only used if the ONNX model has `past_key_values` inputs and `present_key_values` outputs.

**Quantization Support**: Three modes supported - NONE (FP32), DYNAMIC (runtime INT8 conversion), and STATIC (pre-quantized model loading). Falls back gracefully if quantization fails.

**Thread Safety**: The batcher runs a background daemon thread that processes the request queue. All metrics and cache operations are thread-safe.

### Model Requirements

- ONNX model should be placed at `models/gpt2.onnx`
- Model must have `input_ids` input
- For KV caching: model should have `past_key_values.*` inputs and `present_key_values.*` outputs
- For static quantization: pre-quantized model should be named `*_int8.onnx`

### Testing and Validation

The codebase includes several KV cache test files (`test_kv_cache*.py`) that demonstrate different approaches to testing the caching behavior. These are useful references for understanding expected cache performance characteristics.

### Metrics and Monitoring

The system tracks comprehensive metrics including batch statistics, inference timing, quantization performance, and KV cache usage. Access via the `GetMetrics` gRPC endpoint or batcher's `get_metrics()` method.