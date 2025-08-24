#!/usr/bin/env python3
"""
Simple demo client for the Inference Engine gRPC API
"""

import grpc
import time
import logging
import inference_pb2
import inference_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo-client")


def main():
    channel = grpc.insecure_channel("localhost:50051")
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    # Health
    health = stub.Health(inference_pb2.HealthRequest())
    logger.info(f"Health: ok={health.ok}, message={health.message}")
    if not health.ok:
        logger.warning("Service not ready; continuing for demo...")

    # Greedy predict
    req = inference_pb2.PredictRequest(input_ids=[1, 2, 3], max_tokens=16)
    t0 = time.time()
    resp = stub.Predict(req)
    logger.info(f"Greedy tokens ({len(resp.output_ids)}): {list(resp.output_ids)[:20]}")
    logger.info(f"Latency: {(time.time()-t0)*1000:.1f} ms")

    # Beam search predict
    req_beam = inference_pb2.PredictRequest(
        input_ids=[1, 2, 3], max_tokens=16, beam_size=3, length_penalty=1.0
    )
    t0 = time.time()
    resp_beam = stub.Predict(req_beam)
    logger.info(
        f"Beam tokens ({len(resp_beam.output_ids)}): {list(resp_beam.output_ids)[:20]}"
    )
    logger.info(f"Latency: {(time.time()-t0)*1000:.1f} ms")

    # Metrics
    metrics = stub.GetMetrics(inference_pb2.MetricsRequest())
    logger.info(
        f"Metrics: total_requests={metrics.total_requests}, total_batches={metrics.total_batches}, avg_batch_size={metrics.avg_batch_size:.2f}"
    )


if __name__ == "__main__":
    main()
