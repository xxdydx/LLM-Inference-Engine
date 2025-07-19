"""
gRPC Service Module

Handles gRPC service implementation for inference requests.
"""

import grpc
import logging
from typing import Dict, Any
from core.batcher import Batcher
from utils.health import Health
import inference_pb2
import inference_pb2_grpc
import time

logger = logging.getLogger(__name__)


class InferenceService(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC inference service implementation"""

    def __init__(self):
        logger.info("InferenceService starting up")
        try:
            self.batcher = Batcher.instance()
            self.start_time = time.time()  # Track service start time for metrics
            logger.info("InferenceService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize InferenceService: {e}")
            raise

    def Predict(self, request, context):
        try:
            input_ids = list(request.input_ids)
            max_tokens = request.max_tokens

            # Validate input
            if not input_ids:
                logger.warning("empty input_ids")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("empty input_ids")
                return inference_pb2.PredictResponse()

            if max_tokens < 0:
                logger.warning(f"invalid max_tokens: {max_tokens}")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("max_tokens must be non-negative")
                return inference_pb2.PredictResponse()

            logger.info(f"Predict: input_len={len(input_ids)}, max_tokens={max_tokens}")

            # Submit request to batcher
            future = self.batcher.submit_request(input_ids, max_tokens)

            # Wait for result
            try:
                result = future.result(timeout=30)
                response = inference_pb2.PredictResponse(
                    output_ids=result.output_ids, logits=result.logits
                )
                logger.debug(f"Predict responded with {len(result.output_ids)} tokens")
                return response
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Inference error: {e}")
                return inference_pb2.PredictResponse()

        except Exception as ex:
            logger.error(f"Prediction failed: {ex}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Inference error: {ex}")
            return inference_pb2.PredictResponse()

    def Health(self, request, context):
        try:
            ok = Health.check()
            response = inference_pb2.HealthResponse(
                ok=ok, message="OK" if ok else "NOT OK"
            )
            logger.info(f"Health check: {'OK' if ok else 'NOT OK'}")
            return response
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return inference_pb2.HealthResponse(ok=False, message=f"Error: {e}")

    def GetMetrics(self, request, context):
        try:
            metrics = self.batcher.get_metrics()

            uptime_seconds = time.time() - self.start_time
            requests_per_second = metrics["total_requests"] / max(1, uptime_seconds)

            response = inference_pb2.MetricsResponse(
                total_requests=metrics["total_requests"],
                total_batches=metrics["total_batches"],
                avg_batch_size=metrics["avg_batch_size"],
                batch_timeout_ms=metrics["batch_timeout_ms"],
                avg_batch_time_ms=metrics["avg_batch_time_ms"],
                max_batch_size=metrics["max_batch_size"],
                requests_per_second=requests_per_second,
            )
            return response
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error getting metrics: {e}")
            return inference_pb2.MetricsResponse()
