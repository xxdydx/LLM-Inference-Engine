"""
gRPC Service Module

Handles gRPC service implementation for inference requests.
"""

import grpc
import logging
from typing import Dict, Any
from ..core.batcher import Batcher
from ..utils.health import Health
import inference_pb2
import inference_pb2_grpc

logger = logging.getLogger(__name__)


class InferenceService(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC inference service implementation"""

    def __init__(self):
        logger.info("InferenceService starting up")
        try:
            self.batcher = Batcher.instance()
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

            # Enqueue for batching
            result = self.batcher.enqueue(
                {"input_ids": input_ids, "max_tokens": max_tokens}
            )

            response = inference_pb2.PredictResponse(
                output_ids=result.output_ids, logits=result.logits
            )
            logger.debug(f"Predict responded with {len(result.output_ids)} tokens")
            return response

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
