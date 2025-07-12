"""
Inference Batcher

Handles batching of inference requests using singleton pattern.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import logging
from .onnx_infer import ONNXInfer

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of inference operation"""

    output_ids: List[int]
    logits: List[float]


class Batcher:
    """Singleton batcher for handling inference requests"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Batcher, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            try:
                self.engine = ONNXInfer("models/gpt2.onnx")
                self.initialized = True
                logger.info("Batcher initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize batcher: {e}")
                raise

    @classmethod
    def instance(cls):
        """Get singleton instance"""
        return cls()

    def enqueue(self, request: Dict[str, Any]) -> InferenceResult:
        """Enqueue a prediction request and return results"""
        try:
            input_ids = request.get("input_ids", [])

            # Convert input_ids to list of ints if needed
            if isinstance(input_ids, (str, bytes)):
                # Handle if input_ids is passed as string/bytes
                input_ids = [int(x) for x in input_ids]

            if not input_ids:
                raise ValueError("Empty input_ids provided")

            # Run inference
            logits = self.engine.run(input_ids, len(input_ids))

            # For now, return input_ids as output_ids (you can implement actual generation later)
            result = InferenceResult(output_ids=input_ids, logits=logits)
            logger.info(f"Enqueued request processed: {len(input_ids)} input tokens")
            return result

        except Exception as e:
            logger.error(f"Failed to enqueue request: {e}")
            raise
