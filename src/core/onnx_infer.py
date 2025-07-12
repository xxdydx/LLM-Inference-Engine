"""
ONNX Inference Engine

Handles model loading and inference using ONNX Runtime.
"""

import onnxruntime as ort
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class ONNXInfer:
    """ONNX inference engine for running model inference"""

    def __init__(self, model_path: str):
        """Initialize ONNX inference engine with model path"""
        try:
            self.env = ort.Environment(ort.LoggingLevel.WARNING, "infer")
            self.opts = ort.SessionOptions()
            self.opts.intra_op_num_threads = 1  # single thread for inference
            self.session = ort.InferenceSession(
                model_path, self.opts, providers=["CPUExecutionProvider"]
            )
            logger.info(f"ONNX model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model from {model_path}: {e}")
            raise

    def run(self, input_ids: List[int], seq_len: int) -> List[float]:
        """Run inference on input token IDs and return logits"""
        try:
            # Prepare input tensor
            input_tensor = np.array(input_ids, dtype=np.int64).reshape(1, seq_len)

            # Run inference
            outputs = self.session.run(
                output_names=["logits"], input_feed={"input_ids": input_tensor}
            )

            # Extract logits
            logits = outputs[0].flatten().tolist()
            logger.debug(f"Inference completed: {len(logits)} logits")
            return logits

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
