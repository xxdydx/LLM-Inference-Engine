"""
ONNX Inference Engine

Handles model loading and inference using ONNX Runtime.
"""

import onnxruntime as ort
import numpy as np
from typing import List, Optional
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

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference on a batch of input token IDs and return logits"""
        try:
            # input_tensor shape: (batch_size, seq_len)
            outputs = self.session.run(
                output_names=["logits"], input_feed={"input_ids": input_tensor}
            )
            logger.debug(f"Batch Inference completed: {input_tensor.shape[0]} requests")

            return outputs[0]

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def generate_tokens(
        self, input_ids: List[int], max_length: int, eos_token_id: Optional[int] = None
    ) -> List[int]:
        """
        Greedy decode from the ONNX model

        Args:
            input_ids: List of token IDs to seed decoding (e.g. [BOS]).
            max_length: Maximum total length (including prompt).
            eos_token_id: If provided, stop when this token is generated.

        Returns:
            Full list of token IDs (prompt + generated).
        """

        output_ids = input_ids.copy()
        for _ in range(len(input_ids), max_length):
            input_tensor = np.array([output_ids], dtype=np.int64)
            logits = self.run(input_tensor)
            last_logits = logits[0, -1, :]
            next_token = int(np.argmax(last_logits))
            output_ids.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break
        return output_ids
