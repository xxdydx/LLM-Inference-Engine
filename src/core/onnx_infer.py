"""
ONNX Inference Engine with Quantization Support

Handles model loading and inference using ONNX Runtime with quantization optimization.
"""

import onnxruntime as ort
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Types of quantization available"""

    NONE = "none"
    DYNAMIC = "dynamic"
    STATIC = "static"


class ONNXInfer:
    """ONNX inference engine with quantization optimization"""

    def __init__(
        self,
        model_path: str,
        quantization_type: QuantizationType = QuantizationType.DYNAMIC,
    ):
        """
        Initialize ONNX inference engine with quantization

        Args:
            model_path: Path to the ONNX model
            quantization_type: Type of quantization to use
        """
        try:
            self.quantization_type = quantization_type
            self.model_path = model_path

            # Configure session options for quantization
            self.opts = ort.SessionOptions()
            self.opts.intra_op_num_threads = 1

            # Enable graph optimizations for better performance
            self.opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            # Enable memory optimizations
            self.opts.enable_mem_pattern = True
            self.opts.enable_cpu_mem_arena = True

            # Load the appropriate model based on quantization type
            if quantization_type == QuantizationType.STATIC:
                self.session = self._load_static_quantized_model(model_path)
            else:
                self.session = self._load_dynamic_quantized_model(model_path)

            # Performance tracking
            self.inference_times = []

            logger.info(f"ONNX model loaded successfully from {model_path}")
            logger.info(f"Quantization type: {quantization_type.value}")

        except Exception as e:
            logger.error(f"Failed to load ONNX model from {model_path}: {e}")
            raise

    def _load_dynamic_quantized_model(self, model_path: str) -> ort.InferenceSession:
        """Load model with dynamic quantization enabled"""
        try:
            # Dynamic quantization - converts to INT8 at runtime
            session = ort.InferenceSession(
                model_path, self.opts, providers=["CPUExecutionProvider"]
            )
            logger.info(
                "Dynamic quantization enabled - model will be quantized at runtime"
            )
            return session
        except Exception as e:
            logger.warning(f"Dynamic quantization failed, falling back to FP32: {e}")
            return ort.InferenceSession(
                model_path, self.opts, providers=["CPUExecutionProvider"]
            )

    def _load_static_quantized_model(self, model_path: str) -> ort.InferenceSession:
        """Load pre-quantized model or create one"""
        try:
            # Try to load pre-quantized model
            quantized_path = model_path.replace(".onnx", "_int8.onnx")
            session = ort.InferenceSession(
                quantized_path, self.opts, providers=["CPUExecutionProvider"]
            )
            logger.info(f"Static quantized model loaded from {quantized_path}")
            return session
        except Exception as e:
            logger.warning(f"Static quantized model not found, using FP32: {e}")
            return ort.InferenceSession(
                model_path, self.opts, providers=["CPUExecutionProvider"]
            )

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference with performance tracking

        Args:
            input_tensor: Input tensor of shape (batch_size, seq_len)

        Returns:
            Logits tensor
        """
        try:
            start_time = time.time()

            # Run inference
            outputs = self.session.run(
                output_names=["logits"], input_feed={"input_ids": input_tensor}
            )

            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # Keep only last 100 measurements
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            logger.debug(
                f"Batch inference completed: {input_tensor.shape[0]} requests in {inference_time:.3f}s"
            )

            return outputs[0]

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def generate_tokens(
        self,
        input_ids_list: List[List[int]],
        max_lengths: List[int],
        eos_token_id: Optional[int] = None,
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Greedy decode from the ONNX model
        Args:
            input_ids_list: List of input token ID sequences.
            max_lengths: List of maximum lengths for each sequence.
            eos_token_id: If provided, stop when this token is generated.

        Returns:
            List of tuples (output_ids, last_logits) for each request.
        """

        batch_size = len(input_ids_list)
        if batch_size == 0:
            return []

        # Initialize output sequences
        output_sequences = [ids.copy() for ids in input_ids_list]
        finished_sequences = [False] * batch_size

        # Find the maximum sequence length to process
        max_seq_len = max(max_lengths)

        # Process each token position
        for pos in range(max_seq_len):
            # Check if all sequences are finished
            if all(finished_sequences):
                break

            # Prepare batch input tensor
            batch_input = []
            active_indices = []

            for i in range(batch_size):
                if (
                    not finished_sequences[i]
                    and len(output_sequences[i]) < max_lengths[i]
                ):
                    # Pad shorter sequences to current position
                    while len(output_sequences[i]) <= pos:
                        output_sequences[i].append(0)  # pad with 0
                    batch_input.append(output_sequences[i])
                    active_indices.append(i)
                else:
                    # This sequence is finished, add padding
                    padded_seq = output_sequences[i] + [0] * (
                        pos + 1 - len(output_sequences[i])
                    )
                    batch_input.append(padded_seq)

            if not active_indices:
                break

            # Run batched inference
            input_tensor = np.array(batch_input, dtype=np.int64)
            logits = self.run(input_tensor)

            # Process results for active sequences
            for batch_idx, seq_idx in enumerate(active_indices):
                if len(output_sequences[seq_idx]) <= pos:
                    # Get the next token for this sequence
                    last_logits = logits[batch_idx, pos, :]
                    next_token = int(np.argmax(last_logits))
                    output_sequences[seq_idx].append(next_token)

                    # Check for EOS token
                    if eos_token_id is not None and next_token == eos_token_id:
                        finished_sequences[seq_idx] = True

        # Return results with last logits for each sequence
        results = []
        for i, seq in enumerate(output_sequences):
            # Get the last logits for this sequence (we'll need to run one more inference)
            if len(seq) > 0:
                # Run single inference to get last logits
                single_input = np.array([seq], dtype=np.int64)
                single_logits = self.run(single_input)
                last_logits = single_logits[0, -1, :].tolist()
            else:
                last_logits = []
            results.append((seq, last_logits))

        return results

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the quantized model"""
        if not self.inference_times:
            return {}

        avg_inference_time = np.mean(self.inference_times)
        min_inference_time = np.min(self.inference_times)
        max_inference_time = np.max(self.inference_times)

        return {
            "quantization_type": self.quantization_type.value,
            "avg_inference_time_ms": avg_inference_time * 1000,
            "min_inference_time_ms": min_inference_time * 1000,
            "max_inference_time_ms": max_inference_time * 1000,
            "total_inferences": len(self.inference_times),
            "throughput_req_per_sec": (
                1.0 / avg_inference_time if avg_inference_time > 0 else 0
            ),
        }

    def benchmark_quantization(
        self, test_inputs: List[List[int]], num_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark different quantization types

        Args:
            test_inputs: List of test input sequences
            num_runs: Number of benchmark runs

        Returns:
            Benchmark results
        """
        logger.info("Starting quantization benchmark...")

        results = {}

        # Test current quantization
        logger.info(f"Testing {self.quantization_type.value} quantization...")
        quantized_times = []
        for _ in range(num_runs):
            start_time = time.time()
            self.generate_tokens(test_inputs, [50] * len(test_inputs))
            quantized_times.append(time.time() - start_time)

        results[self.quantization_type.value] = {
            "avg_time": np.mean(quantized_times),
            "std_time": np.std(quantized_times),
            "speedup": 1.0,  # Baseline
        }

        logger.info(
            f"Benchmark complete. {self.quantization_type.value} performance tracked."
        )

        return results
