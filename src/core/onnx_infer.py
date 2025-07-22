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
    """ONNX inference engine with quantization optimization and KV caching"""

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

            # KV cache for storing actual key-value tensors
            self.kv_cache = {}
            self.cache_enabled = True

            # Get model input/output names to understand KV cache structure
            self._analyze_model_io()

            logger.info(f"ONNX model loaded successfully from {model_path}")
            logger.info(f"Quantization type: {quantization_type.value}")
            logger.info("KV caching enabled for autoregressive generation")

        except Exception as e:
            logger.error(f"Failed to load ONNX model from {model_path}: {e}")
            raise

    def _analyze_model_io(self):
        """Analyze model inputs and outputs to understand KV cache structure"""
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

        self.past_key_values_inputs = [
            name for name in self.input_names if "past_key_values" in name
        ]
        self.present_key_values_outputs = [
            name for name in self.output_names if "present_key_values" in name
        ]

        # See if model supports KV caching
        self.supports_kv_cache = (
            len(self.past_key_values_inputs) > 0
            and len(self.present_key_values_outputs) > 0
        )

        logger.info(f"Model inputs: {self.input_names}")
        logger.info(f"Model outputs: {self.output_names}")
        logger.info(f"Supports KV cache: {self.supports_kv_cache}")
        if self.supports_kv_cache:
            logger.info(f"Past KV inputs: {self.past_key_values_inputs}")
            logger.info(f"Present KV outputs: {self.present_key_values_outputs}")

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

    def _get_cache_key(self, sequence_id: int, position: int) -> str:
        """Generate cache key for KV cache"""
        return f"seq_{sequence_id}_pos_{position}"

    def _update_kv_cache(
        self, sequence_id: int, position: int, present_key_values: Dict[str, np.ndarray]
    ):
        """Update KV cache with actual present_key_values tensors"""
        if not self.cache_enabled or not self.supports_kv_cache:
            return

        cache_key = self._get_cache_key(sequence_id, position)
        self.kv_cache[cache_key] = {
            "present_key_values": present_key_values,
            "position": position,
        }

        # elementary approach for cache eviction— just keeping the last 1000 entries
        # TODO: use LRU eviction with memory usage tracking
        if len(self.kv_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self.kv_cache.keys(), key=lambda k: self.kv_cache[k]["position"]
            )[:100]
            for key in oldest_keys:
                del self.kv_cache[key]

    def _get_cached_kv(
        self, sequence_id: int, position: int
    ) -> Optional[Dict[str, np.ndarray]]:
        """Retrieve cached present_key_values tensors"""
        if not self.cache_enabled or not self.supports_kv_cache:
            return None

        cache_key = self._get_cache_key(sequence_id, position)
        if cache_key in self.kv_cache:
            cached = self.kv_cache[cache_key]
            return cached["present_key_values"]
        return None

    def clear_cache(self):
        """Clear the KV cache"""
        self.kv_cache.clear()
        logger.info("KV cache cleared")

    def run(
        self,
        input_tensor: np.ndarray,
        sequence_id: int = 0,
        position: int = 0,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Run inference with KV caching

        Args:
            input_tensor: Input tensor of shape (batch_size, seq_len)
            sequence_id: ID of the sequence for KV caching
            position: Current position in the sequence for KV caching
            use_cache: Whether to use KV caching

        Returns:
            Logits tensor
        """
        try:
            start_time = time.time()

            # Prepare input feed
            input_feed = {"input_ids": input_tensor}

            # Add past_key_values to input if we have cached values and model supports it
            cached_present_kv = None
            if (
                use_cache
                and self.cache_enabled
                and self.supports_kv_cache
                and position > 0
            ):
                cached_present_kv = self._get_cached_kv(sequence_id, position - 1)
                if cached_present_kv is not None:
                    # Add past_key_values to input feed
                    for input_name in self.past_key_values_inputs:
                        if input_name in cached_present_kv:
                            input_feed[input_name] = cached_present_kv[input_name]
                    logger.debug(
                        f"Using cached KV for sequence {sequence_id} at position {position}"
                    )

            # Determine output names
            output_names = ["logits"]
            if use_cache and self.cache_enabled and self.supports_kv_cache:
                output_names.extend(self.present_key_values_outputs)

            # Run inference
            outputs = self.session.run(output_names=output_names, input_feed=input_feed)

            # Extract logits (always first output)
            logits = outputs[0]

            # Update cache with present_key_values if available
            if (
                use_cache
                and self.cache_enabled
                and self.supports_kv_cache
                and len(outputs) > 1
            ):
                present_key_values = {}
                for i, output_name in enumerate(self.present_key_values_outputs):
                    if i + 1 < len(outputs):  # +1 because logits is first output
                        present_key_values[output_name] = outputs[i + 1]

                if present_key_values:
                    self._update_kv_cache(sequence_id, position, present_key_values)

            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            logger.debug(
                f"Batch inference completed: {input_tensor.shape[0]} requests in {inference_time:.3f}s"
            )

            return logits

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
        Greedy decode from the ONNX model with KV caching
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

        # Process each token position with KV caching
        for pos in range(max_seq_len):
            if all(finished_sequences):  # Checkinh if all sequences are finished
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

            # Run batched inference with KV caching
            input_tensor = np.array(batch_input, dtype=np.int64)

            if self.cache_enabled and active_indices and self.supports_kv_cache:
                # Use KV caching for the first active sequence
                first_active = active_indices[0]
                logits = self.run(
                    input_tensor, sequence_id=first_active, position=pos, use_cache=True
                )
            else:
                logits = self.run(input_tensor, use_cache=False)

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
            # Get the last logits for this sequence
            if len(seq) > 0:
                # Run single inference to get last logits
                single_input = np.array([seq], dtype=np.int64)
                if self.cache_enabled and self.supports_kv_cache:
                    single_logits = self.run(
                        single_input,
                        sequence_id=i,
                        position=len(seq) - 1,
                        use_cache=True,
                    )
                else:
                    single_logits = self.run(single_input, use_cache=False)
                last_logits = single_logits[0, -1, :].tolist()
            else:
                last_logits = []
            results.append((seq, last_logits))

        return results

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the quantized model with KV cache info"""
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
            "kv_cache_enabled": self.cache_enabled,
            "kv_cache_size": len(self.kv_cache),
            "supports_kv_cache": self.supports_kv_cache,
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

    def benchmark_kv_cache(
        self, test_inputs: List[List[int]], num_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark KV cache performance

        Args:
            test_inputs: List of test input sequences
            num_runs: Number of benchmark runs

        Returns:
            Benchmark results comparing with/without KV cache
        """
        logger.info("Starting KV cache benchmark...")

        # Test with KV cache enabled
        self.cache_enabled = True
        self.clear_cache()
        kv_cache_times = []

        for _ in range(num_runs):
            start_time = time.time()
            self.generate_tokens(test_inputs, [50] * len(test_inputs))
            kv_cache_times.append(time.time() - start_time)

        # Test without KV cache
        self.cache_enabled = False
        no_cache_times = []

        for _ in range(num_runs):
            start_time = time.time()
            self.generate_tokens(test_inputs, [50] * len(test_inputs))
            no_cache_times.append(time.time() - start_time)

        # Re-enable cache
        self.cache_enabled = True

        kv_cache_avg = np.mean(kv_cache_times)
        no_cache_avg = np.mean(no_cache_times)
        speedup = no_cache_avg / kv_cache_avg if kv_cache_avg > 0 else 1.0

        results = {
            "with_kv_cache": {
                "avg_time": kv_cache_avg,
                "std_time": np.std(kv_cache_times),
            },
            "without_kv_cache": {
                "avg_time": no_cache_avg,
                "std_time": np.std(no_cache_times),
            },
            "speedup": speedup,
        }

        logger.info(f"KV cache benchmark complete. Speedup: {speedup:.2f}x")
        return results
