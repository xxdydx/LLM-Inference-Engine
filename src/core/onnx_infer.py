"""
ONNX Inference Engine
"""

import logging
from typing import List, Tuple, Dict, Any

from .kv_cache import KVCache
from .quantization import QuantizationType, ModelLoader, ModelAnalyzer
from .token_generator import TokenGenerator
from .benchmarks import QuantizationBenchmark, KVCacheBenchmark

logger = logging.getLogger(__name__)


class ONNXInfer:
    """ONNX inference engine with quantization optimization and KV caching"""

    def __init__(
        self,
        model_path: str,
        quantization_type: QuantizationType = QuantizationType.DYNAMIC,
        max_cache_size_mb: int = 512,
        mmap_threshold_kb: int = 64,
    ):
        """
        Initialize ONNX inference engine

        Args:
            model_path: Path to the ONNX model
            quantization_type: Type of quantization to use
            max_cache_size_mb: Maximum KV cache size in MB
            mmap_threshold_kb: Memory mapping threshold in KB
        """
        try:
            self.quantization_type = quantization_type
            self.model_path = model_path

            # Initialize model loader and load the model
            self.model_loader = ModelLoader()
            self.session = self.model_loader.load_model(model_path, quantization_type)

            # Analyze model structure
            self.model_analysis = ModelAnalyzer.analyze_model_io(self.session)

            # Initialize KV cache
            self.kv_cache = KVCache(
                max_cache_size_mb=max_cache_size_mb, mmap_threshold_kb=mmap_threshold_kb
            )
            self.cache_enabled = True

            # Initialize token generator
            self.token_generator = TokenGenerator(self)

            # Initialize benchmarking tools (lazy loading)
            self._quantization_benchmark = None
            self._kv_cache_benchmark = None

            logger.info(f"ONNX model loaded successfully from {model_path}")
            logger.info(f"Quantization type: {quantization_type.value}")
            logger.info("KV caching enabled for autoregressive generation")

        except Exception as e:
            logger.error(f"Failed to load ONNX model from {model_path}: {e}")
            raise

    @property
    def supports_kv_cache(self) -> bool:
        """Check if model supports KV caching"""
        return self.model_analysis["supports_kv_cache"]

    def generate_tokens(
        self,
        input_ids_list: List[List[int]],
        max_lengths: List[int],
        eos_token_id: int = None,
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Generate tokens using the token generator

        Args:
            input_ids_list: List of input token ID sequences
            max_lengths: List of maximum lengths for each sequence
            eos_token_id: If provided, stop when this token is generated

        Returns:
            List of tuples (output_ids, last_logits) for each request
        """
        return self.token_generator.generate_tokens(
            input_ids_list, max_lengths, eos_token_id
        )

    def clear_cache(self):
        """Clear the KV cache"""
        self.kv_cache.clear()
        logger.info("KV cache cleared")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        # Get token generation metrics
        token_metrics = self.token_generator.get_performance_metrics()

        # Get cache statistics
        cache_stats = self.kv_cache.get_stats()

        # Combine all metrics
        metrics = {
            "quantization_type": self.quantization_type.value,
            "kv_cache_enabled": self.cache_enabled,
            "supports_kv_cache": self.supports_kv_cache,
            **token_metrics,
            **{f"kv_cache_{k}": v for k, v in cache_stats.items()},
        }

        return metrics

    # Benchmark methods (lazy loading)
    def benchmark_quantization(
        self, test_inputs: List[List[int]], num_runs: int = 10
    ) -> Dict[str, Any]:
        """Benchmark quantization performance"""
        if self._quantization_benchmark is None:
            self._quantization_benchmark = QuantizationBenchmark(self)
        return self._quantization_benchmark.benchmark_quantization(
            test_inputs, num_runs
        )

    def benchmark_kv_cache(
        self, test_inputs: List[List[int]], num_runs: int = 10
    ) -> Dict[str, Any]:
        """Benchmark KV cache performance"""
        if self._kv_cache_benchmark is None:
            self._kv_cache_benchmark = KVCacheBenchmark(self)
        return self._kv_cache_benchmark.benchmark_kv_cache(test_inputs, num_runs)

    def __str__(self) -> str:
        """String representation of the inference engine"""
        return (
            f"ONNXInfer(model={self.model_path}, "
            f"quantization={self.quantization_type.value}, "
            f"cache_enabled={self.cache_enabled}, "
            f"supports_kv_cache={self.supports_kv_cache})"
        )

    def __repr__(self) -> str:
        """Detailed representation of the inference engine"""
        return self.__str__()
