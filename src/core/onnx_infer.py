"""
ONNX Inference Engine
"""

import logging
from typing import List, Tuple, Dict, Any

from .kv_cache import KVCache
from .quantization import QuantizationType, ModelLoader, ModelAnalyzer
from .token_generator import TokenGenerator

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


    def __str__(self) -> str:
        """String representation of the inference engine"""
        return (
            f"ONNXInfer(model={self.model_path}, "
            f"quantization={self.quantization_type.value}, "
            f"cache_enabled={self.cache_enabled}, "
            f"supports_kv_cache={self.supports_kv_cache})"
        )

