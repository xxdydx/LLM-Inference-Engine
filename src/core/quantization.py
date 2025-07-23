"""
Quantization Module

Handles different quantization types and ONNX model loading optimizations.
"""

import onnxruntime as ort
import logging
from enum import Enum
from typing import Dict, Any

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Types of quantization available"""

    NONE = "none"
    DYNAMIC = "dynamic"
    STATIC = "static"


class ModelLoader:
    """Handles ONNX model loading with quantization optimizations"""

    def __init__(self):
        self.session_opts = None
        self._configure_session_options()

    def _configure_session_options(self):
        """Configure ONNX Runtime session options for optimal performance"""
        self.session_opts = ort.SessionOptions()

        self.session_opts.intra_op_num_threads = 1

        self.session_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Enable memory optimizations
        self.session_opts.enable_mem_pattern = True
        self.session_opts.enable_cpu_mem_arena = True

        logger.debug("ONNX session options configured for optimal performance")

    def _load_model_with_fallback(self, model_path: str, quantization_type: QuantizationType) -> ort.InferenceSession:
        """Load model with fallback handling"""
        try:
            if quantization_type == QuantizationType.STATIC:
                # Try to load pre-quantized model
                quantized_path = model_path.replace(".onnx", "_int8.onnx")
                session = ort.InferenceSession(
                    quantized_path, self.session_opts, providers=["CPUExecutionProvider"]
                )
                logger.info(f"Static quantized model loaded from {quantized_path}")
                return session
            else:
                # Load regular model (DYNAMIC/NONE are the same for now)
                session = ort.InferenceSession(
                    model_path, self.session_opts, providers=["CPUExecutionProvider"]
                )
                logger.info(f"Model loaded with {quantization_type.value} configuration")
                return session
        except Exception as e:
            logger.warning(f"Model loading failed, falling back to default: {e}")
            return ort.InferenceSession(
                model_path, self.session_opts, providers=["CPUExecutionProvider"]
            )

    def load_model(
        self, model_path: str, quantization_type: QuantizationType
    ) -> ort.InferenceSession:
        """Load model based on quantization type"""
        return self._load_model_with_fallback(model_path, quantization_type)


class ModelAnalyzer:
    """Analyzes ONNX model structure for KV caching support"""

    @staticmethod
    def analyze_model_io(session: ort.InferenceSession) -> Dict[str, Any]:
        """Analyze model inputs and outputs to understand KV cache structure"""
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]

        past_key_values_inputs = [
            name for name in input_names if "past_key_values" in name
        ]
        present_key_values_outputs = [
            name for name in output_names if "present_key_values" in name
        ]

        # Check if model supports KV caching
        supports_kv_cache = (
            len(past_key_values_inputs) > 0 and len(present_key_values_outputs) > 0
        )

        analysis = {
            "input_names": input_names,
            "output_names": output_names,
            "past_key_values_inputs": past_key_values_inputs,
            "present_key_values_outputs": present_key_values_outputs,
            "supports_kv_cache": supports_kv_cache,
        }

        logger.info(f"Model inputs: {input_names}")
        logger.info(f"Model outputs: {output_names}")
        logger.info(f"Supports KV cache: {supports_kv_cache}")

        if supports_kv_cache:
            logger.info(f"Past KV inputs: {past_key_values_inputs}")
            logger.info(f"Present KV outputs: {present_key_values_outputs}")

        return analysis
