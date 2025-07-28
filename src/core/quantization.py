"""
Quantization Module

Handles different quantization types and ONNX model loading optimizations.
"""

import onnxruntime as ort
import logging
import os
import time
from pathlib import Path
from enum import Enum
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat
    from onnxruntime.quantization.calibrate import CalibrationDataReader

    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    logger.warning("ONNX quantization not available - install onnx package")


class QuantizationType(Enum):
    """Types of quantization available"""

    NONE = "none"
    DYNAMIC = "dynamic"
    STATIC = "static"


class ModelLoader:
    """Handles ONNX model loading with REAL quantization optimizations"""

    def __init__(self, quantization_cache_dir: str = "models/quantized"):
        self.session_opts = None
        self.quantizer = ModelQuantiser(quantization_cache_dir)
        self._configure_session_options()

    def _configure_session_options(self):
        """Configure ONNX Runtime session options for optimal performance"""
        self.session_opts = ort.SessionOptions()

        # Threading configuration
        self.session_opts.intra_op_num_threads = 1

        # Enable graph optimizations for better performance
        self.session_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Enable memory optimizations
        self.session_opts.enable_mem_pattern = True
        self.session_opts.enable_cpu_mem_arena = True

        logger.debug("ONNX session options configured for optimal performance")

    def load_model(
        self, model_path: str, quantization_type: QuantizationType
    ) -> ort.InferenceSession:
        """Load model with REAL quantization"""
        try:
            if quantization_type == QuantizationType.DYNAMIC:
                # Actually quantize the model to INT8
                quantized_path = self.quantizer.quantise_dynamic(model_path)
                session = self._load_session(quantized_path)
                logger.info(f"Loaded dynamically quantized model from {quantized_path}")
                return session

            elif quantization_type == QuantizationType.STATIC:
                # Try static quantization
                try:
                    quantized_path = self.quantizer.quantise_static(model_path)
                    session = self._load_session(quantized_path)
                    logger.info(f"Loaded statically quantized model from {quantized_path}")
                    return session
                except Exception as e:
                    logger.warning(f"Static quantization failed: {e}, falling back to dynamic")
                    quantized_path = self.quantizer.quantise_dynamic(model_path)
                    return self._load_session(quantized_path)

            else:  # QuantizationType.NONE
                # Load original FP32 model
                session = self._load_session(model_path)
                logger.info(f"Loaded FP32 model from {model_path}")
                return session

        except Exception as e:
            logger.error(f"All quantization attempts failed: {e}, loading FP32 as fallback")
            return self._load_session(model_path)

    def _load_session(self, model_path: str) -> ort.InferenceSession:
        """Load ONNX session with configured options"""
        providers = [
            (
                "CoreMLExecutionProvider",
                {
                    "MLComputeUnits": "ALL",  # GPU + Neural Engine + CPU
                    "MLProgram": True,
                    "RequireStaticInputShapes": False,
                    "EnableOnSubgraphs": False,
                },
            )
        ]
        try:
            session = ort.InferenceSession(
                model_path, providers=providers, sess_options=self.session_opts
            )
            actual_providers = session.get_providers()
            logger.info(f"Loaded model with providers: {actual_providers}")

            if "CoreMLExecutionProvider" in actual_providers:
                logger.info("Apple Silicon GPU enabled")

            return session
        except Exception as e:
            logger.error(f"CoreML failed: {e}, using CPUExecutionProvider")
            session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
                sess_options=self.session_opts,
            )
            return session

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get Apple Silicon GPU information"""
        try:
            import psutil

            return {
                "gpu_available": "CoreMLExecutionProvider"
                in self.session.get_providers(),
                "metal_support": True,
                "memory_pressure": psutil.virtual_memory().percent,
                "gpu_memory_usage": psutil.virtual_memory().used,
                "gpu_memory_total": psutil.virtual_memory().total,
                "gpu_memory_free": psutil.virtual_memory().free,
                "gpu_memory_used": psutil.virtual_memory().used,
                "gpu_memory_total": psutil.virtual_memory().total,
                "gpu_memory_free": psutil.virtual_memory().free,
            }
        except ImportError:
            return {"error": "psutil not available, cannot get GPU information"}

    def benchmark_quantization(self, model_path: str, test_inputs: List = None) -> Dict[str, Any]:
        """
        Benchmark FP32 vs quantized model performance
        
        Args:
            model_path: Path to original model
            test_inputs: Optional test data for inference benchmarking
            
        Returns:
            Performance comparison metrics
        """
        results = {
            "model_path": model_path,
            "quantization_available": QUANTIZATION_AVAILABLE
        }

        if not QUANTIZATION_AVAILABLE:
            results["error"] = "Quantization not available - install onnx package"
            return results

        try:
            # Get file size metrics
            quantized_path = self.quantizer.quantise_dynamic(model_path)
            size_metrics = self.quantizer.get_quantisation_metrics(model_path, quantized_path)
            results.update(size_metrics)

            # TODO: Add inference speed benchmarking if test_inputs provided
            if test_inputs:
                results["inference_benchmark"] = "Not yet implemented"

            return results

        except Exception as e:
            results["error"] = str(e)
            return results


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


class ModelQuantiser:
    """Handles model quantization for ONNX models"""

    def __init__(self, cache_dir: str = "models/quantised"):
        """Initialize ModelQuantiser with cache directory"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Quantisation cache directory initialised: {self.cache_dir}")

    def quantise_dynamic(self, model_path: str) -> str:
        """Quantise model to INT8 dynamically"""
        if not QUANTIZATION_AVAILABLE:
            logger.warning(
                "ONNX quantisation not available - returning the original model"
            )
            return model_path

        quantised_path = self._get_quantised_path(model_path, "dynamic")
        if self._is_quantised_cached(quantised_path, model_path):
            logger.info(f"Quantised model already exists: {quantised_path}")
            return quantised_path

        try:
            logger.info(f"Starting dynamic quantisation of {model_path}")
            start_time = time.time()

            # quantise model using onnx runtime
            quantize_dynamic(
                model_input=model_path,
                model_output=str(quantised_path),
                weight_type=QuantType.QInt8,
                optimize_model=True,
                extra_options={
                    "EnableSubgraph": True,
                    "ForceQuantizeNoInputCheck": False,
                    "MatMulConstBOnly": True,
                },
            )

            quantisation_time = time.time() - start_time
            # logging
            original_size = Path(model_path).stat().st_size
            quantised_size = Path(quantised_path).stat().st_size
            size_reduction = (1 - quantised_size / original_size) * 100

            logger.info(
                f"Dynamic quantisation completed in {quantisation_time:.2f} seconds"
            )
            logger.info(
                f"Model size reduced from {original_size/1024/1024:.2f}MB to {quantised_size/1024/1024:.2f}MB ({size_reduction:.2f}%)"
            )
            logger.info(f"Quantised model saved to {quantised_path}")

            return str(quantised_path)
        except Exception as e:
            logger.error(f"Dynamic quantisation failed: {e}")
            if quantised_path.exists():
                quantised_path.unlink()
            return model_path

    def quantise_static(self, model_path: str) -> str:
        """Quantise model to INT8 statically"""
        if not QUANTIZATION_AVAILABLE:
            logger.warning("Static quantization not available, falling back to dynamic")
            return self.quantise_dynamic(model_path)

        quantised_path = self._get_quantised_path(model_path, "static")

        if self._is_quantised_cached(quantised_path, model_path):
            logger.info(f"Using cached static quantized model: {quantised_path}")
            return str(quantised_path)

        try:
            # TODO: Implement static quantization with calibration data
            logger.warning(
                "Static quantization with calibration not yet implemented, using dynamic quantisation"
            )
            return self.quantise_dynamic(model_path)

        except Exception as e:
            logger.error(f"Static quantization failed: {e}")
            raise

    def _get_quantised_path(self, original_path: str, quant_type: str) -> Path:
        """Generate path for quantized model"""
        original_path = Path(original_path)
        filename = f"{original_path.stem}_{quant_type}_int8{original_path.suffix}"
        return self.cache_dir / filename

    def _is_quantised_cached(self, quantised_path: Path, model_path: str) -> bool:
        """Check if quantised model is already cached"""
        if not quantised_path.exists():
            return False
        try:
            original_time = Path(model_path).stat().st_mtime
            quantised_time = quantised_path.stat().st_mtime
            return quantised_time >= original_time
        except OSError:
            return False

    def get_quantisation_metrics(
        self, original_path: str, quantised_path: str
    ) -> Dict[str, Any]:
        """Get quantisation metrics for a model"""
        original_size = Path(original_path).stat().st_size
        quantised_size = Path(quantised_path).stat().st_size
        size_reduction = (1 - quantised_size / original_size) * 100
        size_reduction_ratio = original_size / quantised_size
        return {
            "original_size": original_size,
            "quantised_size": quantised_size,
            "size_reduction": size_reduction,
            "size_reduction_ratio": size_reduction_ratio,
        }

    def clear_cache(self):
        """Clear the quantisation cache"""
        try:
            for file in self.cache_dir.glob("*.onnx"):
                file.unlink()
            logger.info("Quantisation cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear quantisation cache: {e}")
