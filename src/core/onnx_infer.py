"""
Simple ONNX Inference Engine
"""

import logging
import onnxruntime as ort
from typing import List, Tuple, Dict, Any

from .kv_cache import KVCache
from .token_generator import TokenGenerator

logger = logging.getLogger(__name__)


class ONNXInfer:
    """Simple ONNX inference engine with KV caching"""

    def __init__(self, model_path: str, max_cache_size_mb: int = 512):
        """
        Initialize ONNX inference engine

        Args:
            model_path: Path to the ONNX model
            max_cache_size_mb: Maximum KV cache size in MB
        """
        try:
            self.model_path = model_path
            
            # Load ONNX session
            self.session = ort.InferenceSession(model_path)
            
            # Analyze model structure
            self.model_analysis = self._analyze_model()

            # Initialize KV cache
            self.kv_cache = KVCache(max_cache_size_mb=max_cache_size_mb)
            self.cache_enabled = True

            # Initialize token generator
            self.token_generator = TokenGenerator(self)

            logger.info(f"ONNX Inference Engine initialized: {model_path}")

        except Exception as e:
            logger.error(f"Failed to initialize ONNX Inference Engine: {e}")
            raise
    
    def _analyze_model(self) -> Dict[str, Any]:
        """Analyze model inputs/outputs for KV cache support"""
        inputs = [inp.name for inp in self.session.get_inputs()]
        outputs = [out.name for out in self.session.get_outputs()]
        
        # Check for KV cache support
        past_key_inputs = [inp for inp in inputs if "past_key_values" in inp]
        present_key_outputs = [out for out in outputs if "present_key_values" in out]
        
        return {
            "inputs": inputs,
            "outputs": outputs,
            "supports_kv_cache": len(past_key_inputs) > 0 and len(present_key_outputs) > 0,
            "past_key_values_inputs": past_key_inputs,
            "present_key_values_outputs": present_key_outputs,
        }

    @property
    def supports_kv_cache(self) -> bool:
        """Check if model supports KV caching"""
        return self.model_analysis["supports_kv_cache"]

    def generate_tokens(
        self,
        input_ids_list: List[List[int]],
        max_lengths: List[int],
        beam_size: int = 1,
        length_penalty: float = 1.0,
        eos_token_id: int | None = None,
    ) -> List[Tuple[List[int], List[float]]]:
        """Generate tokens using the token generator"""
        return self.token_generator.generate_tokens(
            input_ids_list=input_ids_list,
            max_lengths=max_lengths,
            beam_size=beam_size,
            length_penalty=length_penalty,
            eos_token_id=eos_token_id,
        )