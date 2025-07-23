"""
Benchmarking Module

Utilities for benchmarking quantization and KV cache performance.
"""

import time
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class QuantizationBenchmark:
    """Benchmark quantization performance"""
    
    def __init__(self, inference_engine):
        """
        Initialize benchmark
        
        Args:
            inference_engine: The ONNX inference engine instance
        """
        self.engine = inference_engine
    
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
        logger.info(f"Testing {self.engine.quantization_type.value} quantization...")
        quantized_times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            self.engine.token_generator.generate_tokens(
                test_inputs, [50] * len(test_inputs)
            )
            quantized_times.append(time.time() - start_time)
        
        results[self.engine.quantization_type.value] = {
            "avg_time": np.mean(quantized_times),
            "std_time": np.std(quantized_times),
            "speedup": 1.0,  # Baseline
        }
        
        logger.info(
            f"Benchmark complete. {self.engine.quantization_type.value} performance tracked."
        )
        
        return results


class KVCacheBenchmark:
    """Benchmark KV cache performance"""
    
    def __init__(self, inference_engine):
        """
        Initialize benchmark
        
        Args:
            inference_engine: The ONNX inference engine instance
        """
        self.engine = inference_engine
    
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
        self.engine.cache_enabled = True
        self.engine.kv_cache.clear()
        kv_cache_times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            self.engine.token_generator.generate_tokens(
                test_inputs, [50] * len(test_inputs)
            )
            kv_cache_times.append(time.time() - start_time)
        
        # Test without KV cache
        self.engine.cache_enabled = False
        no_cache_times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            self.engine.token_generator.generate_tokens(
                test_inputs, [50] * len(test_inputs)
            )
            no_cache_times.append(time.time() - start_time)
        
        # Re-enable cache
        self.engine.cache_enabled = True
        
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


