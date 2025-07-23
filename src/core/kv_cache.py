"""
KV Cache Module

Advanced key-value caching with memory mapping for large tensors.
"""

import os
import mmap
import tempfile
import threading
import pickle
import logging
from collections import deque
from typing import Dict, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class KVCache:
    """Advanced KV cache using memory mapping for large tensors"""

    def __init__(self, max_cache_size_mb=512, mmap_threshold_kb=64):
        """
        Initialize KV cache with memory mapping support

        Args:
            max_cache_size_mb: Maximum cache size in MB
            mmap_threshold_kb: Size threshold in KB for using memory mapping
        """
        self.max_size_bytes = max_cache_size_mb * 1024 * 1024
        self.mmap_threshold_bytes = mmap_threshold_kb * 1024
        self.current_size = 0

        # In-memory cache for small tensors
        self.in_memory_cache = {}

        # Memory-mapped cache for large tensors
        self.mmap_cache = {}
        self.mmap_files = {}  # Track open mmap files

        # Thread safety
        self._lock = threading.RLock()

        # Create temp directory for mmap files
        self.temp_dir = tempfile.mkdtemp(prefix="kv_cache_")

        # LRU tracking
        self.access_order = deque()  # Most recent at right
        self.access_count = {}

        logger.info(
            f"KV Cache initialized: max_size={max_cache_size_mb}MB, "
            f"mmap_threshold={mmap_threshold_kb}KB"
        )

    def _calculate_tensor_size(self, tensor_dict: Dict[str, np.ndarray]) -> int:
        """Calculate total memory size of tensor dictionary"""
        total_size = 0
        for tensor in tensor_dict.values():
            total_size += tensor.nbytes
        return total_size

    def _should_use_mmap(self, tensor_dict: Dict[str, np.ndarray]) -> bool:
        """Decide whether to use memory mapping based on size"""
        return self._calculate_tensor_size(tensor_dict) > self.mmap_threshold_bytes

    def _serialize_to_mmap(
        self, cache_key: str, tensor_dict: Dict[str, np.ndarray]
    ) -> Optional[Dict]:
        """Save tensor dict to memory-mapped file"""
        try:
            # Create unique filename
            filename = f"cache_{hash(cache_key) % 1000000}.bin"
            filepath = os.path.join(self.temp_dir, filename)

            # Serialize tensor dict to bytes
            serialized_data = pickle.dumps(tensor_dict)

            # Create memory-mapped file
            with open(filepath, "wb") as f:
                f.write(serialized_data)

            # Open as memory-mapped file
            f = open(filepath, "r+b")
            mm = mmap.mmap(f.fileno(), 0)

            # Store references
            self.mmap_files[cache_key] = (f, mm, filepath)

            # Update mmap cache entry
            self.mmap_cache[cache_key] = {
                "is_mmapped": True,
                "filepath": filepath,
                "size": len(serialized_data),
            }

            return self.mmap_cache[cache_key]

        except Exception as e:
            logger.error(f"Failed to create mmap cache: {e}")
            return None

    def _load_from_mmap(self, cache_key: str) -> Optional[Dict[str, np.ndarray]]:
        """Load tensor dict from memory-mapped file"""
        if cache_key not in self.mmap_files:
            return None

        try:
            f, mm, filepath = self.mmap_files[cache_key]
            mm.seek(0)
            serialized_data = mm.read()
            tensor_dict = pickle.loads(serialized_data)
            return tensor_dict
        except Exception as e:
            logger.error(f"Failed to load from mmap: {e}")
            return None

    def _evict_lru_entries(self, needed_space: int):
        """Evict least recently used entries to free space"""
        freed_space = 0

        while freed_space < needed_space and self.access_order:
            # Get least recently used key (leftmost)
            lru_key = self.access_order.popleft()

            if lru_key in self.in_memory_cache:
                tensor_dict = self.in_memory_cache[lru_key]
                freed_space += self._calculate_tensor_size(tensor_dict)
                del self.in_memory_cache[lru_key]

            elif lru_key in self.mmap_cache:
                cache_entry = self.mmap_cache[lru_key]
                freed_space += cache_entry["size"]

                # Clean up mmap resources
                if lru_key in self.mmap_files:
                    f, mm, filepath = self.mmap_files[lru_key]
                    mm.close()
                    f.close()
                    if os.path.exists(filepath):
                        os.unlink(filepath)  # Delete file
                    del self.mmap_files[lru_key]

                del self.mmap_cache[lru_key]

            # Remove from access tracking
            if lru_key in self.access_count:
                del self.access_count[lru_key]

            self.current_size -= freed_space

    def put(self, cache_key: str, tensor_dict: Dict[str, np.ndarray]):
        """Store tensor dict in cache with automatic mmap decision"""
        with self._lock:
            tensor_size = self._calculate_tensor_size(tensor_dict)

            # Check if we need to evict entries
            if self.current_size + tensor_size > self.max_size_bytes:
                self._evict_lru_entries(tensor_size)

            # Decide storage method
            if self._should_use_mmap(tensor_dict):
                # Use memory mapping for large tensors
                mmap_info = self._serialize_to_mmap(cache_key, tensor_dict)
                if mmap_info:
                    self.current_size += mmap_info["size"]
                    logger.debug(
                        f"Cached {cache_key} to mmap ({mmap_info['size']} bytes)"
                    )
                else:
                    # Fallback to memory if mmap fails
                    self.in_memory_cache[cache_key] = tensor_dict
                    self.current_size += tensor_size
            else:
                # Use in-memory storage for small tensors
                self.in_memory_cache[cache_key] = tensor_dict
                self.current_size += tensor_size

            # Update access tracking
            self.access_order.append(cache_key)
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1

    def get(self, cache_key: str) -> Optional[Dict[str, np.ndarray]]:
        """Retrieve tensor dict from cache"""
        with self._lock:
            # Check memory cache first (faster)
            if cache_key in self.in_memory_cache:
                if cache_key in self.access_order:
                    self.access_order.remove(cache_key)
                self.access_order.append(cache_key)  # update to most recent
                self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
                return self.in_memory_cache[cache_key]

            # Check mmap cache
            if cache_key in self.mmap_cache:
                tensor_dict = self._load_from_mmap(cache_key)
                if tensor_dict is not None:
                    if cache_key in self.access_order:
                        self.access_order.remove(cache_key)
                    self.access_order.append(cache_key)  # update to most recent
                    self.access_count[cache_key] = (
                        self.access_count.get(cache_key, 0) + 1
                    )
                    return tensor_dict

            return None

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            # Clean up mmap resources
            for cache_key in list(self.mmap_files.keys()):
                f, mm, filepath = self.mmap_files[cache_key]
                mm.close()
                f.close()
                if os.path.exists(filepath):
                    os.unlink(filepath)

            self.in_memory_cache.clear()
            self.mmap_cache.clear()
            self.mmap_files.clear()
            self.access_order.clear()
            self.access_count.clear()
            self.current_size = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "memory_entries": len(self.in_memory_cache),
                "mmap_entries": len(self.mmap_cache),
                "total_size_mb": self.current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "hit_counts": dict(self.access_count),
            }

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.clear()
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception as e:
            logger.debug(f"Cache cleanup error: {e}")
