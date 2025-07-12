"""
Thread Pool Module

Provides thread pool functionality for concurrent task handling.
"""

import threading
import queue
import time
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor, Future
import logging

logger = logging.getLogger(__name__)


class ThreadPool:
    """Thread pool for handling concurrent tasks"""

    def __init__(self, num_workers: int = 4):
        """Initialize thread pool with specified number of workers"""
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self._shutdown = False
        logger.info(f"Thread pool initialized with {num_workers} workers")

    def submit(self, func: Callable, *args, **kwargs) -> Future:
        """Submit a task to the thread pool"""
        if self._shutdown:
            raise RuntimeError("ThreadPool is shutdown")

        try:
            future = self.executor.submit(func, *args, **kwargs)
            logger.debug(f"Task submitted to thread pool: {func.__name__}")
            return future
        except Exception as e:
            logger.error(f"Failed to submit task to thread pool: {e}")
            raise

    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool"""
        if not self._shutdown:
            self._shutdown = True
            self.executor.shutdown(wait=wait)
            logger.info("Thread pool shutdown completed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
