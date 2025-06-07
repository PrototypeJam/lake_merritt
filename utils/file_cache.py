"""
File-based caching for expensive operations.
"""
import json
import hashlib
import os
from pathlib import Path
from typing import Any, Optional, Callable
from datetime import datetime, timedelta
import pickle
import logging

logger = logging.getLogger(__name__)


class FileCache:
    """Simple file-based cache for storing results of expensive operations."""

    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 24):
        """Initialize file cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def _get_cache_key(self, key_data: Any) -> str:
        """Generate a cache key from input data."""
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.cache"

    def get(self, key_data: Any) -> Optional[Any]:
        """Retrieve a value from cache."""
        cache_key = self._get_cache_key(key_data)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > self.ttl:
                logger.debug(f"Cache expired for key {cache_key}")
                cache_path.unlink()
                return None

            logger.debug(f"Cache hit for key {cache_key}")
            return cache_data['value']

        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None

    def set(self, key_data: Any, value: Any) -> None:
        """Store a value in cache."""
        cache_key = self._get_cache_key(key_data)
        cache_path = self._get_cache_path(cache_key)

        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'value': value,
            'key_data': key_data,
        }

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.debug(f"Cached value for key {cache_key}")
        except Exception as e:
            logger.error(f"Error writing cache: {e}")

    def clear(self) -> None:
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(f"Error deleting cache file {cache_file}: {e}")
        logger.info("Cache cleared")

    def get_size(self) -> int:
        """Get total size of cache in bytes."""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            total_size += cache_file.stat().st_size
        return total_size

    def cleanup_expired(self) -> int:
        """Remove expired cache entries. Returns number of entries removed."""
        removed = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)

                cached_time = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - cached_time > self.ttl:
                    cache_file.unlink()
                    removed += 1
            except Exception as e:
                logger.error(f"Error checking cache file {cache_file}: {e}")

        if removed > 0:
            logger.info(f"Removed {removed} expired cache entries")
        return removed


def cache_result(
    cache_dir: str = ".cache",
    ttl_hours: int = 24,
    key_func: Optional[Callable] = None,
):
    """Decorator to cache function results."""
    def decorator(func):
        cache = FileCache(cache_dir, ttl_hours)

        def wrapper(*args, **kwargs):
            if key_func:
                key_data = key_func(*args, **kwargs)
            else:
                key_data = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': kwargs,
                }

            cached_value = cache.get(key_data)
            if cached_value is not None:
                return cached_value

            result = func(*args, **kwargs)
            cache.set(key_data, result)
            return result

        async def async_wrapper(*args, **kwargs):
            if key_func:
                key_data = key_func(*args, **kwargs)
            else:
                key_data = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': kwargs,
                }

            cached_value = cache.get(key_data)
            if cached_value is not None:
                return cached_value

            result = await func(*args, **kwargs)
            cache.set(key_data, result)
            return result

        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

    return decorator


_global_cache = FileCache()


def get_cache() -> FileCache:
    """Get the global cache instance."""
    return _global_cache
