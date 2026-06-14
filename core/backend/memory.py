"""Memory manager for efficient resource usage."""

import os
import gc
from typing import Dict, Optional


class MemoryManager:
    """Manages memory allocation and cleanup for efficient operation on any device."""
    
    def __init__(self, max_memory_gb: Optional[float] = None):
        ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        self.total_ram_gb = ram_bytes / 1e9
        self.max_memory_gb = max_memory_gb or (self.total_ram_gb * 0.8)
        self.allocated = 0
        self.cache = {}
    
    def can_allocate(self, size_bytes: int) -> bool:
        return (self.allocated + size_bytes) / 1e9 < self.max_memory_gb
    
    def allocate(self, size_bytes: int, name: str = ""):
        self.allocated += size_bytes
    
    def deallocate(self, size_bytes: int):
        self.allocated = max(0, self.allocated - size_bytes)
    
    def cleanup(self):
        self.cache.clear()
        gc.collect()
    
    def get_memory_usage(self) -> Dict:
        return {
            "allocated_gb": round(self.allocated / 1e9, 2),
            "total_gb": round(self.total_ram_gb, 1),
            "max_gb": round(self.max_memory_gb, 1),
            "usage_percent": round(self.allocated / (self.max_memory_gb * 1e9) * 100, 1),
        }
    
    def estimate_model_memory(self, num_params: int, precision_bytes: int = 4) -> float:
        return num_params * precision_bytes / 1e9
