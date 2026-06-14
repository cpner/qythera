"""Compute backend abstraction layer."""

from core.backend.device import Device, detect_device, get_precision
from core.backend.memory import MemoryManager

__all__ = ["Device", "detect_device", "get_precision", "MemoryManager"]
