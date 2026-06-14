"""Auto-detect compute backend and precision."""

import os
import sys
import platform
from enum import Enum


class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"
    OPENCL = "opencl"
    VULKAN = "vulkan"


class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


def detect_device() -> Device:
    """Auto-detect best available compute backend."""
    try:
        import torch
        if torch.cuda.is_available():
            return Device.CUDA
    except ImportError:
        pass

    if sys.platform == "darwin":
        try:
            import subprocess
            result = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=5)
            if "Apple" in result.stdout:
                return Device.METAL
        except:
            pass

    try:
        import subprocess
        result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5)
        if "NVIDIA" in result.stdout:
            return Device.CUDA
        if "AMD" in result.stdout:
            return Device.OPENCL
    except:
        pass

    return Device.CPU


def get_precision(device: Device, ram_gb: float = 8.0) -> Precision:
    """Auto-select precision based on device and available memory."""
    if device == Device.CUDA:
        if ram_gb >= 24:
            return Precision.FP32
        elif ram_gb >= 8:
            return Precision.FP16
        else:
            return Precision.INT8
    elif device in (Device.METAL, Device.OPENCL):
        if ram_gb >= 8:
            return Precision.FP32
        else:
            return Precision.FP16
    else:
        if ram_gb >= 16:
            return Precision.FP32
        elif ram_gb >= 8:
            return Precision.FP16
        elif ram_gb >= 4:
            return Precision.INT8
        else:
            return Precision.INT4


def get_system_info() -> Dict:
    """Get complete system information."""
    ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    ram_gb = ram_bytes / 1e9
    device = detect_device()
    precision = get_precision(device, ram_gb)
    return {
        "device": device.value,
        "precision": precision.value,
        "ram_gb": round(ram_gb, 1),
        "cpu_cores": os.cpu_count(),
        "platform": platform.machine(),
        "python": platform.python_version(),
        "os": platform.system(),
    }
