import os
import platform
import re
from typing import Dict, List, Optional


def detect_arch() -> str:
    machine = platform.machine().lower()
    if machine.startswith(("aarch64", "arm", "armv")):
        return "ARM"
    elif machine.startswith(("x86", "x64", "amd64", "i686", "i386")):
        return "x86"
    elif "mips" in machine:
        return "MIPS"
    elif "riscv" in machine:
        return "RISCV"
    return "unknown"


def detect_cpu_count() -> int:
    return os.cpu_count() or 1


def detect_ram() -> Dict[str, float]:
    """Parse /proc/meminfo or return fallback estimate."""
    try:
        with open("/proc/meminfo", "r") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    val_kb = int(parts[1])
                    info[key] = val_kb
            total = info.get("MemTotal", 0)
            available = info.get("MemAvailable", info.get("MemFree", 0))
            return {
                "total_gb": round(total / 1048576, 2),
                "available_gb": round(available / 1048576, 2),
                "used_gb": round((total - available) / 1048576, 2),
                "total_kb": total,
            }
    except FileNotFoundError:
        page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
        pages = os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 16384
        total_bytes = page_size * pages
        total_gb = total_bytes / (1024 ** 3)
        return {
            "total_gb": round(total_gb, 2),
            "available_gb": round(total_gb * 0.8, 2),
            "used_gb": round(total_gb * 0.2, 2),
            "total_kb": int(total_bytes / 1024),
        }


def detect_simd() -> List[str]:
    """Detect SIMD capabilities from /proc/cpuinfo."""
    features = []
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("flags") or line.startswith("Features"):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        flags = parts[1].split()
                        simd_flags = {"avx", "avx2", "avx512", "sse", "sse2", "sse4", "sse4_1",
                                      "sse4_2", "neon", "asimd"}
                        for flag in flags:
                            if flag.lower() in simd_flags or flag.lower().startswith("avx"):
                                features.append(flag.upper())
                        break
    except FileNotFoundError:
        pass
    if not features:
        arch = detect_arch()
        if arch == "ARM":
            features = ["NEON"]
        elif arch == "x86":
            features = ["SSE", "SSE2"]
    return features


def auto_config() -> Dict[str, any]:
    """Return model config dict based on detected RAM."""
    ram = detect_ram()
    total_gb = ram["total_gb"]
    if total_gb <= 4:
        return {
            "layers": 6,
            "hidden_dim": 256,
            "attention_heads": 4,
            "batch_size": 4,
            "max_seq_length": 512,
            "precision": "fp32",
            "model_size": "tiny",
        }
    elif total_gb <= 8:
        return {
            "layers": 12,
            "hidden_dim": 512,
            "attention_heads": 8,
            "batch_size": 8,
            "max_seq_length": 1024,
            "precision": "fp32",
            "model_size": "small",
        }
    elif total_gb <= 16:
        return {
            "layers": 18,
            "hidden_dim": 768,
            "attention_heads": 12,
            "batch_size": 16,
            "max_seq_length": 2048,
            "precision": "fp32",
            "model_size": "medium",
        }
    elif total_gb <= 32:
        return {
            "layers": 24,
            "hidden_dim": 1024,
            "attention_heads": 16,
            "batch_size": 32,
            "max_seq_length": 4096,
            "precision": "fp32",
            "model_size": "large",
        }
    else:
        return {
            "layers": 36,
            "hidden_dim": 2048,
            "attention_heads": 32,
            "batch_size": 64,
            "max_seq_length": 8192,
            "precision": "fp16",
            "model_size": "xlarge",
        }


def get_device_info() -> Dict[str, any]:
    """Comprehensive hardware summary."""
    arch = detect_arch()
    cpu_count = detect_cpu_count()
    ram = detect_ram()
    simd = detect_simd()
    python_ver = platform.python_version()
    os_info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }
    config = auto_config()
    return {
        "architecture": arch,
        "cpu_count": cpu_count,
        "ram": ram,
        "simd": simd,
        "python": python_ver,
        "os": os_info,
        "recommended_config": config,
    }
