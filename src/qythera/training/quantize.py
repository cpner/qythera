"""Quantization methods for neural network weights.

Pure Python + NumPy implementations of various quantization techniques
for model compression and efficient inference.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import math


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def quantize_tensor(
    tensor: np.ndarray,
    bits: int,
    group_size: int = 128,
    symmetric: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """General-purpose tensor quantization.

    Args:
        tensor: Input float32 tensor.
        bits: Target bit-width (1-16).
        group_size: Number of elements per quantization group.
        symmetric: If True, use symmetric quantization around zero.

    Returns:
        Tuple of (quantized indices, metadata dict).
    """
    original_shape = tensor.shape
    flat = tensor.reshape(-1).astype(np.float32)

    n = flat.size
    n_groups = max(1, math.ceil(n / group_size))
    padded_n = n_groups * group_size
    if padded_n > n:
        flat_padded = np.zeros(padded_n, dtype=np.float32)
        flat_padded[:n] = flat
    else:
        flat_padded = flat

    groups = flat_padded.reshape(n_groups, group_size)

    qmin = 0
    qmax = (1 << bits) - 1

    if symmetric:
        scales = np.max(np.abs(groups), axis=1, keepdims=True)
        scales = np.maximum(scales, 1e-8)
        scale_factor = scales / (qmax / 2)
        quantized = np.round(groups / scale_factor).astype(np.int32)
        quantized = np.clip(quantized, -qmax / 2, qmax / 2).astype(np.int32)
        zero_point = None
    else:
        mins = np.min(groups, axis=1, keepdims=True)
        maxs = np.max(groups, axis=1, keepdims=True)
        ranges = maxs - mins
        ranges = np.maximum(ranges, 1e-8)
        scale_factor = ranges / qmax
        quantized = np.round((groups - mins) / scale_factor).astype(np.int32)
        quantized = np.clip(quantized, 0, qmax)
        zero_point = (-mins / scale_factor)

    quantized = quantized.reshape(-1)[:n]

    metadata = {
        "original_shape": original_shape,
        "bits": bits,
        "group_size": group_size,
        "symmetric": symmetric,
        "scale_factor": scale_factor.reshape(-1),
        "zero_point": zero_point.reshape(-1) if zero_point is not None else None,
        "n_groups": n_groups,
    }

    return quantized, metadata


def measure_quantization_error(
    original: np.ndarray,
    quantized: np.ndarray,
) -> Dict[str, float]:
    """Measure error metrics between original and quantized tensors.

    Args:
        original: Original float32 tensor.
        quantized: Quantized (possibly dequantized) float32 tensor.

    Returns:
        Dict with MSE, MAE, max absolute error, relative error, and SNR.
    """
    orig_flat = original.reshape(-1).astype(np.float32)
    quant_flat = quantized.reshape(-1).astype(np.float32)

    if orig_flat.size != quant_flat.size:
        raise ValueError(f"Size mismatch: {orig_flat.size} vs {quant_flat.size}")

    diff = orig_flat - quant_flat
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    max_err = float(np.max(np.abs(diff)))

    orig_mean = np.mean(orig_flat)
    signal_power = float(np.mean((orig_flat - orig_mean) ** 2))
    noise_power = mse
    snr_db = float(10 * np.log10(signal_power / (noise_power + 1e-12)))

    relative_error = float(np.mean(np.abs(diff) / (np.abs(orig_flat) + 1e-8)))

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_err,
        "relative_error": relative_error,
        "snr_db": snr_db,
    }


def auto_quantize(
    model_weights: Dict[str, np.ndarray],
    target_bits: float,
    calibration_data: Optional[np.ndarray] = None,
) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
    """Automatically quantize model weights to achieve target average bit-width.

    Uses a greedy strategy: quantizes sensitive layers (by variance) to higher
    precision and compresses low-variance layers more aggressively.

    Args:
        model_weights: Dict mapping layer names to weight arrays.
        target_bits: Target average bit-width across all parameters.
        calibration_data: Optional calibration input for static quantization.

    Returns:
        Dict mapping layer names to (quantized, metadata) tuples.
    """
    n_layers = len(model_weights)
    if n_layers == 0:
        return {}

    total_params = sum(w.size for w in model_weights.values())
    total_bits_budget = int(target_bits * total_params)

    layer_stats = []
    for name, weight in model_weights.items():
        variance = float(np.var(weight))
        layer_stats.append((name, weight, variance))

    layer_stats.sort(key=lambda x: x[2], reverse=True)

    results = {}
    bits_used = 0

    for name, weight, variance in layer_stats:
        param_bits = weight.size * 8
        remaining_layers = n_layers - len(results)
        remaining_budget = total_bits_budget - bits_used

        if remaining_layers > 0:
            bits_per_layer = remaining_budget / remaining_layers / weight.size
        else:
            bits_per_layer = 8

        bits_per_layer = max(2, min(8, int(round(bits_per_layer))))

        if variance < 1e-6:
            bits_per_layer = max(2, bits_per_layer - 2)
        elif variance > 1.0:
            bits_per_layer = min(8, bits_per_layer + 1)

        quantized, meta = quantize_tensor(weight, bits_per_layer, group_size=128)
        dequantized = dequantize_tensor(quantized, meta)

        layer_meta = {
            "method": f"INT{bits_per_layer}",
            "quantized": quantized,
            "metadata": meta,
            "dequantized": dequantized,
            "compression_ratio": param_bits / (bits_per_layer * weight.size),
        }

        results[name] = (quantized, layer_meta)
        bits_used += bits_per_layer * weight.size

    return results


def dequantize_tensor(
    quantized: np.ndarray,
    metadata: Dict[str, Any],
) -> np.ndarray:
    """Dequantize a tensor back to float32.

    Args:
        quantized: Quantized integer tensor.
        metadata: Metadata from quantize_tensor.

    Returns:
        Dequantized float32 tensor.
    """
    scale_factor = metadata["scale_factor"]
    symmetric = metadata["symmetric"]
    n_groups = metadata["n_groups"]
    group_size = metadata["group_size"]
    original_shape = metadata["original_shape"]

    n = quantized.size
    padded_n = n_groups * group_size
    q_padded = np.zeros(padded_n, dtype=np.float32)
    q_padded[:n] = quantized.astype(np.float32)

    groups = q_padded.reshape(n_groups, group_size)

    if symmetric:
        reconstructed = groups * scale_factor
    else:
        zero_point = metadata["zero_point"]
        reconstructed = groups * scale_factor + zero_point

    return reconstructed.reshape(-1)[:n].reshape(original_shape)


# ---------------------------------------------------------------------------
# Dynamic INT8 Quantization
# ---------------------------------------------------------------------------

class DynamicINT8:
    """Dynamic per-tensor INT8 quantization.

    Computes scale on-the-fly: scale = max(abs(tensor)) / 127.
    Supports both per-tensor and per-channel modes.
    """

    def __init__(self, per_channel: bool = False, axis: int = 0):
        self.per_channel = per_channel
        self.axis = axis
        self._last_metadata: Optional[Dict[str, Any]] = None

    def quantize(self, tensor: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        tensor = tensor.astype(np.float32)

        if self.per_channel:
            reduce_axes = tuple(i for i in range(tensor.ndim) if i != self.axis)
            abs_max = np.max(np.abs(tensor), axis=reduce_axes, keepdims=True)
            scale = abs_max / 127.0
            scale = np.maximum(scale, 1e-8)
            quantized = np.round(tensor / scale).astype(np.int8)
        else:
            abs_max = float(np.max(np.abs(tensor)))
            scale = max(abs_max / 127.0, 1e-8)
            quantized = np.round(tensor / scale).astype(np.int8)

        metadata = {
            "scale": scale,
            "dtype": np.int8,
            "per_channel": self.per_channel,
            "axis": self.axis,
            "original_shape": tensor.shape,
            "bits": 8,
        }
        self._last_metadata = metadata
        return quantized, metadata

    def dequantize(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        scale = metadata["scale"]
        return (quantized.astype(np.float32) * scale).astype(np.float32)

    def get_compression_ratio(self) -> float:
        return 32.0 / 8.0


# ---------------------------------------------------------------------------
# Static INT8 Quantization
# ---------------------------------------------------------------------------

class StaticINT8:
    """Static INT8 quantization using calibration data.

    Pre-computes scale and zero-point from a calibration dataset,
    then reuses these values for all future quantizations.
    """

    def __init__(self, num_calibration_batches: int = 100):
        self.num_calibration_batches = num_calibration_batches
        self._calibrated = False
        self._scale: Optional[Union[float, np.ndarray]] = None
        self._zero_point: Optional[Union[int, np.ndarray]] = None
        self._per_channel: bool = False
        self._axis: int = 0

    def calibrate(
        self,
        calibration_batches: List[np.ndarray],
        per_channel: bool = False,
        axis: int = 0,
    ) -> None:
        """Calibrate using representative data.

        Args:
            calibration_batches: List of calibration input arrays.
            per_channel: Whether to compute per-channel scales.
            axis: Channel axis if per_channel is True.
        """
        self._per_channel = per_channel
        self._axis = axis

        if per_channel:
            all_abs_max = None
            for batch in calibration_batches[:self.num_calibration_batches]:
                batch = batch.astype(np.float32)
                reduce_axes = tuple(i for i in range(batch.ndim) if i != axis)
                batch_max = np.max(np.abs(batch), axis=reduce_axes, keepdims=True)
                if all_abs_max is None:
                    all_abs_max = batch_max
                else:
                    all_abs_max = np.maximum(all_abs_max, batch_max)

            self._scale = all_abs_max / 127.0
            self._scale = np.maximum(self._scale, 1e-8)
            self._zero_point = np.zeros_like(self._scale)
        else:
            global_max = 0.0
            for batch in calibration_batches[:self.num_calibration_batches]:
                batch = batch.astype(np.float32)
                batch_max = float(np.max(np.abs(batch)))
                global_max = max(global_max, batch_max)

            self._scale = max(global_max / 127.0, 1e-8)
            self._zero_point = 0

        self._calibrated = True

    def quantize(self, tensor: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self._calibrated:
            raise RuntimeError("Model not calibrated. Call calibrate() first.")

        tensor = tensor.astype(np.float32)
        quantized = np.round(tensor / self._scale).astype(np.int8)

        metadata = {
            "scale": self._scale,
            "zero_point": self._zero_point,
            "dtype": np.int8,
            "per_channel": self._per_channel,
            "axis": self._axis,
            "original_shape": tensor.shape,
            "bits": 8,
            "calibrated": True,
        }
        return quantized, metadata

    def dequantize(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        scale = metadata["scale"]
        zero_point = metadata.get("zero_point", 0)
        return ((quantized.astype(np.float32) - zero_point) * scale).astype(np.float32)

    def get_compression_ratio(self) -> float:
        return 32.0 / 8.0


# ---------------------------------------------------------------------------
# INT4 Group Quantization
# ---------------------------------------------------------------------------

class INT4GroupQuantization:
    """INT4 quantization with per-group scale and zero-point.

    Groups of 128 elements share a scale and zero-point for better
    accuracy at 4-bit precision.
    """

    def __init__(self, group_size: int = 128, symmetric: bool = True):
        self.group_size = group_size
        self.symmetric = symmetric

    def quantize(self, tensor: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        tensor = tensor.astype(np.float32)
        original_shape = tensor.shape
        flat = tensor.reshape(-1)
        n = flat.size

        n_groups = max(1, math.ceil(n / self.group_size))
        padded_n = n_groups * self.group_size
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = flat

        groups = padded.reshape(n_groups, self.group_size)

        if self.symmetric:
            scales = np.max(np.abs(groups), axis=1, keepdims=True)
            scales = np.maximum(scales, 1e-8)
            scale_factor = scales / 7.0
            quantized = np.round(groups / scale_factor).astype(np.int32)
            quantized = np.clip(quantized, -8, 7)
            zero_points = None
        else:
            mins = np.min(groups, axis=1, keepdims=True)
            maxs = np.max(groups, axis=1, keepdims=True)
            ranges = maxs - mins
            ranges = np.maximum(ranges, 1e-8)
            scale_factor = ranges / 15.0
            quantized = np.round((groups - mins) / scale_factor).astype(np.int32)
            quantized = np.clip(quantized, 0, 15)
            zero_points = (-mins / scale_factor).astype(np.float32)

        quantized_flat = quantized.reshape(-1)[:n].astype(np.int8)

        metadata = {
            "original_shape": original_shape,
            "bits": 4,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "n_groups": n_groups,
            "scale": scale_factor.reshape(-1),
            "zero_point": zero_points.reshape(-1) if zero_points is not None else None,
        }
        return quantized_flat, metadata

    def dequantize(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        original_shape = metadata["original_shape"]
        n_groups = metadata["n_groups"]
        n = quantized.size
        padded_n = n_groups * self.group_size
        q_padded = np.zeros(padded_n, dtype=np.float32)
        q_padded[:n] = quantized.astype(np.float32)

        groups = q_padded.reshape(n_groups, self.group_size)

        if metadata["symmetric"]:
            reconstructed = groups * metadata["scale"].reshape(-1, 1)
        else:
            reconstructed = (
                groups * metadata["scale"].reshape(-1, 1)
                + metadata["zero_point"].reshape(-1, 1)
            )

        return reconstructed.reshape(-1)[:n].reshape(original_shape)

    def get_compression_ratio(self) -> float:
        return 32.0 / 4.0


# ---------------------------------------------------------------------------
# INT2 Quantization
# ---------------------------------------------------------------------------

class INT2Quantization:
    """Extreme 2-bit quantization: 4 values packed per byte.

    Uses lookup-table quantization with 4 representable values per group.
    """

    def __init__(self, group_size: int = 64):
        self.group_size = group_size

    def quantize(self, tensor: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        tensor = tensor.astype(np.float32)
        original_shape = tensor.shape
        flat = tensor.reshape(-1)
        n = flat.size

        n_groups = max(1, math.ceil(n / self.group_size))
        padded_n = n_groups * self.group_size
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = flat

        groups = padded.reshape(n_groups, self.group_size)

        codes_all = np.zeros(padded_n, dtype=np.uint8)
        luts = np.zeros((n_groups, 4), dtype=np.float32)

        for g in range(n_groups):
            group = groups[g]
            vmin, vmax = float(np.min(group)), float(np.max(group))
            if vmax - vmin < 1e-8:
                lut = np.array([vmin, vmin, vmin, vmin], dtype=np.float32)
            else:
                lut = np.linspace(vmin, vmax, 4).astype(np.float32)

            luts[g] = lut

            distances = np.abs(group.reshape(-1, 1) - lut.reshape(1, -1))
            codes = np.argmin(distances, axis=1).astype(np.uint8)
            codes_all[g * self.group_size : (g + 1) * self.group_size] = codes

        packed = self._pack_int2(codes_all[:n])

        metadata = {
            "original_shape": original_shape,
            "bits": 2,
            "group_size": self.group_size,
            "n_groups": n_groups,
            "luts": luts,
            "n_original": n,
            "packed_shape": packed.shape,
        }
        return packed, metadata

    def _pack_int2(self, codes: np.ndarray) -> np.ndarray:
        n = codes.size
        padded_n = math.ceil(n / 4) * 4
        padded = np.zeros(padded_n, dtype=np.uint8)
        padded[:n] = codes

        n_bytes = padded_n // 4
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for i in range(n_bytes):
            offset = i * 4
            packed[i] = (
                (padded[offset] & 0x03)
                | ((padded[offset + 1] & 0x03) << 2)
                | ((padded[offset + 2] & 0x03) << 4)
                | ((padded[offset + 3] & 0x03) << 6)
            )
        return packed

    def _unpack_int2(self, packed: np.ndarray, n: int) -> np.ndarray:
        unpadded = np.zeros(packed.size * 4, dtype=np.uint8)
        for i in range(packed.size):
            byte = packed[i]
            unpadded[i * 4] = byte & 0x03
            unpadded[i * 4 + 1] = (byte >> 2) & 0x03
            unpadded[i * 4 + 2] = (byte >> 4) & 0x03
            unpadded[i * 4 + 3] = (byte >> 6) & 0x03
        return unpadded[:n]

    def dequantize(self, packed: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        original_shape = metadata["original_shape"]
        n = metadata["n_original"]
        luts = metadata["luts"]

        codes = self._unpack_int2(packed, n)

        n_groups = metadata["n_groups"]
        group_size = metadata["group_size"]
        result = np.zeros(n, dtype=np.float32)

        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, n)
            if start >= n:
                break
            group_codes = codes[start:end]
            result[start:end] = luts[g][group_codes]

        return result.reshape(original_shape)

    def get_compression_ratio(self) -> float:
        return 32.0 / 2.0


# ---------------------------------------------------------------------------
# GPTQ Quantization
# ---------------------------------------------------------------------------

class GPTQ:
    """GPTQ: layer-wise quantization using Hessian information.

    Uses optimal brain quantization (OBQ) ordering with Cholesky decomposition
    of the Hessian (H = X^T @ X) for optimal weight quantization.
    """

    def __init__(self, bits: int = 4, group_size: int = 128, damp: float = 0.01):
        self.bits = bits
        self.group_size = group_size
        self.damp = damp

    def quantize(
        self,
        weight: np.ndarray,
        hessian: Optional[np.ndarray] = None,
        calibration_data: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize weight matrix using GPTQ.

        Args:
            weight: Weight matrix of shape (out_features, in_features).
            hessian: Pre-computed Hessian (X^T @ X). If None, computed from calibration_data.
            calibration_data: Input data for computing Hessian if not provided.

        Returns:
            Tuple of (quantized_weight, metadata).
        """
        weight = weight.astype(np.float32)
        rows, cols = weight.shape

        if hessian is None:
            if calibration_data is None:
                raise ValueError("Either hessian or calibration_data must be provided")
            X = calibration_data.astype(np.float32)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            hessian = X.T @ X

        H = hessian.astype(np.float32)
        assert H.shape == (cols, cols), f"Hessian shape {H.shape} != ({cols}, {cols})"

        diag = np.diag(H)
        damp_val = self.damp * np.mean(np.abs(diag)) + 1e-6
        H += damp_val * np.eye(cols, dtype=np.float32)

        H_inv = np.linalg.inv(H)
        H_inv_sym = (H_inv + H_inv.T) / 2.0

        W = weight.copy()
        W_quantized = np.zeros_like(W)

        qmin = 0
        qmax = (1 << self.bits) - 1
        n_groups = max(1, math.ceil(cols / self.group_size))

        scale_factors = np.zeros((rows, n_groups), dtype=np.float32)
        zero_points = np.zeros((rows, n_groups), dtype=np.float32)

        for g in range(n_groups):
            start = g * self.group_size
            end = min(start + self.group_size, cols)
            group_W = W[:, start:end]

            abs_max = np.max(np.abs(group_W), axis=1, keepdims=True)
            abs_max = np.maximum(abs_max, 1e-8)
            sf = abs_max / (qmax / 2)
            scale_factors[:, g] = sf[:, 0]

            q_group = np.round(group_W / sf).astype(np.int32)
            q_group = np.clip(q_group, -qmax / 2, qmax / 2)
            W_quantized[:, start:end] = (q_group * sf).astype(np.float32)

            error = group_W - W_quantized[:, start:end]

            H_block = H_inv_sym[start:end, start:end]
            correction = error @ H_block
            W[:, end:] += correction @ H_inv_sym[end:, start:end]

        metadata = {
            "original_shape": weight.shape,
            "bits": self.bits,
            "group_size": self.group_size,
            "scale": scale_factors,
            "zero_point": zero_points,
            "damp": self.damp,
            "method": "GPTQ",
        }

        return W_quantized, metadata

    def dequantize(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        return quantized.astype(np.float32)

    def get_compression_ratio(self) -> float:
        return 32.0 / self.bits


# ---------------------------------------------------------------------------
# AWQ (Activation-aware Weight Quantization)
# ---------------------------------------------------------------------------

class AWQ:
    """Activation-aware Weight Quantization.

    Searches for optimal per-channel scaling factors that minimize
    quantization error, weighted by activation magnitudes.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        n_grid: int = 20,
        search_range: float = 2.0,
    ):
        self.bits = bits
        self.group_size = group_size
        self.n_grid = n_grid
        self.search_range = search_range

    def _quantize_with_scale(
        self,
        weight: np.ndarray,
        scales: np.ndarray,
        bits: int,
        group_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        qmax = (1 << (bits - 1)) - 1
        scaled_weight = weight * scales
        rows, cols = scaled_weight.shape
        n_groups = max(1, math.ceil(cols / group_size))
        padded_cols = n_groups * group_size

        padded = np.zeros((rows, padded_cols), dtype=np.float32)
        padded[:, :cols] = scaled_weight

        groups = padded.reshape(rows, n_groups, group_size)
        abs_max = np.max(np.abs(groups), axis=2, keepdims=True)
        abs_max = np.maximum(abs_max, 1e-8)
        group_scale = abs_max / qmax
        quantized = np.round(groups / group_scale).astype(np.int32)
        quantized = np.clip(quantized, -qmax - 1, qmax)

        dequantized = (quantized * group_scale).reshape(rows, padded_cols)[:, :cols]
        dequantized = dequantized / scales

        return dequantized, group_scale.reshape(rows, n_groups)

    def quantize(
        self,
        weight: np.ndarray,
        activation_scales: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize weight with optimal per-channel scaling.

        Args:
            weight: Weight matrix of shape (out_features, in_features).
            activation_scales: Per-channel activation scales of shape (1, in_features).
                              If None, uses uniform importance.

        Returns:
            Tuple of (quantized_weight, metadata).
        """
        weight = weight.astype(np.float32)
        rows, cols = weight.shape

        if activation_scales is None:
            activation_scales = np.ones((1, cols), dtype=np.float32)
        else:
            activation_scales = activation_scales.reshape(1, -1).astype(np.float32)

        act_importance = activation_scales / np.max(activation_scales)
        weight_importance = np.abs(weight) * act_importance
        weight_importance_norm = np.mean(weight_importance, axis=0)
        weight_importance_norm = np.maximum(weight_importance_norm, 1e-8)

        best_scales = np.ones((1, cols), dtype=np.float32)
        best_error = float("inf")

        grid = np.linspace(
            1.0 / self.search_range,
            self.search_range,
            self.n_grid,
        )

        for alpha in grid:
            candidate_scales = (
                (weight_importance_norm ** alpha).reshape(1, -1)
                + 1e-8
            )
            candidate_scales = candidate_scales / np.max(candidate_scales) * 2.0

            dequantized, _ = self._quantize_with_scale(
                weight, candidate_scales, self.bits, self.group_size
            )

            err = float(np.mean((weight - dequantized) ** 2 * act_importance))
            if err < best_error:
                best_error = err
                best_scales = candidate_scales

        dequantized, group_scales = self._quantize_with_scale(
            weight, best_scales, self.bits, self.group_size
        )

        qmax = (1 << (self.bits - 1)) - 1
        scaled_weight = weight * best_scales
        n_groups = max(1, math.ceil(cols / self.group_size))
        padded_cols = n_groups * self.group_size
        padded = np.zeros((rows, padded_cols), dtype=np.float32)
        padded[:, :cols] = scaled_weight
        groups = padded.reshape(rows, n_groups, self.group_size)
        abs_max = np.max(np.abs(groups), axis=2, keepdims=True)
        abs_max = np.maximum(abs_max, 1e-8)
        group_scale = abs_max / qmax
        quantized = np.round(groups / group_scale).astype(np.int8)

        metadata = {
            "original_shape": weight.shape,
            "bits": self.bits,
            "group_size": self.group_size,
            "channel_scales": best_scales,
            "group_scales": group_scales,
            "quantized_shape": quantized.shape,
            "method": "AWQ",
        }

        return quantized, metadata

    def dequantize(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        channel_scales = metadata["channel_scales"]
        rows, cols = metadata["original_shape"]
        n_groups = max(1, math.ceil(cols / self.group_size))
        padded_cols = n_groups * self.group_size

        float_q = quantized.astype(np.float32)
        if float_q.ndim == 2 and float_q.shape[0] == rows and float_q.shape[1] == padded_cols:
            groups = float_q.reshape(rows, n_groups, self.group_size)
        else:
            padded = np.zeros((rows, padded_cols), dtype=np.float32)
            flat = quantized.reshape(-1)
            n = min(flat.size, rows * padded_cols)
            padded.flat[:n] = flat[:n]
            groups = padded.reshape(rows, n_groups, self.group_size)

        group_scales = metadata["group_scales"].reshape(rows, n_groups, 1)
        dequantized = (groups * group_scales).reshape(rows, padded_cols)[:, :cols]
        dequantized = dequantized / channel_scales

        return dequantized.reshape(metadata["original_shape"])

    def get_compression_ratio(self) -> float:
        return 32.0 / self.bits


# ---------------------------------------------------------------------------
# SmoothQuant
# ---------------------------------------------------------------------------

class SmoothQuant:
    """SmoothQuant: migration of quantization difficulty from activations to weights.

    Applies per-channel scaling to smooth the input distribution, making
    both weights and activations easier to quantize.
    """

    def __init__(self, alpha: float = 0.5):
        """Initialize SmoothQuant.

        Args:
            alpha: Smoothness parameter. Higher values shift more difficulty
                   to weights. Range [0, 1].
        """
        self.alpha = alpha
        self._weight_scales: Optional[np.ndarray] = None
        self._activation_scales: Optional[np.ndarray] = None

    def compute_scales(
        self,
        weight: np.ndarray,
        activation_scales: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute smooth quantization scales.

        Args:
            weight: Weight matrix (out_features, in_features).
            activation_scales: Per-input-channel activation scales (in_features,).

        Returns:
            Tuple of (weight_scales, smoothed_activation_scales).
        """
        weight = weight.astype(np.float32)
        activation_scales = activation_scales.astype(np.float32).reshape(-1)

        weight_scales = np.max(np.abs(weight), axis=0) ** self.alpha
        weight_scales = np.maximum(weight_scales, 1e-8)

        activation_scales_smoothed = activation_scales ** (1.0 - self.alpha)
        activation_scales_smoothed = np.maximum(activation_scales_smoothed, 1e-8)

        self._weight_scales = weight_scales
        self._activation_scales = activation_scales_smoothed

        return weight_scales, activation_scales_smoothed

    def smooth_weight(
        self,
        weight: np.ndarray,
        weight_scales: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply smoothing to weights.

        Args:
            weight: Weight matrix.
            weight_scales: Pre-computed scales. If None, recomputed.

        Returns:
            Smoothed weight matrix.
        """
        if weight_scales is None:
            weight_scales = self._weight_scales
        if weight_scales is None:
            raise ValueError("No weight scales available. Call compute_scales first.")

        return weight / weight_scales.reshape(1, -1)

    def quantize(
        self,
        weight: np.ndarray,
        activation_scales: Optional[np.ndarray] = None,
        int8_quantizer: Optional[StaticINT8] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Full SmoothQuant pipeline: smooth then quantize.

        Args:
            weight: Weight matrix.
            activation_scales: Per-channel activation scales.
            int8_quantizer: Optional pre-calibrated quantizer.

        Returns:
            Tuple of (quantized_weight, metadata).
        """
        weight = weight.astype(np.float32)

        if activation_scales is None:
            activation_scales = np.max(np.abs(weight), axis=0)

        weight_scales, act_scales_smoothed = self.compute_scales(weight, activation_scales)
        smoothed_weight = self.smooth_weight(weight, weight_scales)

        if int8_quantizer is None:
            int8_quantizer = StaticINT8()
            int8_quantizer.calibrate([smoothed_weight], per_channel=True, axis=0)

        quantized, q_meta = int8_quantizer.quantize(smoothed_weight)

        metadata = {
            "original_shape": weight.shape,
            "weight_scales": weight_scales,
            "activation_scales": act_scales_smoothed,
            "alpha": self.alpha,
            "quantizer_metadata": q_meta,
            "method": "SmoothQuant",
            "bits": 8,
        }

        return quantized, metadata

    def dequantize(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        q_meta = metadata["quantizer_metadata"]
        int8_q = StaticINT8()
        smoothed = int8_q.dequantize(quantized, q_meta)

        weight_scales = metadata["weight_scales"]
        return smoothed * weight_scales.reshape(1, -1)

    def get_compression_ratio(self) -> float:
        return 32.0 / 8.0


# ---------------------------------------------------------------------------
# LLM.int8() Quantization
# ---------------------------------------------------------------------------

class LLM_int8:
    """LLM.int8(): mixed-precision decomposition.

    Detects outlier channels (>6 sigma) and keeps them in FP16,
    while quantizing the remaining channels to INT8.
    """

    def __init__(self, outlier_threshold: float = 6.0, per_channel: bool = True):
        self.outlier_threshold = outlier_threshold
        self.per_channel = per_channel

    def _find_outliers(self, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find outlier channels based on magnitude statistics.

        Args:
            tensor: Input tensor.

        Returns:
            Tuple of (outlier_mask, channel_magnitudes).
        """
        if tensor.ndim == 1:
            abs_vals = np.abs(tensor)
            mean = np.mean(abs_vals)
            std = np.std(abs_vals)
            threshold = mean + self.outlier_threshold * std
            mask = abs_vals > threshold
            return mask, abs_vals

        if self.per_channel:
            channel_mag = np.max(np.abs(tensor), axis=0)
        else:
            channel_mag = np.max(np.abs(tensor), axis=1)

        mean = np.mean(channel_mag)
        std = np.std(channel_mag)
        threshold = mean + self.outlier_threshold * std
        mask = channel_mag > threshold

        return mask, channel_mag

    def quantize(self, tensor: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize tensor with outlier detection.

        Args:
            tensor: Input weight or activation tensor.

        Returns:
            Tuple of (quantized_result, metadata).
        """
        tensor = tensor.astype(np.float32)
        original_shape = tensor.shape

        outlier_mask, channel_mag = self._find_outliers(tensor)

        if tensor.ndim == 1:
            outlier_channels = tensor[outlier_mask].copy()
            normal_channels = tensor.copy()
            normal_channels[outlier_mask] = 0.0
        else:
            if self.per_channel:
                outlier_channels = np.zeros_like(tensor)
                normal_channels = tensor.copy()
                for i in range(tensor.shape[1]):
                    if outlier_mask[i]:
                        outlier_channels[:, i] = tensor[:, i]
                        normal_channels[:, i] = 0.0
            else:
                outlier_channels = np.zeros_like(tensor)
                normal_channels = tensor.copy()
                for i in range(tensor.shape[0]):
                    if outlier_mask[i]:
                        outlier_channels[i] = tensor[i]
                        normal_channels[i] = 0.0

        abs_max = np.max(np.abs(normal_channels))
        abs_max = max(abs_max, 1e-8)
        scale = abs_max / 127.0
        quantized_normal = np.round(normal_channels / scale).astype(np.int8)

        metadata = {
            "original_shape": original_shape,
            "outlier_mask": outlier_mask,
            "outlier_channels": outlier_channels.astype(np.float16),
            "normal_scale": scale,
            "outlier_count": int(np.sum(outlier_mask)),
            "total_channels": tensor.size if tensor.ndim == 1 else tensor.shape[1 if self.per_channel else 0],
            "method": "LLM.int8",
            "bits": 8,
            "has_fp16_outliers": True,
        }

        return quantized_normal, metadata

    def dequantize(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        scale = metadata["normal_scale"]
        reconstructed = quantized.astype(np.float32) * scale

        outlier_mask = metadata["outlier_mask"]
        outlier_channels = metadata["outlier_channels"].astype(np.float32)

        original_shape = metadata["original_shape"]

        if quantized.ndim == 1:
            reconstructed[outlier_mask] = outlier_channels
        else:
            if self.per_channel:
                for i in range(quantized.shape[1]):
                    if outlier_mask[i]:
                        reconstructed[:, i] = outlier_channels[:, i]
            else:
                for i in range(quantized.shape[0]):
                    if outlier_mask[i]:
                        reconstructed[i] = outlier_channels[i]

        return reconstructed.reshape(original_shape)

    def get_compression_ratio(self) -> float:
        return 32.0 / 8.0


# ---------------------------------------------------------------------------
# KV Cache Quantization
# ---------------------------------------------------------------------------

class KVCacheQuantization:
    """Per-head KV cache quantization.

    Quantizes keys and values with per-head scaling for efficient
    attention computation in autoregressive models.
    """

    def __init__(
        self,
        bits: int = 8,
        per_head_scale: bool = True,
        group_size: int = 128,
    ):
        self.bits = bits
        self.per_head_scale = per_head_scale
        self.group_size = group_size

    def quantize(
        self,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        """Quantize key and value tensors.

        Args:
            keys: Key tensor of shape (batch, n_heads, seq_len, head_dim).
            values: Value tensor of shape (batch, n_heads, seq_len, head_dim).

        Returns:
            Tuple of ((quantized_keys, quantized_values), metadata).
        """
        keys = keys.astype(np.float32)
        values = values.astype(np.float32)

        qmax = (1 << (self.bits - 1)) - 1

        if self.per_head_scale and keys.ndim >= 2:
            k_scales = self._compute_head_scales(keys, qmax)
            v_scales = self._compute_head_scales(values, qmax)
        else:
            k_abs_max = max(float(np.max(np.abs(keys))), 1e-8)
            v_abs_max = max(float(np.max(np.abs(values))), 1e-8)
            k_scales = k_abs_max / qmax
            v_scales = v_abs_max / qmax

        k_quantized = np.round(keys / k_scales).astype(np.int8)
        k_quantized = np.clip(k_quantized, -qmax - 1, qmax)

        v_quantized = np.round(values / v_scales).astype(np.int8)
        v_quantized = np.clip(v_quantized, -qmax - 1, qmax)

        metadata = {
            "original_key_shape": keys.shape,
            "original_value_shape": values.shape,
            "key_scales": k_scales,
            "value_scales": v_scales,
            "bits": self.bits,
            "per_head_scale": self.per_head_scale,
            "group_size": self.group_size,
            "method": "KVCacheQuantization",
        }

        return (k_quantized, v_quantized), metadata

    def _compute_head_scales(
        self,
        tensor: np.ndarray,
        qmax: int,
    ) -> np.ndarray:
        """Compute per-head scales for a tensor.

        Args:
            tensor: Input tensor.
            qmax: Maximum quantization value.

        Returns:
            Per-head scales.
        """
        if tensor.ndim == 4:
            abs_max = np.max(np.abs(tensor), axis=(0, 2, 3), keepdims=True)
        elif tensor.ndim == 3:
            abs_max = np.max(np.abs(tensor), axis=(0, 2), keepdims=True)
        else:
            abs_max = np.max(np.abs(tensor))

        abs_max = np.maximum(abs_max, 1e-8)
        return abs_max / qmax

    def dequantize(
        self,
        quantized_data: Tuple[np.ndarray, np.ndarray],
        metadata: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dequantize keys and values.

        Args:
            quantized_data: Tuple of (quantized_keys, quantized_values).
            metadata: Metadata from quantize().

        Returns:
            Tuple of (dequantized_keys, dequantized_values).
        """
        k_quantized, v_quantized = quantized_data
        k_scales = metadata["key_scales"]
        v_scales = metadata["value_scales"]

        keys = k_quantized.astype(np.float32) * k_scales
        values = v_quantized.astype(np.float32) * v_scales

        return keys, values

    def get_compression_ratio(self) -> float:
        return 32.0 / self.bits


# ---------------------------------------------------------------------------
# QuaRot: Random Rotation Quantization
# ---------------------------------------------------------------------------

class QuaRot:
    """QuaRot: apply random orthogonal rotation to weights before quantization.

    Reduces quantization error by rotating weights into a space where they
    are more uniformly distributed (incoherence principle). The rotation
    matrix Q is generated via QR decomposition of a random Gaussian matrix.
    """

    def __init__(self, dim: int):
        self.dim = dim
        rng = np.random.RandomState(42)
        H = rng.randn(dim, dim).astype(np.float32)
        self.Q, _ = np.linalg.qr(H)
        self.Q = self.Q.astype(np.float32)

    def rotate(self, weight: np.ndarray) -> np.ndarray:
        return self.Q @ weight.astype(np.float32)

    def quantize(
        self,
        weight: np.ndarray,
        bits: int = 4,
        group_size: int = 128,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        weight = weight.astype(np.float32)
        rotated = self.Q @ weight
        quantized, meta = quantize_tensor(rotated, bits, group_size)
        metadata = {
            "method": "QuaRot",
            "bits": bits,
            "group_size": group_size,
            "original_shape": weight.shape,
            "quantized_shape": quantized.shape,
            "Q": self.Q,
            "quantizer_metadata": meta,
        }
        return quantized, metadata

    def dequantize(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        meta = metadata["quantizer_metadata"]
        Q = metadata["Q"]
        dequantized = dequantize_tensor(quantized, meta)
        return Q.T @ dequantized

    def get_compression_ratio(self) -> float:
        return 32.0 / 4.0


# ---------------------------------------------------------------------------
# EXL2: Mixed-Precision Per-Layer Quantization
# ---------------------------------------------------------------------------

class EXL2:
    """EXL2: mixed-precision quantization with per-layer bit allocation.

    Analyzes weight distributions across model layers and assigns per-layer
    bit-widths based on importance (variance) to meet a target average bitrate.
    """

    def __init__(self, model: Optional[Dict[str, np.ndarray]] = None, target_bits: float = 3.5):
        self.target_bits = target_bits
        self._weights: Dict[str, np.ndarray] = {}
        self._layer_bits: Dict[str, int] = {}
        self._quantized_layers: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
        if model is not None:
            self._analyze_model(model)

    def _analyze_model(self, model: Dict[str, np.ndarray]) -> None:
        self._weights = {k: v.astype(np.float32) for k, v in model.items()}

    def _layer_importance(self, weight: np.ndarray) -> float:
        return float(np.var(weight))

    def _allocate_bits(self, layer_stats: List[Tuple[str, np.ndarray, float]]) -> Dict[str, int]:
        n_layers = len(layer_stats)
        total_params = sum(w.size for _, w, _ in layer_stats)
        total_budget = self.target_bits * total_params
        importances = np.array([imp for _, _, imp in layer_stats])
        max_imp = np.max(importances) + 1e-8
        norm_imp = importances / max_imp
        allocations = {}
        bits_used = 0
        sorted_layers = sorted(enumerate(layer_stats), key=lambda x: -x[1][2])
        for idx, (name, weight, imp) in sorted_layers:
            remaining_layers = n_layers - len(allocations)
            remaining_budget = total_budget - bits_used
            if remaining_layers > 0:
                bits_per_param = remaining_budget / remaining_layers / weight.size
            else:
                bits_per_param = self.target_bits
            base_bits = int(round(np.clip(bits_per_param, 2, 8)))
            imp_boost = int(round(norm_imp[idx] * 2))
            layer_bits = int(np.clip(base_bits + imp_boost, 2, 8))
            allocations[name] = layer_bits
            bits_used += layer_bits * weight.size
        return allocations

    def quantize(
        self,
        weights: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        if weights is not None:
            self._analyze_model(weights)
        layer_stats = []
        for name, weight in self._weights.items():
            importance = self._layer_importance(weight)
            layer_stats.append((name, weight, importance))
        self._layer_bits = self._allocate_bits(layer_stats)
        results = {}
        for name, weight, _ in layer_stats:
            bits = self._layer_bits[name]
            quantized, meta = quantize_tensor(weight, bits, group_size=128)
            layer_meta = {
                "method": f"EXL2-{bits}bit",
                "bits": bits,
                "group_size": 128,
                "original_shape": weight.shape,
                "quantized_shape": quantized.shape,
                "quantizer_metadata": meta,
            }
            results[name] = (quantized, layer_meta)
        self._quantized_layers = results
        return results

    def dequantize(self) -> Dict[str, np.ndarray]:
        results = {}
        for name, (quantized, meta) in self._quantized_layers.items():
            qmeta = meta["quantizer_metadata"]
            results[name] = dequantize_tensor(quantized, qmeta)
        return results

    def get_compression_ratio(self) -> float:
        if not self._layer_bits:
            return 32.0 / self.target_bits
        total_params = 0
        total_bits = 0
        for name, bits in self._layer_bits.items():
            w = self._weights[name]
            total_params += w.size
            total_bits += bits * w.size
        return 32.0 / (total_bits / total_params)


# ---------------------------------------------------------------------------
# OmniQuant: Learnable Equivalent Transformation
# ---------------------------------------------------------------------------

class OmniQuant:
    """OmniQuant: learnable equivalent transformation for quantization.

    Learns per-channel scales and zero-points via gradient-free optimization
    (Hill-climbing search) to minimize quantization error while keeping the
    weight distribution well-suited for low-bit integer representation.
    """

    def __init__(self, bits: int = 8, group_size: int = 128, n_search: int = 20):
        self.bits = bits
        self.group_size = group_size
        self.n_search = n_search

    def quantize(self, weight: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantize weight using learnable equivalent transformation.

        Args:
            weight: Weight matrix of any shape.

        Returns:
            Tuple of (quantized, metadata).
        """
        weight = weight.astype(np.float32)
        original_shape = weight.shape
        flat = weight.reshape(-1)
        n = flat.size

        n_groups = max(1, math.ceil(n / self.group_size))
        padded_n = n_groups * self.group_size
        padded = np.zeros(padded_n, dtype=np.float32)
        padded[:n] = flat
        groups = padded.reshape(n_groups, self.group_size)

        qmax = (1 << (self.bits - 1)) - 1

        best_scale = np.ones(n_groups, dtype=np.float32)
        best_zp = np.zeros(n_groups, dtype=np.float32)

        best_err = float("inf")
        rng = np.random.RandomState(0)

        for _ in range(self.n_search):
            cand_scale = best_scale * (0.8 + 0.4 * rng.rand(n_groups))
            cand_zp = best_zp + rng.randn(n_groups).astype(np.float32) * 0.01

            cand_scale = np.maximum(cand_scale, 1e-8)
            scaled = groups / cand_scale.reshape(-1, 1) + cand_zp.reshape(-1, 1)
            q = np.round(scaled).astype(np.int32)
            q = np.clip(q, -qmax - 1, qmax)
            recon = (q - cand_zp.reshape(-1, 1)) * cand_scale.reshape(-1, 1)
            err = float(np.mean((groups - recon) ** 2))

            if err < best_err:
                best_err = err
                best_scale = cand_scale
                best_zp = cand_zp

        scaled = groups / best_scale.reshape(-1, 1) + best_zp.reshape(-1, 1)
        q = np.round(scaled).astype(np.int32)
        q = np.clip(q, -qmax - 1, qmax)
        q_flat = q.reshape(-1)[:n]

        self._metadata = {
            "original_shape": original_shape,
            "bits": self.bits,
            "group_size": self.group_size,
            "scale": best_scale,
            "zero_point": best_zp,
            "n_groups": n_groups,
            "method": "OmniQuant",
        }
        return q_flat.astype(np.float32)

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        metadata = self._metadata
        scale = metadata["scale"]
        zp = metadata["zero_point"]
        n_groups = metadata["n_groups"]
        original_shape = metadata["original_shape"]
        n = quantized.size
        padded_n = n_groups * metadata["group_size"]
        q_padded = np.zeros(padded_n, dtype=np.float32)
        q_padded[:n] = quantized.astype(np.float32)
        groups = q_padded.reshape(n_groups, metadata["group_size"])
        recon = (groups - zp.reshape(-1, 1)) * scale.reshape(-1, 1)
        return recon.reshape(-1)[:n].reshape(original_shape)


# ---------------------------------------------------------------------------
# SpQR: Sparse-Quantized Representation
# ---------------------------------------------------------------------------

class SpQR:
    """SpQR: sparse-quantized representation.

    Identifies outlier weights by magnitude and keeps them in FP16,
    while quantizing remaining weights to low-bit integers. This
    preserves accuracy on critical weights while compressing the rest.
    """

    def __init__(self, bits: int = 4, outlier_threshold: float = 0.01):
        self.bits = bits
        self.threshold = outlier_threshold

    def quantize(self, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize weight with outlier preservation in FP16.

        Args:
            weight: Weight matrix of any shape.

        Returns:
            Tuple of (quantized_weight, outlier_indices) where
            quantized_weight has FP16 outliers mixed in, and
            outlier_indices has shape (n_outliers,) with flat indices.
        """
        weight = weight.astype(np.float32)
        flat = weight.reshape(-1)

        abs_weights = np.abs(flat)
        threshold_val = self.threshold * np.max(abs_weights)
        if threshold_val < 1e-8:
            threshold_val = self.threshold * np.mean(abs_weights) + 1e-8

        outlier_mask = abs_weights > threshold_val
        outlier_indices = np.where(outlier_mask)[0]

        non_outlier_flat = flat.copy()
        non_outlier_flat[outlier_mask] = 0.0

        qmax = (1 << (self.bits - 1)) - 1
        abs_max = np.max(np.abs(non_outlier_flat))
        abs_max = max(abs_max, 1e-8)
        scale = abs_max / qmax
        q = np.round(non_outlier_flat / scale).astype(np.int32)
        q = np.clip(q, -qmax - 1, qmax)

        result = q.astype(np.float32) * scale
        result[outlier_mask] = flat[outlier_mask]
        result = result.astype(np.float16)

        metadata = {
            "original_shape": weight.shape,
            "bits": self.bits,
            "threshold": self.threshold,
            "scale": scale,
            "outlier_count": int(np.sum(outlier_mask)),
            "outlier_indices": outlier_indices,
            "method": "SpQR",
        }

        return result, outlier_indices

    def dequantize(self, quantized: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        return quantized.astype(np.float32).reshape(metadata["original_shape"])


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "DynamicINT8",
    "StaticINT8",
    "INT4GroupQuantization",
    "INT2Quantization",
    "GPTQ",
    "AWQ",
    "SmoothQuant",
    "LLM_int8",
    "KVCacheQuantization",
    "QuaRot",
    "EXL2",
    "OmniQuant",
    "SpQR",
    "quantize_tensor",
    "dequantize_tensor",
    "measure_quantization_error",
    "auto_quantize",
]
