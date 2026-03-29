"""
Adaptive communication protocol for FedSynth-Engine.

Implements quantization, delta encoding, and sparse transmission to reduce
bandwidth consumption during marginal exchange in federated synthesis.
"""

import struct
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import zstandard as zstd


@dataclass
class CompressedPayload:
    """Wire-format payload for a compressed marginal update."""
    marginal_key: str
    is_delta: bool
    num_entries: int
    min_val: float
    max_val: float
    quant_bits: int
    sparse_indices: np.ndarray
    sparse_values: np.ndarray
    raw_bytes: Optional[bytes] = None

    def size_bytes(self) -> int:
        if self.raw_bytes is not None:
            return len(self.raw_bytes)
        idx_bytes = self.sparse_indices.nbytes
        val_bytes = self.sparse_values.nbytes
        return idx_bytes + val_bytes + 32  # overhead for metadata


@dataclass
class CommunicationStats:
    """Tracks communication statistics for a synthesis round."""
    total_uncompressed_bytes: int = 0
    total_compressed_bytes: int = 0
    num_transmissions: int = 0
    num_skipped: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.total_uncompressed_bytes == 0:
            return 1.0
        return self.total_compressed_bytes / self.total_uncompressed_bytes

    @property
    def savings_pct(self) -> float:
        return (1.0 - self.compression_ratio) * 100.0


class AdaptiveCommunicationProtocol:
    """
    Compresses marginal updates using three complementary strategies:
    1. Quantization: Reduce float32 to b-bit representation
    2. Delta encoding: Transmit only changes from previous round
    3. Sparse transmission: Omit zero-delta entries
    """

    def __init__(
        self,
        quant_bits: int = 8,
        delta_threshold: float = 1e-4,
        enable_delta: bool = True,
        enable_quantization: bool = True,
        enable_sparse: bool = True,
        compression_level: int = 3,
    ):
        self.quant_bits = quant_bits
        self.delta_threshold = delta_threshold
        self.enable_delta = enable_delta
        self.enable_quantization = enable_quantization
        self.enable_sparse = enable_sparse
        self.compression_level = compression_level

        self._prev_marginals: Dict[str, Dict[int, np.ndarray]] = {}
        self._compressor = zstd.ZstdCompressor(level=compression_level)
        self._decompressor = zstd.ZstdDecompressor()
        self._stats = CommunicationStats()

    def reset_stats(self):
        self._stats = CommunicationStats()

    @property
    def stats(self) -> CommunicationStats:
        return self._stats

    def compress_marginal(
        self,
        party_id: int,
        marginal_key: str,
        marginal: np.ndarray,
    ) -> Optional[CompressedPayload]:
        """Compress a marginal for transmission. Returns None if skipped."""
        uncompressed_size = marginal.nbytes
        self._stats.total_uncompressed_bytes += uncompressed_size

        data_to_send = marginal.copy()
        is_delta = False

        if self.enable_delta:
            prev_key = f"{party_id}_{marginal_key}"
            if prev_key in self._prev_marginals:
                prev = self._prev_marginals.get(prev_key, {}).get(party_id)
                if prev is not None and prev.shape == marginal.shape:
                    delta = marginal - prev
                    if np.max(np.abs(delta)) < self.delta_threshold:
                        self._stats.num_skipped += 1
                        return None
                    data_to_send = delta
                    is_delta = True

            if marginal_key not in self._prev_marginals:
                self._prev_marginals[marginal_key] = {}
            self._prev_marginals[marginal_key][party_id] = marginal.copy()

        sparse_indices, sparse_values = self._sparsify(data_to_send)

        if self.enable_quantization:
            sparse_values = self._quantize(sparse_values)

        payload = CompressedPayload(
            marginal_key=marginal_key,
            is_delta=is_delta,
            num_entries=len(marginal),
            min_val=float(np.min(data_to_send)) if len(data_to_send) > 0 else 0.0,
            max_val=float(np.max(data_to_send)) if len(data_to_send) > 0 else 0.0,
            quant_bits=self.quant_bits,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
        )

        payload.raw_bytes = self._serialize(payload)
        self._stats.total_compressed_bytes += payload.size_bytes()
        self._stats.num_transmissions += 1

        return payload

    def decompress_marginal(
        self,
        payload: CompressedPayload,
        party_id: int,
    ) -> np.ndarray:
        """Decompress a received payload back to a full marginal vector."""
        full = np.zeros(payload.num_entries, dtype=np.float64)

        values = payload.sparse_values
        if self.enable_quantization:
            values = self._dequantize(values, payload.min_val, payload.max_val)

        if len(payload.sparse_indices) > 0:
            full[payload.sparse_indices] = values

        if payload.is_delta:
            prev = self._prev_marginals.get(payload.marginal_key, {}).get(party_id)
            if prev is not None:
                full = prev + full

        return full

    def _sparsify(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract non-zero entries for sparse transmission."""
        if not self.enable_sparse:
            indices = np.arange(len(data), dtype=np.int32)
            return indices, data.astype(np.float64)

        threshold = self.delta_threshold if self.enable_delta else 0.0
        mask = np.abs(data) > threshold
        indices = np.where(mask)[0].astype(np.int32)
        values = data[mask].astype(np.float64)
        return indices, values

    def _quantize(self, values: np.ndarray) -> np.ndarray:
        """Quantize floating-point values to b-bit representation."""
        if len(values) == 0:
            return values

        v_min = np.min(values)
        v_max = np.max(values)
        v_range = v_max - v_min

        if v_range < 1e-12:
            return values

        max_int = (1 << self.quant_bits) - 1
        normalized = (values - v_min) / v_range
        quantized_int = np.round(normalized * max_int).astype(np.int32)
        dequantized = v_min + (quantized_int / max_int) * v_range

        return dequantized

    def _dequantize(
        self, values: np.ndarray, v_min: float, v_max: float
    ) -> np.ndarray:
        """Values are already dequantized during quantization step."""
        return values

    def _serialize(self, payload: CompressedPayload) -> bytes:
        """Serialize payload to bytes with zstd compression."""
        idx_bytes = payload.sparse_indices.tobytes()
        val_bytes = payload.sparse_values.tobytes()

        header = struct.pack(
            "<?iiffii",
            payload.is_delta,
            payload.num_entries,
            payload.quant_bits,
            payload.min_val,
            payload.max_val,
            len(payload.sparse_indices),
            len(payload.sparse_values),
        )

        raw = header + idx_bytes + val_bytes
        compressed = self._compressor.compress(raw)
        return compressed

    def _deserialize(self, data: bytes) -> CompressedPayload:
        """Deserialize bytes back into a CompressedPayload."""
        raw = self._decompressor.decompress(data)
        header_size = struct.calcsize("<?iiffii")
        header = struct.unpack("<?iiffii", raw[:header_size])

        is_delta, num_entries, quant_bits, min_val, max_val, n_idx, n_val = header
        offset = header_size

        idx_size = n_idx * 4
        indices = np.frombuffer(raw[offset:offset + idx_size], dtype=np.int32).copy()
        offset += idx_size

        val_size = n_val * 8
        values = np.frombuffer(raw[offset:offset + val_size], dtype=np.float64).copy()

        return CompressedPayload(
            marginal_key="",
            is_delta=is_delta,
            num_entries=num_entries,
            min_val=min_val,
            max_val=max_val,
            quant_bits=quant_bits,
            sparse_indices=indices,
            sparse_values=values,
            raw_bytes=data,
        )


def compute_uncompressed_size(marginals: Dict[str, np.ndarray]) -> int:
    """Compute total size in bytes if marginals were sent uncompressed (float32)."""
    return sum(m.size * 4 for m in marginals.values())
