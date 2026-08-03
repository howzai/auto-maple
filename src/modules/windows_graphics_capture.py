"""Read BGRA frames published by MapleCaptureHost through shared memory.

This backend never falls back to desktop capture. If the host stops publishing,
returns an invalid header, or the frame becomes stale, callers receive ``None`` so
Auto Maple can pause safely instead of processing Chrome/CMD pixels as game data.
"""

from __future__ import annotations

import mmap
import os
import struct
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


MAPPING_NAME = r"Local\AutoMaple.GraphicsCapture.Frame"
HEADER_SIZE = 64
MAGIC = 0x50414D41
VERSION = 1
HEADER = struct.Struct("<IIqiiiiqii")
MAX_FRAME_BYTES = 2560 * 1440 * 4
STALE_AFTER_SECONDS = 1.0
WINDOWS_EPOCH_OFFSET_SECONDS = 11644473600.0
INVALID_HEADER_RECONNECTS = 8


@dataclass(frozen=True)
class SharedFrameInfo:
    sequence: int
    width: int
    height: int
    stride: int
    payload_size: int
    timestamp_ticks: int
    producer_pid: int
    header_size: int


class WindowsGraphicsCaptureReader:
    """Lock-free latest-frame reader for the C# capture host."""

    def __init__(self, mapping_name: str = MAPPING_NAME):
        if os.name != "nt":
            raise OSError("Windows Graphics Capture is only available on Windows")
        self.mapping_name = mapping_name
        self._mapping: Optional[mmap.mmap] = None
        self.last_sequence = 0
        self.last_error: Optional[str] = None
        self.last_frame_time = 0.0
        self._invalid_header_reads = 0

    @property
    def connected(self) -> bool:
        return self._mapping is not None

    def connect(self) -> bool:
        if self._mapping is not None:
            return True
        try:
            self._mapping = mmap.mmap(
                -1,
                HEADER_SIZE + MAX_FRAME_BYTES,
                tagname=self.mapping_name,
                access=mmap.ACCESS_READ,
            )
            self.last_error = None
            return True
        except (OSError, ValueError) as exc:
            self.last_error = f"Capture host shared memory unavailable: {exc}"
            self._mapping = None
            return False

    def close(self) -> None:
        mapping, self._mapping = self._mapping, None
        self._invalid_header_reads = 0
        if mapping is not None:
            mapping.close()

    def _mark_invalid_header(self, message: str) -> None:
        """Reconnect if Python attached before the host initialized the mapping."""
        self._invalid_header_reads += 1
        self.last_error = message
        if self._invalid_header_reads >= INVALID_HEADER_RECONNECTS:
            self.close()

    def read_latest(self) -> Optional[np.ndarray]:
        if not self.connect() or self._mapping is None:
            return None

        mapping = self._mapping
        try:
            mapping.seek(0)
            first = mapping.read(HEADER.size)
            info1 = self._parse_header(first)
            if info1 is None:
                self._mark_invalid_header(
                    "Waiting for MapleCaptureHost to initialize shared memory"
                )
                return None
            if info1.sequence <= 0 or info1.sequence % 2:
                self._mark_invalid_header(
                    "Waiting for MapleCaptureHost to publish a complete frame"
                )
                return None

            self._invalid_header_reads = 0
            if info1.sequence == self.last_sequence:
                return None
            if self._is_stale(info1.timestamp_ticks):
                self.last_error = "Windows Graphics Capture frame is stale"
                self.close()
                return None

            mapping.seek(info1.header_size)
            payload = mapping.read(info1.payload_size)

            mapping.seek(0)
            second = mapping.read(HEADER.size)
            info2 = self._parse_header(second)
            if info2 is None or info1.sequence != info2.sequence or info2.sequence % 2:
                return None
            if len(payload) != info2.payload_size:
                self.last_error = "Shared frame payload was truncated"
                return None

            raw = np.frombuffer(payload, dtype=np.uint8)
            rows = raw.reshape((info2.height, info2.stride))
            packed = rows[:, : info2.width * 4]
            frame = packed.reshape((info2.height, info2.width, 4)).copy()

            self.last_sequence = info2.sequence
            self.last_frame_time = time.monotonic()
            self.last_error = None
            return frame
        except (BufferError, OSError, ValueError, struct.error) as exc:
            self.last_error = f"Shared frame read failed: {exc}"
            self.close()
            return None

    @staticmethod
    def _parse_header(data: bytes) -> Optional[SharedFrameInfo]:
        if len(data) < HEADER.size:
            return None
        (
            magic,
            version,
            sequence,
            width,
            height,
            stride,
            payload_size,
            timestamp_ticks,
            producer_pid,
            header_size,
        ) = HEADER.unpack_from(data)

        valid = (
            magic == MAGIC
            and version == VERSION
            and header_size == HEADER_SIZE
            and 0 < width <= 2560
            and 0 < height <= 1440
            and width * 4 <= stride <= 2560 * 4
            and payload_size == stride * height
            and payload_size <= MAX_FRAME_BYTES
            and producer_pid > 0
        )
        if not valid:
            return None
        return SharedFrameInfo(
            sequence=sequence,
            width=width,
            height=height,
            stride=stride,
            payload_size=payload_size,
            timestamp_ticks=timestamp_ticks,
            producer_pid=producer_pid,
            header_size=header_size,
        )

    @staticmethod
    def _is_stale(timestamp_ticks: int) -> bool:
        if timestamp_ticks <= 0:
            return True
        unix_seconds = timestamp_ticks / 10_000_000.0 - WINDOWS_EPOCH_OFFSET_SECONDS
        return time.time() - unix_seconds > STALE_AFTER_SECONDS

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
