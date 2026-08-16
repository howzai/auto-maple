"""USB HID keyboard backend for Auto Maple.

The Windows synthetic-input experiments (PostMessage/keybd_event/SendInput) are
not used here. Patrol commands are sent over a serial link to a small USB HID
microcontroller (for example an ATmega32U4 Pro Micro/Leonardo). The device then
emits normal USB keyboard reports.

Safety remains in PatrolController/manual_foreground_input: commands are only
issued while the captured MapleStory window is the active application.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None
    list_ports = None


BAUDRATE = 115200
PROTOCOL = "AUTO_MAPLE_HID_V1"
VALID_KEYS = {"left", "right", "up", "down", "shift", "space", "z", "a"}

_lock = threading.RLock()
_serial = None
_port: Optional[str] = None
_last_error = "HID not initialized"
_last_probe = 0.0
_pressed = set()


def _candidate_ports():
    forced = os.environ.get("AUTO_MAPLE_HID_PORT", "").strip()
    if forced:
        yield forced
        return
    if list_ports is None:
        return
    ports = list(list_ports.comports())
    preferred = []
    other = []
    for item in ports:
        text = f"{item.description} {item.manufacturer or ''} {item.hwid}".casefold()
        if any(token in text for token in ("arduino", "leonardo", "pro micro", "32u4", "sparkfun")):
            preferred.append(item.device)
        else:
            other.append(item.device)
    for device in preferred + other:
        yield device


def _close_locked():
    global _serial, _port
    ser = _serial
    _serial = None
    _port = None
    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass


def _handshake(ser) -> bool:
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception:
        pass
    ser.write(b"PING\n")
    ser.flush()
    deadline = time.monotonic() + 1.2
    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        text = raw.decode("utf-8", errors="ignore").strip()
        if text == f"PONG {PROTOCOL}" or text == f"READY {PROTOCOL}":
            return True
    return False


def initialize(force: bool = False) -> bool:
    global _serial, _port, _last_error, _last_probe
    with _lock:
        if _serial is not None and getattr(_serial, "is_open", False) and not force:
            return True
        now = time.monotonic()
        if not force and now - _last_probe < 1.5:
            return False
        _last_probe = now
        _close_locked()

        if serial is None:
            _last_error = "pyserial is not installed; run setup_wgc.bat again"
            return False

        errors = []
        for port in _candidate_ports() or ():
            try:
                ser = serial.Serial(port, BAUDRATE, timeout=0.15, write_timeout=0.25)
                time.sleep(1.6)
                if _handshake(ser):
                    _serial = ser
                    _port = port
                    _last_error = ""
                    print(f"\n[~] USB HID keyboard connected on {port} ({PROTOCOL})")
                    return True
                ser.close()
                errors.append(f"{port}: handshake failed")
            except Exception as exc:
                errors.append(f"{port}: {type(exc).__name__}: {exc}")

        _last_error = "No Auto Maple HID device found"
        if errors:
            _last_error += " | " + "; ".join(errors[:4])
        return False


def is_ready() -> bool:
    with _lock:
        if _serial is not None and getattr(_serial, "is_open", False):
            return True
    return initialize(force=False)


def port_name() -> str:
    with _lock:
        return _port or ""


def last_error() -> str:
    with _lock:
        return _last_error


def status_text() -> str:
    return f"USB HID ready on {port_name()}" if is_ready() else f"USB HID unavailable: {last_error()}"


def _send(command: str) -> bool:
    global _last_error
    with _lock:
        if _serial is not None and getattr(_serial, "is_open", False):
            try:
                _serial.write((command.strip() + "\n").encode("ascii"))
                _serial.flush()
                return True
            except Exception as exc:
                _last_error = f"HID write failed: {type(exc).__name__}: {exc}"
                _close_locked()
                return False
    if not initialize(force=False):
        return False
    with _lock:
        try:
            _serial.write((command.strip() + "\n").encode("ascii"))
            _serial.flush()
            return True
        except Exception as exc:
            _last_error = f"HID write failed: {type(exc).__name__}: {exc}"
            _close_locked()
            return False


def _normalize(key: str) -> str:
    normalized = str(key).strip().lower()
    if normalized not in VALID_KEYS:
        raise ValueError(f"Unsupported HID key: {key!r}")
    return normalized


def key_down(key: str) -> bool:
    key = _normalize(key)
    if _send(f"KD {key.upper()}"):
        with _lock:
            _pressed.add(key)
        return True
    return False


def key_up(key: str) -> bool:
    key = _normalize(key)
    ok = _send(f"KU {key.upper()}")
    with _lock:
        _pressed.discard(key)
    return ok


def press(key: str, n: int = 1, down_time: float = 0.05, up_time: float = 0.03) -> bool:
    key = _normalize(key)
    if n < 1:
        return False
    sent = False
    hold_ms = max(1, min(2000, int(round(max(0.0, down_time) * 1000))))
    for _ in range(n):
        if not _send(f"PRESS {key.upper()} {hold_ms}"):
            return sent
        sent = True
        time.sleep(max(0.0, down_time) + max(0.0, up_time))
    return sent


def combo(first: str, second: str, first_lead: float = 0.03, hold: float = 0.08) -> bool:
    first = _normalize(first)
    second = _normalize(second)
    if not key_down(first):
        return False
    try:
        time.sleep(max(0.0, first_lead))
        if not key_down(second):
            return False
        try:
            time.sleep(max(0.0, hold))
        finally:
            key_up(second)
    finally:
        key_up(first)
    return True


def release_all() -> None:
    with _lock:
        _pressed.clear()
    _send("RELEASE_ALL")


def close() -> None:
    try:
        release_all()
    finally:
        with _lock:
            _close_locked()
