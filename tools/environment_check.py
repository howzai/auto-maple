"""Preflight checks for the Auto Maple development environment.

Run this before starting the application:

    python tools/environment_check.py

The script only inspects the local environment. It does not start capture, register
hotkeys, or send keyboard input.
"""

from __future__ import annotations

import importlib
import platform
import struct
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ASSETS = (
    "assets/icon.png",
    "assets/icon.ico",
    "assets/minimap_tl_template.png",
    "assets/minimap_br_template.png",
    "assets/player_template.png",
)
REQUIRED_MODULES = (
    ("cv2", "opencv-python-headless"),
    ("git", "GitPython"),
    ("keyboard", "keyboard"),
    ("mss", "mss"),
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
    ("pygame", "pygame"),
    ("win32api", "pywin32"),
)
OPTIONAL_MODULES = (("tensorflow", "tensorflow (Rune model support)"),)


def _result(ok: bool, label: str, detail: str = "") -> bool:
    marker = "OK" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{marker:4}] {label}{suffix}")
    return ok


def _module_version(module) -> str:
    return str(getattr(module, "__version__", "version unknown"))


def main() -> int:
    print("Auto Maple environment check")
    print("=" * 48)

    checks = []
    checks.append(_result(platform.system() == "Windows", "Operating system", platform.platform()))
    checks.append(
        _result(
            sys.version_info[:2] == (3, 10),
            "Python version",
            f"{platform.python_version()} (recommended: 3.10.x)",
        )
    )
    checks.append(
        _result(
            struct.calcsize("P") * 8 == 64,
            "Python architecture",
            f"{struct.calcsize('P') * 8}-bit",
        )
    )

    print("\nPython packages")
    print("-" * 48)
    for import_name, package_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(import_name)
        except Exception as exc:
            checks.append(_result(False, package_name, str(exc)))
        else:
            checks.append(_result(True, package_name, _module_version(module)))

    for import_name, package_name in OPTIONAL_MODULES:
        try:
            module = importlib.import_module(import_name)
        except Exception as exc:
            _result(False, package_name, f"optional; unavailable: {exc}")
        else:
            _result(True, package_name, _module_version(module))

    print("\nRequired assets")
    print("-" * 48)
    for relative_path in REQUIRED_ASSETS:
        path = PROJECT_ROOT / relative_path
        checks.append(_result(path.is_file(), relative_path, str(path)))

    writable_targets = (PROJECT_ROOT / "logs", PROJECT_ROOT / "settings")
    print("\nWritable directories")
    print("-" * 48)
    for directory in writable_targets:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError as exc:
            checks.append(_result(False, str(directory), str(exc)))
        else:
            checks.append(_result(True, str(directory)))

    print("\n" + "=" * 48)
    if all(checks):
        print("Environment check passed. You can run: python main.py")
        return 0

    print("Environment check failed. Fix the FAIL items before starting.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
