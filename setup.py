"""Create a desktop shortcut that launches this checkout safely on Windows."""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
from pathlib import Path

import win32com.client as client


MAX_DEPTH = 1
PROJECT_ROOT = Path(__file__).resolve().parent


def quote_cmd(value: str) -> str:
    """Quote one command-line value for cmd.exe."""
    return '"' + value.replace('"', '""') + '"'


def run_as_admin(args) -> None:
    if args.depth >= MAX_DEPTH:
        raise PermissionError("Unable to create or modify the desktop shortcut")

    print("\n[!] Insufficient privileges; requesting administrator access")
    forwarded = [str(Path(__file__).resolve()), "--depth", str(args.depth + 1)]
    if args.stay:
        forwarded.append("--stay")

    parameters = subprocess.list2cmdline(forwarded)
    result = ctypes.windll.shell32.ShellExecuteW(
        None,
        "runas",
        sys.executable,
        parameters,
        str(PROJECT_ROOT),
        1,
    )
    if result <= 32:
        raise OSError(f"Administrator launch failed with code {result}")


def create_desktop_shortcut(args) -> Path:
    """Create a shortcut using stable absolute paths and the active Python."""
    print("\n[~] Creating desktop shortcut for Auto Maple")

    shell = client.Dispatch("WScript.Shell")
    desktop = Path(shell.SpecialFolders("Desktop"))
    shortcut_path = desktop / "Auto Maple.lnk"
    cmd_path = Path(os.environ.get("COMSPEC", Path(os.environ["WINDIR"]) / "System32" / "cmd.exe"))

    flag = "/k" if args.stay else "/c"
    command = (
        f"cd /d {quote_cmd(str(PROJECT_ROOT))} && "
        f"{quote_cmd(sys.executable)} {quote_cmd(str(PROJECT_ROOT / 'main.py'))}"
    )

    shortcut = shell.CreateShortCut(str(shortcut_path))
    shortcut.TargetPath = str(cmd_path)
    shortcut.Arguments = f"{flag} {quote_cmd(command)}"
    shortcut.WorkingDirectory = str(PROJECT_ROOT)
    shortcut.IconLocation = str(PROJECT_ROOT / "assets" / "icon.ico")
    shortcut.Description = "Launch Auto Maple from this checkout"

    try:
        shortcut.save()
    except Exception:
        run_as_admin(args)
        return shortcut_path

    print(f"[~] Shortcut created: {shortcut_path}")
    if args.stay:
        print("[~] Command Prompt will remain open after the application exits")
    return shortcut_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--stay", action="store_true", help="Keep the console open after exit")
    args = parser.parse_args()

    if os.name != "nt":
        print("[!] Shortcut setup is supported only on Windows")
        return 1

    try:
        create_desktop_shortcut(args)
    except Exception as exc:
        print(f"[!] Shortcut setup failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
