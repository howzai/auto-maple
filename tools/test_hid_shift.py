import time

from src.modules import game_input


def main():
    print("Auto Maple USB HID Shift test")
    print("================================")
    print("[~] Looking for HID keyboard bridge...")
    if not game_input.initialize(force=True):
        print(f"[!] HID not ready: {game_input.last_error()}")
        print("[~] Flash hid_bridge/AutoMapleHID/AutoMapleHID.ino first, then reconnect the board.")
        return 1

    print(f"[~] HID connected on {game_input.port_name()}")
    print("[~] In 5 seconds the HID device will press LEFT SHIFT once.")
    print("[~] Click MapleStory now. Auto Maple will NOT change window focus.")
    for remaining in range(5, 0, -1):
        print(f"    {remaining}...")
        time.sleep(1)

    ok = game_input.press("shift", 1, down_time=0.12, up_time=0.04)
    print("[~] Shift command sent." if ok else f"[!] Shift command failed: {game_input.last_error()}")
    time.sleep(0.2)
    game_input.close()
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
