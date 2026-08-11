# Auto Maple USB HID Keyboard Bridge

This backend replaces Windows synthetic keyboard injection with a small USB HID
microcontroller. Auto Maple sends simple serial commands; the board presents
itself to Windows as a normal USB keyboard.

## Supported boards

Use a board with native USB HID support, such as:

- Arduino Leonardo (ATmega32U4)
- Arduino Micro (ATmega32U4)
- Pro Micro / compatible ATmega32U4 board

A classic Uno/Nano with ATmega328P is not suitable for this sketch because it
cannot normally present itself as a USB keyboard directly.

## Flash the firmware

1. Install Arduino IDE.
2. Connect the HID board by USB.
3. Open `hid_bridge/AutoMapleHID/AutoMapleHID.ino`.
4. Select the correct board and COM port.
5. Upload the sketch.
6. Leave the board connected.

The sketch uses both USB HID Keyboard and the serial COM interface at 115200 baud.

## Update Auto Maple

From the project directory:

```bat
git pull
setup_wgc.bat
```

`setup_wgc.bat` installs `pyserial`, which Auto Maple uses to find the board.

## First test

Before running patrol, test only Shift:

```bat
test_hid_shift.bat
```

The test finds the HID board, waits five seconds, then sends one Left Shift press.
It never changes foreground windows, so manually click MapleStory during the
countdown.

Expected output includes:

```text
[~] HID connected on COMx
[~] In 5 seconds the HID device will press LEFT SHIFT once.
[~] Shift command sent.
```

If multiple serial devices are connected and auto-detection chooses the wrong
one, set the port before launching:

```bat
set AUTO_MAPLE_HID_PORT=COM6
```

Replace `COM6` with the board's actual port shown in Windows Device Manager.

## Patrol

Once `test_hid_shift.bat` successfully triggers the in-game Shift action:

```bat
run_auto_maple_wgc.bat
```

Auto Maple will print the detected HID COM port. Click MapleStory and press
Insert. The patrol controller then uses the HID backend for Left/Right/Up/Down,
Space, Shift and Z.

For safety, the software still checks that MapleStory is the active window before
issuing patrol commands. It never calls SetForegroundWindow or forces the game to
the front. F12 remains the emergency stop.
