# Windows setup

## Recommended environment

- Windows 10 or Windows 11
- 64-bit Python 3.13
- A local checkout in a normal writable folder

## Install

Open Command Prompt in the project folder:

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python tools\environment_check.py
```

All required core checks should pass before starting the application.

## Optional Rune model dependency

The core capture and GUI features do not require TensorFlow. Install the optional
machine-learning dependency only after the core environment works:

```bat
python -m pip install -r requirements-ml.txt
python tools\environment_check.py
```

The Python 3.13 profile uses TensorFlow 2.21 on Windows x86-64. The legacy Rune
model still requires live loading and prediction tests; TensorFlow installation alone
does not guarantee that the old model is compatible.

## Start

```bat
python main.py
```

## Desktop shortcut

```bat
python setup.py --stay
```

The `--stay` option leaves Command Prompt open after exit so errors remain visible.

## Diagnostic information

When reporting a problem, include:

- Windows version
- output from `python --version`
- output from `python tools\environment_check.py`
- relevant lines from `logs/auto-maple.log`
- exact reproduction steps
