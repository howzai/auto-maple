# Windows setup

## Recommended environment

- Windows 10 or Windows 11
- 64-bit Python 3.10
- A local checkout in a normal writable folder

## Install

Open Command Prompt in the project folder:

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tools\environment_check.py
```

All required checks should pass before starting the application.

## Optional model dependency

Install the optional machine-learning dependency only after the core environment works:

```bat
python -m pip install -r requirements-ml.txt
python tools\environment_check.py
```

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
- Python version
- output from `python tools\environment_check.py`
- relevant lines from `logs/auto-maple.log`
- exact reproduction steps
