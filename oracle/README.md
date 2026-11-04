# Oracle implementation

Python oracle for Nous: a Hugging Face baseline and a NumPy implementation of the LLM forward pass. It is the oracle used to validate the C++ engine, and it produces the traces that viewers consume.

> Status: Phase 0 (Foundations): See the ALICE project board for the current sprint.

## Requirements

- macOS or Linux
- Python 3.12 (recommended)

## Setup

```bash
cd oracle
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip pip-tools
pip-sync requirements.txt requirements-dev.txt
```

It prints the versions of the main dependencies and the available device (`cuda`, `mps` or `cpu`).

## Managing dependencies

Edit the `.in` files, never the `.txt` files: the latter are generated.

```bash
pip-compile requirements.in -o requirements.txt
pip-compile requirements-dev.in -o requirements-dev.txt
pip-sync requirements.txt requirements-dev.txt
```

## Layout

```
oracle/
    ├── check_env.py
    ├── requirements.in
    ├── requirements-dev.in
    ├── requirements.txt        # generated
    └── requirements-dev.txt    # generated
```