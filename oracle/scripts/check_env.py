# ==========================================================================================
# Author: Pablo González García
# Created: 2026/11/06
# ==========================================================================================


# ==============================
# IMPORTS
# ==============================

# Default:
from __future__ import annotations

# Standard:
import sys
import shutil
import platform
from pathlib import Path

# External:
import torch
import transformers




# ==============================
# CONSTANTS
# ==============================

# Path to the file .python-version file at the root of the oracle folder.
PYTHON_VERSION_FILE: Path = Path(__file__).resolve().parents[1] / ".python-version"




# ==============================
# FUNCTIONS
# ==============================

def read_required_python() -> tuple[int, ...]:
    """
    Read the required Python version from the `.python-version` file.

    Returns:
        tuple[int, ...]: The required version as a tuple of integers.
            (e.g. `(3, 12)`).

    Raises:
        ValueError: If the file does not contain a version.
    """

    for line in PYTHON_VERSION_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if line and not line.startswith("#"):
            return tuple(int(part) for part in line.split("."))

    raise ValueError(f"No Python version found in {PYTHON_VERSION_FILE}")



def check_python_version() -> None:
    """
    Exit with an error if the running Python does not match `.python-version`.
    """

    required: tuple[int, ...] = read_required_python()

    if sys.version_info[:len(required)] != required:
        sys.exit(
            f"Python {'.'.join(map(str, required))} is required, "
            f"found {platform.python_version()}."
        )



def detect_device() -> str:
    """
    Detect he best available computation device.

    Determines the device to use for PyTorch computations by checking
    available hardware accelerators in the following priority order:
    CUDA, Apple Metal Performance Shaders (MPS), and CPU.

    Returns:
        str: The selected computation device:
            - `"cuda"`: NVIDIA GPU with CUDA support.
            - `"mps"`: Apple GPU with MPS support.
            - `"cpu"`: CPU (fallback when no supported accelerator
                is available).
    """

    if torch.cuda.is_available():
        return "cuda"

    if torch.mps.is_available():
        return "mps"

    return "cpu"




# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    # Check if the version is correct.
    check_python_version()

    # Calculate the size of information separators.
    columns: int = shutil.get_terminal_size().columns // 2

    # Print platform information.
    print(f"{'='*columns}")
    print(f"{'Platform':<15}: {sys.version.split()[0]}")
    print(f"{'Platform':<15}: {platform.platform()}")
    print(f"{'PyTorch':<15}: {torch.__version__}")
    print(f"{'Transformers':<15}: {transformers.__version__}")
    print(f"{'Device':<15}: {detect_device()}")
    print(f"{'='*columns}")
