"""GPU (CuPy) utilities for accelerated array operations.

Falls back gracefully to NumPy when CUDA is unavailable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

# Attempt CuPy import; silently fall back to NumPy on failure or when disabled.
_GPU_ENABLED: bool = False
xp: "type | None" = None  # array module (numpy or cupy)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOCALAPPDATA = os.environ.get("LOCALAPPDATA")
_CUPY_ROOT = (
    Path(_LOCALAPPDATA) / "Temp" / "qkd_optical_network_cupy"
    if _LOCALAPPDATA
    else _PROJECT_ROOT / ".cupy_cache"
)
_CUPY_KERNEL_CACHE = _CUPY_ROOT / "kernel"
_CUPY_TEMP = _CUPY_ROOT / "tmp"
try:
    _CUPY_KERNEL_CACHE.mkdir(parents=True, exist_ok=True)
    _CUPY_TEMP.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUPY_CACHE_DIR", str(_CUPY_KERNEL_CACHE))
    os.environ.setdefault("TEMP", str(_CUPY_TEMP))
    os.environ.setdefault("TMP", str(_CUPY_TEMP))
    os.environ.setdefault("TMPDIR", str(_CUPY_TEMP))
    tempfile.tempdir = str(_CUPY_TEMP)
except Exception:
    pass


class _ReusableTemporaryDirectory:
    """Fallback for restricted Windows ACLs that block temp subdirectory writes."""

    def __init__(self, *args, **kwargs) -> None:
        _ = args, kwargs
        _CUPY_TEMP.mkdir(parents=True, exist_ok=True)
        self.name = str(_CUPY_TEMP)

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def cleanup(self) -> None:
        return None


def _temp_subdir_is_writable() -> bool:
    try:
        with tempfile.TemporaryDirectory(dir=str(_CUPY_TEMP)) as tmp_dir:
            probe = Path(tmp_dir) / "probe.txt"
            probe.write_text("ok", encoding="utf-8")
            return probe.read_text(encoding="utf-8") == "ok"
    except Exception:
        return False


if not _temp_subdir_is_writable():
    tempfile.TemporaryDirectory = _ReusableTemporaryDirectory

if TYPE_CHECKING:
    import cupy as cp
    from cupy import ndarray

try:
    _env_disabled = os.environ.get("CUDA_ENABLED", "").lower()
    if _env_disabled not in ("0", "false", "no"):
        import cupy as cp

        # Verify CUDA runtime and kernel compilation are actually functional
        # (avoids reporting GPU when CuPy can import but cannot launch kernels).
        try:
            cp.cuda.runtime.getDeviceCount()
            _probe = cp.arange(1, dtype=cp.float32)
            cp.cuda.Stream.null.synchronize()
            _probe.get()
            _GPU_ENABLED = True
            xp = cp
        except Exception:
            # No GPU visible or CUDA runtime broken — fall back to NumPy.
            pass
except Exception:
    # CuPy not installed or CUDA not available.
    pass

import numpy as np

# Public API.
GPU_ENABLED: bool = _GPU_ENABLED


def get_array_module():
    """Return the active array module (CuPy if GPU available, else NumPy).

    Returns
    -------
    module
        ``cupy`` or ``numpy``.
    """
    return xp if xp is not None else np


def to_device(arr: np.ndarray) -> "ndarray | np.ndarray":
    """Copy a NumPy array to GPU memory (no-op if GPU disabled).

    Parameters
    ----------
    arr : np.ndarray
        Source array on CPU.

    Returns
    -------
    ndarray or np.ndarray
        GPU array if GPU_ENABLED, otherwise the original NumPy array.
    """
    if _GPU_ENABLED:
        return xp.asarray(arr)
    return arr


def to_host(arr: "ndarray | np.ndarray") -> np.ndarray:
    """Copy an array from GPU to CPU (no-op if GPU disabled or already on CPU).

    Parameters
    ----------
    arr : ndarray or np.ndarray
        Source array (GPU or CPU).

    Returns
    -------
    np.ndarray
        NumPy array on CPU.
    """
    if _GPU_ENABLED and hasattr(arr, "get"):
        # CuPy ndarray — .get() copies back to CPU.
        return arr.get()
    return np.asarray(arr)


def has_cupy() -> bool:
    """Return True if CuPy with a functional GPU is available."""
    return _GPU_ENABLED


def get_gpu_module() -> "tuple[type, bool]":
    """Return ``(xp, is_gpu)`` where *xp* is ``cupy`` or ``numpy``.

    Caches the result so repeated calls are cheap.  Use this instead of
    ``get_array_module()`` when you also need the boolean ``is_gpu`` flag
    (e.g. to decide between a CPU and a GPU code path).
    """
    return (xp, True) if _GPU_ENABLED else (np, False)
