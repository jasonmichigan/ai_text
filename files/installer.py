"""ai_text_files.installer — dependency installer.

Installs pypdf (read PDFs) and reportlab (write PDFs) alongside the core deps.

Public entry point:
    ensure_dependencies()
"""

from __future__ import annotations

import importlib
import subprocess
import sys


TORCH_CUDA_INDEXES = [
    "https://download.pytorch.org/whl/cu128",
    "https://download.pytorch.org/whl/cu126",
    "https://download.pytorch.org/whl/cu130",
]

PACKAGES = {
    "ollama":         "ollama",
    "python-docx":    "docx",
    "python-pptx":    "pptx",
    "pypdf":          "pypdf",        # read PDFs
    "reportlab":      "reportlab",    # write PDFs
    "matplotlib":     "matplotlib",
    "pandas":         "pandas",
    "numpy":          "numpy",
    "Pillow":         "PIL",
    "requests":       "requests",
    "ipywidgets":     "ipywidgets",
    "openpyxl":       "openpyxl",
    "ipyfilechooser": "ipyfilechooser",
    "anthropic":      "anthropic",
}


def _torch_status():
    try:
        import torch
    except ImportError:
        return "missing"
    return "cuda_ok" if torch.cuda.is_available() else "cpu_only"


def _try_pip_install_torch(index_url):
    cmd = [sys.executable, "-m", "pip", "install",
           "torch", "torchvision", "--index-url", index_url]
    print(f"   Trying: {index_url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    return (result.returncode == 0, output[-600:])


def _install_torch_cpu():
    """Install CPU-only PyTorch from the default PyPI index."""
    status = _torch_status()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    if status in ("cuda_ok", "cpu_only"):
        import torch
        print(f"✅ PyTorch already installed (torch {torch.__version__}, CPU mode)")
        return True

    print(f"📦 Installing CPU-only PyTorch for Python {py_ver}...")
    cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "-q"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ PyTorch (CPU) installed.")
        print()
        print("🔄 IMPORTANT: restart the Jupyter kernel now, then re-run this cell.")
        raise SystemExit("Restart kernel and re-run to continue.")

    print("❌ CPU PyTorch install failed. Last pip output:")
    print("─" * 60)
    print(((result.stdout or "") + (result.stderr or ""))[-600:])
    print("─" * 60)
    return False


def _install_torch():
    status = _torch_status()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    if status == "cuda_ok":
        import torch
        print(f"✅ PyTorch with CUDA already installed "
              f"(torch {torch.__version__}, CUDA {torch.version.cuda})")
        return True

    if status == "cpu_only":
        print("⚠️  CPU-only PyTorch is installed. Reinstalling with CUDA support...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y",
                               "torch", "torchvision"])
    else:
        print(f"📦 Installing PyTorch with CUDA support for Python {py_ver}...")
        print("   (~2.5 GB download, takes 5-10 min)")

    last_output = ""
    for idx_url in TORCH_CUDA_INDEXES:
        ok, last_output = _try_pip_install_torch(idx_url)
        if ok:
            print(f"   ✓ Success with {idx_url}")
            print()
            print("✅ PyTorch installed.")
            print()
            print("🔄 IMPORTANT: restart the Jupyter kernel now, then re-run this cell.")
            print("    (PyTorch was just installed — Python won't see it without restart.)")
            raise SystemExit("Restart kernel and re-run to continue.")
        print("   ✗ Failed — trying next channel...")

    print()
    print("❌ All CUDA channels failed. Last pip output:")
    print("─" * 60)
    print(last_output)
    print("─" * 60)
    print()
    print("    Diagnosis steps:")
    print("    1. Run `nvidia-smi` in a terminal — confirm your driver version.")
    print(f"    2. Python version: this kernel is {py_ver} "
          "(PyTorch supports 3.10–3.14 on Windows/Linux).")
    print("    3. Continuing without torch — text features still work.")
    return False


def _install_packages():
    missing, present = [], []
    for pip_name, import_name in PACKAGES.items():
        try:
            importlib.import_module(import_name)
            present.append(pip_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"📦 Installing {len(missing)} missing package(s): {', '.join(missing)}")
        for pkg in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        print(f"✅ Installed {len(missing)} package(s).")
    else:
        print(f"✅ All {len(present)} dependencies already installed — skipping pip.")


def _gpu_summary(use_gpu=True):
    try:
        import torch
    except ImportError:
        print("ℹ️  Torch not installed — text features still work.")
        return

    if not use_gpu:
        print(f"💻 CPU mode active (torch {torch.__version__}) — GPU packages skipped.")
        return

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎨 CUDA ready: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("ℹ️  Torch installed, CUDA not available. Text features work fine.")


def ensure_dependencies(use_gpu=True):
    """Install everything ai_text needs. Idempotent — safe to re-run.

    Parameters
    ----------
    use_gpu : bool
        If True (default), install PyTorch with CUDA support.
        If False, install CPU-only PyTorch and skip CUDA-related setup.
    """
    if use_gpu:
        _install_torch()
    else:
        _install_torch_cpu()
    _install_packages()
    _gpu_summary(use_gpu=use_gpu)
