"""ai_text_files.logger — comprehensive diagnostic logger.

NOTEBOOK_VERSION bumped to ai_text (so log files become
ai_text_log_*.txt). Output-quality and validate-section events
are gone (the corresponding features were removed).

Public API:
    start_session()
    log_write(line, prefix="·")
    log_section(title)
    log_event(name, **kwargs)
    log_exception(where)
    install_log_wrapper(target_module, target_name, ...)
    install_log_handler_wrapper(button_widget, handler_fn, name)
"""

from __future__ import annotations

import datetime as _dt
import importlib
import os
import platform
import sys
import threading
import time
import traceback

from . import config

LOGGER_VERSION   = "5.0"
NOTEBOOK_VERSION = "ai_text"
SECRET_PATTERNS  = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY")

# Every kernel session gets its own fresh log file. Files accumulate in
# LOGS_DIR — open the most recent one to see what's happening in the
# current run; older files remain as historical records.
_new_ts  = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = config.LOGS_DIR / f"{NOTEBOOK_VERSION}_log_{_new_ts}.txt"

_log_lock = threading.Lock()


def _is_secret_name(name):
    up = str(name).upper()
    return any(p in up for p in SECRET_PATTERNS)


def log_write(line, prefix="·"):
    ts = _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    full = f"[{ts}] {prefix} {line}\n"
    try:
        with _log_lock:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(full)
    except Exception:
        pass


def log_section(title):
    bar = "═" * 70
    log_write(bar, prefix=" ")
    log_write(f"  {title}", prefix=" ")
    log_write(bar, prefix=" ")


def log_event(event_name, **kwargs):
    parts = []
    for k, v in kwargs.items():
        if _is_secret_name(k):
            parts.append(f"{k}=<redacted>")
        else:
            sv = str(v)
            if len(sv) > 800:
                sv = sv[:800] + f"...<+{len(sv)-800} chars>"
            sv = sv.replace("\n", "\\n")
            parts.append(f"{k}={sv!r}")
    log_write(f"EVENT {event_name} | " + " · ".join(parts), prefix="◆")


def log_exception(where):
    tb = traceback.format_exc()
    log_write(f"EXCEPTION in {where}:", prefix="❌")
    for line in tb.splitlines():
        log_write(f"    {line}", prefix=" ")


def _safe_call(fn, *args, default="<error>", **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return f"<error: {type(e).__name__}: {e}>"


def _dump_environment():
    log_section("ENVIRONMENT")
    log_write(f"Python      : {sys.version.split()[0]} ({sys.executable})")
    log_write(f"Platform    : {platform.platform()}")
    log_write(f"Machine     : {platform.machine()} | CPU count: {os.cpu_count()}")
    log_write(f"Working dir : {os.getcwd()}")
    log_write(f"Notebook    : {NOTEBOOK_VERSION}.ipynb (logger v{LOGGER_VERSION})")
    log_write(f"Package dir : {config.PACKAGE_DIR}")

    try:
        import shutil
        usage = shutil.disk_usage(str(config.OUTPUT_DIR))
        log_write(f"Disk free   : {usage.free/1e9:.1f} / {usage.total/1e9:.1f} GB on {config.OUTPUT_DIR.anchor}")
    except Exception as e:
        log_write(f"Disk free   : <error: {e}>")

    log_write("")
    log_write("Package versions:")
    for pkg in ("torch", "torchvision", "ollama", "anthropic",
                "ipywidgets", "matplotlib", "numpy", "pandas",
                "docx", "pptx", "PIL", "openpyxl",
                "reportlab", "pypdf"):
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "<no __version__>")
            log_write(f"  {pkg:<14} {ver}")
        except ImportError:
            log_write(f"  {pkg:<14} <NOT INSTALLED>")
        except Exception as e:
            log_write(f"  {pkg:<14} <error: {e}>")


def _dump_cuda():
    log_section("CUDA / GPU (informational)")
    try:
        import torch
        log_write(f"CUDA available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log_write(f"Device count   : {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                log_write(f"  [{i}] {props.name}  |  {props.total_memory/1e9:.1f} GB total")
            log_write(f"CUDA version   : {torch.version.cuda}")
            log_write(f"cuDNN          : {torch.backends.cudnn.version()}")
    except ImportError:
        log_write("torch not installed")
    except Exception:
        log_exception("_dump_cuda")


def _dump_ollama():
    log_section("OLLAMA")
    try:
        import ollama
        listing = ollama.list()
        models = []
        for m in (listing.get("models", []) if isinstance(listing, dict)
                  else getattr(listing, "models", [])):
            name = (m.get("name") if isinstance(m, dict)
                    else getattr(m, "name", None) or getattr(m, "model", None))
            if name:
                models.append(name)
        log_write(f"Service        : reachable, {len(models)} model(s)")
        for m in models:
            log_write(f"  • {m}")
    except Exception as e:
        log_write(f"Service        : UNREACHABLE — {type(e).__name__}: {e}")


def _dump_claude_code():
    log_section("CLAUDE CODE CLI")
    try:
        import shutil as _sh
        import subprocess as _sub
        path = _sh.which("claude")
        log_write(f"`claude` on PATH : {path or '<NOT FOUND>'}")
        if path:
            try:
                r = _sub.run([path, "--version"], capture_output=True, text=True, timeout=10)
                log_write(f"Version output   : {r.stdout.strip() or '<empty>'}")
                if r.stderr.strip():
                    log_write(f"Version stderr   : {r.stderr.strip()}")
            except Exception as e:
                log_write(f"Version check    : <error: {type(e).__name__}: {e}>")
    except Exception:
        log_exception("_dump_claude_code")


def _dump_globals_snapshot():
    log_section("CONFIG SNAPSHOT")
    interesting = (
        "PROVIDER", "CLAUDE_API_KEY", "CLAUDE_MODELS", "CLAUDE_MODEL",
        "CLAUDECODE_MODELS", "CLAUDECODE_MODEL", "CLAUDECODE_BIN",
        "PLAN_WITH_CLAUDECODE", "PLANNER_CLAUDECODE_MODEL",
        "ACTIVE_TIER", "MODEL_TIERS",
        "AUTO_PULL_MISSING_MODELS", "AUTO_PULL_TIERS",
        "PROMPT_LLM", "PROMPT_LLM_CLAUDECODE_MODEL",
        "ENHANCE_NON_IMAGE_PROMPTS",
        "_BEST_CONTENT_MODEL", "_INSTALLED_OLLAMA",
        "OUTPUT_DIR", "LOGS_DIR", "WORK_DIR", "PACKAGE_DIR",
        "SUPPORTED_EXTS",
    )
    for name in interesting:
        if not hasattr(config, name):
            continue
        if _is_secret_name(name):
            v_str = "<redacted>"
        else:
            v = getattr(config, name)
            try:
                if isinstance(v, dict) and len(v) > 8:
                    v_str = f"<dict, {len(v)} keys: {list(v.keys())[:5]}...>"
                elif isinstance(v, (list, tuple)) and len(v) > 12:
                    v_str = f"<{type(v).__name__}, {len(v)} items, first: {list(v)[:4]}>"
                else:
                    v_str = repr(v)
                if len(v_str) > 500:
                    v_str = v_str[:500] + f"...<+{len(v_str)-500} chars>"
            except Exception:
                v_str = "<unprintable>"
        log_write(f"  config.{name} = {v_str}")
    log_write(f"  LOG_FILE = {LOG_FILE}")


def _dump_env_vars():
    log_section("ENVIRONMENT VARIABLES (filtered)")
    interesting = sorted(k for k in os.environ if any(
        s in k.upper() for s in ("PATH", "PYTHON", "CUDA", "HF_", "TRANSFORMERS",
                                  "OLLAMA", "ANTHROPIC", "USER", "HOME",
                                  "SHELL", "LANG")
    ))
    for k in interesting:
        if _is_secret_name(k):
            v = os.environ[k]
            log_write(f"  {k} = <redacted, len={len(v)}>")
        else:
            v = os.environ[k]
            if len(v) > 300:
                v = v[:300] + f"...<+{len(v)-300}>"
            log_write(f"  {k} = {v}")


def _dump_tier_availability():
    log_section("TIER AVAILABILITY")
    installed = set(config._INSTALLED_OLLAMA)
    for tier_name, models in config.MODEL_TIERS.items():
        needed = set(models.values())
        missing = sorted(needed - installed)
        if not missing:
            log_write(f"  {tier_name:<10} ✅ all {len(needed)} model(s) present")
        else:
            log_write(f"  {tier_name:<10} ⚠️  missing: {', '.join(missing)}")


def _write_log_header():
    now_iso = _dt.datetime.now().isoformat()
    banner = [
        "═" * 70,
        f"  {NOTEBOOK_VERSION} — DIAGNOSTIC LOG",
        f"  Created: {now_iso}",
        f"  Logger version: {LOGGER_VERSION}",
        f"  Notes: Output Quality removed (no profiles/validation),",
        f"              PDF read+write added, package renamed to ai_text_files.",
        f"  This file contains:",
        f"    • Notebook state, all globals, environment",
        f"    • Live events: button clicks, intent decisions, model picks,",
        f"      tier changes, build_*_structure timing",
        f"    • Prompts and enhanced prompts (full text)",
        f"  Secrets (KEY/TOKEN/SECRET/PASSWORD vars) are redacted.",
        f"  Tracebacks from any error are recorded.",
        f"  One fresh file is created per kernel session (logs accumulate",
        f"  in LOGS_DIR; the most recent file is always this run).",
        "═" * 70,
    ]
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            for line in banner:
                f.write(line + "\n")
            f.write("\n")
    except Exception as e:
        print(f"⚠️  Logger could not write banner: {e}")


_WRAPPED_REGISTRY = {}


def install_log_wrapper(target_module, target_name, *,
                        capture_args=True, capture_return=True,
                        arg_names=None, return_name="result"):
    key = (target_module.__name__, target_name)
    if not hasattr(target_module, target_name):
        log_write(f"  install_log_wrapper: {key} not defined yet — skipping",
                  prefix="⚠")
        return False
    if key in _WRAPPED_REGISTRY:
        return True
    original = getattr(target_module, target_name)
    if not callable(original):
        return False
    _WRAPPED_REGISTRY[key] = original

    def make_wrapper(orig, name, ans, rn):
        def wrapped(*args, **kwargs):
            t0 = time.time()
            kw = {}
            if capture_args:
                if ans:
                    for i, val in enumerate(args):
                        if i < len(ans):
                            kw[ans[i]] = val
                else:
                    for i, val in enumerate(args):
                        kw[f"arg{i}"] = val
                for k, v in kwargs.items():
                    kw[k] = v
            try:
                result = orig(*args, **kwargs)
                if capture_return:
                    kw[rn] = result
                kw["_ms"] = f"{(time.time()-t0)*1000:.1f}"
                log_event(name, **kw)
                return result
            except Exception:
                kw["_ms"] = f"{(time.time()-t0)*1000:.1f}"
                log_event(name + "!ERROR", **kw)
                log_exception(name)
                raise
        wrapped.__name__ = orig.__name__
        wrapped.__doc__  = orig.__doc__
        wrapped.__wrapped__ = orig
        return wrapped

    setattr(target_module, target_name,
            make_wrapper(original, target_name, arg_names, return_name))
    return True


def install_log_handler_wrapper(button_widget, handler_fn, name):
    def logged(b):
        log_event(f"CLICK_{name}")
        try:
            return handler_fn(b)
        except Exception:
            log_exception(f"button:{name}")
            raise
    return logged


def start_session():
    _write_log_header()
    _dump_environment()
    _dump_cuda()
    _dump_ollama()
    _dump_claude_code()
    _dump_env_vars()
    _dump_globals_snapshot()
    _dump_tier_availability()
    log_write("")
    log_write("Startup capture complete. Live events follow.", prefix=" ")
    log_write("─" * 70, prefix=" ")

    print(f"📋 Diagnostic logger ready (v{LOGGER_VERSION}) for {NOTEBOOK_VERSION}")
    print(f"   Log file: {LOG_FILE}")
    print(f"   (fresh file for this kernel session)")
    print(f"   Open the file at any time to see what's recorded.")
