"""ai_text_files.model_installer — pull missing Ollama models automatically.

Wraps `ollama pull` as a subprocess so the notebook can fetch models
without you typing in a terminal. Two entry points:

    pull_missing_for_tier(tier, on_line=None) → dict
        Pulls every model in MODEL_TIERS[tier] that isn't installed.

    auto_pull_on_startup(on_line=None) → dict
        Reads config.AUTO_PULL_MISSING_MODELS / config.AUTO_PULL_TIERS,
        decides what to pull, calls pull_missing_for_tier for each.
        This is what runs from the notebook's startup cell.

Both stream `ollama pull` stdout line-by-line via on_line callback so
you see progress in the aiconsole. Failures are non-fatal — a missing
model just stays missing and the UI reflects that.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Callable, Iterable, Optional

from . import config, logger, tier_manager


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _ollama_available() -> bool:
    """True if the `ollama` CLI is on PATH AND the service is reachable."""
    if not shutil.which("ollama"):
        return False
    try:
        # `ollama list` is the cheapest call that touches the service.
        r = subprocess.run(["ollama", "list"],
                            capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def refresh_ollama_state():
    """Re-probe Ollama and update config._INSTALLED_OLLAMA / _BEST_CONTENT_MODEL.
    Other modules (intent_router.pick_model, tier_manager.verify_tier_models)
    read from these globals, so every push to config refreshes the UI's
    model awareness on the next call."""
    installed = config._list_installed_ollama()
    config._INSTALLED_OLLAMA = installed
    # Recompute best content model in case we just pulled a better one
    config._BEST_CONTENT_MODEL = config._best_local_content_model("qwen3:8b")
    return installed


# ─────────────────────────────────────────────────────────────────────
# Single-model pull
# ─────────────────────────────────────────────────────────────────────
def _pull_one(model: str, on_line: Optional[Callable[[str], None]] = None) -> dict:
    """Run `ollama pull <model>` and stream stdout. Returns dict with ok flag."""
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            ["ollama", "pull", model],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
    except FileNotFoundError:
        msg = f"`ollama` not found on PATH — cannot pull {model}"
        if on_line: on_line(f"❌ {msg}")
        logger.log_event("model_pull_error", model=model, reason="ollama_not_found")
        return {"model": model, "ok": False, "reason": "ollama_not_found",
                "ms": (time.time() - t0) * 1000}

    last_line = ""
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            last_line = line
            if on_line:
                on_line(line)
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        if on_line: on_line(f"⚠️  Pull of {model} interrupted")
        return {"model": model, "ok": False, "reason": "interrupted",
                "ms": (time.time() - t0) * 1000}

    ok = (proc.returncode == 0)
    elapsed = time.time() - t0
    if ok:
        if on_line: on_line(f"✅ Pulled {model} in {elapsed:.1f}s")
        logger.log_event("model_pull_done", model=model, ok=True,
                         seconds=f"{elapsed:.1f}")
    else:
        if on_line: on_line(f"❌ Failed to pull {model} (exit {proc.returncode}): {last_line}")
        logger.log_event("model_pull_error", model=model,
                         exit_code=proc.returncode, last_line=last_line)
    return {"model": model, "ok": ok, "exit_code": proc.returncode,
            "last_line": last_line, "ms": elapsed * 1000}


# ─────────────────────────────────────────────────────────────────────
# Public: pull missing models for a tier
# ─────────────────────────────────────────────────────────────────────
def pull_missing_for_tier(tier: str,
                           on_line: Optional[Callable[[str], None]] = None
                           ) -> dict:
    """Pull every model the given tier needs that isn't already installed.
    Returns {model: result_dict} for each one attempted. Empty dict if
    nothing was missing or Ollama isn't available."""
    if not _ollama_available():
        msg = "Ollama service unreachable — cannot auto-pull"
        if on_line: on_line(f"⚠️  {msg}")
        logger.log_event("auto_pull_skipped", reason="ollama_unavailable", tier=tier)
        return {}

    if tier == "custom":
        # Custom tier's models are user-picked; we don't auto-pull them
        # because they may be intentionally light.
        if on_line: on_line(f"  (skipping 'custom' tier — user-managed)")
        return {}

    # Refresh state first so we know what's truly missing
    refresh_ollama_state()
    avail = tier_manager.verify_tier_models()
    info = avail.get(tier, {})
    missing = info.get("missing", [])

    if not missing:
        if on_line: on_line(f"  '{tier}' tier — already complete, nothing to pull")
        logger.log_event("auto_pull_skipped", reason="already_complete", tier=tier)
        return {}

    if on_line:
        on_line(f"📥 '{tier}' tier — pulling {len(missing)} missing model(s):")
        for m in missing:
            on_line(f"     • {m}")

    logger.log_event("auto_pull_start", tier=tier, missing=missing)
    t0 = time.time()
    results = {}
    for m in missing:
        if on_line: on_line(f"\n── Pulling {m} ──")
        results[m] = _pull_one(m, on_line=on_line)
        # Refresh after each pull so subsequent missingness checks are accurate
        refresh_ollama_state()

    # Final refresh so config / verify_tier_models report the new state
    refresh_ollama_state()
    elapsed = time.time() - t0
    n_ok   = sum(1 for r in results.values() if r["ok"])
    n_fail = sum(1 for r in results.values() if not r["ok"])
    summary = (f"\n📋 Tier '{tier}' pull complete in {elapsed:.1f}s: "
               f"{n_ok} succeeded, {n_fail} failed")
    if on_line: on_line(summary)
    logger.log_event("auto_pull_done", tier=tier, n_ok=n_ok, n_fail=n_fail,
                     seconds=f"{elapsed:.1f}")
    return results


# ─────────────────────────────────────────────────────────────────────
# Public: startup auto-pull (driven by config flags)
# ─────────────────────────────────────────────────────────────────────
def auto_pull_on_startup(on_line: Optional[Callable[[str], None]] = None
                          ) -> dict:
    """Read config flags and pull whatever is missing for the configured tiers.

    Driven by:
        config.AUTO_PULL_MISSING_MODELS  (bool, default True)
            Master switch. Set False to skip entirely.
        config.AUTO_PULL_TIERS  (list[str], default ['active'])
            Which tiers to ensure are complete. 'active' is a synonym for
            config.ACTIVE_TIER. Other valid values: 'fast', 'balanced',
            'best'. Use ['active'] for minimum download (recommended);
            use ['fast', 'balanced', 'best'] to pull everything.

    Returns {tier: {model: result}} for tiers attempted.
    """
    if not getattr(config, "AUTO_PULL_MISSING_MODELS", True):
        if on_line: on_line("ℹ️  Auto-pull disabled (config.AUTO_PULL_MISSING_MODELS=False)")
        logger.log_event("auto_pull_skipped", reason="disabled_in_config")
        return {}

    if not _ollama_available():
        msg = ("⚠️  Ollama service unreachable. Start it (open the Ollama app "
               "or run `ollama serve`), then restart the notebook kernel "
               "to retry auto-pull.")
        if on_line: on_line(msg)
        logger.log_event("auto_pull_skipped", reason="ollama_unavailable")
        return {}

    tiers_cfg = getattr(config, "AUTO_PULL_TIERS", ["active"])
    # Resolve 'active' → real tier name; deduplicate; drop 'custom'
    resolved = []
    for t in tiers_cfg:
        actual = config.ACTIVE_TIER if t == "active" else t
        if actual == "custom":
            continue
        if actual not in resolved and actual in config.MODEL_TIERS:
            resolved.append(actual)

    if not resolved:
        if on_line: on_line("ℹ️  No tiers to auto-pull")
        return {}

    if on_line:
        on_line("─" * 60)
        on_line(f"🔄 Auto-pull on startup — checking {len(resolved)} tier(s): "
                + ", ".join(resolved))
        on_line("─" * 60)

    out = {}
    for tier in resolved:
        out[tier] = pull_missing_for_tier(tier, on_line=on_line)

    # Final state summary — re-verify and report
    if on_line:
        on_line("\n" + "─" * 60)
        on_line("✅ Auto-pull finished. Updated tier availability:")
        avail = tier_manager.verify_tier_models()
        for tier_name in ("fast", "balanced", "best"):
            info = avail.get(tier_name, {})
            if info.get("all_present", False):
                on_line(f"  {tier_name:<10} ✅ all {len(info['needed'])} model(s) present")
            else:
                miss = info.get("missing", [])
                on_line(f"  {tier_name:<10} ⚠️  still missing: {', '.join(miss)}")
    return out
