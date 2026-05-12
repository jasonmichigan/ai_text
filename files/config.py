"""ai_text_files.config — all runtime globals.

Notes:
  • OUTPUT_QUALITY_PROFILES, OUTPUT_QUALITY, get_active_profile,
    VALIDATE_VIA_CLAUDECODE, all EXTENDED_* knobs are GONE. The model
    writes whatever length your prompt asks for.
  • Each tier now has a 'pdf' intent (alongside chat/docx/pptx/python/qa
    /planner) — used by the new PDF-creation pipeline.
  • SUPPORTED_EXTS gains '.pdf'.

Mutate attributes directly on the module:
    from ai_text_files import config
    config.PROVIDER = "claudecode"
    config.ACTIVE_TIER = "best"
"""

from __future__ import annotations

import os
import shutil as _shutil
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────
PACKAGE_DIR = Path(__file__).resolve().parent          # ai_text_files/
WORK_DIR    = PACKAGE_DIR.parent                        # project root
OUTPUT_DIR  = PACKAGE_DIR / "generated_files"
LOGS_DIR    = OUTPUT_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# PROVIDER STATE
# ─────────────────────────────────────────────────────────────────────
PROVIDER = "local"   # 'local' or 'claudecode'

# ─────────────────────────────────────────────────────────────────────
# CLAUDE — direct API path (legacy)
# ─────────────────────────────────────────────────────────────────────
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

CLAUDE_MODELS = {
    "Claude Opus 4.7   (most capable)":     "claude-opus-4-7",
    "Claude Opus 4.6   (previous flagship)":"claude-opus-4-6",
    "Claude Sonnet 4.6 (best default)":     "claude-sonnet-4-6",
    "Claude Sonnet 4.5 (proven workhorse)": "claude-sonnet-4-5-20250929",
    "Claude Haiku 4.5  (fastest, cheapest)":"claude-haiku-4-5-20251001",
}
CLAUDE_MODEL = "claude-sonnet-4-6"

# ─────────────────────────────────────────────────────────────────────
# CLAUDE CODE (subprocess)
# ─────────────────────────────────────────────────────────────────────
CLAUDECODE_MODELS = {
    "Default (whatever your CLI is set to)": "",
    "Opus    (most capable)":                "opus",
    "Sonnet  (best default)":                "sonnet",
    "Haiku   (fastest, cheapest)":           "haiku",
}
CLAUDECODE_MODEL = ""
CLAUDECODE_BIN   = _shutil.which("claude") or "claude"

PLAN_WITH_CLAUDECODE     = False
PLANNER_CLAUDECODE_MODEL = "haiku"

# ─────────────────────────────────────────────────────────────────────
# OLLAMA — installed model probe
# ─────────────────────────────────────────────────────────────────────
_PREFERRED_CONTENT_MODELS = [
    "qwen3:14b", "qwen3:8b",
    "llama3.1:70b", "llama3.3:70b", "llama3:70b",
    "qwen2.5:72b", "qwen2.5:32b", "qwen2.5:14b",
    "mixtral:8x7b", "mixtral:8x22b",
    "llama3.1:8b", "llama3:latest", "llama3:8b",
    "mistral:latest",
]


def _list_installed_ollama():
    try:
        import ollama
        listing = ollama.list()
        installed = []
        for m in (listing.get("models", []) if isinstance(listing, dict)
                  else getattr(listing, "models", [])):
            name = (m.get("name") if isinstance(m, dict)
                    else getattr(m, "name", None) or getattr(m, "model", None))
            if name:
                installed.append(name)
        return installed
    except Exception:
        return []


def _best_local_content_model(default="qwen3:8b"):
    installed = _list_installed_ollama()
    for pref in _PREFERRED_CONTENT_MODELS:
        if pref in installed:
            return pref
    for n in installed:
        low = n.lower()
        if "qwen" in low or "llama" in low or "mistral" in low:
            return n
    return default


_BEST_CONTENT_MODEL = _best_local_content_model("qwen3:8b")
_INSTALLED_OLLAMA   = _list_installed_ollama()

# ─────────────────────────────────────────────────────────────────────
# MODEL TIERS — every tier maps every intent to a model
# ─────────────────────────────────────────────────────────────────────
# 'pdf' intent added (uses same model as docx — they're both prose)

MODEL_TIERS = {
    "fast": {
        "chat":    "mistral:latest",
        "docx":    "llama3.1:8b",
        "pptx":    "llama3.1:8b",
        "pdf":     "llama3.1:8b",
        "python":  "qwen2.5-coder:7b",
        "qa":      "llama3-groq-tool-use:latest",
        "planner": "llama3.1:8b",
    },
    "balanced": {
        "chat":    "qwen3:8b",
        "docx":    "qwen3:8b",
        "pptx":    "qwen3:8b",
        "pdf":     "qwen3:8b",
        "python":  "qwen2.5-coder:7b",
        "qa":      "qwen3:8b",
        "planner": "qwen3:8b",
    },
    "best": {
        "chat":    "qwen3:14b",
        "docx":    "qwen3:14b",
        "pptx":    "qwen3:14b",
        "pdf":     "qwen3:14b",
        "python":  "qwen2.5-coder:14b",
        "qa":      "qwen3:14b",
        # F-08: outline planning is a tiny JSON task; using the 14B for it
        # was overkill. qwen3:8b produces equally-valid outlines ~3x faster.
        "planner": "qwen3:8b",
    },
    "custom": {
        "chat":    "qwen3:8b",
        "docx":    "qwen3:8b",
        "pptx":    "qwen3:8b",
        "pdf":     "qwen3:8b",
        "python":  "qwen2.5-coder:7b",
        "qa":      "qwen3:8b",
        "planner": "qwen3:8b",
    },
}
ACTIVE_TIER = "balanced"

# ─────────────────────────────────────────────────────────────────────
# AUTO-PULL
# ─────────────────────────────────────────────────────────────────────
AUTO_PULL_MISSING_MODELS = True
AUTO_PULL_TIERS          = ["fast", "balanced", "best"]

# ─────────────────────────────────────────────────────────────────────
# PROMPT LLM
# ─────────────────────────────────────────────────────────────────────
PROMPT_LLM = (_BEST_CONTENT_MODEL
              if _BEST_CONTENT_MODEL in _INSTALLED_OLLAMA
              else "qwen3:8b")
PROMPT_LLM_CLAUDECODE_MODEL = "haiku"
ENHANCE_NON_IMAGE_PROMPTS   = False

# ─────────────────────────────────────────────────────────────────────
# MISC
# ─────────────────────────────────────────────────────────────────────
SUPPORTED_EXTS = {
    ".docx", ".pptx", ".pdf", ".py", ".txt", ".md", ".json", ".csv",
    ".xlsx", ".html", ".js", ".ts", ".r", ".sql",
}


def resolve_prompt_llm():
    if PROMPT_LLM == "claudecode":
        return ("claudecode", PROMPT_LLM_CLAUDECODE_MODEL)
    return ("ollama", PROMPT_LLM)


def status_summary():
    lines = [
        f"  WORK_DIR        : {WORK_DIR}",
        f"  OUTPUT_DIR      : {OUTPUT_DIR}",
        f"  LOGS_DIR        : {LOGS_DIR}",
        f"  PROVIDER        : {PROVIDER}",
        f"  ACTIVE_TIER     : {ACTIVE_TIER}",
        f"  PROMPT_LLM      : {PROMPT_LLM}",
        f"  Best local model: {_BEST_CONTENT_MODEL}",
        f"  Installed ollama: {len(_INSTALLED_OLLAMA)} model(s)",
        f"  `claude` CLI    : {CLAUDECODE_BIN if _shutil.which('claude') else 'NOT FOUND'}",
    ]
    return "\n".join(lines)


print(f"✅ Config ready  |  {WORK_DIR}")
print(f"   Output → {OUTPUT_DIR}")
print(f"   Active tier: {ACTIVE_TIER}  ·  (Output Quality removed — "
      f"specify length in your prompt)")
print(f"   Best installed local: {_BEST_CONTENT_MODEL}  "
      f"({len(_INSTALLED_OLLAMA)} model(s) found)")
print(f"   `claude` CLI: {CLAUDECODE_BIN if _shutil.which('claude') else 'NOT FOUND on PATH'}")
