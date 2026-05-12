"""ai_text_files.intent_router — intent detection and model routing.

Notes:
  • New 'pdf' intent for PDF file generation. Detected by phrases like
    "create a pdf", "make a pdf", ".pdf", etc.
  • Patterns ordered: python > pdf > pptx > docx (PDF before docx so
    "create a pdf report" doesn't fall through to docx).

Public API:
    detect_intent(text, has_files=False) → str
        Returns one of: chat, docx, pptx, pdf, python, qa
    pick_model(intent) → (model_name, reason)
    parse_chart_request(text) → int
"""

from __future__ import annotations

import re

from . import config


# Pattern order matters — more specific first.
# PDF placed BEFORE docx to win on "create a pdf report"-style asks.
INTENT_PATTERNS = [
    # Python
    (r'\b(?:create|generate|make|write|build)\b.{0,80}\b(?:python|\.py|script|function)\b', 'python'),
    (r'\b(?:more complex|extend|improve|rewrite)\b.{0,80}(?:function|script|code|python)', 'python'),
    # PDF — must come before docx
    (r'\b(?:create|generate|make|write|produce|save\s+as)\b.{0,80}\b(?:pdf|\.pdf)\b', 'pdf'),
    # PPTX
    (r'\b(?:create|generate|make|build)\b.{0,80}\b(?:pptx|powerpoint|presentation|slides?|deck)\b', 'pptx'),
    # DOCX
    (r'\b(?:create|generate|make|write|produce)\b.{0,80}\b(?:docx|word doc|word document|report|memo|letter)\b', 'docx'),
]


def detect_intent(text, has_files=False):
    """Returns: 'chat', 'docx', 'pptx', 'pdf', 'python', or 'qa'."""
    t = text.lower()
    for pattern, kind in INTENT_PATTERNS:
        if re.search(pattern, t):
            return kind
    return 'qa' if has_files else 'chat'


def pick_model(intent):
    """Returns (model_name, reason). Tier-aware with fallbacks."""
    if config.PROVIDER == 'claude':
        label = next((k for k, v in config.CLAUDE_MODELS.items()
                      if v == config.CLAUDE_MODEL), config.CLAUDE_MODEL)
        return config.CLAUDE_MODEL, label

    if config.PROVIDER == 'claudecode':
        label = next((k for k, v in config.CLAUDECODE_MODELS.items()
                      if v == config.CLAUDECODE_MODEL),
                     config.CLAUDECODE_MODEL or 'CLI default')
        return config.CLAUDECODE_MODEL, f"Claude Code → {label}"

    tier = config.ACTIVE_TIER
    tier_map = config.MODEL_TIERS.get(tier, {})
    chosen  = tier_map.get(intent)
    installed = set(config._INSTALLED_OLLAMA)

    if chosen and chosen in installed:
        return chosen, f"tier={tier}, intent={intent}"

    bal = config.MODEL_TIERS["balanced"].get(intent)
    if bal and bal in installed:
        msg = f"tier={tier} model '{chosen}' missing → fell back to balanced tier ({bal})"
        print(f"  ⚠️  {msg}")
        return bal, msg

    if config._BEST_CONTENT_MODEL in installed:
        msg = (f"tier={tier} and balanced both unavailable for intent={intent} → "
               f"using auto-detected {config._BEST_CONTENT_MODEL}")
        print(f"  ⚠️  {msg}")
        return config._BEST_CONTENT_MODEL, msg

    msg = (f"no preferred models installed for intent={intent} → "
           f"trying mistral:latest (may also fail)")
    print(f"  ⚠️  {msg}")
    return "mistral:latest", msg


_CHART_NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3,
                       "four": 4, "five": 5, "six": 6}


def parse_chart_request(text):
    t = text.lower()
    if re.search(r'\b(?:no|without|skip|exclude)\s+(?:charts?|graphs?|plots?|figures?)\b', t):
        return 0
    m = re.search(r'\b(\d+)\s+(?:charts?|graphs?|plots?|figures?|visualizations?)\b', t)
    if m:
        return min(int(m.group(1)), 6)
    m = re.search(r'\b(a|an|one|two|three|four|five|six)\s+'
                  r'(?:charts?|graphs?|plots?|figures?|visualizations?)\b', t)
    if m:
        return _CHART_NUMBER_WORDS.get(m.group(1), 0)
    if re.search(r'\b(?:with|include|add|including)\b.{0,30}'
                 r'\b(?:charts?|graphs?|plots?|figures?|visualizations?)\b', t):
        return 1
    return 0
