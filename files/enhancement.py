"""ai_text_files.enhancement — prompt enhancement for non-image intents.

added 'pdf' template (mirrors docx).

Public API:
    enhance_only_generic(short_prompt, intent) → (enhanced_text, intent)
"""

from __future__ import annotations

import shutil as _shutil

from . import config, conversation


_GENERIC_ENHANCE_TEMPLATES = {
    "chat": (
        "You expand short user questions into clearer, more structured prompts. "
        "Take the user's message and rewrite it as a single self-contained prompt "
        "that: (1) preserves every fact, name, and constraint they mentioned; "
        "(2) makes implicit context explicit (audience, format, depth); "
        "(3) is at most 2-3 sentences longer than the original. "
        "Output ONLY the rewritten prompt — no preamble, no explanation, no quotes."
    ),
    "docx": (
        "You expand short Word-document requests into structured prompts. "
        "Take the user's topic and produce a richer prompt of 80-150 words that "
        "specifies: (1) the audience and reading level; (2) the structure "
        "(sections, headings); (3) tone (formal/casual/technical); (4) approximate "
        "length; (5) any factual specifics the user mentioned. PRESERVE every "
        "concrete detail. If the user mentioned a target length (e.g. '2000 words', "
        "'5 pages'), preserve it verbatim. "
        "Output ONLY the expanded prompt — no preamble, no quotes."
    ),
    "pdf": (  # same as docx but the deliverable is a PDF
        "You expand short PDF-creation requests into structured prompts. "
        "Take the user's topic and produce a richer prompt of 80-150 words that "
        "specifies: (1) the audience and reading level; (2) the structure "
        "(sections, headings); (3) tone (formal/casual/technical); (4) approximate "
        "length; (5) any factual specifics the user mentioned. PRESERVE every "
        "concrete detail. If the user mentioned a target length (e.g. '2000 words', "
        "'2 pages'), preserve it verbatim. "
        "Output ONLY the expanded prompt — no preamble, no quotes."
    ),
    "pptx": (
        "You expand short slide-deck requests into structured prompts. "
        "Take the user's topic and produce a richer prompt of 80-150 words that "
        "specifies: (1) the audience; (2) approximate slide count; (3) main "
        "sections / talking points; (4) tone (pitch / training / executive / "
        "academic); (5) any factual details the user mentioned. PRESERVE every "
        "concrete detail. If the user mentioned a slide count (e.g. '8 slides'), "
        "preserve it verbatim. "
        "Output ONLY the expanded prompt — no preamble, no quotes."
    ),
    "python": (
        "You expand short Python script requests into specification-style prompts. "
        "Take the user's ask and produce a richer prompt of 60-130 words that "
        "specifies: (1) the script's purpose and inputs/outputs; (2) any libraries "
        "to prefer or avoid; (3) error-handling expectations; (4) code style; "
        "(5) any specific behavior the user mentioned. PRESERVE every concrete detail. "
        "Output ONLY the expanded prompt — no preamble, no quotes, no code."
    ),
    "qa": (
        "You expand short questions about attached files into clearer prompts. "
        "Rewrite the user's question as a single self-contained prompt that "
        "preserves every fact and constraint, makes implicit context explicit, "
        "and is at most 2 sentences longer than the original. "
        "Output ONLY the rewritten prompt — no preamble, no explanation, no quotes."
    ),
}


def _call_prompt_llm(system, user_msg, label="prompt"):
    kind, model = config.resolve_prompt_llm()
    try:
        if kind == "claudecode":
            if not _shutil.which("claude"):
                print(f"  ⚠️  Claude Code chosen for {label} but `claude` not on PATH; "
                      f"falling back to Ollama {config._BEST_CONTENT_MODEL}")
                kind, model = "ollama", config._BEST_CONTENT_MODEL
            else:
                print(f"  ✨ {label} via Claude Code ({model or 'CLI default'})...")
                on_tok = (lambda t: print(".", end="", flush=True))
                out = conversation._stream_claudecode(
                    model, system, [{"role": "user", "content": user_msg}], on_tok)
                print()
                return out
        print(f"  ✨ {label} via Ollama ({model})...")
        return conversation.call_plain(user_msg, model=model, system=system)
    except Exception as e:
        print(f"  ⚠️  {label} call failed ({e}); using fallback Ollama "
              f"{config._BEST_CONTENT_MODEL}")
        try:
            return conversation.call_plain(
                user_msg, model=config._BEST_CONTENT_MODEL, system=system)
        except Exception as e2:
            print(f"  ⚠️  Fallback also failed ({e2})")
            raise


# F-11: phrases that signal a meta-discussion preamble the enhancer
# sometimes prepends (most often Claude Code, despite the system prompt).
# When such a preamble is detected ahead of a '---' separator, we strip
# everything before the last '---' and use the trailing rewrite.
_PREAMBLE_MARKERS = (
    "i notice", "i'll treat", "i will treat",
    "i'm happy", "i am happy",
    "benign", "i need to clarify",
    "these instructions", "system directive",
    "i can help", "appears to be",
)


def _strip_enhancer_preamble(text):
    """If the response begins with a meta-discussion paragraph followed by
    '---' (a common Claude Code pattern when it sees instruction-like input),
    strip everything before the last '---' separator."""
    if not text or "---" not in text:
        return text
    parts = text.split("---")
    preamble = "---".join(parts[:-1])
    head     = preamble[:600].lower()
    if any(marker in head for marker in _PREAMBLE_MARKERS):
        stripped = parts[-1].lstrip()
        print(f"  🧹 Stripped enhancer preamble ({len(preamble)} chars "
              f"before '---' separator).")
        return stripped
    return text


def enhance_only_generic(short_prompt, intent):
    if not config.ENHANCE_NON_IMAGE_PROMPTS:
        return short_prompt, intent
    system = _GENERIC_ENHANCE_TEMPLATES.get(intent, _GENERIC_ENHANCE_TEMPLATES["chat"])
    user_msg = f"User prompt:\n{short_prompt}\n\nRewritten prompt:"
    try:
        enhanced = _call_prompt_llm(system, user_msg, label=f"enhance {intent}")
        enhanced = _strip_enhancer_preamble(enhanced)  # F-11
        enhanced = enhanced.strip().strip('"').strip("'")
        if len(enhanced) < 10 or len(enhanced) > 4000:
            print(f"  ⚠️  Generic enhancer produced unusable output "
                  f"({len(enhanced)} chars); using original.")
            return short_prompt, intent
        return enhanced, intent
    except Exception as e:
        print(f"  ⚠️  Generic enhancement failed ({e}); using original.")
        return short_prompt, intent
