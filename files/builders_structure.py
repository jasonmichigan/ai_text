"""ai_text_files.builders_structure — two-pass content builders.

Notes:
  • OUTPUT_QUALITY_PROFILES removed. The model writes whatever length
    your prompt requests. Internal defaults (sections, slides, bullets)
    are sane fallbacks; mention specific numbers in your prompt to
    override (e.g. "8 sections", "10 slides", "2000 words").
  • Validation pass removed entirely.
  • Added build_pdf_structure (mirrors docx structure for PDF rendering).

Two-pass for local; single-pass for Claude.

Public API:
    build_docx_structure(task, analysis, model, n_chart_sections) → dict
    build_pptx_structure(task, analysis, model, n_chart_slides)   → dict
    build_pdf_structure(task, analysis, model, n_chart_sections)  → dict
    build_python_code(task, analysis, model)                      → str
    analyze_files_for_task(task, file_context, model)             → str
"""

from __future__ import annotations

import json
import re
import shutil as _shutil

from . import config, conversation


# ─── Sane default targets — used as guidance, not hard limits ────────
# Mention specific numbers in your prompt to override:
#   "write a 2000-word essay"     → model targets ~2000 words
#   "create an 8-section report"  → model targets 8 sections
#   "10-slide deck"               → 10 slides
DEFAULT_N_SECTIONS = 5
DEFAULT_N_SLIDES   = 6
DEFAULT_BULLETS    = "4-6"   # min-max range as string for prompt
REF_WINDOW         = 8000    # legacy upper bound; see _ref_window() for live sizing

# F-09: chars of analysis text included in each per-section / per-slide
# prompt. When the user attached files, the analysis carries real content
# and a bigger window helps. When they didn't, the 'analysis' is just the
# (short) enhanced prompt and padding it is pure waste.
REF_WINDOW_WITH_FILES = 12000
REF_WINDOW_NO_FILES   = 2500


def _ref_window(analysis, has_files):
    """Return how many chars of `analysis` to include in a per-section prompt."""
    cap = REF_WINDOW_WITH_FILES if has_files else REF_WINDOW_NO_FILES
    return min(len(analysis or ""), cap)

# ─── Word / page target parsing (F-03, F-05) ─────────────────────────
# These are used to compute per-section word budgets so local models
# (qwen3:14b in particular) don't blow past length targets by 4-6x.
_WORD_PATTERNS = (
    r'(\d+)[-\s]?word',
    r'(\d+)\s*words?\b',
    r'(\d+)w\b',
)
_PAGE_PATTERN  = r'(\d+)[-\s]?page'
_WORDS_PER_PAGE = 250  # rough density at our reportlab settings (10.5pt / 14pt leading / 1in margin)


def parse_word_target(text, include_page_fallback=True):
    """Extract a word target from a prompt. Returns int or None.

    Matches '500-word', '1500 words', '2000w'. If include_page_fallback is True
    (default), also matches '3-page' / '5 pages' and converts via _WORDS_PER_PAGE."""
    t = (text or "").lower()
    for pat in _WORD_PATTERNS:
        m = re.search(pat, t)
        if m:
            return int(m.group(1))
    if include_page_fallback:
        m = re.search(_PAGE_PATTERN, t)
        if m:
            return int(m.group(1)) * _WORDS_PER_PAGE
    return None


def parse_page_target(text):
    """Extract a page target (PDF). Returns int or None."""
    t = (text or "").lower()
    m = re.search(_PAGE_PATTERN, t)
    if m:
        return int(m.group(1))
    return None


def analyze_files_for_task(task, file_context, model):
    system = (
        "You are an expert analyst. The user has attached file(s) below. "
        "Read carefully and produce a detailed analysis for the task. "
        "Use ONLY information from the actual file content — never invent filenames. "
        "Quote function names, classes, and identifiers exactly as they appear."
    )
    prompt = (
        f"TASK: {task}\n\n"
        f"FILE CONTENTS:\n{file_context}\n\n"
        "Produce a thorough, specific analysis that covers everything needed for the task."
    )
    analysis = conversation.call_plain(prompt, model=model, system=system)
    print(f"\n  📝 Analysis: {len(analysis):,} chars")
    return analysis


def _clean_section_text(text, expected_heading=None):
    if not text:
        return text
    t = text.strip()
    t = re.sub(r'^(?:here(?:\'?s| is)|below is|sure[,!]?\s*here(?:\'?s| is)?)\s*'
               r'(?:the\s+)?(?:requested\s+)?(?:prose|content|text|paragraphs?|section|summary|conclusion)[\s:]*\n+',
               '', t, flags=re.IGNORECASE)
    if expected_heading:
        eh = re.escape(expected_heading.strip())
        t = re.sub(rf'^\*{{1,3}}\s*{eh}\s*\*{{1,3}}\s*\n+', '', t, flags=re.IGNORECASE)
        t = re.sub(rf'^#{{1,6}}\s*{eh}\s*\n+', '', t, flags=re.IGNORECASE)
        t = re.sub(rf'^{eh}\s*[:\.]?\s*\n+', '', t, flags=re.IGNORECASE)
    t = re.sub(r'^\*{2,3}([^\*\n]{1,50})\*{2,3}\s*\n+', '', t)
    return t.strip()


def _clean_conclusion_text(text):
    """Truncate the conclusion at the first heading / pseudo-heading / numbered-bold
    marker so the local model can't write a 'mini-document inside the conclusion'.
    F-04: TC4_local's Conclusion started with its own title and contained three
    numbered sub-sections repeating the whole document; this cuts that off."""
    if not text:
        return text
    t = _clean_section_text(text, 'Conclusion')
    cut_patterns = (
        r'\n#{1,6}\s',                       # markdown heading
        r'\n\*\*[^*\n]{1,80}\*\*\s*\n',      # bold pseudo-heading on its own line
        r'\n\d+\.\s+\*\*',                   # numbered list opening with bold
        r'\n[A-Z][\w ,/&-]{3,60}\?\s*\n',    # "What is X?" pseudo-heading
    )
    for pat in cut_patterns:
        m = re.search(pat, t)
        if m:
            t = t[:m.start()].rstrip()
            break
    return t


# ─── Planner routing (Claude Code optional) ──────────────────────────
def _planner_model_for_local():
    """Resolve which model handles outline/JSON planning.

    F-08: read the per-tier 'planner' slot from config.MODEL_TIERS instead
    of always using the content model. For tier=best this is qwen3:8b
    (faster), and outline JSON is a small task the 8B handles well. Falls
    back to _BEST_CONTENT_MODEL if the planner model isn't installed."""
    if config.PLAN_WITH_CLAUDECODE and _shutil.which("claude"):
        return ('cc', config.PLANNER_CLAUDECODE_MODEL)
    tier_map      = config.MODEL_TIERS.get(config.ACTIVE_TIER, {})
    planner_model = tier_map.get('planner') or config._BEST_CONTENT_MODEL
    if planner_model not in (config._INSTALLED_OLLAMA or []):
        planner_model = config._BEST_CONTENT_MODEL
    return ('local', planner_model)


def _call_planner(prompt, system):
    kind, model_arg = _planner_model_for_local()
    if kind == 'cc':
        label = model_arg if model_arg else 'CLI default'
        print(f"  🧠 [planner: Claude Code → {label}] ", end="", flush=True)
        on_tok = (lambda t: print(".", end="", flush=True))
        try:
            result = conversation._stream_claudecode(
                model_arg, system,
                [{"role": "user", "content": prompt}], on_tok)
            print()
            return result
        except Exception as e:
            print(f"\n  ⚠️  Claude Code planner failed: {e}")
            print(f"  ↪ falling back to local model {config._BEST_CONTENT_MODEL}")
            return conversation.call_plain(prompt,
                                            model=config._BEST_CONTENT_MODEL,
                                            system=system)
    return conversation.call_plain(prompt, model=model_arg, system=system)


def _call_planner_json(prompt, system):
    raw = _call_planner(prompt, system)
    m = re.search(r'(\{.*\}|\[.*\])', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return {"raw": raw}


# ─── DOCX structure ──────────────────────────────────────────────────
def build_docx_structure(task, analysis, model, n_chart_sections,
                         word_target_override=None, has_files=False):
    """Two-pass for local; single-pass for Claude.

    F-02: single-pass path moves all formatting instructions into the SYSTEM
          message and wraps user content in <user_task>/<reference> XML so
          Claude doesn't read it as nested system directives.
    F-03: local two-pass path enforces hard per-section word budgets derived
          from a target parsed out of the task (or `word_target_override`).
    F-04: conclusion prompt is tightened to prevent the mini-document-inside-
          conclusion runaway pattern, plus _clean_conclusion_text on the output.
    F-09: per-prompt reference window scales with whether files are attached
          (controlled by `has_files`).
    """
    is_local = (config.PROVIDER == 'local')
    rw       = _ref_window(analysis, has_files)
    ref      = (analysis or "")[:rw]

    if not is_local:
        # F-02: instructions in SYSTEM, user content in XML-tagged blocks.
        no_charts_note = ('Do NOT mention figures, charts, graphs, or include '
                          'caption phrases like "Figure 1:".'
                          if n_chart_sections == 0 else '')
        system = (
            'You are a Document formatter that returns ONLY valid JSON. '
            'Schema: {"title":str,"subtitle":str,"executive_summary":str,'
            '"sections":[{"heading":str,"content":str,"chart_topic":str|null}],'
            '"conclusion":str}. '
            'Read the <user_task> and <reference> blocks below and produce '
            'the JSON document structure. Choose section count and length '
            'based on the task description. Mark exactly '
            f'{n_chart_sections} section(s) with a non-null chart_topic; '
            'the rest must be null. Substantive prose drawn from the '
            f'reference; no placeholders. {no_charts_note}'
        )
        prompt = (f"<user_task>{task}</user_task>\n\n"
                  f"<reference>{ref}</reference>")
        return conversation.call_json(prompt, model=model, system=system)

    # ── LOCAL PATH: two-pass with per-section word budgets ──
    print(f"\n  STEP 2a — Planning outline...")
    outline_system = (
        "You are an expert technical writer. Plan a Word document outline. "
        "Return ONLY a JSON object, no prose, no markdown. "
        'Schema: {"title":str,"subtitle":str,"section_headings":[strings]} '
        "Pick a section count appropriate to the task — if the user "
        f"specified a count, honor it; otherwise default to about {DEFAULT_N_SECTIONS}. "
        "Each heading should be a concrete topic, not a label like 'Introduction' "
        "or 'Section 1'. Tailor headings tightly to the task."
    )
    # Outline only needs a thin slice of the reference (small task).
    outline_ref = ref[:min(len(ref), 4000)]
    outline_prompt = (
        f"Task: {task}\n\n"
        f"Reference material:\n{outline_ref}\n\n"
        "Plan title, subtitle, and section headings."
    )
    outline = _call_planner_json(outline_prompt, outline_system)
    title    = outline.get("title", task[:80].title())
    subtitle = outline.get("subtitle", "")
    headings = outline.get("section_headings", [])
    if not isinstance(headings, list) or len(headings) < 2:
        defaults = ["Background", "Key Concepts", "Discussion",
                    "Implications", "Considerations", "Outlook"]
        headings = defaults[:DEFAULT_N_SECTIONS]

    print(f"  ✓ Outline: {title}")
    print(f"    Sections: {' / '.join(headings)}")

    # F-03: compute word budgets. Split: 15% exec / 70% body / 15% conclusion.
    # Falls back to 1500 if the task has no parseable target.
    total_target = (word_target_override
                    or parse_word_target(task)
                    or 1500)
    n_sections = max(1, len(headings))
    per_section_budget  = max(60, int(total_target * 0.70 / n_sections))
    exec_budget         = max(80, int(total_target * 0.15))
    conclusion_budget   = max(60, int(total_target * 0.15))
    print(f"  📏 Budget: total={total_target}w  exec={exec_budget}w  "
          f"per-section={per_section_budget}w  conclusion={conclusion_budget}w")

    writer_system = (
        "You are an expert writer. Respect word limits exactly — when a "
        "hard limit is given, stop at that limit even mid-sentence rather "
        "than continuing. Produce only the requested prose, nothing else."
    )

    # F-10: when no charts are requested, ask the writer to avoid figure
    # callouts / "Figure N:" captions that would dangle without an image.
    no_charts_clause = (' Do NOT mention figures, charts, graphs, or '
                        'include captions like "Figure 1:".'
                        if n_chart_sections == 0 else '')

    # PASS 2a — executive summary (F-03: hard word cap)
    print("\n  STEP 2b — Writing executive summary...")
    es_prompt = (
        f"User's request: {task}\n\n"
        f"Topic: {title}\n"
        f"Reference material:\n{ref}\n\n"
        f"Write an executive summary in 1-2 short paragraphs. "
        f"HARD WORD LIMIT: {exec_budget} words. If you reach the limit, "
        "stop mid-sentence rather than continuing. "
        "Be specific and substantive. Do NOT include phrases like "
        '"this document", "this report", or "in this section". Do NOT include '
        "headings, bullets, or any scaffolding — just polished prose."
        + no_charts_clause
    )
    executive_summary = conversation.call_plain(
        es_prompt, model=model, system=writer_system)
    executive_summary = _clean_section_text(executive_summary, 'Executive Summary')

    # PASS 2b — per-section content (F-03: hard word cap per section)
    sections = []
    chart_marks = ([True] * n_chart_sections
                   + [False] * max(0, len(headings) - n_chart_sections))

    for idx, heading in enumerate(headings):
        print(f"\n  STEP 2c.{idx+1} — Writing section: {heading}")
        section_no_charts = ('' if chart_marks[idx] else no_charts_clause)
        sec_prompt = (
            f"User's request: {task}\n\n"
            f"Document topic: {title}\n"
            f"Section heading: {heading}\n\n"
            f"Reference material:\n{ref}\n\n"
            f'Write the content for the section titled "{heading}". '
            f"HARD WORD LIMIT: {per_section_budget}-{per_section_budget + 50} "
            "words. If you reach the upper limit, stop mid-sentence rather "
            "than continuing. "
            "Be concrete and specific. Use facts and details from the reference. "
            "Do NOT repeat the heading. Do NOT include any meta-text like "
            '"this section will discuss" or "in this part". Do NOT include '
            "scaffolding, JSON, or markdown — just clean prose paragraphs "
            "separated by blank lines."
            + section_no_charts
        )
        content = conversation.call_plain(
            sec_prompt, model=model, system=writer_system)
        content = _clean_section_text(content, heading)

        sections.append({
            "heading":     heading,
            "content":     content,
            "chart_topic": (heading if chart_marks[idx] else None),
        })

    # PASS 2c — conclusion (F-04: prevent mini-document runaway)
    print("\n  STEP 2d — Writing conclusion...")
    conc_prompt = (
        f"User's request: {task}\n\n"
        f"Topic: {title}\n\n"
        f"Write a CONCLUSION of EXACTLY 2 short paragraphs, {conclusion_budget} "
        "words maximum. Rules: do NOT include a title or heading. Do NOT use "
        "bold or markdown. Do NOT use numbered lists or bullets. Do NOT "
        "introduce any new topic, example, or section. Only synthesize what "
        "was already covered. If you find yourself starting a new section "
        "(e.g. '## ...', '**Heading**', a numbered list, or 'What is X?'), "
        "STOP immediately."
    )
    conclusion = conversation.call_plain(
        conc_prompt, model=model,
        system=writer_system + " No headings, no markdown, no lists.")
    conclusion = _clean_conclusion_text(conclusion)

    return {
        "title":             title,
        "subtitle":          subtitle,
        "executive_summary": executive_summary,
        "sections":          sections,
        "conclusion":        conclusion,
    }


# ─── PDF structure ───────────────────────────────────────────────────
# F-05: page-target aware. Converts "N-page" to a word target before
# delegating to build_docx_structure (which renders structurally identical
# output; only the renderer differs).
def build_pdf_structure(task, analysis, model, n_chart_sections,
                        has_files=False):
    """Resolve a word target from the task's page count (if any) and delegate
    to build_docx_structure with that target. Falls back to the docx builder's
    own default when neither pages nor words are specified.

    F-09: passes `has_files` through so the docx builder can size its
    reference window correctly."""
    pages          = parse_page_target(task)
    explicit_words = parse_word_target(task, include_page_fallback=False)

    if pages is not None:
        page_word_target = pages * _WORDS_PER_PAGE
        # If the user gave BOTH a page count and a word count, trust whichever
        # is consistent with the other. Otherwise prefer pages (that's what the
        # user is rendering at).
        if (explicit_words is not None
                and abs(explicit_words - page_word_target) <= 0.5 * page_word_target):
            word_target = explicit_words
        else:
            word_target = page_word_target
    else:
        word_target = explicit_words  # may be None — docx builder will default

    print(f"  📕 PDF target: pages={pages}  words={word_target}")
    return build_docx_structure(task, analysis, model, n_chart_sections,
                                word_target_override=word_target,
                                has_files=has_files)


# ─── PPTX structure ──────────────────────────────────────────────────
def build_pptx_structure(task, analysis, model, n_chart_slides,
                         has_files=False):
    """F-09: per-prompt reference window now scales with `has_files`."""
    is_local = (config.PROVIDER == 'local')
    rw       = _ref_window(analysis, has_files)
    ref      = (analysis or "")[:rw]

    if not is_local:
        # F-02: same defense as build_docx_structure single-pass.
        system = (
            'You are a Presentation formatter that returns ONLY valid JSON. '
            'Schema: {"title":str,"subtitle":str,'
            '"slides":[{"title":str,"bullets":[str],"chart_topic":str|null}]}. '
            'Read the <user_task> and <reference> blocks below and produce the '
            'JSON deck structure. Pick slide count based on the task; mark '
            f'exactly {n_chart_slides} slide(s) with a non-null chart_topic; '
            'the rest must be null. Each slide should have specific bullets — '
            'no placeholder text.'
        )
        prompt = (f"<user_task>{task}</user_task>\n\n"
                  f"<reference>{ref}</reference>")
        return conversation.call_json(prompt, model=model, system=system)

    # ── LOCAL PATH: two-pass ──
    print(f"\n  STEP 2a — Planning slide outline...")
    outline_system = (
        "You plan presentation slide decks. Return ONLY JSON, no prose. "
        'Schema: {"title":str,"subtitle":str,"slide_titles":[strings]} '
        "Pick a slide count appropriate to the task — if the user specified "
        f"one, honor it; otherwise default to about {DEFAULT_N_SLIDES}. "
        "Specific, content-focused slide titles. No 'Introduction' or 'Conclusion' titles."
    )
    outline_ref = ref[:min(len(ref), 4000)]
    outline_prompt = (
        f"Task: {task}\n\nReference:\n{outline_ref}\n\n"
        "Plan deck title, subtitle, and content-slide titles."
    )
    outline = _call_planner_json(outline_prompt, outline_system)
    title    = outline.get("title", task[:80].title())
    subtitle = outline.get("subtitle", "")
    slide_titles = outline.get("slide_titles", [])
    if not isinstance(slide_titles, list) or len(slide_titles) < 2:
        defaults = ["Background", "Key Concepts", "Details", "Examples",
                    "Implications", "Considerations", "Outlook", "Conclusion"]
        slide_titles = defaults[:DEFAULT_N_SLIDES]

    print(f"  ✓ Outline: {title}")
    print(f"    Slides: {' / '.join(slide_titles)}")

    slides = []
    chart_marks = ([True] * n_chart_slides
                   + [False] * max(0, len(slide_titles) - n_chart_slides))

    for idx, st in enumerate(slide_titles):
        print(f"\n  STEP 2b.{idx+1} — Bullets for: {st}")
        bullet_prompt = (
            f"User's request: {task}\n\n"
            f"Deck topic: {title}\n"
            f"Slide title: {st}\n\n"
            f"Reference:\n{ref}\n\n"
            f'Write bullet points for the slide titled "{st}". Aim for {DEFAULT_BULLETS} '
            "bullets, but adjust if the user asked for a specific count or density. "
            "Each bullet: one specific, substantive sentence (not a fragment, not a label). "
            "Output one bullet per line. Do NOT include markdown bullets (• or -), "
            "do NOT echo the slide title, do NOT add any preamble."
        )
        raw = conversation.call_plain(
            bullet_prompt, model=model,
            system="Output only the requested bullets, one per line. Nothing else."
        ).strip()
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        bullets = []
        for ln in lines:
            ln = re.sub(r'^[\-\*\•\d\.\)]+\s*', '', ln)
            if ln and len(ln) > 6 and st.lower() not in ln.lower()[:len(st)+5]:
                bullets.append(ln)
        bullets = bullets[:10] if bullets else [raw[:120]]

        slides.append({
            "title":       st,
            "bullets":     bullets,
            "chart_topic": (st if chart_marks[idx] else None),
        })

    return {"title": title, "subtitle": subtitle, "slides": slides}


# ─── Python code ─────────────────────────────────────────────────────
def build_python_code(task, analysis, model, has_files=False):
    """Generate a Python script. Length and quality driven by the prompt;
    no profile-driven extras anymore.

    F-09: per-prompt reference window scales with `has_files`."""
    rw  = _ref_window(analysis, has_files)
    ref = (analysis or "")[:rw]
    system = ("Expert Python developer. Output ONLY raw Python code, "
              "no markdown fences, no commentary. Match the scope and depth "
              "the user asked for.")
    prompt = (f"Task: {task}\n\nReference (from attached files):\n{ref}\n\n"
              "Write the script. Include imports and named functions. "
              "Be complete and runnable — no placeholders or '...' to fill in.")

    print(f"🤖 [{model}] ", end="", flush=True)
    on_tok = (lambda t: print(".", end="", flush=True))

    if conversation._is_claude_model(model) or config.PROVIDER == 'claudecode':
        if config.PROVIDER == 'claudecode':
            code = conversation._stream_claudecode(
                model, system,
                [{"role": "user", "content": prompt}], on_tok)
        else:
            code = conversation._stream_claude(
                model, system,
                [{"role": "user", "content": prompt}], on_tok)
    else:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
        code = conversation._stream_ollama(model, msgs, on_tok)
    print()
    code = re.sub(r'^```[\w]*\n?', '', code, flags=re.MULTILINE)
    return re.sub(r'```$', '', code, flags=re.MULTILINE).strip()
