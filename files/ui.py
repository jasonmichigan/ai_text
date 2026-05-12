"""ai_text_files.ui — main UI (ipywidgets).

  • Quality Preset panel REMOVED entirely (no Output radio, no Extended
    sub-panel, no validate-via-CC checkbox). The model writes whatever
    length your prompt requests — say "write a 2000-word essay" or
    "create a 3-page PDF" and it targets that.
  • Tier picker remains, in a slimmer "Model Tier" panel that also hosts
    the Pull missing models button.
  • PDF intent wired into _run_intent and the topic placeholder.

Public API:
    render_ui()
"""

from __future__ import annotations

import os
import re
import shutil as _shutil
import subprocess as _subprocess
import traceback
from pathlib import Path

import ipywidgets as widgets
from IPython.display import HTML, Javascript, clear_output, display, display as _display

from . import (builders_files, conversation, config, enhancement,
               file_picker, file_readers, intent_router, logger,
               tier_manager)

attached_files = []
PENDING_GEN    = {}


def _rank_ollama_model(name):
    """Score an Ollama model for prose / instruction-following tasks
    (the kind 'Enhance my prompt' needs). Returns (score, stars, suffix).
    Higher score = stronger. Used to sort the Prompt LLM dropdown
    strongest-first and to label each entry with stars + descriptor."""
    n = name.lower()

    # Parameter size in billions: matches "14b", "8b", "7b", "8x7b", etc.
    size_b = 7  # conservative default if size isn't in the name
    mx = re.search(r'(\d+)x(\d+)b\b', n)
    if mx:
        size_b = int(mx.group(1)) * int(mx.group(2))
    else:
        ms = re.search(r'(\d+)b\b', n)
        if ms:
            size_b = int(ms.group(1))

    # Family rank: newer/stronger families score higher for prose tasks
    if 'qwen3' in n:
        family, fr = 'qwen3', 50
    elif 'qwen2.5' in n:
        family, fr = 'qwen2.5', 40
    elif 'llama3.3' in n or 'llama3.1' in n:
        family, fr = 'llama3.1', 30
    elif 'mixtral' in n:
        family, fr = 'mixtral', 30
    elif 'llama3' in n:
        family, fr = 'llama3', 20
    elif 'mistral' in n:
        family, fr = 'mistral', 15
    else:
        family, fr = name.split(':')[0], 10

    # Specialization penalty: code / tool-use models are weaker at prose
    spec_note = ""
    spec_penalty = 0
    if 'coder' in n:
        spec_penalty = 8
        spec_note = ", code-specialist"
    elif 'tool-use' in n or 'tool_use' in n:
        spec_penalty = 8
        spec_note = ", tool-use specialist"

    score = fr + size_b - spec_penalty

    if score >= 60:
        stars = "★★★★★"
    elif score >= 48:
        stars = "★★★★"
    elif score >= 35:
        stars = "★★★"
    elif score >= 22:
        stars = "★★"
    else:
        stars = "★"

    suffix = f"  ({size_b}B {family}{spec_note})"
    return score, stars, suffix


def _aiconsole_print(msg, *, kind="info"):
    icons = {
        "settings": "⚙️ ", "info": "ℹ️ ", "topic": "📝", "enhanced": "✨",
        "output": "🎉", "warn": "⚠️ ", "error": "❌", "model": "🧠",
        "tier": "🎯",
    }
    icon = icons.get(kind, "·")
    line = f"{icon} {msg}"
    try:
        with _W["output"]:
            print(line)
    except Exception:
        print(line)
    try:
        logger.log_event(f"aiconsole_{kind}", msg=msg)
    except Exception:
        pass


_W = {}


# ─── Provider panel ──────────────────────────────────────────────────
def _build_provider_panel():
    w_provider_radio = widgets.RadioButtons(
        options=[('🖥️  Local Ollama          (private, free)', 'local'),
                 ('🔌 Claude Code (subprocess) — uses your CLI auth', 'claudecode')],
        value='local', description='Provider:',
        layout=widgets.Layout(width='540px'), style={'description_width': '90px'})

    w_claudecode_model = widgets.Dropdown(
        options=list(config.CLAUDECODE_MODELS.items()),
        value=config.CLAUDECODE_MODEL,
        description='CLI model:',
        layout=widgets.Layout(width='540px', display='none'),
        style={'description_width': '120px'})

    w_planner_radio = widgets.RadioButtons(
        options=[('Local llama only (faster, fully offline)', False),
                 ('Claude Code plans, local llama writes (better quality)', True)],
        value=False, description='Planner:',
        layout=widgets.Layout(width='540px'), style={'description_width': '90px'})

    w_planner_model = widgets.Dropdown(
        options=list(config.CLAUDECODE_MODELS.items()),
        value=config.PLANNER_CLAUDECODE_MODEL,
        description='Planner model:',
        layout=widgets.Layout(width='540px', display='none'),
        style={'description_width': '120px'})

    w_test_cli_btn = widgets.Button(description='🧪 Test `claude` CLI',
        button_style='info', layout=widgets.Layout(width='180px', height='30px'))
    w_diag_btn = widgets.Button(description='🔍 Diagnose',
        layout=widgets.Layout(width='130px', height='30px'))
    w_provider_status = widgets.HTML()

    def _refresh_status():
        if config.PROVIDER == 'claudecode':
            if not _shutil.which('claude'):
                w_provider_status.value = ("<span style='font-size:11px;color:#cf222e'>"
                    "⚠️  `claude` CLI not found on PATH.</span>")
            else:
                label = next((k for k, v in config.CLAUDECODE_MODELS.items()
                              if v == config.CLAUDECODE_MODEL),
                             config.CLAUDECODE_MODEL or 'CLI default')
                w_provider_status.value = (
                    f"<span style='font-size:11px;color:#1a7f37'>"
                    f"✅ Claude Code (subprocess): <b>{label}</b> "
                    f"·&nbsp; binary: <code>{config.CLAUDECODE_BIN}</code></span>")
        else:
            planner_note = ""
            if config.PLAN_WITH_CLAUDECODE:
                if _shutil.which('claude'):
                    pm_label = next((k for k, v in config.CLAUDECODE_MODELS.items()
                                     if v == config.PLANNER_CLAUDECODE_MODEL),
                                    config.PLANNER_CLAUDECODE_MODEL or 'CLI default')
                    planner_note = f" · <b>planner:</b> Claude Code → {pm_label}"
                else:
                    planner_note = " · <span style='color:#cf222e'>planner: `claude` not on PATH</span>"
            else:
                planner_note = " · planner: local llama"
            w_provider_status.value = (f"<span style='font-size:11px;color:#1a7f37'>"
                f"✅ Local mode · tier: <b>{config.ACTIVE_TIER}</b>{planner_note}</span>")

    def _on_provider_change(change):
        config.PROVIDER = change['new']
        is_local = (config.PROVIDER == 'local')
        is_cc    = (config.PROVIDER == 'claudecode')
        w_claudecode_model.layout.display = ('' if is_cc else 'none')
        w_planner_radio.layout.display    = ('' if is_local else 'none')
        w_planner_model.layout.display    = ('' if is_local and config.PLAN_WITH_CLAUDECODE else 'none')
        _refresh_status()
        _aiconsole_print(f"Provider → {config.PROVIDER}", kind="model")

    def _on_claudecode_model_change(change):
        config.CLAUDECODE_MODEL = change['new']
        _refresh_status()
        label = next((k for k, v in config.CLAUDECODE_MODELS.items()
                      if v == config.CLAUDECODE_MODEL),
                     config.CLAUDECODE_MODEL or 'CLI default')
        _aiconsole_print(f"Claude Code model → {label}", kind="model")

    def _on_planner_change(change):
        config.PLAN_WITH_CLAUDECODE = bool(change['new'])
        w_planner_model.layout.display = ('' if config.PLAN_WITH_CLAUDECODE
                                          and config.PROVIDER == 'local' else 'none')
        _refresh_status()
        _aiconsole_print(f"Planner → {'Claude Code' if config.PLAN_WITH_CLAUDECODE else 'local llama only'}",
                         kind="settings")

    def _on_planner_model_change(change):
        config.PLANNER_CLAUDECODE_MODEL = change['new']
        _refresh_status()
        _aiconsole_print(f"Planner CC model → {config.PLANNER_CLAUDECODE_MODEL or 'CLI default'}",
                         kind="model")

    def _on_test_cli(b):
        with _W["output"]:
            clear_output(wait=True)
            print(f"🧪 Testing CLI at: {config.CLAUDECODE_BIN}")
            try:
                r = _subprocess.run([config.CLAUDECODE_BIN, "--version"],
                                    capture_output=True, text=True, timeout=15)
                print(f"  exit code: {r.returncode}")
                print(f"  stdout   : {r.stdout.strip() or '(empty)'}")
                if r.stderr.strip():
                    print(f"  stderr   : {r.stderr.strip()}")
                if r.returncode == 0:
                    print("✅ CLI is reachable.")
                else:
                    print("⚠️  CLI returned non-zero — see stderr above.")
            except FileNotFoundError:
                print("❌ `claude` not found.")
            except _subprocess.TimeoutExpired:
                print("❌ Timed out after 15s.")
            except Exception as ex:
                print(f"❌ {type(ex).__name__}: {ex}")

    def _on_diagnose(b):
        with _W["output"]:
            clear_output(wait=True)
            print("🔍 Diagnostic")
            print(config.status_summary())
            print()
            print(f"  PLAN_WITH_CLAUDECODE         : {config.PLAN_WITH_CLAUDECODE}")
            print(f"  ENHANCE_NON_IMAGE_PROMPTS    : {config.ENHANCE_NON_IMAGE_PROMPTS}")
            print()
            print("  [Local Ollama]")
            try:
                installed = config._list_installed_ollama()
                print(f"    Service          : reachable ({len(installed)} models)")
                print(f"    Best content     : {config._BEST_CONTENT_MODEL}")
                print(f"    All installed    : "
                      f"{', '.join(installed) if installed else '(none)'}")
            except Exception as ex:
                print(f"    Service          : unreachable ({type(ex).__name__})")
            print()
            print("  [Claude Code (subprocess)]")
            print(f"    `claude` on PATH : "
                  f"{'yes — ' + (_shutil.which('claude') or '') if _shutil.which('claude') else 'NO'}")
            print()
            print("  [Tier availability]")
            tier_manager.print_tier_status()

    w_provider_radio.observe(_on_provider_change, names='value')
    w_claudecode_model.observe(_on_claudecode_model_change, names='value')
    w_planner_radio.observe(_on_planner_change, names='value')
    w_planner_model.observe(_on_planner_model_change, names='value')
    w_test_cli_btn.on_click(_on_test_cli)
    w_diag_btn.on_click(_on_diagnose)

    panel = widgets.VBox([
        widgets.HTML("<b style='color:#1F3864;font-size:14px'>🌐 Model Provider</b>"),
        w_provider_radio, w_claudecode_model, w_planner_radio, w_planner_model,
        widgets.HBox([w_test_cli_btn, w_diag_btn], layout=widgets.Layout(gap='6px')),
        w_provider_status,
    ], layout=widgets.Layout(border='1px solid #b3c6e0', padding='10px',
                              margin='4px 0', border_radius='6px'))

    _W.update({
        "provider_radio":       w_provider_radio,
        "claudecode_model":     w_claudecode_model,
        "planner_radio":        w_planner_radio,
        "planner_model":        w_planner_model,
        "test_cli_btn":         w_test_cli_btn,
        "diag_btn":             w_diag_btn,
        "provider_status":      w_provider_status,
        "refresh_provider_status": _refresh_status,
    })
    return panel


# ─── Model Tier panel (slim — no Output Quality anymore) ─────────────
def _build_tier_panel():
    w_tier_radio = widgets.RadioButtons(
        options=[('Fast (7-8B; ~5-15s)', 'fast'),
                 ('Balanced (qwen3:8b; recommended default)', 'balanced'),
                 ('Best (14B; slower, higher quality)', 'best'),
                 ('Custom (pick each model individually)', 'custom')],
        value=config.ACTIVE_TIER,
        description='Tier:',
        layout=widgets.Layout(width='720px'),
        style={'description_width': '90px'})

    w_recommendation_html = widgets.HTML()

    def _ollama_options():
        installed = config._INSTALLED_OLLAMA or []
        if not installed:
            return [("(no Ollama models found — start the service and pull some)", "")]
        return [(name, name) for name in installed]

    custom_dropdowns = {}
    for intent in ("chat", "docx", "pptx", "pdf", "python", "qa", "planner"):
        opts = _ollama_options()
        cur  = config.MODEL_TIERS["custom"][intent]
        valid_values = [v for _, v in opts]
        if cur not in valid_values and valid_values and valid_values[0]:
            cur = valid_values[0]
        custom_dropdowns[intent] = widgets.Dropdown(
            options=opts, value=cur, description=f'{intent}:',
            layout=widgets.Layout(width='540px'),
            style={'description_width': '90px'})

    w_custom_panel = widgets.VBox(
        [widgets.HTML("<div style='font-size:11px;color:#666;margin-bottom:4px'>"
                       "Custom tier — pick each model from your installed Ollama models. "
                       "Falls back to balanced tier and to the auto-detected best model "
                       "if a chosen model becomes unavailable.</div>")]
        + list(custom_dropdowns.values()),
        layout=widgets.Layout(display=('' if config.ACTIVE_TIER == 'custom' else 'none'),
                              border='1px dashed #b3c6e0',
                              padding='8px', margin='6px 0 0 0',
                              border_radius='4px'))

    def _refresh_recommendation():
        avail = tier_manager.verify_tier_models()
        rec, msg = tier_manager.recommend_tier()
        rows = []
        for tier in ("fast", "balanced", "best"):
            info = avail[tier]
            if info["all_present"]:
                rows.append(f"<li><b>{tier}</b> ✅ all {len(info['needed'])} model(s) present</li>")
            else:
                miss = ', '.join(f"<code>{m}</code>" for m in info['missing'])
                rows.append(f"<li><b>{tier}</b> ⚠️  missing: {miss}</li>")
        unlock_lines = []
        for tier in ("balanced", "best"):
            cmd = tier_manager.pull_command_for(tier)
            if cmd:
                unlock_lines.append(
                    f"<div style='font-size:11px;color:#666;margin-left:18px'>"
                    f"To unlock <b>{tier}</b>: <code>{cmd}</code></div>")
        w_recommendation_html.value = (
            f"<div style='font-size:12px;line-height:1.5;color:#1F3864;"
            f"padding:6px 10px;border-left:3px solid #1F3864;background:#f4f7fb;"
            f"margin:6px 0'>"
            f"<b>💡 Recommendation:</b> {msg}<br>"
            f"<ul style='margin:6px 0 0 0;padding-left:20px'>"
            f"{''.join(rows)}"
            f"</ul>"
            f"{''.join(unlock_lines)}"
            f"</div>"
        )

    def _on_tier_change(change):
        old = config.ACTIVE_TIER
        config.ACTIVE_TIER = change['new']
        w_custom_panel.layout.display = ('' if config.ACTIVE_TIER == 'custom' else 'none')
        try:
            logger.log_event("tier_change", old=old, new=config.ACTIVE_TIER)
        except Exception:
            pass
        _aiconsole_print(f"Tier → {config.ACTIVE_TIER}", kind="tier")
        _W["refresh_provider_status"]()

    def _refresh_custom_dropdowns():
        for intent, dd in custom_dropdowns.items():
            cur = dd.value
            new_opts = (
                [(f"Ollama: {m}", m) for m in (config._INSTALLED_OLLAMA or [])]
                or [("(no Ollama models found)", "")]
            )
            new_values = [v for _, v in new_opts]
            dd.options = new_opts
            if cur in new_values:
                dd.value = cur
            else:
                default = config.MODEL_TIERS["custom"][intent]
                dd.value = default if default in new_values else (new_values[0] if new_values else "")

    def _make_custom_handler(intent):
        def _h(change):
            new = change['new']
            if new:
                config.MODEL_TIERS["custom"][intent] = new
                _aiconsole_print(f"Custom tier · {intent} → {new}", kind="model")
        return _h

    w_tier_radio.observe(_on_tier_change, names='value')
    for intent, dd in custom_dropdowns.items():
        dd.observe(_make_custom_handler(intent), names='value')

    _refresh_recommendation()

    panel = widgets.VBox([
        widgets.HTML("<b style='color:#1F3864;font-size:14px'>🎯 Model Tier</b> "
                     "<span style='color:#888;font-size:11px'>"
                     "Missing models are auto-pulled on startup (Cell 2). "
                     "Tip: control output length by saying it in your prompt "
                     "(e.g. \"write a 2000-word essay\", \"create a 3-page PDF\").</span>"),
        w_tier_radio,
        w_recommendation_html,
        w_custom_panel,
    ], layout=widgets.Layout(border='1px solid #b3c6e0', padding='10px',
                              margin='4px 0', border_radius='6px'))

    _W.update({
        "tier_radio":            w_tier_radio,
        "recommendation_html":   w_recommendation_html,
        "custom_panel":          w_custom_panel,
        "custom_dropdowns":      custom_dropdowns,
        "refresh_recommendation":   _refresh_recommendation,
        "refresh_custom_dropdowns": _refresh_custom_dropdowns,
    })
    return panel


# ─── Prompt Enhancement panel ────────────────────────────────────────
def _build_enhance_panel():
    w_enhance_my_prompt = widgets.Checkbox(
        value=config.ENHANCE_NON_IMAGE_PROMPTS,
        description=('✨ Enhance my prompt before sending '
                     '(uses the Prompt LLM picked below; works for chat/docx/pptx/pdf/python/qa)'),
        layout=widgets.Layout(width='820px'), indent=False)

    def _prompt_llm_options():
        ranked = []
        for m in (config._INSTALLED_OLLAMA or []):
            score, stars, suffix = _rank_ollama_model(m)
            label = f"{stars}  Ollama: {m}{suffix}"
            ranked.append((score, label, m))
        ranked.sort(key=lambda x: -x[0])  # strongest first
        opts = [(label, m) for _, label, m in ranked]
        cli_label = "★★★★★  Claude Code (subprocess) — frontier-grade"
        if not _shutil.which("claude"):
            cli_label = "Claude Code (subprocess)  [⚠ not found on PATH]"
        opts.append((cli_label, "claudecode"))
        if not opts:
            opts.append(("(no models available)", ""))
        return opts

    opts = _prompt_llm_options()
    valid_values = [v for _, v in opts]
    initial = (config.PROMPT_LLM if config.PROMPT_LLM in valid_values
               else (valid_values[0] if valid_values else "qwen3:8b"))

    w_prompt_llm = widgets.Dropdown(
        options=opts, value=initial, description='Prompt LLM:',
        layout=widgets.Layout(width='720px'),
        style={'description_width': '120px'})

    w_prompt_llm_cc = widgets.Dropdown(
        options=list(config.CLAUDECODE_MODELS.items()),
        value=config.PROMPT_LLM_CLAUDECODE_MODEL,
        description='CC model:',
        layout=widgets.Layout(width='540px',
            display=('' if config.PROMPT_LLM == 'claudecode' else 'none')),
        style={'description_width': '120px'})

    w_prompt_llm_help = widgets.HTML(
        "<div style='font-size:11px;color:#666;margin-left:120px;line-height:1.5'>"
        "Which model handles 'Enhance my prompt'. Independent from the main provider above. "
        "Models are sorted strongest-first; ★ rating reflects suitability for prose / "
        "instruction-following (size + family + general-vs-specialist). "
        "Code-specialists (qwen2.5-coder) and tool-use specialists are penalized "
        "because they're tuned for code/tools, not prose rewriting."
        "</div>")

    def _on_enhance_change(change):
        config.ENHANCE_NON_IMAGE_PROMPTS = bool(change['new'])
        _aiconsole_print(f"Enhance my prompt → "
                         f"{'ON' if config.ENHANCE_NON_IMAGE_PROMPTS else 'OFF'}",
                         kind="settings")

    def _on_prompt_llm_change(change):
        config.PROMPT_LLM = change['new']
        w_prompt_llm_cc.layout.display = ('' if config.PROMPT_LLM == 'claudecode' else 'none')
        _aiconsole_print(f"Prompt LLM → {config.PROMPT_LLM}", kind="model")

    def _on_prompt_llm_cc_change(change):
        config.PROMPT_LLM_CLAUDECODE_MODEL = change['new']
        _aiconsole_print(f"Prompt LLM CC model → "
                         f"{config.PROMPT_LLM_CLAUDECODE_MODEL or 'CLI default'}",
                         kind="model")

    w_enhance_my_prompt.observe(_on_enhance_change, names='value')
    w_prompt_llm.observe(_on_prompt_llm_change, names='value')
    w_prompt_llm_cc.observe(_on_prompt_llm_cc_change, names='value')

    def _refresh_prompt_llm_dropdown():
        cur = w_prompt_llm.value
        new_opts = _prompt_llm_options()
        new_values = [v for _, v in new_opts]
        w_prompt_llm.options = new_opts
        if cur in new_values:
            w_prompt_llm.value = cur
        elif new_values:
            w_prompt_llm.value = new_values[0]
            config.PROMPT_LLM = w_prompt_llm.value

    panel = widgets.VBox([
        widgets.HTML("<b style='color:#1F3864;font-size:14px'>✨ Prompt Enhancement</b>"),
        w_enhance_my_prompt,
        w_prompt_llm,
        w_prompt_llm_cc,
        w_prompt_llm_help,
    ], layout=widgets.Layout(border='1px solid #b3c6e0', padding='10px',
                              margin='4px 0', border_radius='6px'))

    _W.update({
        "enhance_my_prompt":           w_enhance_my_prompt,
        "prompt_llm":                  w_prompt_llm,
        "prompt_llm_cc":               w_prompt_llm_cc,
        "refresh_prompt_llm_dropdown": _refresh_prompt_llm_dropdown,
    })
    return panel


# ─── File attach panel ───────────────────────────────────────────────
def _build_file_panel():
    w_browse_btn = widgets.Button(description='📂 Browse & Attach Files',
        button_style='info', layout=widgets.Layout(width='220px', height='36px'))
    w_clear_files_btn = widgets.Button(description='🗑 Clear All Files',
        button_style='warning', layout=widgets.Layout(width='160px', height='36px'))
    w_file_list = widgets.HTML("<i style='color:#888'>No files attached.</i>")

    def _refresh_list():
        if not attached_files:
            w_file_list.value = "<i style='color:#888'>No files attached.</i>"
            return
        rows = ""
        for i, fp in enumerate(attached_files):
            p = Path(fp)
            ex = p.exists()
            icon = "✅" if ex else "❌"
            color = "#1a7f37" if ex else "#cf222e"
            size = f"{p.stat().st_size/1024:.1f} KB" if ex else "not found"
            rows += (f"<tr><td style='padding:2px 6px;color:#888'>{i+1}</td>"
                     f"<td style='padding:2px 6px'>{icon}</td>"
                     f"<td style='padding:2px 6px;color:{color};"
                     f"font-family:monospace;font-size:12px'>{fp}</td>"
                     f"<td style='padding:2px 6px;color:#888;font-size:11px'>{size}</td></tr>")
        w_file_list.value = (
            "<table style='border-collapse:collapse'>"
            "<tr><th style='padding:2px 6px;font-size:11px;color:#888'>#</th><th></th>"
            "<th style='padding:2px 6px;text-align:left;font-size:11px;color:#888'>Path</th>"
            "<th style='padding:2px 6px;text-align:left;font-size:11px;color:#888'>Size</th></tr>"
            f"{rows}</table>"
            f"<span style='font-size:12px;color:#555'>"
            f"<b>{len(attached_files)}</b> file(s) attached</span>")

    def _on_browse(b):
        paths = file_picker.open_native_file_picker()
        if not paths:
            print("(no files selected)")
            return
        added = 0
        for p in paths:
            if len(attached_files) >= 20:
                print("⚠️  Max 20 files reached.")
                break
            if p not in attached_files:
                attached_files.append(p)
                added += 1
        print(f"➕ Added {added} file(s).  Total attached: {len(attached_files)}")
        _refresh_list()
        if "refresh_followup_files" in _W:
            _W["refresh_followup_files"]()

    def _on_clear_files(b):
        attached_files.clear()
        _refresh_list()
        if "refresh_followup_files" in _W:
            _W["refresh_followup_files"]()
        print("🗑 All files cleared.")

    w_browse_btn.on_click(_on_browse)
    w_clear_files_btn.on_click(_on_clear_files)

    panel = widgets.VBox([
        widgets.HTML("<b style='color:#1F3864;font-size:14px'>📎 Attached Files</b> "
                     "<span style='color:#888;font-size:11px'>"
                     "supports docx, pptx, pdf, csv, xlsx, txt, code; "
                     "up to 20, multi-select with Ctrl/Shift+click</span>"),
        widgets.HBox([w_browse_btn, w_clear_files_btn],
                     layout=widgets.Layout(gap='10px')),
        w_file_list,
    ], layout=widgets.Layout(border='1px solid #b3c6e0', padding='10px',
                              margin='4px 0', border_radius='6px'))

    _W.update({
        "browse_btn":      w_browse_btn,
        "clear_files_btn": w_clear_files_btn,
        "file_list":       w_file_list,
        "refresh_list":    _refresh_list,
    })
    return panel


# ─── Topic + buttons + preview ───────────────────────────────────────
def _build_topic_and_actions():
    w_topic = widgets.Textarea(
        placeholder=(
            'Type anything. Examples:\n'
            '  • "explain blockchain in simple terms"                 (→ chat)\n'
            '  • "create a 2000-word docx about AI in healthcare"     (→ Word doc)\n'
            '  • "generate a 3-page pdf summarizing the attached file"(→ PDF)\n'
            '  • "make an 8-slide pptx explaining the attached file"  (→ PowerPoint)\n'
            '  • "write a python script that scrapes news"            (→ .py file)\n'
            '  • "list functions in the attached file"                (→ Q&A about files)'
        ),
        description='📝 Topic / Question:',
        layout=widgets.Layout(width='820px', height='130px'),
        style={'description_width': '160px'})

    w_btn = widgets.Button(description='🚀 Generate Response',
        button_style='success',
        layout=widgets.Layout(width='230px', height='44px'))
    # F-12: cancel an in-progress local generation between LLM tokens
    w_cancel_gen_btn = widgets.Button(description='✖ Cancel Generation',
        button_style='danger',
        layout=widgets.Layout(width='180px', height='44px'))
    w_clear_conv_btn = widgets.Button(description='🧹 Clear Conversation',
        button_style='warning',
        layout=widgets.Layout(width='200px', height='44px'))

    w_aiconsole_label = widgets.HTML(
        "<div style='background:#1F3864;color:#fff;padding:6px 12px;"
        "border-radius:6px 6px 0 0;font-weight:bold;font-size:13px;margin-top:8px'>"
        "🖥️ aiconsole — operational log "
        "<span style='font-weight:normal;font-size:11px;color:#cce'>"
        "(settings, topic, enhanced prompts, generated outputs)</span></div>"
    )
    w_output = widgets.Output(layout=widgets.Layout(
        border='1px solid #1F3864', border_top='none',
        padding='10px', max_height='500px', overflow='auto',
        border_radius='0 0 6px 6px'))
    w_aiconsole = widgets.VBox([w_aiconsole_label, w_output])

    _display(HTML("""
    <style>
      .jupyter-widgets .widget-output pre, .jupyter-widgets .output_subarea pre,
      .widget-output pre {
        white-space: pre-wrap !important; word-wrap: break-word !important;
        overflow-wrap: anywhere !important;
      }
      .widget-output { word-wrap: break-word !important; }
    </style>
    """))

    w_preview_label    = widgets.HTML()
    w_preview_textarea = widgets.Textarea(
        placeholder='(enhanced prompt will appear here — edit if you want)',
        layout=widgets.Layout(width='820px', height='150px'))
    w_preview_send_btn   = widgets.Button(description='✅ Send', button_style='success',
        layout=widgets.Layout(width='180px', height='38px'))
    w_preview_cancel_btn = widgets.Button(description='✖ Cancel',
        layout=widgets.Layout(width='100px', height='38px'))
    w_preview_box = widgets.VBox([
        w_preview_label,
        w_preview_textarea,
        widgets.HBox([w_preview_send_btn, w_preview_cancel_btn],
                     layout=widgets.Layout(gap='8px', margin='4px 0')),
    ], layout=widgets.Layout(display='none', border='2px solid #1a7f37',
                              padding='12px', margin='8px 0', border_radius='6px'))

    w_followup_text = widgets.Textarea(
        placeholder="Type follow-up or answer the model's question...",
        layout=widgets.Layout(width='680px', height='65px'))
    w_followup_btn = widgets.Button(description='↩ Send Reply',
        button_style='info',
        layout=widgets.Layout(width='130px', height='65px'))
    w_followup_browse_btn = widgets.Button(description='📂 Attach Files',
        button_style='info',
        layout=widgets.Layout(width='160px', height='32px'))
    w_followup_clear_btn = widgets.Button(description='🗑 Clear',
        button_style='warning',
        layout=widgets.Layout(width='90px', height='32px'))
    w_followup_files_html = widgets.HTML(
        "<i style='color:#888;font-size:11px'>(no extra files for this follow-up)</i>")

    def _refresh_followup_files():
        if not attached_files:
            w_followup_files_html.value = (
                "<i style='color:#888;font-size:11px'>(no files attached)</i>")
            return
        items = ""
        for fp in attached_files:
            p = Path(fp)
            icon = "✅" if p.exists() else "❌"
            items += (f"<div style='font-size:11px;color:#555;font-family:monospace'>"
                      f"{icon} {fp}</div>")
        w_followup_files_html.value = (
            f"<div style='font-size:12px;color:#1F3864;margin-top:4px'>"
            f"<b>{len(attached_files)}</b> file(s) attached:</div>{items}")

    def _on_followup_browse(b):
        paths = file_picker.open_native_file_picker()
        if not paths:
            return
        added = 0
        for p in paths:
            if len(attached_files) >= 20:
                break
            if p not in attached_files:
                attached_files.append(p)
                added += 1
        print(f"➕ Added {added} file(s).  Total: {len(attached_files)}")
        _W["refresh_list"]()
        _refresh_followup_files()

    def _on_followup_clear(b):
        attached_files.clear()
        _W["refresh_list"]()
        _refresh_followup_files()
        print("🗑 All files cleared.")

    w_followup_browse_btn.on_click(_on_followup_browse)
    w_followup_clear_btn.on_click(_on_followup_clear)

    w_followup_box = widgets.VBox([
        widgets.HTML("<div style='margin-top:10px;border-top:2px solid #1F3864;"
                     "padding-top:8px'><b style='color:#1F3864'>🔁 Follow-up</b></div>"),
        widgets.HBox([w_followup_browse_btn, w_followup_clear_btn],
                     layout=widgets.Layout(gap='6px', margin='4px 0')),
        w_followup_files_html,
        widgets.HBox([w_followup_text, w_followup_btn],
                     layout=widgets.Layout(gap='8px', align_items='flex-start',
                                           margin='6px 0 0 0')),
    ], layout=widgets.Layout(display='none'))

    _W.update({
        "topic":                 w_topic,
        "btn":                   w_btn,
        "cancel_gen_btn":        w_cancel_gen_btn,
        "clear_conv_btn":        w_clear_conv_btn,
        "aiconsole":             w_aiconsole,
        "output":                w_output,
        "preview_box":           w_preview_box,
        "preview_label":         w_preview_label,
        "preview_textarea":      w_preview_textarea,
        "preview_send_btn":      w_preview_send_btn,
        "preview_cancel_btn":    w_preview_cancel_btn,
        "followup_box":          w_followup_box,
        "followup_text":         w_followup_text,
        "followup_btn":          w_followup_btn,
        "followup_browse_btn":   w_followup_browse_btn,
        "followup_clear_btn":    w_followup_clear_btn,
        "refresh_followup_files": _refresh_followup_files,
    })


def _scroll_output_to_bottom():
    _display(Javascript("""
        setTimeout(function(){
          var els = document.querySelectorAll('.jp-OutputArea-output, .widget-output, .output_subarea');
          els.forEach(function(el){ el.scrollTop = el.scrollHeight; });
        }, 50);
    """))


META_FILE_PATTERNS = [
    r'how many files', r'(?:list|show|name)\s+(?:the\s+)?(?:attached\s+)?files?',
    r'what files (?:did i|have i|are)', r'which files (?:did i|have i|are)',
    r'tell (?:me )?how many files',
]


def _is_meta_file_question(text):
    t = text.lower()
    return any(re.search(p, t) for p in META_FILE_PATTERNS)


def _answer_meta_file_question():
    n = len(attached_files)
    if n == 0:
        print("📎 You have no files attached. Use the Browse button to add some.")
        return
    print(f"📎 You attached **{n}** file(s):")
    for i, fp in enumerate(attached_files, 1):
        p = Path(fp)
        size = f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "NOT FOUND"
        print(f"  {i}. {p.name}    ({size})    [{fp}]")


def _run_intent(user_msg, intent, file_ctx):
    """Dispatch a request to the right builder."""
    print(f"⚙️  _run_intent (intent={intent!r})", flush=True)
    try:
        model, reason = intent_router.pick_model(intent)
    except KeyError as e:
        print(f"❌ pick_model({intent!r}) failed: KeyError {e}")
        return
    except Exception as e:
        print(f"❌ pick_model({intent!r}) failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return

    print(f"💡 Intent: {intent.upper()}  →  using {model} ({reason})")
    print(f"💡 Tier: {config.ACTIVE_TIER}  ·  (length determined by your prompt)")
    print("=" * 60)

    if intent == 'docx':
        path = builders_files.create_docx(user_msg, model=model, file_context=file_ctx)
        _aiconsole_print(f"Generated DOCX → {path}", kind="output")
    elif intent == 'pptx':
        path = builders_files.create_pptx(user_msg, model=model, file_context=file_ctx)
        _aiconsole_print(f"Generated PPTX → {path}", kind="output")
    elif intent == 'pdf':
        path = builders_files.create_pdf(user_msg, model=model, file_context=file_ctx)
        _aiconsole_print(f"Generated PDF → {path}", kind="output")
    elif intent == 'python':
        path = builders_files.create_python_script(user_msg, model=model, file_context=file_ctx)
        _aiconsole_print(f"Generated Python script → {path}", kind="output")
    elif intent == 'qa':
        names = [Path(fp).name for fp in attached_files]
        n = len(names)
        system = (f"You are an expert assistant. The user has attached {n} file(s): "
                  f"{', '.join(names) if names else '(none)'}.\n"
                  "Answer questions using the file contents below.")
        if file_ctx:
            full = (f"Attached files ({n}): {', '.join(names)}\n\n"
                    f"File contents:\n{file_ctx}\n\n---\nUser question: {user_msg}")
        else:
            full = user_msg
        conversation.chat_turn(full, model=model, system=system, stream_print=True)
    else:
        conversation.chat_turn(user_msg, model=model, stream_print=True)


def _show_preview(short_prompt, enhanced, intent, file_ctx):
    PENDING_GEN.clear()
    PENDING_GEN.update({
        "short":    short_prompt,
        "intent":   intent,
        "file_ctx": file_ctx,
    })
    _W["preview_label"].value = (
        f"<div style='color:#1a7f37;font-weight:bold;font-size:13px'>"
        f"📝 Preview enhanced prompt (intent: {intent})</div>"
        f"<div style='font-size:11px;color:#666'>Edit if you want, then click Send.</div>"
    )
    _W["preview_textarea"].value = enhanced
    _W["preview_box"].layout.display = ''


def _on_preview_cancel(b):
    PENDING_GEN.clear()
    _W["preview_box"].layout.display = 'none'
    with _W["output"]:
        print("\n✖ Cancelled.")


def _on_preview_send(b):
    if not PENDING_GEN:
        return
    final_prompt = _W["preview_textarea"].value.strip()
    if not final_prompt:
        with _W["output"]:
            print("⚠️  Empty prompt — please type something or cancel.")
        return
    pending = dict(PENDING_GEN)
    PENDING_GEN.clear()
    _W["preview_box"].layout.display = 'none'

    # F-12: clear any stale cancel flag before starting
    conversation.reset_cancel()
    with _W["output"]:
        try:
            _aiconsole_print(
                f"Sending enhanced prompt to {pending['intent']} pipeline …",
                kind="info")
            _run_intent(final_prompt, pending['intent'], pending.get('file_ctx', ''))
        except conversation.CancelledError as e:
            print(f"\n✖ {e}")
        except Exception as e:
            print(f"\n❌ ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
    _scroll_output_to_bottom()


def _on_cancel_generation(b):
    """F-12: signal any in-flight LLM call to stop between tokens."""
    conversation.request_cancel()
    with _W["output"]:
        print("\n⚠️  Cancel requested. Generation will stop between tokens.")


def _on_generate(b):
    topic = _W["topic"].value.strip()
    if not topic:
        with _W["output"]:
            clear_output(wait=True)
            print("⚠️  Please enter a topic or question.")
        return

    # F-12: clear any stale cancel flag from a prior aborted run
    conversation.reset_cancel()
    _W["preview_box"].layout.display = 'none'
    PENDING_GEN.clear()

    with _W["output"]:
        clear_output(wait=True)
        if _is_meta_file_question(topic):
            _answer_meta_file_question()
            _W["followup_box"].layout.display = ''
            _W["refresh_followup_files"]()
            _scroll_output_to_bottom()
            return

        file_ctx = ""
        if attached_files:
            print(f"📎 Loading {len(attached_files)} file(s)...")
            file_ctx = file_readers.build_file_context(attached_files)

        intent = intent_router.detect_intent(topic, has_files=bool(attached_files))
        _aiconsole_print(f"Topic → {topic}", kind="topic")
        _aiconsole_print(f"Provider: {config.PROVIDER.upper()}  ·  "
                         f"Intent: {intent.upper()}  ·  "
                         f"Tier: {config.ACTIVE_TIER}",
                         kind="info")

        is_read, fname = file_readers.detect_read_request(topic)
        if is_read:
            print(f"🔍 Looking for '{fname}'...")
            p = file_readers.find_local_file(fname)
            if p is None:
                print(f"\n❌ Cannot find '{fname}'.")
                try:
                    found = sorted(f.name for f in config.WORK_DIR.iterdir()
                                   if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTS)
                    if found:
                        print("   Files here:", ", ".join(found))
                except Exception:
                    pass
            else:
                content = file_readers.read_file_by_path(p)
                print(f"\n📄 Loaded '{fname}'  ({len(content):,} chars)\n" + "=" * 60)
                model, _ = intent_router.pick_model('qa')
                conversation.chat_turn(
                    f"User asked about '{fname}'. Full content:\n{content[:8000]}\n\n"
                    f"User request: {topic}",
                    model=model, stream_print=True)
            _W["followup_box"].layout.display = ''
            _W["refresh_followup_files"]()
            _scroll_output_to_bottom()
            return

        try:
            if (config.ENHANCE_NON_IMAGE_PROMPTS
                    and intent in ('chat', 'docx', 'pptx', 'pdf', 'python', 'qa')
                    and not _is_meta_file_question(topic)):
                _aiconsole_print(f"Enhancing prompt for intent='{intent}' …", kind="info")
                enhanced, _ = enhancement.enhance_only_generic(topic, intent)
                if enhanced and enhanced != topic:
                    _aiconsole_print(f"Enhanced prompt: {enhanced}", kind="enhanced")
                    _show_preview(topic, enhanced, intent, file_ctx)
                else:
                    _aiconsole_print("Enhancement returned no change; running with original prompt.",
                                     kind="info")
                    _run_intent(topic, intent, file_ctx)
            else:
                _run_intent(topic, intent, file_ctx)
        except conversation.CancelledError as e:
            print(f"\n✖ {e}")

        _W["followup_box"].layout.display = ''
        _W["refresh_followup_files"]()
        _scroll_output_to_bottom()


def _on_followup(b):
    reply = _W["followup_text"].value.strip()
    if not reply:
        return
    _W["preview_box"].layout.display = 'none'
    PENDING_GEN.clear()

    # F-12: clear any stale cancel flag before starting
    conversation.reset_cancel()

    file_ctx = ""
    if attached_files:
        file_ctx = file_readers.build_file_context(attached_files)
    intent = intent_router.detect_intent(reply, has_files=bool(attached_files))

    with _W["output"]:
        print(f"\n{'─'*60}\n👤 You: {reply}\n" + "─" * 60)
        try:
            if intent in ('docx', 'pptx', 'pdf', 'python'):
                _run_intent(reply, intent, file_ctx)
            else:
                model, _ = intent_router.pick_model('qa' if attached_files else 'chat')
                msg = (f"[Attached files]:\n{file_ctx[:4000]}\n\n{reply}"
                       if file_ctx and len(conversation.conversation_history) <= 2
                       else reply)
                system = "You are a helpful assistant. Continue the conversation."
                conversation.chat_turn(msg, model=model, system=system, stream_print=True)
        except conversation.CancelledError as e:
            print(f"\n✖ {e}")
    _W["followup_text"].value = ""
    _scroll_output_to_bottom()


def _on_clear_conv(b):
    conversation.clear_history()
    PENDING_GEN.clear()
    _W["preview_box"].layout.display = 'none'
    with _W["output"]:
        clear_output(wait=True)
        print("🧹 Conversation cleared. Type a new topic above.")
    _W["followup_box"].layout.display = 'none'
    _W["followup_text"].value = ''


def _install_log_wrappers():
    logger.log_write("")
    logger.log_section("WRAPPER INSTALLATION (ui)")

    logger.install_log_wrapper(intent_router, "detect_intent",
                                arg_names=["text", "has_files"],
                                return_name="intent")
    logger.install_log_wrapper(intent_router, "pick_model",
                                arg_names=["intent"],
                                return_name="model_and_reason")

    logger.install_log_wrapper(tier_manager, "verify_tier_models",
                                return_name="availability")
    logger.install_log_wrapper(tier_manager, "recommend_tier",
                                return_name="recommendation")

    logger.install_log_wrapper(builders_structure_module(), "build_docx_structure",
                                arg_names=["task", "analysis", "model", "n_chart_sections"],
                                capture_return=False)
    logger.install_log_wrapper(builders_structure_module(), "build_pptx_structure",
                                arg_names=["task", "analysis", "model", "n_chart_slides"],
                                capture_return=False)
    logger.install_log_wrapper(builders_structure_module(), "build_pdf_structure",
                                arg_names=["task", "analysis", "model", "n_chart_sections"],
                                capture_return=False)
    logger.install_log_wrapper(builders_structure_module(), "build_python_code",
                                arg_names=["task", "analysis", "model"],
                                capture_return=False)
    logger.install_log_wrapper(builders_structure_module(), "_call_planner",
                                arg_names=["prompt", "system"],
                                capture_return=False)
    logger.install_log_wrapper(builders_structure_module(), "analyze_files_for_task",
                                arg_names=["task", "file_context", "model"],
                                capture_return=False)

    # F-01: wrap the render/save functions so exceptions in reportlab,
    # python-docx, python-pptx, or matplotlib chart insertion are logged
    # instead of vanishing into widget output. Previously TC6_local and
    # TC12_local hit failures here with no log trace.
    logger.install_log_wrapper(builders_files, "create_docx",
                                arg_names=["topic", "model", "file_context", "n_charts"],
                                capture_return=False)
    logger.install_log_wrapper(builders_files, "create_pptx",
                                arg_names=["topic", "model", "file_context", "n_charts"],
                                capture_return=False)
    logger.install_log_wrapper(builders_files, "create_pdf",
                                arg_names=["topic", "model", "file_context", "n_charts"],
                                capture_return=False)
    logger.install_log_wrapper(builders_files, "create_python_script",
                                arg_names=["description", "model", "file_context"],
                                capture_return=False)

    logger.install_log_wrapper(enhancement, "enhance_only_generic",
                                arg_names=["short_prompt", "intent"],
                                return_name="enhanced_and_intent")
    logger.install_log_wrapper(enhancement, "_call_prompt_llm",
                                arg_names=["system", "user_msg", "label"],
                                capture_return=False)

    button_log_map = [
        (_W["btn"],                  "GENERATE"),
        (_W["cancel_gen_btn"],       "CANCEL_GENERATION"),
        (_W["clear_conv_btn"],       "CLEAR_CONVERSATION"),
        (_W["browse_btn"],           "BROWSE_FILES"),
        (_W["clear_files_btn"],      "CLEAR_FILES"),
        (_W["followup_btn"],         "SEND_REPLY"),
        (_W["followup_browse_btn"],  "FOLLOWUP_BROWSE"),
        (_W["followup_clear_btn"],   "FOLLOWUP_CLEAR"),
        (_W["preview_send_btn"],     "PREVIEW_SEND"),
        (_W["preview_cancel_btn"],   "PREVIEW_CANCEL"),
        (_W["test_cli_btn"],         "TEST_CLAUDE_CLI"),
        (_W["diag_btn"],             "DIAGNOSE"),
    ]
    for btn, name in button_log_map:
        try:
            _wrap_button(btn, name)
        except Exception as _e:
            logger.log_write(f"  Could not wrap {name}: {_e}", prefix="⚠")

    logger.log_write("")
    logger.log_write("✅ Logging instrumentation complete. All future events recorded.",
                     prefix=" ")
    print(f"📋 logging active → {logger.LOG_FILE}")


def builders_structure_module():
    from . import builders_structure
    return builders_structure


def _wrap_button(btn, handler_name):
    if not hasattr(btn, "_click_handlers") or btn._click_handlers is None:
        return
    handlers = (list(btn._click_handlers.callbacks)
                if hasattr(btn._click_handlers, "callbacks") else [])
    if not handlers:
        return
    if any(getattr(h, "_logged", False) for h in handlers):
        return
    btn._click_handlers.callbacks = []
    for h in handlers:
        def make_logged(orig_h, name):
            def logged(b):
                logger.log_event(f"CLICK_{name}")
                try:
                    return orig_h(b)
                except Exception:
                    logger.log_exception(f"button:{name}")
                    raise
            logged._logged = True
            logged.__wrapped__ = orig_h
            return logged
        btn._click_handlers.callbacks.append(make_logged(h, handler_name))


def render_ui():
    """Build and display the full UI. Single call from the notebook."""
    provider_panel = _build_provider_panel()
    tier_panel     = _build_tier_panel()
    enhance_panel  = _build_enhance_panel()
    file_panel     = _build_file_panel()
    _build_topic_and_actions()

    _W["btn"].on_click(_on_generate)
    _W["cancel_gen_btn"].on_click(_on_cancel_generation)
    _W["clear_conv_btn"].on_click(_on_clear_conv)
    _W["preview_send_btn"].on_click(_on_preview_send)
    _W["preview_cancel_btn"].on_click(_on_preview_cancel)
    _W["followup_btn"].on_click(_on_followup)

    _W["refresh_provider_status"]()

    display(
        widgets.HTML("""
        <h2 style='color:#1F3864;margin-bottom:2px'>
          🤖 ai_text — Personal AI Workspace (Text Edition)
        </h2>
        <p style='color:#555;margin-top:0;font-size:13px'>
          Local Ollama + Claude Code. Generates DOCX, PPTX, PDF, and Python files.
          Reads DOCX, PPTX, PDF, CSV, XLSX, and plain text. <b>Output length is
          driven by your prompt</b> — say "write a 2000-word essay" or
          "create a 3-page PDF" and the model targets that. Successor to ai_text_v1.
        </p>
        """),
        provider_panel,
        tier_panel,
        enhance_panel,
        _W["topic"],
        file_panel,
        widgets.HBox([_W["btn"], _W["cancel_gen_btn"], _W["clear_conv_btn"]],
                     layout=widgets.Layout(gap='10px')),
        _W["preview_box"],
        _W["aiconsole"],
        _W["followup_box"],
    )

    _install_log_wrappers()
