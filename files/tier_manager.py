"""ai_text_files.tier_manager — verify tier availability + recommend.

Reads what's installed in Ollama, checks each tier in config.MODEL_TIERS
to see if all its models are present, and recommends the highest tier
that's fully available.

Public API:
    verify_tier_models() → dict {tier: {"all_present": bool,
                                         "missing": [model_names]}}
    recommend_tier()     → (recommended_tier, message_str)
    print_tier_status()  → console summary for the UI banner
    pull_command_for(tier) → "ollama pull X Y Z" string
"""

from __future__ import annotations

from . import config


def verify_tier_models():
    """Check each tier against installed Ollama models.

    Returns dict keyed by tier name with:
        all_present (bool)
        missing     (list[str])
        needed      (list[str])  — every model the tier requires
    """
    # Refresh in case models were pulled during the session
    installed = set(config._list_installed_ollama())
    config._INSTALLED_OLLAMA = sorted(installed)
    out = {}
    for tier_name, models in config.MODEL_TIERS.items():
        if tier_name == "custom":
            # Custom is always "user's responsibility"
            out[tier_name] = {
                "all_present": True,
                "missing":     [],
                "needed":      sorted(set(models.values())),
            }
            continue
        needed = set(models.values())
        missing = sorted(needed - installed)
        out[tier_name] = {
            "all_present": (len(missing) == 0),
            "missing":     missing,
            "needed":      sorted(needed),
        }
    return out


def recommend_tier():
    """Pick the highest fully-available tier and return (tier, message).

    Priority order: best > balanced > fast > custom (always last because
    its membership depends on user choice).
    """
    avail = verify_tier_models()
    priority = ["best", "balanced", "fast"]
    for tier in priority:
        if avail[tier]["all_present"]:
            return tier, f"'{tier}' tier — all required models present"
    # Nothing fully available: recommend whatever has the fewest missing
    fallback = min(priority, key=lambda t: len(avail[t]["missing"]))
    miss = avail[fallback]["missing"]
    return fallback, (f"no tier fully available; '{fallback}' is closest "
                      f"(missing: {', '.join(miss)})")


def pull_command_for(tier):
    """Return the `ollama pull` command that would unlock a tier.
    Returns empty string if the tier is already complete or is 'custom'."""
    avail = verify_tier_models()
    if tier == "custom":
        return ""
    info = avail.get(tier, {})
    if info.get("all_present", False):
        return ""
    miss = info.get("missing", [])
    if not miss:
        return ""
    return "ollama pull " + " ".join(miss)


def print_tier_status():
    """Print a multi-line tier-availability summary. Returns the same
    info as a string for embedding in UI HTML widgets."""
    avail = verify_tier_models()
    lines = []
    lines.append("🎯 Tier availability:")
    for tier in ("fast", "balanced", "best"):
        info = avail[tier]
        if info["all_present"]:
            lines.append(f"   {tier:<10} ✅ all {len(info['needed'])} model(s) present")
        else:
            lines.append(f"   {tier:<10} ⚠️  missing: {', '.join(info['missing'])}")
    rec, msg = recommend_tier()
    lines.append("")
    lines.append(f"💡 Recommendation: {msg}")
    for tier in ("balanced", "best"):
        cmd = pull_command_for(tier)
        if cmd:
            lines.append(f"   To unlock '{tier}':  {cmd}")
    output = "\n".join(lines)
    print(output)
    return output, rec


def apply_recommendation():
    """Set config.ACTIVE_TIER to the recommended tier. Returns the tier
    name that was applied."""
    tier, msg = recommend_tier()
    config.ACTIVE_TIER = tier
    return tier
