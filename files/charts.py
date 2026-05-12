"""ai_files_v1.charts — chart-type picker and matplotlib chart generator.

Direct port of ai_my_v4.1 cell 9. No logic changes.
"""

from __future__ import annotations

from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import config, conversation


def ai_pick_charts(topic, model, n_charts=1):
    """Pick chart types for a topic. Returns at most n_charts entries."""
    if n_charts <= 0:
        return []
    system = ('Data viz expert. Decide ' + str(n_charts) + ' chart(s). '
              'pie=proportions, line=trends, bar=comparisons, scatter=correlations. '
              'Reply ONLY with JSON array: '
              '[{"chart_type":"bar"|"line"|"pie"|"scatter","topic":str},...]')
    result = conversation.call_json(
        f"Best {n_charts} chart(s) for: {topic}", model=model, system=system)
    charts = result if isinstance(result, list) else result.get(
        "charts", [{"chart_type": "bar", "topic": topic}])
    valid = {"bar", "line", "pie", "scatter"}
    cleaned = [
        {"chart_type": (c.get("chart_type", "bar").lower()
                        if c.get("chart_type", "bar").lower() in valid else "bar"),
         "topic":      c.get("topic", topic)}
        for c in charts[:n_charts]
    ]
    while len(cleaned) < n_charts:
        cleaned.append({"chart_type": "bar", "topic": topic})
    print(f"  🤖 AI chose {len(cleaned)} chart(s): {[c['chart_type'] for c in cleaned]}")
    return cleaned


def generate_chart(topic, chart_type="bar", save_path=None):
    system = ('Reply ONLY with JSON. '
              'Schema:{"title":str,"xlabel":str,"ylabel":str,'
              '"labels":[str,...],"values":[number,...],'
              '"chart_type":"bar"|"line"|"pie"|"scatter"}')
    data = conversation.call_json(
        f"Realistic data for {chart_type} chart about: {topic}. 5-7 points.",
        model="mistral:latest", system=system)
    labels = data.get("labels", ["A", "B", "C", "D", "E"])
    values = data.get("values", [10, 20, 15, 30, 25])
    ctype  = data.get("chart_type", chart_type)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    C = plt.cm.Set2.colors
    if ctype == "bar":
        bars = ax.bar(labels, values, color=C[:len(labels)])
        ax.bar_label(bars, fmt='%.1f', padding=3)
        ax.set_xlabel(data.get("xlabel", ""))
        ax.set_ylabel(data.get("ylabel", ""))
    elif ctype == "line":
        ax.plot(labels, values, marker='o', linewidth=2, color=C[0])
        ax.fill_between(labels, values, alpha=0.15, color=C[0])
        ax.set_xlabel(data.get("xlabel", ""))
        ax.set_ylabel(data.get("ylabel", ""))
    elif ctype == "pie":
        ax.pie(values, labels=labels, autopct='%1.1f%%',
               colors=C[:len(labels)], startangle=90)
    elif ctype == "scatter":
        ax.scatter(np.linspace(1, len(values), len(values)),
                   values, s=100, color=C[:len(values)])
        ax.set_xlabel(data.get("xlabel", ""))
        ax.set_ylabel(data.get("ylabel", ""))
    ax.set_title(data.get("title", topic), fontsize=13, fontweight='bold')
    fig.tight_layout()
    if save_path is None:
        save_path = str(config.OUTPUT_DIR / f"chart_{datetime.now().strftime('%H%M%S%f')}.png")
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Chart → {save_path}")
    return save_path
