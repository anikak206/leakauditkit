"""Report generation: JSON, Markdown, and self-contained HTML scorecards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LeakAudit Report</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; color: #222; }
h1 { border-bottom: 2px solid #333; }
.card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
.metric { display: inline-block; margin-right: 2rem; }
.metric .val { font-size: 2rem; font-weight: 700; }
.bad { color: #c0392b; }
.warn { color: #d68910; }
.good { color: #1e8449; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 0.9rem; }
th { background: #f4f4f4; }
code { background: #eee; padding: 2px 4px; border-radius: 3px; }
</style>
</head>
<body>
<h1>LeakAudit Report</h1>
<div class="card">
  <div class="metric"><div>Train items</div><div class="val">__N_TRAIN__</div></div>
  <div class="metric"><div>Eval items</div><div class="val">__N_EVAL__</div></div>
  <div class="metric"><div>Contaminated eval</div><div class="val __RATE_CLS__">__N_CONTAM__</div></div>
  <div class="metric"><div>Contamination rate</div><div class="val __RATE_CLS__">__RATE_PCT__</div></div>
</div>
<h2>Detector: __DETECTOR__</h2>
<p>Threshold: __THRESHOLD__ &nbsp;|&nbsp; Matches: __N_MATCHES__</p>
<h3>Top matches</h3>
<table>
<tr><th>Eval ID</th><th>Train ID</th><th>Score</th><th>Evidence</th></tr>
__ROWS__
</table>
</body>
</html>
"""


def write_json(result: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict() if hasattr(result, "to_dict") else result,
                              indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_markdown(result: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = result.to_dict() if hasattr(result, "to_dict") else result
    lines = [
        f"# LeakAudit Report — {d['detector']}",
        "",
        f"- Train items: **{d['n_train']}**",
        f"- Eval items: **{d['n_eval']}**",
        f"- Matches: **{d['n_matches']}**",
        f"- Contamination rate: **{d['contamination_rate']:.2%}**",
        f"- Threshold: {d['threshold']}",
        "",
        "## Top matches",
        "",
        "| Eval ID | Train ID | Score | Evidence |",
        "|---|---|---|---|",
    ]
    for m in d.get("matches", [])[:50]:
        ev = (m.get("evidence") or "").replace("|", "\\|")[:80]
        lines.append(f"| {m['eval_id']} | {m['train_id']} | {m['score']} | {ev} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_html(result: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = result.to_dict() if hasattr(result, "to_dict") else result
    rate = d.get("contamination_rate", 0.0)
    if rate >= 0.2:
        rate_cls = "bad"
    elif rate >= 0.05:
        rate_cls = "warn"
    else:
        rate_cls = "good"
    rows = []
    for m in d.get("matches", [])[:100]:
        ev = (m.get("evidence") or "")[:120]
        rows.append(
            f"<tr><td>{m['eval_id']}</td><td>{m['train_id']}</td>"
            f"<td>{m['score']}</td><td><code>{ev}</code></td></tr>"
        )
    # try interactive plotly report
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        scores = [m["score"] for m in d.get("matches", [])[:50]]
        labels = [m["eval_id"] for m in d.get("matches", [])[:50]]
        fig = go.Figure(data=[go.Bar(x=labels, y=scores, marker_color="#c0392b")])
        fig.update_layout(title=f"{d['detector']} — match scores",
                          xaxis_title="eval id", yaxis_title="score",
                          template="plotly_white")
        chart = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
        html = (_HTML_TEMPLATE
                .replace("__N_TRAIN__", str(d["n_train"]))
                .replace("__N_EVAL__", str(d["n_eval"]))
                .replace("__N_CONTAM__", str(len({m["eval_id"] for m in d.get("matches", [])})))
                .replace("__RATE_PCT__", f"{rate:.1%}")
                .replace("__RATE_CLS__", rate_cls)
                .replace("__DETECTOR__", str(d["detector"]))
                .replace("__THRESHOLD__", str(d["threshold"]))
                .replace("__N_MATCHES__", str(d["n_matches"]))
                .replace("__ROWS__", "\n".join(rows) if rows else "<tr><td colspan='4'>No matches above threshold.</td></tr>"))
        html = html.replace("</body>", chart + "\n</body>")
        path.write_text(html, encoding="utf-8")
        return path
    except Exception:
        pass
    html = (_HTML_TEMPLATE
            .replace("__N_TRAIN__", str(d["n_train"]))
            .replace("__N_EVAL__", str(d["n_eval"]))
            .replace("__N_CONTAM__", str(len({m["eval_id"] for m in d.get("matches", [])})))
            .replace("__RATE_PCT__", f"{rate:.1%}")
            .replace("__RATE_CLS__", rate_cls)
            .replace("__DETECTOR__", str(d["detector"]))
            .replace("__THRESHOLD__", str(d["threshold"]))
            .replace("__N_MATCHES__", str(d["n_matches"]))
            .replace("__ROWS__", "\n".join(rows) if rows else "<tr><td colspan='4'>No matches above threshold.</td></tr>"))
    path.write_text(html, encoding="utf-8")
    return path