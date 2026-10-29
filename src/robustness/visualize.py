"""Standalone, escaped attribution HTML for technical presentations."""

from __future__ import annotations

import html
import json


def attribution_html(explanation: dict) -> str:
    text = explanation["text"]
    words = []
    position = 0
    scale = max((abs(token["score"]) for token in explanation["tokens"]), default=1)
    scale = max(scale, 1e-12)
    rows = []
    for token in explanation["tokens"]:
        words.append(html.escape(text[position : token["start"]]))
        colour = "34,197,94" if token["score"] >= 0 else "244,114,182"
        opacity = (
            0 if token["score"] == 0 else 0.12 + 0.45 * abs(token["score"]) / scale
        )
        value = f"{100 * token['score']:+.3f} percentage points"
        words.append(
            f'<mark style="background:rgba({colour},{opacity:.3f})" '
            f'title="{value}">{html.escape(token["text"])}</mark>'
        )
        rows.append(f"<tr><td>{html.escape(token['text'])}</td><td>{value}</td></tr>")
        position = token["end"]
    words.append(html.escape(text[position:]))
    metadata = html.escape(json.dumps(explanation["model"], indent=2))
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>DarkPatternGuard Robustness evaluation attribution</title>"
        "<style>body{font:17px system-ui;max-width:900px;margin:40px auto;padding:20px;"
        "color:#182238}article{white-space:pre-wrap;line-height:2;padding:20px;"
        "border:1px solid #ddd}mark{color:inherit}td{padding:6px 16px}pre{"
        "white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<h1>Robustness evaluation — model sensitivity</h1><p>"
        + html.escape(explanation["label"])
        + f" · {100 * explanation['confidence']:.2f}% model confidence</p>"
        + "<article>"
        + "".join(words)
        + "</article>"
        + "<p>Green supports the predicted class; pink opposes it. "
        "Intensity is relative within this text.</p><p>"
        + html.escape(explanation["description"])
        + "</p><p>"
        + html.escape(explanation["truncation_warning"])
        + "</p>"
        + "<table><tr><th>Word</th><th>Signed contribution</th></tr>"
        + "".join(rows)
        + "</table><h2>Actual model identity</h2><pre>"
        + metadata
        + "</pre></html>"
    )
