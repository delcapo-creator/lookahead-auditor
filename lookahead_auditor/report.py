"""Text, JSON and self-contained HTML reports."""
from __future__ import annotations

import html
import json
import math
from typing import Dict, List

import numpy as np

from .audit import AuditResult

ROWS = [
    ("trades", "Trades", "{:.0f}"),
    ("net", "Net P&L (price units x volume)", "{:,.2f}"),
    ("profit_factor", "Profit factor", "{:.2f}"),
    ("win_rate", "Win rate", "{:.1%}"),
    ("payoff", "Avg win / avg loss", "{:.2f}"),
    ("expectancy", "Expectancy per trade", "{:,.4f}"),
    ("t_stat", "t-stat of mean", "{:.2f}"),
    ("max_drawdown", "Max drawdown", "{:,.2f}"),
    ("avg_bars", "Avg bars in trade", "{:.2f}"),
]


def _fmt(v, f):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return f.format(v)


def to_text(res: AuditResult) -> str:
    m = res.meta
    out: List[str] = []
    out.append("=" * 72)
    out.append(f"LOOKAHEAD AUDIT  ->  {res.verdict}")
    out.append("=" * 72)
    out.append(f"Trades: {m['n_trades']} (mapped to bars: {m['n_mapped']}, dropped rows: {m['dropped_rows']})")
    out.append(f"Bars: {m['n_bars']} @ {m['bar_interval']}  [{m['price_start']} .. {m['price_end']}]")
    out.append(f"Price tolerance: {m['tolerance']:.5g}   Cost/trade: {m['cost']:.5g}")
    mix = m["entry_fill_mix"]
    out.append(f"Entry fills: at open {mix['open']}, mid-bar {mix['mid']}, at close {mix['close']}, open~close {mix['ambiguous']}")
    out.append("")
    out.append(f"{'Metric':<34}{'Original':>16}{'Corrected':>16}")
    out.append("-" * 66)
    for key, label, f in ROWS:
        out.append(f"{label:<34}{_fmt(res.original[key], f):>16}{_fmt(res.corrected[key], f):>16}")
    out.append("")
    if not res.findings:
        out.append("No findings.")
    for fd in res.findings:
        out.append(f"[{fd.severity}] {fd.title}  -  {fd.count} trades ({fd.rate:.1%})")
        out.append(f"        {fd.explanation}")
        if fd.examples:
            out.append(f"        example trade rows: {', '.join(str(int(x)) for x in fd.examples)}")
    out.append("")
    out.append("Corrected = pessimistic re-simulation from OHLC: fills after the signal bar, SL wins")
    out.append("whenever SL and TP share a bar, gaps fill at the open, impossible prices clamped.")
    return "\n".join(out)


def to_json(res: AuditResult) -> str:
    def clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return clean(float(v))
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v

    payload = {
        "verdict": res.verdict,
        "verdict_level": res.verdict_level,
        "meta": res.meta,
        "original": res.original,
        "corrected": res.corrected,
        "findings": [
            {"code": f.code, "severity": f.severity, "title": f.title, "count": f.count,
             "rate": f.rate, "examples": [int(x) for x in f.examples]}
            for f in res.findings
        ],
    }
    return json.dumps(clean(payload), indent=2, default=str)


def _equity_svg(a: np.ndarray, b: np.ndarray, w=860, h=260, pad=36) -> str:
    ea, eb = np.concatenate([[0], np.cumsum(a)]), np.concatenate([[0], np.cumsum(b)])
    lo, hi = float(min(ea.min(), eb.min())), float(max(ea.max(), eb.max()))
    if hi == lo:
        hi = lo + 1
    n = max(len(ea) - 1, 1)

    def pts(e):
        return " ".join(
            f"{pad + (w - 2 * pad) * i / n:.1f},{h - pad - (h - 2 * pad) * (v - lo) / (hi - lo):.1f}"
            for i, v in enumerate(e)
        )

    zero_y = h - pad - (h - 2 * pad) * (0 - lo) / (hi - lo)
    return f"""<svg viewBox="0 0 {w} {h}" role="img" aria-label="Equity curves" class="eq">
<line x1="{pad}" x2="{w - pad}" y1="{zero_y:.1f}" y2="{zero_y:.1f}" class="zero"/>
<polyline points="{pts(ea)}" class="orig"/>
<polyline points="{pts(eb)}" class="corr"/>
<text x="{pad}" y="{pad - 12}" class="lbl">{hi:,.0f}</text>
<text x="{pad}" y="{h - 10}" class="lbl">{lo:,.0f}</text>
<text x="{w - pad}" y="{h - 10}" class="lbl" text-anchor="end">trade #{n}</text>
</svg>"""


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e4e3de;--err:#b3261e;--warn:#9a6700;--info:#3d5a80;--ok:#1f7a4d;--orig:#9a9a94;--corr:#2f5fd0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--fg:#ecebe6;--mut:#a2a19b;--card:#1f1f1d;--line:#34332f;--err:#f2867e;--warn:#e3b341;--info:#8fb3e6;--ok:#5cc28f;--orig:#77766f;--corr:#7ea2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:920px;margin:0 auto;padding:28px 16px 60px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 10px}
.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}
.verdict{font-weight:650;font-size:17px}.l0{color:var(--ok)}.l1{color:var(--warn)}.l2{color:var(--err)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right}
td:first-child,th:first-child{text-align:left}th{font-weight:600;color:var(--mut);font-size:13px}
.sev{font-size:12px;font-weight:700;letter-spacing:.04em;padding:2px 7px;border-radius:5px;border:1px solid currentColor;margin-right:8px}
.ERROR{color:var(--err)}.WARN{color:var(--warn)}.INFO{color:var(--info)}
.eq{width:100%;height:auto}.eq polyline{fill:none;stroke-width:1.8}.eq .orig{stroke:var(--orig)}.eq .corr{stroke:var(--corr)}
.eq .zero{stroke:var(--line)}.eq .lbl{fill:var(--mut);font-size:11px}.legend span{display:inline-block;width:14px;height:3px;vertical-align:middle;margin:0 6px 0 14px}
.scroll{overflow-x:auto}code{font-size:13px}
"""


def to_html(res: AuditResult, title: str = "Lookahead Audit") -> str:
    m = res.meta
    e = html.escape
    rows = "".join(
        f"<tr><td>{e(label)}</td><td>{_fmt(res.original[k], f)}</td><td>{_fmt(res.corrected[k], f)}</td></tr>"
        for k, label, f in ROWS
    )
    finds = "".join(
        f"""<div class="card"><div><span class="sev {fd.severity}">{fd.severity}</span><b>{e(fd.title)}</b>
<span class="mut"> &middot; {fd.count} trades ({fd.rate:.1%})</span></div><p>{e(fd.explanation)}</p>
{f'<p class="mut">Example rows: <code>{", ".join(str(int(x)) for x in fd.examples)}</code></p>' if fd.examples else ''}</div>"""
        for fd in res.findings
    ) or '<div class="card">No findings.</div>'

    t = res.trades
    flagged = t[t["flags"] != ""].head(40)
    cols = ["entry_time", "direction", "entry_price", "exit_price", "sl", "tp", "entry_fill",
            "corr_exit_price", "corr_reason", "pnl_calc", "corr_pnl", "flags"]
    thead = "".join(f"<th>{c}</th>" for c in ["row"] + cols)
    tbody = ""
    for idx, r in flagged.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                v = "" if math.isnan(v) else f"{v:,.4f}".rstrip("0").rstrip(".")
            cells.append(f"<td>{e(str(v))}</td>")
        tbody += f"<tr><td>{idx}</td>{''.join(cells)}</tr>"
    mix = m["entry_fill_mix"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title><style>{CSS}</style></head>
<body><main>
<h1>{e(title)}</h1>
<div class="mut">{m['n_trades']} trades &middot; {m['n_bars']} bars @ {e(m['bar_interval'])} &middot; {e(m['price_start'])} &rarr; {e(m['price_end'])}</div>
<div class="card"><div class="verdict l{res.verdict_level}">{e(res.verdict)}</div>
<div class="mut">Trades with an ERROR flag: {m['error_trade_rate']:.1%} &middot; Profit-factor drop after correction: {m['pf_drop']:.1%}
&middot; Tolerance {m['tolerance']:.4g} &middot; Entry fills open/mid/close/ambiguous: {mix['open']}/{mix['mid']}/{mix['close']}/{mix['ambiguous']}</div></div>
<h2>Original vs corrected</h2>
<div class="card scroll"><table><tr><th>Metric</th><th>Original</th><th>Corrected</th></tr>{rows}</table></div>
<div class="card"><div class="legend mut"><span style="background:var(--orig)"></span>original<span style="background:var(--corr)"></span>corrected (pessimistic)</div>
{_equity_svg(t['pnl_calc'].to_numpy(float), t['corr_pnl'].to_numpy(float))}</div>
<h2>Findings</h2>{finds}
<h2>Flagged trades (first 40)</h2>
<div class="card scroll"><table><tr>{thead}</tr>{tbody or '<tr><td>None</td></tr>'}</table></div>
<p class="mut">Corrected results re-simulate each trade from OHLC bars under pessimistic assumptions: a fill at the bar close can only exit
from the next bar; when SL and TP share a bar, SL is assumed first; gaps through the SL fill at the open; prices that never traded are clamped to the bar.
These checks catch common execution and look-ahead bugs. They cannot prove a strategy is free of bias in its signals.</p>
</main></body></html>"""
