"""Core audit engine: bar mapping, bias checks and pessimistic re-simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .metrics import compute_metrics

ERROR, WARN, INFO = "ERROR", "WARN", "INFO"

CHECKS: Dict[str, Tuple[str, str, str]] = {
    # code: (severity, title, explanation)
    "EXIT_ON_ENTRY_BAR_AFTER_CLOSE_FILL": (
        ERROR, "Exit on the entry bar after a close-price fill",
        "The trade was filled at the bar's close, yet it exited inside that same bar. "
        "Everything in that bar happened before the fill, so the backtest used the bar's "
        "High/Low from the past. This is the classic SL/TP look-ahead bug.",
    ),
    "SL_SKIPPED": (
        ERROR, "Stop-loss touched but ignored",
        "Price reached the stop-loss on a bar after entry, but the trade stayed open and "
        "closed later. Either the SL was not simulated or future bars were peeked at.",
    ),
    "ENTRY_OUTSIDE_BAR": (
        ERROR, "Entry price outside the bar's range",
        "The fill price never traded during that bar (beyond tolerance). Impossible fill, "
        "or a timezone/bar-alignment mismatch.",
    ),
    "EXIT_OUTSIDE_BAR": (
        ERROR, "Exit price outside the bar's range",
        "The exit price never traded during that bar (beyond tolerance). Impossible fill, "
        "gap ignored, or timezone mismatch.",
    ),
    "EXIT_BEFORE_ENTRY": (
        ERROR, "Exit time before entry time", "Corrupted or mis-parsed timestamps.",
    ),
    "AMBIGUOUS_TP": (
        WARN, "Take-profit credited when stop-loss was also inside the bar",
        "Both SL and TP were inside the exit bar's range and the backtest chose TP. "
        "OHLC data cannot tell which came first; the optimistic choice inflates results.",
    ),
    "SAME_BAR_EXIT": (
        WARN, "Entry and exit inside the same bar",
        "Intrabar order is unknowable from OHLC data. Fine for tick-level backtests, "
        "suspicious for bar-level ones.",
    ),
    "TP_SKIPPED": (
        INFO, "Take-profit touched but not taken",
        "Price reached TP after entry but the trade stayed open. Not optimistic, but "
        "indicates the simulator does not model the TP order.",
    ),
    "UNMAPPED": (
        INFO, "Trade time not covered by price data",
        "Entry or exit falls outside the price file or in a data gap; trade kept as-is.",
    ),
}


@dataclass
class AuditConfig:
    tol: Optional[float] = None        # price tolerance; auto = 2% of median bar range
    cost: float = 0.0                  # round-trip cost per unit volume, price units
    contract_size: float = 1.0
    max_hold_bars: int = 5000
    perfect_rate_threshold: float = 0.10


@dataclass
class Finding:
    code: str
    severity: str
    title: str
    explanation: str
    count: int
    rate: float
    examples: List[int] = field(default_factory=list)


@dataclass
class AuditResult:
    trades: pd.DataFrame
    original: Dict[str, float]
    corrected: Dict[str, float]
    findings: List[Finding]
    meta: Dict[str, object]
    verdict: str
    verdict_level: int  # 0 clean, 1 suspicious, 2 likely biased


def _fmt_interval(td: pd.Timedelta) -> str:
    sec = int(td.total_seconds())
    for size, unit in ((86400, "d"), (3600, "h"), (60, "min")):
        if sec and sec % size == 0:
            return f"{sec // size}{unit}"
    return f"{sec}s"


def _map_bars(bar_times: np.ndarray, bar_delta: np.timedelta64, times: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(bar_times, times, side="right") - 1
    ok = idx >= 0
    safe = np.clip(idx, 0, len(bar_times) - 1)
    ok &= times < bar_times[safe] + bar_delta
    return np.where(ok, idx, -1)


def _outside_rate(bar_times, bar_delta, H, L, times, prices, tol) -> float:
    idx = _map_bars(bar_times, bar_delta, times)
    m = idx >= 0
    if m.sum() == 0:
        return 1.0
    i = idx[m]
    p = prices[m]
    return float(((p > H[i] + tol) | (p < L[i] - tol)).mean())


def _eval_bar(i, d, sl, tp, O, H, L, gap_ok):
    """Pessimistic single-bar evaluation. Returns (price, reason) or None."""
    has_sl, has_tp = not np.isnan(sl), not np.isnan(tp)
    if d == 1:
        if gap_ok and has_sl and O[i] <= sl:
            return O[i], "SL_GAP"
        if gap_ok and has_tp and O[i] >= tp:
            return tp, "TP"
        if has_sl and L[i] <= sl:
            return sl, "SL"
        if has_tp and H[i] >= tp:
            return tp, "TP"
    else:
        if gap_ok and has_sl and O[i] >= sl:
            return O[i], "SL_GAP"
        if gap_ok and has_tp and O[i] <= tp:
            return tp, "TP"
        if has_sl and H[i] >= sl:
            return sl, "SL"
        if has_tp and L[i] <= tp:
            return tp, "TP"
    return None


def audit(trades: pd.DataFrame, prices: pd.DataFrame, cfg: Optional[AuditConfig] = None) -> AuditResult:
    cfg = cfg or AuditConfig()
    t = trades.copy().reset_index(drop=True)
    bt = prices["time"].values.astype("datetime64[ns]")
    O, H, L, C = (prices[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    diffs = np.diff(bt)
    bar_delta = np.median(diffs).astype("timedelta64[ns]")
    rng = H - L
    tol = cfg.tol if cfg.tol is not None else 0.02 * float(np.median(rng[rng > 0]) if (rng > 0).any() else 0.0)
    tol = max(tol, 1e-9)

    ent_t = t["entry_time"].values.astype("datetime64[ns]")
    ext_t = t["exit_time"].values.astype("datetime64[ns]")
    eb_all = _map_bars(bt, bar_delta, ent_t)
    xb_all = _map_bars(bt, bar_delta, ext_t)

    # --- timezone sanity -------------------------------------------------
    base_rate = _outside_rate(bt, bar_delta, H, L, ent_t, t["entry_price"].to_numpy(float), tol)
    tz_hint = None
    if base_rate > 0.20:
        best = (base_rate, 0)
        for h in range(-14, 15):
            if h == 0:
                continue
            r = _outside_rate(bt + np.timedelta64(h, "h"), bar_delta, H, L, ent_t,
                              t["entry_price"].to_numpy(float), tol)
            if r < best[0]:
                best = (r, h)
        if best[1] != 0 and best[0] < base_rate * 0.5:
            tz_hint = {"shift_hours": best[1], "outside_rate_now": base_rate, "outside_rate_shifted": best[0]}

    n = len(t)
    d_arr = t["direction"].to_numpy(int)
    ep_arr = t["entry_price"].to_numpy(float)
    xp_arr = t["exit_price"].to_numpy(float)
    sl_arr = t["sl"].to_numpy(float)
    tp_arr = t["tp"].to_numpy(float)
    vol = t["volume"].to_numpy(float)
    vol = np.where(np.isnan(vol), 1.0, vol)

    flags: List[List[str]] = [[] for _ in range(n)]
    c_ep = ep_arr.copy()
    c_xp = xp_arr.copy()
    c_xb = xb_all.copy()
    c_reason = np.array([""] * n, dtype=object)
    entry_pos = np.array([""] * n, dtype=object)
    perfect_entry = np.zeros(n, bool)
    perfect_exit = np.zeros(n, bool)
    disc_exit = np.zeros(n, bool)
    mapped = np.zeros(n, bool)

    for k in range(n):
        eb, xb, d = int(eb_all[k]), int(xb_all[k]), int(d_arr[k])
        ep, xp, sl, tp = ep_arr[k], xp_arr[k], sl_arr[k], tp_arr[k]
        if eb < 0 or xb < 0:
            flags[k].append("UNMAPPED")
            c_reason[k] = "UNCHANGED"
            continue
        mapped[k] = True
        f = flags[k]

        # entry position within the bar
        near_c = abs(ep - C[eb]) <= tol
        near_o = abs(ep - O[eb]) <= tol
        if near_o and near_c:
            pos = "ambiguous"  # open ~= close: cannot tell, give the benefit of the doubt
        else:
            pos = "open" if near_o else ("close" if near_c else "mid")
        entry_pos[k] = pos

        if ext_t[k] < ent_t[k]:
            f.append("EXIT_BEFORE_ENTRY")
        if ep > H[eb] + tol or ep < L[eb] - tol:
            f.append("ENTRY_OUTSIDE_BAR")
            c_ep[k] = min(max(ep, L[eb]), H[eb])
        if xp > H[xb] + tol or xp < L[xb] - tol:
            f.append("EXIT_OUTSIDE_BAR")
        if xb == eb:
            f.append("EXIT_ON_ENTRY_BAR_AFTER_CLOSE_FILL" if pos == "close" else "SAME_BAR_EXIT")

        exit_is_tp = (not np.isnan(tp)) and abs(xp - tp) <= tol
        exit_is_sl = (not np.isnan(sl)) and abs(xp - sl) <= tol
        if exit_is_tp and not np.isnan(sl):
            sl_in = L[xb] <= sl if d == 1 else H[xb] >= sl
            if sl_in:
                f.append("AMBIGUOUS_TP")

        # Only mid-bar fills can be "suspiciously perfect": a fill at the open or
        # close legitimately coincides with the bar's extreme quite often.
        if H[eb] > L[eb] and pos == "mid":
            perfect_entry[k] = abs(ep - (L[eb] if d == 1 else H[eb])) <= tol
        xp_at_oc = abs(xp - O[xb]) <= tol or abs(xp - C[xb]) <= tol
        if H[xb] > L[xb] and not exit_is_tp and not exit_is_sl and not xp_at_oc:
            disc_exit[k] = True
            perfect_exit[k] = abs(xp - (H[xb] if d == 1 else L[xb])) <= tol

        # --- pessimistic re-simulation ----------------------------------
        invalid = ("EXIT_BEFORE_ENTRY" in f) or ("EXIT_ON_ENTRY_BAR_AFTER_CLOSE_FILL" in f)
        start = eb if pos == "open" else eb + 1
        has_orders = not (np.isnan(sl) and np.isnan(tp))
        result = None

        if not invalid:
            for i in range(start, xb):
                r = _eval_bar(i, d, sl, tp, O, H, L, gap_ok=i > eb)
                if r:
                    result = (r[0], i, r[1])
                    f.append("SL_SKIPPED" if r[1].startswith("SL") else "TP_SKIPPED")
                    break
            if result is None:
                if exit_is_tp or exit_is_sl:
                    r = _eval_bar(xb, d, sl, tp, O, H, L, gap_ok=xb > eb)
                    if r:
                        result = (r[0], xb, r[1])
                if result is None:
                    result = (min(max(xp, L[xb]), H[xb]), xb, "ORIGINAL")
        else:
            i = start
            last = len(O) - 1
            if not has_orders:
                j = min(eb + 1, last)
                result = (O[j], j, "NEXT_OPEN")
            while result is None:
                if i > last:
                    result = (C[last], last, "END_OF_DATA")
                    break
                r = _eval_bar(i, d, sl, tp, O, H, L, gap_ok=i > eb)
                if r:
                    result = (r[0], i, r[1])
                elif i - start >= cfg.max_hold_bars:
                    result = (C[i], i, "MAX_HOLD")
                i += 1

        c_xp[k], c_xb[k], c_reason[k] = result

    # --- assemble ------------------------------------------------------------
    cs = cfg.contract_size
    t["entry_bar"] = eb_all
    t["exit_bar"] = xb_all
    t["entry_fill"] = entry_pos
    t["pnl_calc"] = (xp_arr - ep_arr) * d_arr * vol * cs
    t["corr_entry_price"] = c_ep
    t["corr_exit_price"] = c_xp
    t["corr_exit_bar"] = c_xb
    t["corr_exit_time"] = [bt[i] if i >= 0 else ext_t[k] for k, i in enumerate(c_xb)]
    t["corr_reason"] = c_reason
    t["corr_pnl"] = (c_xp - c_ep) * d_arr * vol * cs - cfg.cost * vol * cs
    t["flags"] = [";".join(f) for f in flags]

    bars_orig = np.where(mapped, xb_all - eb_all, np.nan)
    bars_corr = np.where(mapped, c_xb - eb_all, np.nan)
    original = compute_metrics(t["pnl_calc"].to_numpy(), bars_orig)
    corrected = compute_metrics(t["corr_pnl"].to_numpy(), bars_corr)

    findings: List[Finding] = []
    for code, (sev, title, expl) in CHECKS.items():
        idx = [k for k in range(n) if code in flags[k]]
        if idx:
            findings.append(Finding(code, sev, title, expl, len(idx), len(idx) / n, idx[:10]))

    n_disc = int(disc_exit.sum())
    px_rate = perfect_exit.sum() / n_disc if n_disc >= 20 else 0.0
    n_mid = int((entry_pos == "mid").sum())
    pe_rate = perfect_entry.sum() / n_mid if n_mid >= 20 else 0.0
    if pe_rate > cfg.perfect_rate_threshold:
        findings.append(Finding(
            "PERFECT_ENTRIES", WARN, "Too many entries at the bar's best price",
            f"{pe_rate:.0%} of mid-bar entries were filled at the bar's most favourable extreme "
            "(buys at the Low, sells at the High). Real fills rarely do this; it usually means "
            "the entry used information from the completed bar.",
            int(perfect_entry.sum()), float(pe_rate), list(np.where(perfect_entry)[0][:10]),
        ))
    if px_rate > cfg.perfect_rate_threshold:
        findings.append(Finding(
            "PERFECT_EXITS", WARN, "Too many discretionary exits at the bar's best price",
            f"{px_rate:.0%} of mid-bar, non-SL/TP exits happened at the bar's most favourable extreme. "
            "Typical sign of exits decided with the completed bar in hand.",
            int(perfect_exit.sum()), float(px_rate), list(np.where(perfect_exit)[0][:10]),
        ))
    if original["avg_bars"] is not None and not np.isnan(original["avg_bars"]) and original["avg_bars"] < 1.0:
        findings.append(Finding(
            "SHORT_HOLD", WARN, "Average holding time below one bar",
            f"Average bars in trade = {original['avg_bars']:.2f}. On bar data this almost always "
            "means exits are resolved inside the entry bar.",
            n, 1.0,
        ))
    if tz_hint:
        findings.append(Finding(
            "TIMEZONE_SUSPECT", WARN, "Trade and price timestamps look misaligned",
            f"{tz_hint['outside_rate_now']:.0%} of entries fall outside their bar's range; shifting "
            f"prices by {tz_hint['shift_hours']:+d}h reduces this to {tz_hint['outside_rate_shifted']:.0%}. "
            "Align timezones before trusting any other finding.",
            n, tz_hint["outside_rate_now"],
        ))
    if trades.attrs.get("has_pnl"):
        pc = t["pnl"].to_numpy(float)
        ok = ~np.isnan(pc) & (np.abs(pc) > 0) & (np.abs(t["pnl_calc"].to_numpy()) > 0)
        if ok.sum() >= 5:
            agree = float((np.sign(pc[ok]) == np.sign(t["pnl_calc"].to_numpy()[ok])).mean())
            if agree < 0.95:
                findings.append(Finding(
                    "PNL_MISMATCH", WARN, "Reported P&L disagrees with prices",
                    f"Only {agree:.0%} of trades have a reported P&L with the same sign as "
                    "(exit - entry) x direction. Check direction mapping, or fees/swaps dominating.",
                    int((~(np.sign(pc[ok]) == np.sign(t['pnl_calc'].to_numpy()[ok]))).sum()), 1 - agree,
                ))
    if not trades.attrs.get("has_sl") and not trades.attrs.get("has_tp"):
        findings.append(Finding(
            "NO_ORDERS", INFO, "No SL/TP columns",
            "Without SL/TP levels, trades cannot be re-simulated; only fill validity is checked.",
            n, 1.0,
        ))

    sev_rank = {ERROR: 0, WARN: 1, INFO: 2}
    findings.sort(key=lambda x: (sev_rank[x.severity], -x.count))

    err_rate = sum(1 for k in range(n) if any(CHECKS.get(c, (INFO,))[0] == ERROR for c in flags[k])) / max(n, 1)
    pf_o, pf_c = original["profit_factor"], corrected["profit_factor"]
    pf_drop = (pf_o - pf_c) / pf_o if (pf_o and np.isfinite(pf_o) and pf_o > 0) else 0.0
    warn_hit = any(f.severity == WARN for f in findings)
    if err_rate > 0.05 or pf_drop > 0.20:
        verdict, level = "LOOK-AHEAD / EXECUTION BIAS LIKELY", 2
    elif err_rate > 0 or warn_hit or pf_drop > 0.05:
        verdict, level = "SUSPICIOUS - REVIEW FLAGGED TRADES", 1
    else:
        verdict, level = "NO BIAS DETECTED BY THESE CHECKS", 0

    meta = {
        "n_trades": n,
        "n_mapped": int(mapped.sum()),
        "dropped_rows": trades.attrs.get("dropped_rows", 0),
        "bar_interval": _fmt_interval(pd.Timedelta(bar_delta)),
        "n_bars": len(prices),
        "price_start": str(prices["time"].iloc[0]),
        "price_end": str(prices["time"].iloc[-1]),
        "tolerance": tol,
        "cost": cfg.cost,
        "error_trade_rate": err_rate,
        "pf_drop": pf_drop,
        "entry_fill_mix": {p: int((entry_pos == p).sum()) for p in ("open", "mid", "close", "ambiguous")},
        "timezone_hint": tz_hint,
    }
    return AuditResult(t, original, corrected, findings, meta, verdict, level)
