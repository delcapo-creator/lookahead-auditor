"""Trade-level performance metrics."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def compute_metrics(pnl: np.ndarray, bars: Optional[np.ndarray] = None) -> Dict[str, float]:
    pnl = np.asarray(pnl, dtype=float)
    pnl = pnl[~np.isnan(pnl)]
    n = len(pnl)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gp, gl = float(wins.sum()), float(-losses.sum())
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else float("nan"))
    avg_w = float(wins.mean()) if len(wins) else 0.0
    avg_l = float(-losses.mean()) if len(losses) else 0.0
    eq = np.cumsum(pnl) if n else np.array([0.0])
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    mdd = float((peak - eq).max()) if n else 0.0
    sd = float(pnl.std(ddof=1)) if n > 1 else float("nan")
    mean = float(pnl.mean()) if n else float("nan")
    avg_bars = None
    if bars is not None:
        b = np.asarray(bars, dtype=float)
        b = b[~np.isnan(b)]
        avg_bars = float(b.mean()) if len(b) else float("nan")
    return {
        "trades": n,
        "net": float(pnl.sum()),
        "profit_factor": pf,
        "win_rate": len(wins) / n if n else float("nan"),
        "avg_win": avg_w,
        "avg_loss": avg_l,
        "payoff": avg_w / avg_l if avg_l > 0 else float("nan"),
        "expectancy": mean,
        "t_stat": mean / sd * np.sqrt(n) if (n > 1 and sd > 0) else float("nan"),
        "max_drawdown": mdd,
        "avg_bars": avg_bars,
    }
