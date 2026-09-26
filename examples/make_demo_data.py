"""Generate a synthetic H1 market and two backtests of the SAME no-edge strategy.

* trades_clean.csv  - correct simulation: fill at the signal bar's close, SL/TP
                      checked from the NEXT bar, SL first when both are hit.
* trades_leaky.csv  - the classic bug: fill at the close, but SL/TP checked
                      against the signal bar's own High/Low, TP first when both hit.

The strategy has no edge by construction (random direction), so any profit in
the leaky log is pure look-ahead.

    python examples/make_demo_data.py --out examples/data
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def make_prices(n_bars: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2023-01-02", periods=int(n_bars * 1.5), freq="h")
    times = times[times.dayofweek < 5][:n_bars]  # weekdays only -> weekend gaps
    sub = 12
    steps = rng.standard_t(df=4, size=(n_bars, sub)) * 0.0009 / np.sqrt(2)
    path = 1900 * np.exp(np.cumsum(steps.ravel())).reshape(n_bars, sub)
    opens = np.concatenate([[1900.0], path[:-1, -1]])
    highs = np.maximum(path.max(axis=1), opens)
    lows = np.minimum(path.min(axis=1), opens)
    closes = path[:, -1]
    ohlc = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}).round(2)
    ohlc.insert(0, "time", times)
    return ohlc


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    pc = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"], np.maximum((df["high"] - pc).abs(), (df["low"] - pc).abs()))
    return tr.rolling(n).mean().to_numpy()


def backtest(df: pd.DataFrame, leaky: bool, seed: int, sl_atr=1.0, tp_atr=0.5, p_entry=0.15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)  # same seed -> same signals in both logs
    O, H, L, C = (df[c].to_numpy() for c in ("open", "high", "low", "close"))
    a = atr(df)
    trades = []
    t = 20
    while t < len(df) - 2:
        if rng.random() > p_entry:
            t += 1
            continue
        d = 1 if rng.random() < 0.5 else -1
        ep = C[t]
        sl = round(ep - d * sl_atr * a[t], 2)
        tp = round(ep + d * tp_atr * a[t], 2)
        i = t if leaky else t + 1
        exit_px = reason = None
        while i < len(df):
            hit_sl = L[i] <= sl if d == 1 else H[i] >= sl
            hit_tp = H[i] >= tp if d == 1 else L[i] <= tp
            if leaky:
                if hit_tp:
                    exit_px, reason = tp, "TP"
                elif hit_sl:
                    exit_px, reason = sl, "SL"
            else:
                gap_sl = O[i] <= sl if d == 1 else O[i] >= sl
                gap_tp = O[i] >= tp if d == 1 else O[i] <= tp
                if gap_sl:
                    exit_px, reason = O[i], "SL"
                elif gap_tp:
                    exit_px, reason = tp, "TP"
                elif hit_sl:
                    exit_px, reason = sl, "SL"
                elif hit_tp:
                    exit_px, reason = tp, "TP"
            if exit_px is not None:
                break
            i += 1
        if exit_px is None:
            break
        trades.append({
            "entry_time": df["time"].iloc[t], "exit_time": df["time"].iloc[i],
            "direction": "buy" if d == 1 else "sell", "entry_price": ep, "exit_price": round(exit_px, 2),
            "sl": sl, "tp": tp, "volume": 0.1, "exit_reason": reason,
        })
        t = i + 1
    return pd.DataFrame(trades)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--bars", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prices = make_prices(args.bars, args.seed)
    prices.to_csv(os.path.join(args.out, "prices_h1.csv"), index=False)
    backtest(prices, leaky=False, seed=args.seed).to_csv(os.path.join(args.out, "trades_clean.csv"), index=False)
    backtest(prices, leaky=True, seed=args.seed).to_csv(os.path.join(args.out, "trades_leaky.csv"), index=False)
    print(f"wrote demo data to {args.out}")


if __name__ == "__main__":
    main()
