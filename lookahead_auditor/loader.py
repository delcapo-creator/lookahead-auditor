"""Load trade logs and OHLC price data from CSV with automatic column detection.

Supports common layouts: generic CSV exports, MetaTrader 5 history exports
(``<DATE>``/``<TIME>`` columns, ``2024.01.02`` dates, tab separated), epoch
timestamps and timezone-aware ISO strings.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower().strip().strip("<>"))


TRADE_ALIASES: Dict[str, Iterable[str]] = {
    "entry_time": ["entry_time", "open_time", "time_open", "entrytime", "opened", "open time", "entry_date"],
    "exit_time": ["exit_time", "close_time", "time_close", "exittime", "closed", "close time", "exit_date"],
    "direction": ["direction", "type", "side", "action", "dir", "position"],
    "entry_price": ["entry_price", "open_price", "price_open", "entryprice", "price_in"],
    "exit_price": ["exit_price", "close_price", "price_close", "exitprice", "price_out"],
    "sl": ["sl", "stop_loss", "stoploss", "s/l", "stop"],
    "tp": ["tp", "take_profit", "takeprofit", "t/p", "target"],
    "volume": ["volume", "lots", "lot", "size", "qty", "quantity"],
    "pnl": ["pnl", "profit", "net_profit", "netprofit", "pl", "p&l", "result"],
    "exit_reason": ["exit_reason", "reason", "close_reason", "comment"],
}
TRADE_REQUIRED = ["entry_time", "exit_time", "direction", "entry_price", "exit_price"]

PRICE_ALIASES: Dict[str, Iterable[str]] = {
    "time": ["time", "datetime", "timestamp", "date_time", "bar_time", "date"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c"],
}


class DataError(ValueError):
    """Raised when input data cannot be interpreted."""


def read_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _resolve(df: pd.DataFrame, aliases: Dict[str, Iterable[str]], overrides: Optional[Dict[str, str]]) -> Dict[str, str]:
    overrides = overrides or {}
    normed = {_norm(c): c for c in df.columns}
    found: Dict[str, str] = {}
    for canon, names in aliases.items():
        if canon in overrides:
            col = overrides[canon]
            if col not in df.columns:
                raise DataError(f"Column '{col}' (mapped to '{canon}') not found. Available: {list(df.columns)}")
            found[canon] = col
            continue
        for n in [canon, *names]:
            if _norm(n) in normed:
                found[canon] = normed[_norm(n)]
                break
    return found


def parse_time(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s
    elif pd.api.types.is_numeric_dtype(s):
        unit = "ms" if s.dropna().abs().median() > 1e11 else "s"
        out = pd.to_datetime(s, unit=unit)
    else:
        txt = s.astype(str).str.strip().str.replace(r"^(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3", regex=True)
        out = pd.to_datetime(txt, errors="coerce", format="mixed", utc=True)
        out = out.dt.tz_convert(None)
        return out.astype("datetime64[ns]")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert(None)
    return out.astype("datetime64[ns]")


_DIR_MAP = {
    "buy": 1, "long": 1, "b": 1, "l": 1, "bull": 1, "up": 1,
    "sell": -1, "short": -1, "s": -1, "bear": -1, "down": -1,
}


def parse_direction(s: pd.Series) -> pd.Series:
    def one(x):
        t = str(x).strip().lower()
        if not t or t == "nan":
            return np.nan
        first = re.split(r"[\s_\-]+", t)[0] or t
        if first in _DIR_MAP:
            return _DIR_MAP[first]
        try:
            v = float(t)
        except ValueError:
            return np.nan
        return 1 if v > 0 else (-1 if v < 0 else np.nan)

    out = s.map(one)
    bad = s[out.isna()].astype(str).unique()[:10]
    if len(bad):
        raise DataError(
            f"Unrecognised direction values: {list(bad)}. Use buy/sell, long/short or +1/-1."
        )
    return out.astype(int)


def load_prices(path_or_df, overrides: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    df = read_table(path_or_df) if isinstance(path_or_df, str) else path_or_df.copy()
    normed = {_norm(c): c for c in df.columns}
    cols = _resolve(df, PRICE_ALIASES, overrides)
    # MT5 export: separate <DATE> and <TIME> columns
    if "date" in normed and "time" in normed and (overrides or {}).get("time") is None:
        t = df[normed["date"]].astype(str).str.strip() + " " + df[normed["time"]].astype(str).str.strip()
        time = parse_time(t)
    else:
        if "time" not in cols:
            raise DataError(f"Price file needs a time column. Found: {list(df.columns)}")
        time = parse_time(df[cols["time"]])
    missing = [c for c in ("open", "high", "low", "close") if c not in cols]
    if missing:
        raise DataError(f"Price file missing columns {missing}. Found: {list(df.columns)}")
    out = pd.DataFrame({"time": time})
    for c in ("open", "high", "low", "close"):
        out[c] = pd.to_numeric(df[cols[c]], errors="coerce")
    out = out.dropna().sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    if len(out) < 3:
        raise DataError("Price file has fewer than 3 valid bars.")
    return out


def load_trades(path_or_df, overrides: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    df = read_table(path_or_df) if isinstance(path_or_df, str) else path_or_df.copy()
    cols = _resolve(df, TRADE_ALIASES, overrides)
    missing = [c for c in TRADE_REQUIRED if c not in cols]
    if missing:
        raise DataError(
            f"Trade file missing required columns {missing}. Found: {list(df.columns)}. "
            "Map them explicitly, e.g. --map-trades \"entry_time=Open Time,exit_price=Close Price\"."
        )
    out = pd.DataFrame(index=df.index)
    out["entry_time"] = parse_time(df[cols["entry_time"]])
    out["exit_time"] = parse_time(df[cols["exit_time"]])
    out["direction"] = parse_direction(df[cols["direction"]])
    for c in ("entry_price", "exit_price", "sl", "tp", "volume", "pnl"):
        out[c] = pd.to_numeric(df[cols[c]], errors="coerce") if c in cols else np.nan
    out["exit_reason"] = df[cols["exit_reason"]].astype(str) if "exit_reason" in cols else ""
    # SL/TP of 0 means "not set" in MT5
    for c in ("sl", "tp"):
        out.loc[out[c] <= 0, c] = np.nan
    bad = out[["entry_time", "exit_time", "entry_price", "exit_price"]].isna().any(axis=1)
    if bad.all():
        raise DataError("No trade row has valid times and prices.")
    out = out[~bad].reset_index(drop=True)
    out.attrs["dropped_rows"] = int(bad.sum())
    out.attrs["has_sl"] = bool(out["sl"].notna().any())
    out.attrs["has_tp"] = bool(out["tp"].notna().any())
    out.attrs["has_volume"] = bool(out["volume"].notna().any())
    out.attrs["has_pnl"] = bool(out["pnl"].notna().any())
    return out


def parse_mapping(text: Optional[str]) -> Dict[str, str]:
    """Parse 'canon=Column Name,canon2=Other' into a dict."""
    if not text:
        return {}
    out = {}
    for part in text.split(","):
        if "=" not in part:
            raise DataError(f"Bad mapping '{part}'. Expected canonical=column.")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out
