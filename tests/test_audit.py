import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from lookahead_auditor import AuditConfig, audit, load_prices, load_trades
from lookahead_auditor.loader import DataError, parse_direction, parse_time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bars(rows):
    t = pd.date_range("2024-01-01", periods=len(rows), freq="h")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.insert(0, "time", t)
    return df


def trade(i_entry, i_exit, d, ep, xp, sl=np.nan, tp=np.nan, prices=None):
    t = prices["time"]
    return {"entry_time": t[i_entry], "exit_time": t[i_exit], "direction": d,
            "entry_price": ep, "exit_price": xp, "sl": sl, "tp": tp, "volume": np.nan,
            "pnl": np.nan, "exit_reason": ""}


def frame(rows):
    df = pd.DataFrame(rows)
    df.attrs.update(has_sl=df["sl"].notna().any(), has_tp=df["tp"].notna().any(), has_pnl=False)
    return df


FLAT = [[100, 101, 99, 100]] * 10


def test_close_fill_same_bar_exit_is_error_and_resimulated():
    p = bars([[98, 105, 95, 100], [100, 101, 99, 100.5], [100.5, 103, 100, 102.5]] + FLAT)
    # buy at close of bar 0 (100), TP 104 "hit" using bar 0's own high -> impossible
    t = frame([trade(0, 0, 1, 100, 104, sl=96, tp=104, prices=p)])
    r = audit(t, p, AuditConfig(tol=0.01))
    assert "EXIT_ON_ENTRY_BAR_AFTER_CLOSE_FILL" in r.trades.loc[0, "flags"]
    assert r.verdict_level == 2
    # re-simulated from bar 1: TP 104 never reached, SL 96 never reached -> runs to end of data
    assert r.trades.loc[0, "corr_reason"] in ("END_OF_DATA", "MAX_HOLD")


def test_sl_skipped_detected():
    p = bars([[100, 100.5, 99.5, 100], [100, 100.2, 94, 95], [95, 106, 95, 105]] + FLAT)
    t = frame([trade(0, 2, 1, 100, 105, sl=96, tp=105, prices=p)])
    r = audit(t, p, AuditConfig(tol=0.01))
    assert "SL_SKIPPED" in r.trades.loc[0, "flags"]
    assert r.trades.loc[0, "corr_exit_price"] == 96
    assert r.corrected["net"] < 0 < r.original["net"]


def test_gap_through_sl_fills_at_open():
    p = bars([[100, 100.5, 99.5, 100], [93, 94, 92, 93.5]] + FLAT)
    t = frame([trade(0, 1, 1, 100, 96, sl=96, tp=110, prices=p)])
    r = audit(t, p, AuditConfig(tol=0.01))
    assert "EXIT_OUTSIDE_BAR" in r.trades.loc[0, "flags"]
    assert r.trades.loc[0, "corr_exit_price"] == 93


def test_ambiguous_tp_goes_to_sl():
    p = bars([[100, 100.5, 99.5, 100], [100, 106, 94, 100]] + FLAT)
    t = frame([trade(0, 1, 1, 100, 105, sl=95, tp=105, prices=p)])
    r = audit(t, p, AuditConfig(tol=0.01))
    assert "AMBIGUOUS_TP" in r.trades.loc[0, "flags"]
    assert r.trades.loc[0, "corr_exit_price"] == 95


def test_clean_trade_untouched():
    p = bars([[100, 100.5, 99.5, 100], [100, 102, 99, 101.5], [101.5, 105.5, 101, 105]] + FLAT)
    t = frame([trade(0, 2, 1, 100, 105, sl=97, tp=105, prices=p)])
    r = audit(t, p, AuditConfig(tol=0.01))
    assert r.trades.loc[0, "flags"] == ""
    assert r.verdict_level == 0
    assert r.original["net"] == pytest.approx(r.corrected["net"])


def test_sell_side_mirror():
    p = bars([[100, 100.5, 99.5, 100], [100, 106, 99.8, 105], [105, 105, 94, 95]] + FLAT)
    t = frame([trade(0, 2, -1, 100, 95, sl=104, tp=95, prices=p)])
    r = audit(t, p, AuditConfig(tol=0.01))
    assert "SL_SKIPPED" in r.trades.loc[0, "flags"]
    assert r.trades.loc[0, "corr_exit_price"] == 104


def test_entry_outside_bar():
    p = bars([[100, 101, 99, 100], [100, 101, 99, 100]] + FLAT)
    t = frame([trade(0, 1, 1, 97, 100, prices=p)])
    r = audit(t, p, AuditConfig(tol=0.01))
    assert "ENTRY_OUTSIDE_BAR" in r.trades.loc[0, "flags"]


def test_parsers():
    s = parse_time(pd.Series(["2024.01.02 10:00:00", "2024.01.02 11:00"]))
    assert s.iloc[0] == pd.Timestamp("2024-01-02 10:00")
    assert list(parse_direction(pd.Series(["BUY", "sell", "Long", "-1", "buy limit"]))) == [1, -1, 1, -1, 1]
    with pytest.raises(DataError):
        parse_direction(pd.Series(["maybe"]))


def test_mt5_price_format(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\n"
                 "2024.01.02\t00:00:00\t2060.1\t2062.0\t2059.0\t2061.0\t100\n"
                 "2024.01.02\t01:00:00\t2061.0\t2063.0\t2060.0\t2062.5\t120\n"
                 "2024.01.02\t02:00:00\t2062.5\t2064.0\t2061.0\t2063.0\t90\n")
    p = load_prices(str(f))
    assert list(p.columns) == ["time", "open", "high", "low", "close"]
    assert p["time"].iloc[1] == pd.Timestamp("2024-01-02 01:00")


def test_demo_end_to_end(tmp_path):
    out = tmp_path / "data"
    subprocess.check_call([sys.executable, os.path.join(ROOT, "examples", "make_demo_data.py"),
                           "--out", str(out), "--bars", "4000"])
    prices = load_prices(str(out / "prices_h1.csv"))
    clean = audit(load_trades(str(out / "trades_clean.csv")), prices)
    leaky = audit(load_trades(str(out / "trades_leaky.csv")), prices)
    assert clean.verdict_level == 0
    assert leaky.verdict_level == 2
    assert leaky.original["profit_factor"] > leaky.corrected["profit_factor"] + 0.2
    code = subprocess.call([sys.executable, "-m", "lookahead_auditor", "--quiet",
                            "--trades", str(out / "trades_leaky.csv"), "--prices", str(out / "prices_h1.csv"),
                            "--html", str(tmp_path / "r.html"), "--json", str(tmp_path / "r.json"),
                            "--csv", str(tmp_path / "r.csv")], cwd=ROOT)
    assert code == 2
    assert (tmp_path / "r.html").read_text().startswith("<!doctype html>")
