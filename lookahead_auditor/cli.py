"""Command-line entry point: ``lookahead-audit``."""
from __future__ import annotations

import argparse
import sys

from .audit import AuditConfig, audit
from .loader import DataError, load_prices, load_trades, parse_mapping
from .report import to_html, to_json, to_text


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lookahead-audit",
        description="Audit a backtest trade log against OHLC bars for look-ahead and execution bias.",
    )
    p.add_argument("--trades", required=True, help="CSV of trades (entry/exit time, direction, prices, optional SL/TP)")
    p.add_argument("--prices", required=True, help="CSV of OHLC bars used by the backtest")
    p.add_argument("--tol", type=float, default=None,
                   help="Price tolerance for 'equal' prices (default: 2%% of median bar range). Use about your spread.")
    p.add_argument("--cost", type=float, default=0.0,
                   help="Round-trip cost per unit volume in price units, applied to corrected results")
    p.add_argument("--contract-size", type=float, default=1.0, help="Multiplier for P&L (e.g. 100 for XAUUSD lots)")
    p.add_argument("--max-hold", type=int, default=5000, help="Max bars when re-simulating an invalid trade")
    p.add_argument("--map-trades", default=None, help='Column overrides, e.g. "entry_time=Open Time,direction=Type"')
    p.add_argument("--map-prices", default=None, help='Column overrides, e.g. "time=Date"')
    p.add_argument("--html", default=None, help="Write an HTML report to this path")
    p.add_argument("--json", default=None, help="Write a JSON summary to this path")
    p.add_argument("--csv", default=None, help="Write per-trade flags and corrected prices to this CSV")
    p.add_argument("--quiet", action="store_true", help="Do not print the text report")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prices = load_prices(args.prices, parse_mapping(args.map_prices))
        trades = load_trades(args.trades, parse_mapping(args.map_trades))
    except (DataError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    cfg = AuditConfig(tol=args.tol, cost=args.cost, contract_size=args.contract_size, max_hold_bars=args.max_hold)
    res = audit(trades, prices, cfg)
    if not args.quiet:
        print(to_text(res))
    if args.html:
        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(to_html(res))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(to_json(res))
    if args.csv:
        res.trades.to_csv(args.csv, index_label="row")
    # exit code usable in CI: 0 clean, 1 suspicious, 2 bias likely
    return res.verdict_level


if __name__ == "__main__":
    sys.exit(main())
