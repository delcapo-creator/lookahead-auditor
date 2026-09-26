# lookahead-auditor

**Find look-ahead and execution bias in a backtest before it costs you money.**

`lookahead-auditor` takes the trade log your backtester produced and the OHLC bars it ran on, checks every trade against what was actually possible on those bars, and re-simulates the suspicious ones under pessimistic assumptions. You get the original and the *corrected* profit factor side by side, and a list of exactly which trades are impossible.

It works with any backtester that can export trades to CSV, including MetaTrader 5, Python/pandas engines, and custom RL environments.

## Why this exists

I built an RL trading bot for XAUUSD. A backtest reported a **profit factor of 2.7–3.9** and the account balance reaching over **$200k**. The real numbers were a **profit factor of ~1.1–1.2** and **$17–21k**.

The cause was one line: stop-loss, take-profit, break-even and trailing were checked against the **entry bar's own High/Low**, while the entry was filled at that bar's **close**. Everything in that bar had already happened before the fill. The backtest was harvesting price moves from the past.

This bug is common, silent, and it looks exactly like a great strategy. This tool detects it automatically, along with the other usual suspects.

## Demo: a strategy with zero edge that "passes" a t > 3 test

`examples/make_demo_data.py` builds a synthetic gold-like H1 market and backtests a strategy that picks **random directions**, so it has no edge by construction. The same strategy is simulated two ways:

| | Leaky backtest (SL/TP checked on the signal bar) | After audit (pessimistic re-simulation) |
|---|---:|---:|
| Profit factor | **1.27** | **0.80** |
| Win rate | 72.0% | 61.8% |
| Net P&L | +6,077 | −6,061 |
| t-stat of mean | **3.61** | −3.33 |
| Avg bars in trade | 0.81 | 1.79 |

The leaky version clears the "t > 3" bar that many quants use to accept a strategy, yet its entire profit is look-ahead. The auditor flags **54% of trades** as impossible and returns the verdict `LOOK-AHEAD / EXECUTION BIAS LIKELY`. The correctly simulated backtest of the same strategy returns `NO BIAS DETECTED` with identical original and corrected numbers.

Try it yourself:

```bash
pip install -e .
python examples/make_demo_data.py
lookahead-audit --trades examples/data/trades_leaky.csv --prices examples/data/prices_h1.csv \
                --contract-size 100 --html report.html
```

Sample reports are in [`examples/report_leaky.html`](examples/report_leaky.html) and [`examples/report_clean.html`](examples/report_clean.html).

## What it checks

| Code | Severity | What it means |
|---|---|---|
| `EXIT_ON_ENTRY_BAR_AFTER_CLOSE_FILL` | ERROR | Filled at the bar's close but exited inside the same bar, so it used that bar's past High/Low. |
| `SL_SKIPPED` | ERROR | Price touched the stop-loss after entry, but the trade stayed open and closed later. |
| `ENTRY_OUTSIDE_BAR` / `EXIT_OUTSIDE_BAR` | ERROR | The fill price never traded in that bar (impossible fill, ignored gap, or timezone mismatch). |
| `EXIT_BEFORE_ENTRY` | ERROR | Corrupted or mis-parsed timestamps. |
| `AMBIGUOUS_TP` | WARN | SL and TP were both inside the exit bar and the backtest chose TP. OHLC cannot tell which came first. |
| `SAME_BAR_EXIT` | WARN | Entry and exit in the same bar with an intrabar fill, so the order of events is unknowable. |
| `SHORT_HOLD` | WARN | Average bars in trade is below 1, which is almost always same-bar resolution. |
| `PERFECT_ENTRIES` / `PERFECT_EXITS` | WARN | Too many mid-bar fills exactly at the bar's best price. |
| `TIMEZONE_SUSPECT` | WARN | Many fills fall outside their bar; the tool tests hour shifts and suggests the offset. |
| `PNL_MISMATCH` | WARN | The reported P&L sign disagrees with prices (direction mapping, or fees dominating). |
| `TP_SKIPPED` | INFO | TP was touched but not taken. This is not optimistic, but the simulator ignores the order. |

## How the correction works

Each trade is replayed on the OHLC bars with deliberately pessimistic rules:

- A fill at a bar's **close** can only start exiting on the **next** bar.
- When **SL and TP fall in the same bar**, the SL is assumed to have hit first.
- A bar that **gaps through the SL** fills at the open, not at the SL price.
- Prices that never traded are **clamped** to the bar's range.
- Legitimate discretionary exits (model/indicator closes) are kept unchanged.
- An optional `--cost` (spread + commission, per unit volume) is subtracted from the corrected results.

A clean backtest comes out unchanged. A leaky one shows how much of its edge was never real.

## Input format

**Trades CSV.** Column names are detected automatically; override them with `--map-trades` if needed.

| Field | Required | Accepted names (examples) |
|---|---|---|
| entry_time | yes | `entry_time`, `open_time`, `time_open` |
| exit_time | yes | `exit_time`, `close_time`, `time_close` |
| direction | yes | `direction`, `type`, `side` with values buy/sell, long/short, +1/−1 |
| entry_price, exit_price | yes | `entry_price`/`open_price`, `exit_price`/`close_price` |
| sl, tp | recommended | `sl`/`stop_loss`, `tp`/`take_profit` (0 = not set) |
| volume, pnl, exit_reason | optional | `volume`/`lots`, `profit`, `reason`/`comment` |

**Prices CSV.** Columns `time, open, high, low, close`. MetaTrader 5 exports (`<DATE>`, `<TIME>`, tab-separated, `2024.01.02` dates) are read directly. Bars must be labelled by their **open** time, which is the MT5 default, and use the same timezone as the trades.

```bash
lookahead-audit --trades my_trades.csv --prices XAUUSD_H1.csv \
    --tol 0.30 --cost 0.25 --contract-size 100 \
    --map-trades "entry_time=Open Time,exit_time=Close Time,direction=Type" \
    --html audit.html --json audit.json --csv audit_trades.csv
```

- `--tol` is the price distance treated as "equal". Set it to about your spread; the default is 2% of the median bar range.
- The exit code is `0` if clean, `1` if suspicious, `2` if bias is likely, and `3` on an input error. This lets you use it as a CI gate on every backtest.

### Python API

```python
from lookahead_auditor import audit, load_prices, load_trades, AuditConfig, to_text

res = audit(load_trades("trades.csv"), load_prices("prices.csv"), AuditConfig(tol=0.3, cost=0.25))
print(to_text(res))
res.trades.query("flags != ''")      # per-trade flags and corrected prices
```

## Limits

- It audits **execution**: fills, exits and intrabar ordering. It cannot see look-ahead inside your **signals**, such as indicators computed on unfinished bars, labels leaking into features, or future-normalised data. For that, compare live and backtest decisions bar by bar.
- It works with OHLC bars, so intrabar paths are unknown by design. When in doubt, it is pessimistic.
- Break-even and trailing stops are not re-simulated; their exits are validated as discretionary exits.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. Need a full audit of your trading bot or backtester? Open an issue or contact me.
