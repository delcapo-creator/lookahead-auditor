"""lookahead-auditor: find look-ahead and execution bias in backtest trade logs."""
from .audit import AuditConfig, AuditResult, audit
from .loader import load_prices, load_trades
from .report import to_html, to_json, to_text

__all__ = ["audit", "AuditConfig", "AuditResult", "load_prices", "load_trades", "to_text", "to_html", "to_json"]
__version__ = "0.1.0"
