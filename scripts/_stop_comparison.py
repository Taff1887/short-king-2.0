"""Side-by-side: backtest with 15 % stop vs no stop. Long-short quintile only.

Pulls backtest_summary_monthly_withstop.csv and backtest_summary_monthly_nostop.csv
(produced by re-running 06 with default stop and --stop-loss-pct 1.0), trims to
the L/S quintile rows for each model, and writes a tidy markdown table.
"""

import pandas as pd
from short_king.utils.config import settings

a = pd.read_csv(settings.reports_dir / "backtest_summary_monthly_withstop.csv")
b = pd.read_csv(settings.reports_dir / "backtest_summary_monthly_nostop.csv")
a["variant"] = "with_stop"
b["variant"] = "no_stop"
df = pd.concat([a, b], ignore_index=True)

keep = df[df["strategy"] == "long_short_quintile"].copy()

# Pivot for display: rows = model, columns = (period, variant, metric).
def _row(model, period):
    sub = keep[(keep["model"] == model) & (keep["period"] == period)].set_index("variant")
    with_s = sub.loc["with_stop"]
    no_s = sub.loc["no_stop"]
    return {
        "model": model, "period": period,
        "Sharpe (with)": round(with_s["Sharpe"], 2),
        "Sharpe (no)": round(no_s["Sharpe"], 2),
        "Δ Sharpe": round(with_s["Sharpe"] - no_s["Sharpe"], 2),
        "CAGR (with)": f"{with_s['CAGR']*100:+.1f}%",
        "CAGR (no)": f"{no_s['CAGR']*100:+.1f}%",
        "MaxDD (with)": f"{with_s['MaxDD']*100:.1f}%",
        "MaxDD (no)": f"{no_s['MaxDD']*100:.1f}%",
    }

rows = []
for m in ("naive", "logit", "gbm_rank", "gbm_cls", "ew"):
    for p in ("IS", "OOS"):
        rows.append(_row(m, p))
out = pd.DataFrame(rows)
out_md = settings.reports_dir / "stop_comparison.md"
md = "# Stop-loss impact — long-short quintile only\n\n"
md += "Side-by-side OOS / IS comparison of the same 5 models with the 15 % "
md += "per-position hard stop (default) vs with the stop disabled "
md += "(`--stop-loss-pct 1.0`). Everything else is identical: monthly "
md += "rebalance, 25 bps commission, 1.5 % p.a. borrow, 5 bps slippage, "
md += "Friday-release rebalance dates.\n\n"
try:
    md += out.to_markdown(index=False)
except (ImportError, ValueError):
    hdr = "| " + " | ".join(out.columns) + " |\n"
    sep = "|" + "|".join(["---"] * len(out.columns)) + "|\n"
    body = "\n".join("| " + " | ".join(str(v) for v in r.values) + " |" for _, r in out.iterrows())
    md += hdr + sep + body
out_md.write_text(md + "\n", encoding="utf-8")
print(md)
