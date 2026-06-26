"""
app.py — AlgoForge Strategy Lab backend
Run: python app.py → http://localhost:5001
"""
import json, os, sys, time, tempfile, subprocess, traceback
from pathlib import Path
from datetime import datetime

import numpy  as np
import pandas as pd
from flask import Flask, jsonify, request, send_file

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
app  = Flask(__name__)
app.json.sort_keys = False

# ── Storage ───────────────────────────────────────────────────────────────────
STRAT_FILE = BASE / "strategies.json"
STRAT_DIR  = BASE / "strategies"
STRAT_DIR.mkdir(exist_ok=True)

BUILTIN_STRATEGIES = [
    {
        "id": "fvg", "name": "FVG Pullback", "ticker": "QQQ", "tf": "15m",
        "dateFrom": "2019-01-01", "dateTo": "2024-12-31",
        "capital": 10000, "riskPct": 0.7, "rr": 3.0, "dailyLim": 5.0,
        "htfEma": 50, "htfSwing": 20, "ltfAtr": 14, "ltfSwing": 10,
        "wickThresh": 0.60, "atrFilt": 0.3, "commPct": 0.01, "slipPts": 2.0,
        "entryRules": (
            "Enter long when a bullish Fair Value Gap forms on the 15-minute chart "
            "during the NY session (13:30–20:00 UTC).\n\n"
            "Wait for price to retrace into the FVG zone (between the gap high and gap low).\n\n"
            "The 4H bias must be bullish — price above the 50 EMA on the 4H chart.\n\n"
            "Only take gaps with minimum displacement of 0.5 ATR."
        ),
        "exitRules": (
            "Take profit at 3:1 reward-to-risk.\n"
            "Stop loss is placed 0.1 ATR below the bottom of the FVG zone.\n"
            "Daily loss limit: 5% of capital — halt new entries for the day."
        ),
        "filterRules": (
            "4H EMA-50 trend filter — only trade in the trend direction.\n"
            "FVG minimum size: 0.3 ATR.\n"
            "Maximum FVG age: 20 bars.\n"
            "Displacement confirmation: 0.5 ATR minimum move before the gap."
        ),
        "cachedMetrics": {
            "annReturn": 15.27, "sharpe": 0.97, "maxDD": -15.69,
            "winRate": 52.0, "profitFactor": 1.44, "nTrades": 680,
            "yearly": {"2019": 12.1, "2020": 18.3, "2021": 14.6,
                       "2022": -3.2, "2023": 21.8, "2024": 9.9},
        },
    },
    {
        "id": "orb", "name": "ORB Breakout", "ticker": "QQQ", "tf": "15m",
        "dateFrom": "2019-01-01", "dateTo": "2024-12-31",
        "capital": 10000, "riskPct": 0.7, "rr": 3.0, "dailyLim": 5.0,
        "htfEma": 50, "htfSwing": 20, "ltfAtr": 14, "ltfSwing": 10,
        "wickThresh": 0.60, "atrFilt": 0.3, "commPct": 0.01, "slipPts": 2.0,
        "entryRules": (
            "Define the opening range as the first 4 × 15M candles of the NY session "
            "(13:30–14:15 UTC = 1 hour).\n\n"
            "Enter long when price closes above the opening range high for the first time.\n"
            "Enter short when price closes below the opening range low for the first time.\n\n"
            "4H EMA-50 trend filter: only trade breakouts in the trend direction."
        ),
        "exitRules": (
            "Take profit at 3:1 reward-to-risk.\n"
            "Stop loss at the opposite side of the opening range.\n"
            "One trade per day maximum."
        ),
        "filterRules": (
            "4H trend must align with the breakout direction.\n"
            "Entry window: after OR is complete until 18:00 UTC.\n"
            "Daily loss limit: 5% of capital."
        ),
        "cachedMetrics": {
            "annReturn": 11.13, "sharpe": 0.91, "maxDD": -8.63,
            "winRate": 35.0, "profitFactor": 1.41, "nTrades": 345,
            "yearly": {"2019": 8.4, "2020": 15.1, "2021": 11.0,
                       "2022": 4.7, "2023": 13.2, "2024": 14.4},
        },
    },
    {
        "id": "amd", "name": "AMD — ICT Sweep", "ticker": "QQQ", "tf": "15m",
        "dateFrom": "2019-01-01", "dateTo": "2024-12-31",
        "capital": 10000, "riskPct": 0.7, "rr": 2.5, "dailyLim": 5.0,
        "htfEma": 50, "htfSwing": 20, "ltfAtr": 14, "ltfSwing": 10,
        "wickThresh": 0.60, "atrFilt": 0.3, "commPct": 0.01, "slipPts": 2.0,
        "entryRules": (
            "Identify the overnight range: high and low of bars outside the NY session "
            "(before 13:30 UTC or after 20:00 UTC).\n\n"
            "Accumulation/Manipulation: during the NY kill zone (13:30–16:00 UTC), "
            "price sweeps below the overnight low by at least 0.05 ATR.\n\n"
            "Distribution entry: first 15M bar that closes BACK ABOVE the overnight low "
            "after the sweep — this is the signal."
        ),
        "exitRules": (
            "Take profit at 2.5:1 reward-to-risk.\n"
            "Stop loss at the wick extreme of the sweep candle (lowest low during session).\n"
            "SL buffer: 0.1 ATR beyond the wick extreme."
        ),
        "filterRules": (
            "4H EMA-50 trend filter.\n"
            "Overnight range must be clearly defined.\n"
            "Entry window: NY kill zone only (13:30–16:00 UTC).\n"
            "Daily loss limit: 5%."
        ),
        "cachedMetrics": {
            "annReturn": 19.74, "sharpe": 1.36, "maxDD": -9.21,
            "winRate": 58.0, "profitFactor": 1.72, "nTrades": 412,
            "yearly": {"2019": 14.2, "2020": 22.4, "2021": 18.1,
                       "2022": 45.9, "2023": 12.3, "2024": 15.3},
        },
    },
]

if not STRAT_FILE.exists():
    STRAT_FILE.write_text(json.dumps(BUILTIN_STRATEGIES, indent=2))

# ── Engine import ─────────────────────────────────────────────────────────────
ENGINE_OK        = False
ENGINE_ERR       = ""
ENGINE_STRATEGIES = {}

try:
    from engine import (
        load_alpaca_csv, build_indicators, run_backtest, DEFAULT_PARAMS,
        build_indicators_fvg, run_backtest_fvg, FVG_PARAMS,
        build_indicators_orb, run_backtest_orb, ORB_PARAMS,
        build_indicators_amd, run_backtest_amd, AMD_PARAMS,
    )
    ENGINE_OK = True

    ENGINE_STRATEGIES = {
        "fvg": {
            "params":   FVG_PARAMS,
            "build_fn": lambda ltf, h1, h4, p: build_indicators_fvg(ltf, h4, p, htf_1h=h1),
            "run_fn":   run_backtest_fvg,
        },
        "orb": {
            "params":   ORB_PARAMS,
            "build_fn": lambda ltf, h1, h4, p: build_indicators_orb(ltf, h4, p),
            "run_fn":   run_backtest_orb,
        },
        "amd": {
            "params":   AMD_PARAMS,
            "build_fn": lambda ltf, h1, h4, p: build_indicators_amd(ltf, h4, p),
            "run_fn":   run_backtest_amd,
        },
    }
except Exception as _e:
    ENGINE_ERR = str(_e)

# ── Data file map ─────────────────────────────────────────────────────────────
TF_LTF = {"15m": "QQQ_15M.csv", "1h": "QQQ_1H.csv", "4h": "QQQ_4H.csv"}
TF_H1  = {"15m": "QQQ_1H.csv",  "1h": "QQQ_1H.csv", "4h": "QQQ_1H.csv"}
TF_H4  = {"15m": "QQQ_4H.csv",  "1h": "QQQ_4H.csv", "4h": "QQQ_4H.csv"}


def _load_bars(d_from: str, d_to: str, tf: str = "15m"):
    ltf_path = BASE / TF_LTF.get(tf, "QQQ_15M.csv")
    h1_path  = BASE / TF_H1.get(tf,  "QQQ_1H.csv")
    h4_path  = BASE / TF_H4.get(tf,  "QQQ_4H.csv")
    if not ltf_path.exists():
        raise FileNotFoundError(f"Data file not found: {ltf_path.name}")
    ltf = load_alpaca_csv(str(ltf_path)).loc[d_from:d_to]
    h1  = load_alpaca_csv(str(h1_path)).loc[d_from:d_to]
    h4  = load_alpaca_csv(str(h4_path)).loc[d_from:d_to]
    return ltf, h1, h4


# ── Metric helpers ────────────────────────────────────────────────────────────
def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    equity = equity.dropna()
    if len(equity) < 10:
        return {"error": "Not enough equity data points"}

    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1)
    days      = max((equity.index[-1] - equity.index[0]).days, 1)
    ann_ret   = float((1 + total_ret) ** (365.25 / days) - 1)

    daily  = equity.resample("D").last().ffill().pct_change().dropna()
    sharpe = (
        float(daily.mean() / daily.std() * np.sqrt(252))
        if len(daily) > 30 and daily.std() > 0 else 0.0
    )

    rolling_max = equity.cummax()
    max_dd = float(((equity - rolling_max) / rolling_max).min())

    exits    = trades[trades["pnl"].notna() & (trades["pnl"] != 0)] if "pnl" in trades.columns else pd.DataFrame()
    n_trades = len(exits)
    wins     = exits[exits["pnl"] > 0] if n_trades else pd.DataFrame()
    losses   = exits[exits["pnl"] < 0] if n_trades else pd.DataFrame()
    wr = float(len(wins) / n_trades) if n_trades else 0.0
    pf = (
        float(abs(wins["pnl"].sum() / losses["pnl"].sum()))
        if len(losses) > 0 and losses["pnl"].sum() != 0 else 0.0
    )

    yearly = {}
    for yr in sorted(equity.index.year.unique()):
        yr_eq = equity[equity.index.year == yr]
        if len(yr_eq) >= 2:
            yearly[str(yr)] = round(float(yr_eq.iloc[-1] / yr_eq.iloc[0] - 1) * 100, 2)

    step      = max(1, len(equity) // 400)
    ec_sample = equity.iloc[::step]

    return {
        "annReturn":    round(ann_ret * 100, 2),
        "sharpe":       round(sharpe, 2),
        "maxDD":        round(max_dd * 100, 2),
        "winRate":      round(wr * 100, 1),
        "profitFactor": round(pf, 2),
        "nTrades":      n_trades,
        "finalEquity":  round(float(equity.iloc[-1]), 2),
        "yearly":       yearly,
        "equityCurve":  [round(float(v), 2) for v in ec_sample.tolist()],
        "equityDates":  [str(d.date()) for d in ec_sample.index],
    }


def pair_trades(trades_df: pd.DataFrame) -> list:
    pairs, pending = [], None
    for _, row in trades_df.iterrows():
        t   = row.to_dict()
        typ = str(t.get("type", ""))
        if "ENTER" in typ:
            pending = t
        elif typ == "EXIT" and pending is not None:
            d        = pending["date"]
            date_str = d.strftime("%d %b %y") if hasattr(d, "strftime") else str(d)[:10]
            pairs.append({
                "date":   date_str,
                "side":   "L" if "LONG" in str(pending.get("type", "")).upper() else "S",
                "entry":  round(float(pending.get("entry", 0)), 2),
                "reason": t.get("exit_reason", ""),
                "pnl":    round(float(t.get("pnl", 0)), 2),
            })
            pending = None
    return list(reversed(pairs))


def run_custom_code(code_str: str, ltf_path: str, htf_path: str,
                    d_from: str, d_to: str, params: dict) -> dict:
    runner = f"""import sys, json, types
sys.path.insert(0, r"{str(BASE)}")
from engine import load_alpaca_csv, build_indicators

{code_str}

user_fns = [
    (name, obj) for name, obj in list(globals().items())
    if callable(obj) and isinstance(obj, types.FunctionType)
    and name.startswith("run_") and name != "run_backtest"
]
if not user_fns:
    print(json.dumps({{"error": "No run_* function found in generated code"}}))
    sys.exit(1)

fn_name, fn = user_fns[0]
p   = {json.dumps(params)}
ltf = load_alpaca_csv(r"{ltf_path}")
htf = load_alpaca_csv(r"{htf_path}")
ltf = ltf.loc["{d_from}":"{d_to}"]
htf = htf.loc["{d_from}":"{d_to}"]
df  = build_indicators(ltf, htf, p)
equity, trades = fn(df, p)

print(json.dumps({{
    "equity":      equity.tolist(),
    "equityDates": [str(d.date()) for d in equity.index],
    "trades":      trades.to_dict("records"),
}}, default=str))
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=str(STRAT_DIR)
    ) as f:
        f.write(runner)
        tmp = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=120, cwd=str(BASE),
        )
        if result.returncode != 0:
            raise ValueError(result.stderr[-3000:] or "Strategy execution failed")
        return json.loads(result.stdout)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_file("strategy_lab.html")


@app.route("/api/strategies", methods=["GET"])
def list_strategies():
    return jsonify(json.loads(STRAT_FILE.read_text()))


@app.route("/api/strategies", methods=["POST"])
def save_strategy():
    data = request.json
    strategies = json.loads(STRAT_FILE.read_text())
    sid  = data.get("id") or f"strat_{int(time.time())}"
    data["id"]        = sid
    data["updatedAt"] = datetime.utcnow().isoformat()
    idx = next((i for i, s in enumerate(strategies) if s.get("id") == sid), None)
    if idx is not None:
        strategies[idx] = data
    else:
        strategies.append(data)
    STRAT_FILE.write_text(json.dumps(strategies, indent=2))
    return jsonify({"ok": True, "id": sid})


@app.route("/api/strategies/<sid>", methods=["DELETE"])
def delete_strategy(sid):
    strategies = [s for s in json.loads(STRAT_FILE.read_text()) if s.get("id") != sid]
    STRAT_FILE.write_text(json.dumps(strategies, indent=2))
    return jsonify({"ok": True})


TRANSLATE_SYSTEM = """You are an expert algorithmic trading developer. Convert plain-English trading rules into Python strategy code that is compatible with the user's engine.py backtesting framework.

=== engine.py API ===
load_alpaca_csv(path) → pd.DataFrame  (OHLCV with Datetime index)
build_indicators(ltf_df, htf_df, p) → pd.DataFrame with these columns:
  Open, High, Low, Close, Volume
  htf_ema50, htf_swing_high, htf_swing_low
  htf_bias_bull  (int: 1 when Close > htf_ema50)
  htf_bias_bear  (int: 1 when Close < htf_ema50)
  atr            (LTF ATR-14, EMA-smoothed)
  ema20          (LTF 20-period EMA)
  swing_high, swing_low  (LTF rolling max/min, shifted 1 bar)
  lower_wick, upper_wick, lower_wick_prev, upper_wick_prev  (0–1 ratio of candle range)
  bos_long       (int: 1 when Close > swing_high — Break of Structure long)
  bos_short      (int: 1 when Close < swing_low  — Break of Structure short)

run_backtest(df, p) → (equity_curve: pd.Series, trade_log: pd.DataFrame)
trade_log row types:
  ENTER_LONG / ENTER_SHORT → {date, type, entry, sl, tp, cost, pnl: None}
  EXIT                     → {date, type, exit_reason: "SL"|"TP", pnl, cost}

PARAMS dict keys:
  initial_capital, risk_pct, reward_ratio, daily_loss_limit,
  htf_ema_span, htf_swing_bars, ltf_atr_span, ltf_swing_bars,
  wick_threshold, atr_filter_pct, commission_pct, slippage_pts

=== Requirements ===
1. Define PARAMS_STRATNAME = { ...all keys from above with values from user config... }
2. Add any helper functions needed above the main run function
3. Define run_stratname(df, p=None) -> tuple[pd.Series, pd.DataFrame]:
   - Loop over df.iterrows() exactly like engine.py's run_backtest()
   - Append to equity_curve list each bar
   - Append entry/exit dicts to trade_log list with the EXACT same schema
   - Return (pd.Series(equity_curve, index=df.index[:len(equity_curve)]), pd.DataFrame(trade_log))
4. Add a if __name__ == "__main__": usage block
5. Return ONLY Python code — no markdown fences, no explanation text"""


@app.route("/api/translate", methods=["POST"])
def translate():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not set.\nRun: export ANTHROPIC_API_KEY=sk-ant-..."}), 500
    try:
        import anthropic as _ant
    except ImportError:
        return jsonify({"error": "anthropic package missing. Run: pip install anthropic"}), 500

    data = request.json
    cfg  = data.get("config", {})

    user_msg = f"""Strategy name : {data.get('stratName', 'MyStrategy')}
Ticker        : {cfg.get('ticker', 'QQQ')}   Timeframe: {cfg.get('tf', '15m')}

Config params to embed in PARAMS dict:
  initial_capital  = {cfg.get('capital', 10000)}
  risk_pct         = {float(cfg.get('riskPct', 0.5)) / 100:.4f}
  reward_ratio     = {cfg.get('rr', 2.0)}
  daily_loss_limit = {float(cfg.get('dailyLim', 6)) / 100:.3f}
  htf_ema_span     = {cfg.get('htfEma', 50)}
  htf_swing_bars   = {cfg.get('htfSwing', 20)}
  ltf_atr_span     = {cfg.get('ltfAtr', 14)}
  ltf_swing_bars   = {cfg.get('ltfSwing', 10)}
  wick_threshold   = {cfg.get('wickThresh', 0.60)}
  atr_filter_pct   = {float(cfg.get('atrFilt', 0.3)) / 100:.4f}
  commission_pct   = {float(cfg.get('commPct', 0.01)) / 100:.5f}
  slippage_pts     = {cfg.get('slipPts', 2.0)}

--- ENTRY RULES ---
{data.get('entryRules', '')}

--- EXIT RULES ---
{data.get('exitRules', '')}

--- FILTERS & CONDITIONS ---
{data.get('filterRules', '')}

Generate the complete Python strategy code now."""

    try:
        client = _ant.Anthropic(api_key=api_key)
        resp   = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=4096,
            system=TRANSLATE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        code = resp.content[0].text.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code  = "\n".join(lines[1:])
        if code.rstrip().endswith("```"):
            code = code.rstrip()[:-3].rstrip()
        return jsonify({"code": code})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest", methods=["POST"])
def backtest():
    if not ENGINE_OK:
        return jsonify({"error": f"engine.py failed to import: {ENGINE_ERR}"}), 500

    data        = request.json
    tf          = data.get("tf", "15m")
    d_from      = data.get("dateFrom", "2019-01-01")
    d_to        = data.get("dateTo",   "2024-12-31")
    custom_code = data.get("generatedCode", "").strip()
    cfg         = data.get("config", {})
    strat_id    = data.get("strategyId", "")
    mode        = "custom" if custom_code else ("engine" if strat_id in ENGINE_STRATEGIES else "wick_bos")

    ltf_file = TF_LTF.get(tf)
    if not ltf_file:
        return jsonify({"error": f'No data file for TF "{tf}". Supported: {list(TF_LTF.keys())}'}), 400

    ltf_path = BASE / ltf_file
    h1_path  = BASE / TF_H1.get(tf, "QQQ_1H.csv")
    h4_path  = BASE / TF_H4.get(tf, "QQQ_4H.csv")
    if not ltf_path.exists():
        return jsonify({"error": f"Data file not found: {ltf_file}\nRun fetch_alpaca.py to download data."}), 400

    try:
        # Build the param dict (int_keys must stay integers for pandas rolling())
        int_keys = {"htf_ema_span", "htf_swing_bars", "ltf_atr_span", "ltf_swing_bars"}

        if custom_code:
            p = dict(DEFAULT_PARAMS)
            for key, eng_key, div in [
                ("capital",    "initial_capital",  1),
                ("rr",         "reward_ratio",      1),
                ("slipPts",    "slippage_pts",       1),
                ("htfEma",     "htf_ema_span",       1),
                ("htfSwing",   "htf_swing_bars",     1),
                ("ltfAtr",     "ltf_atr_span",       1),
                ("ltfSwing",   "ltf_swing_bars",     1),
                ("wickThresh", "wick_threshold",     1),
                ("riskPct",    "risk_pct",         100),
                ("dailyLim",   "daily_loss_limit", 100),
                ("atrFilt",    "atr_filter_pct",   100),
                ("commPct",    "commission_pct",   100),
            ]:
                if cfg.get(key) not in (None, ""):
                    v = float(cfg[key]) / div
                    p[eng_key] = int(v) if eng_key in int_keys else v

            raw = run_custom_code(
                custom_code, str(ltf_path), str(h1_path), d_from, d_to, p
            )
            if "error" in raw:
                return jsonify({"error": raw["error"]}), 400
            equity = pd.Series(raw["equity"], index=pd.to_datetime(raw["equityDates"]))
            trades_df = pd.DataFrame(raw.get("trades", []))
            if not trades_df.empty and "exit_reason" not in trades_df.columns:
                trades_df["exit_reason"] = trades_df.get("reason", "")

        elif strat_id in ENGINE_STRATEGIES:
            strat    = ENGINE_STRATEGIES[strat_id]
            p        = dict(strat["params"])
            build_fn = strat["build_fn"]
            run_fn   = strat["run_fn"]

            # Override with any user-supplied config fields that exist in params
            field_map = {
                "capital": "initial_capital", "rr": "reward_ratio",
                "riskPct": ("risk_pct", 100), "dailyLim": ("daily_loss_limit", 100),
                "commPct": ("commission_pct", 100),
            }
            for ui_key, eng_info in field_map.items():
                if cfg.get(ui_key) not in (None, ""):
                    if isinstance(eng_info, tuple):
                        eng_key, div = eng_info
                        p[eng_key] = float(cfg[ui_key]) / div
                    else:
                        p[eng_info] = float(cfg[ui_key])

            ltf = load_alpaca_csv(str(ltf_path)).loc[d_from:d_to]
            h1  = load_alpaca_csv(str(h1_path)).loc[d_from:d_to]
            h4  = load_alpaca_csv(str(h4_path)).loc[d_from:d_to]
            if len(ltf) < 50:
                return jsonify({"error": "Not enough bars in that date range"}), 400
            df       = build_fn(ltf, h1, h4, p)
            equity, trades_df = run_fn(df, p)

        else:
            # Fallback: original wick-BOS strategy
            p = dict(DEFAULT_PARAMS)
            for key, eng_key, div in [
                ("capital",    "initial_capital",  1),
                ("rr",         "reward_ratio",      1),
                ("slipPts",    "slippage_pts",       1),
                ("htfEma",     "htf_ema_span",       1),
                ("htfSwing",   "htf_swing_bars",     1),
                ("ltfAtr",     "ltf_atr_span",       1),
                ("ltfSwing",   "ltf_swing_bars",     1),
                ("wickThresh", "wick_threshold",     1),
                ("riskPct",    "risk_pct",         100),
                ("dailyLim",   "daily_loss_limit", 100),
                ("atrFilt",    "atr_filter_pct",   100),
                ("commPct",    "commission_pct",   100),
            ]:
                if cfg.get(key) not in (None, ""):
                    v = float(cfg[key]) / div
                    p[eng_key] = int(v) if eng_key in int_keys else v

            ltf = load_alpaca_csv(str(ltf_path)).loc[d_from:d_to]
            htf = load_alpaca_csv(str(h1_path)).loc[d_from:d_to]
            if len(ltf) < 50:
                return jsonify({"error": "Not enough bars in that date range"}), 400
            df = build_indicators(ltf, htf, p)
            equity, trades_df = run_backtest(df, p)

        metrics = compute_metrics(equity, trades_df)
        if "error" in metrics:
            return jsonify(metrics), 400

        metrics["trades"] = pair_trades(trades_df)
        metrics["mode"]   = mode
        return jsonify(metrics)

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Backtest timed out (>120 s). Simplify the strategy or narrow the date range."}), 408
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()[-3000:]}), 500


@app.route("/api/edge", methods=["POST"])
def edge():
    if not ENGINE_OK:
        return jsonify({"error": f"engine.py import failed: {ENGINE_ERR}"}), 500

    data     = request.json
    strat_id = data.get("strategyId", "fvg")
    d_from   = data.get("dateFrom", "2019-01-01")
    d_to     = data.get("dateTo", "2024-12-31")

    strat = ENGINE_STRATEGIES.get(strat_id)
    if not strat:
        return jsonify({
            "error": f"Edge analysis only available for built-in strategies: {list(ENGINE_STRATEGIES.keys())}"
        }), 400

    params   = dict(strat["params"])
    build_fn = strat["build_fn"]
    run_fn   = strat["run_fn"]

    try:
        ltf_path = BASE / "QQQ_15M.csv"
        h1_path  = BASE / "QQQ_1H.csv"
        h4_path  = BASE / "QQQ_4H.csv"
        if not ltf_path.exists():
            return jsonify({"error": "QQQ_15M.csv not found — run fetch_alpaca.py first"}), 400

        ltf = load_alpaca_csv(str(ltf_path)).loc[d_from:d_to]
        h1  = load_alpaca_csv(str(h1_path)).loc[d_from:d_to]
        h4  = load_alpaca_csv(str(h4_path)).loc[d_from:d_to]

        # Build indicators once on the full period
        df_full = build_fn(ltf, h1, h4, params)

        # IS/OOS split at 70%
        split_idx  = int(len(df_full) * 0.70)
        split_date = df_full.index[split_idx]
        df_is      = df_full.iloc[:split_idx]
        df_oos     = df_full.iloc[split_idx:]

        eq_is,  tr_is  = run_fn(df_is,  params)
        eq_oos, tr_oos = run_fn(df_oos, params)
        m_is  = compute_metrics(eq_is, tr_is)
        m_oos = compute_metrics(eq_oos, tr_oos)

        # Full-period run for monthly heatmap
        eq_full, _ = run_fn(df_full, params)

        # Monthly returns
        monthly_eq  = eq_full.resample("ME").last().ffill()
        monthly_ret = monthly_eq.pct_change().dropna()
        monthly = {}
        for ts, ret in monthly_ret.items():
            yr = str(ts.year)
            mo = ts.strftime("%b")
            monthly.setdefault(yr, {})[mo] = round(float(ret) * 100, 2)

        # R:R sensitivity sweep (full period)
        rr_sweep = []
        for rr in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
            p_s = dict(params)
            p_s["reward_ratio"] = rr
            eq_s, tr_s = run_fn(df_full, p_s)
            m_s = compute_metrics(eq_s, tr_s)
            rr_sweep.append({
                "rr":           rr,
                "annReturn":    m_s.get("annReturn",    0),
                "sharpe":       m_s.get("sharpe",       0),
                "profitFactor": m_s.get("profitFactor", 0),
                "winRate":      m_s.get("winRate",      0),
                "nTrades":      m_s.get("nTrades",      0),
            })

        # Normalized IS/OOS equity curves (both start at 100)
        def _norm_curve(eq):
            if len(eq) < 2:
                return [], []
            step = max(1, len(eq) // 300)
            s    = eq.iloc[::step]
            norm = (s / s.iloc[0] * 100).round(2).tolist()
            dates = [str(d.date()) for d in s.index]
            return norm, dates

        is_curve,  is_dates  = _norm_curve(eq_is)
        oos_curve, oos_dates = _norm_curve(eq_oos)

        # Strip heavy fields from IS/OOS metrics
        trim = ("equityCurve", "equityDates", "yearly")
        return jsonify({
            "is":        {k: v for k, v in m_is.items()  if k not in trim},
            "oos":       {k: v for k, v in m_oos.items() if k not in trim},
            "splitDate": str(split_date.date()),
            "isCurve":   is_curve,
            "isCurveDates":  is_dates,
            "oosCurve":  oos_curve,
            "oosCurveDates": oos_dates,
            "rrSweep":   rr_sweep,
            "monthly":   monthly,
            "defaultRR": params.get("reward_ratio", 2.5),
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()[-3000:]}), 500


@app.route("/api/export", methods=["POST"])
def export_code():
    data = request.json
    code = data.get("code", "").strip()
    if not code:
        return jsonify({"error": "No code to export"}), 400
    name = data.get("name", "strategy")
    safe = "".join(
        c if c.isalnum() or c in "_-" else "_"
        for c in name.lower().replace(" ", "_")
    ).strip("_") or "strategy"
    path = STRAT_DIR / f"{safe}.py"
    path.write_text(code)
    return jsonify({"ok": True, "path": str(path), "filename": f"{safe}.py"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n  AlgoForge Strategy Lab  →  http://localhost:{port}\n")
    app.run(debug=True, port=port, use_reloader=False)
