# AlgoForge

AI-powered strategy lab. Describe a trading strategy in plain English — AlgoForge translates it into working Python code, runs a backtest, and returns results in seconds.

## How it works

```
Plain English description
        ↓
  Claude API (AI translation)
        ↓
  Generated Python strategy code
        ↓
  Backtesting engine
        ↓
  Results: Sharpe, MaxDD, equity curve, yearly breakdown
```

## Features

- **Natural language → strategy code** via Claude API
- **Live backtesting** — generated code runs against real historical data
- **Strategy library** — save, load, and compare strategies
- **Pre-built examples** — FVG Pullback, ORB Breakout included
- **Yearly breakdown** — see performance by year to spot consistency

## Quick start

```bash
pip install -r requirements_lab.txt
export ANTHROPIC_API_KEY=your_key_here
python app.py
# Open http://localhost:5001
```

Or using a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
python app.py
```

## Architecture

- `app.py` — Flask backend: handles Claude API calls, backtest execution, strategy storage
- `strategy_lab.html` — single-page UI (vanilla JS, no framework)
- `strategies/` — strategy library (JSON metadata + generated Python code)
- `engine.py` — vectorised backtesting engine (subset of [quant-engine](https://github.com/jmb/quant-engine))

## Example

Type in the UI:
> "Enter long when the 20-period EMA crosses above the 50-period EMA on the daily chart. Exit when it crosses back below. Apply to QQQ from 2019 to 2024."

AlgoForge generates the strategy code, runs the backtest, and returns:
- Equity curve chart
- Sharpe ratio, MaxDD, annual return
- Yearly PnL breakdown
- Editable generated code

## Tech stack

Python · Flask · Claude API (claude-sonnet) · pandas · numpy · vanilla JS
