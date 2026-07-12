# Autonomous Self-Learning Trading Machine

A fully autonomous trading machine that learns to trade from raw price data only — no human indicators, no preconceived trading concepts. The machine discovers market structure through unsupervised learning and reinforcement learning, then continuously improves through loss forensics and auto-adaptation.

## Architecture

```
trading_machine/
├── data/           Data fetching, storage, validation
├── models/         PriceVAE (world model) + PPO RL agent
├── evolution/      Loss forensics, auto-adaptation, backtesting, regime detection
├── live/           Real-time screening, position management, alerts
├── dashboard/      Streamlit dashboard with 5 panels
├── utils/          Configuration, logging, scheduling
├── run.py          CLI entry point
└── setup.py        Package setup
```

## How It Works

1. **World Model (PriceVAE)**: A variational autoencoder compresses 500-tick price windows into 256-dim latent states. It learns to reconstruct future prices, discovering market structure without any human concepts.

2. **RL Agent (PPO)**: A reinforcement learning agent maps latent states to trading actions (Long/Short/Hold). It never sees prices — only the compressed latent representation.

3. **Loss Forensics**: Every losing trade is classified into 4 types (stop too tight, wrong direction, regime change, noise). The system auto-adapts parameters based on findings.

4. **Regime Detection**: Market regime changes are detected purely through cosine distance between latent vectors — no volatility or trend indicators needed.

5. **Dashboard**: Streamlit dashboard with price charts, latent space visualization, live signals, backtest results, and learning logs.

## Setup

### Prerequisites

- Python 3.12+
- FMP (Financial Modeling Prep) API key

### Installation

```bash
cd trading_machine
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Set your FMP API key:

```bash
export FMP_API_KEY=your_key_here
```

Or create a `.env` file in the project root:

```
FMP_API_KEY=your_key_here
```

All configuration defaults are in `utils/config.py`.

## Usage

### Fetch Historical Data

```bash
python run.py fetch
```

Fetches 1-minute intraday data for all configured tickers (TSLA, AMZN, NVDA, CSCO, SPY, QQQ, IWM, SPX, NFLX) from 2018-01-01 to 2026-07-10.

### Train Models

```bash
python run.py train                    # Train all tickers
python run.py train --tickers SPY AAPL # Train specific tickers
python run.py train --epochs 50         # Fewer world model epochs
python run.py train --skip-rl           # Only train world model
```

### Run Backtests

```bash
python run.py backtest                           # Backtest all trained tickers
python run.py backtest --tickers SPY             # Backtest specific ticker
python run.py backtest --forensics               # Run loss forensics after backtest
python run.py backtest --forensics --adapt        # Auto-adapt parameters
```

### Launch Dashboard

```bash
python run.py dashboard
```

Opens at http://localhost:8501 with 5 panels:
- **Summary**: Equity curve, P&L stats, win rate
- **Live Signals**: Current signals ranked by confidence
- **Backtest**: Historical backtest performance
- **Learning Log**: Forensics reports, model versions, discoveries
- **Charts**: OHLCV price charts, latent space PCA, regime map

### Live Trading

```bash
python run.py live
```

Starts real-time trading with:
- 1-second polling during market hours (09:30-16:00 EST)
- Automatic signal generation via world model + RL agent
- Position and risk management with max drawdown enforcement
- Overnight and weekend processing (forensics, retraining)

### System Status

```bash
python run.py status
```

Shows model versions, data availability, and ticker readiness.

### Export Data

```bash
python run.py export --output ./reports
```

Exports trades, backtests, and forensics to Excel files.

## Key Design Decisions

- **Input is ONLY closing prices**: `DataStorage.get_close_prices()` returns raw numpy arrays. No open, high, low, volume, or derived indicators are ever passed to models.
- **Per-ticker model isolation**: Each ticker gets completely separate PriceVAE and RLAgent — no shared weights.
- **Latent space decisions**: The RL agent operates entirely in latent space (256-dim vectors). It never sees prices.
- **Machine-discovered stops**: Stop distances are computed from favorable excursion distributions — NOT ATR.
- **Regime detection via cosine distance**: Market shifts are detected by measuring drift in latent space, not by coding volatility/trend indicators.

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| TICKERS | TSLA,AMZN,NVDA,CSCO,SPY,QQQ,IWM,SPX,NFLX | Traded tickers |
| START_DATE | 2018-01-01 | Historical data start |
| END_DATE | 2026-07-10 | Historical data end |
| LATENT_DIM | 256 | VAE latent dimension |
| INPUT_WINDOW_TICKS | 500 | Past price window size |
| FUTURE_WINDOW_TICKS | 100 | Future price window size |
| BATCH_SIZE | 64 | Training batch size |
| LEARNING_RATE | 0.0001 | World model learning rate |
| PPO_EPOCHS | 10 | PPO epochs per update |
| INITIAL_CAPITAL | 100000.0 | Starting capital for backtesting |
| MAX_DRAWDOWN_PCT | 0.50 | Maximum allowed drawdown |
| DASHBOARD_PORT | 8501 | Streamlit dashboard port |
| POLLING_INTERVAL_SECONDS | 1 | Live trading polling interval |
