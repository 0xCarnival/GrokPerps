# HyperGrok Trading Bot

An AI-powered automated trading bot for Hyperliquid perpetual futures using xAI Grok for market analysis and trading decisions.

## Features

- 🤖 **AI-Powered Trading**: Uses xAI Grok API for market analysis and trading recommendations
- 📊 **Real-time Data**: Integrates with Hyperliquid REST and WebSocket APIs
- 🛡️ **Risk Management**: Configurable position sizing and risk limits
- 📝 **Comprehensive Logging**: Detailed logs for all trading activities
- 🧪 **Testnet Ready**: Configurable for testnet or mainnet trading

## Setup

### Prerequisites

- Python 3.11+
- Hyperliquid account with API access
- xAI Grok API key

### Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env` file:
```env
# Hyperliquid Configuration
HYPERLIQUID_API_URL=https://api.hyperliquid-testnet.xyz  # Use testnet for safety
HYPERLIQUID_WALLET_ADDRESS=0x...  # Your wallet address
HYPERLIQUID_PRIVATE_KEY=0x...     # Your private key (NEVER commit to git!)
HYPERLIQUID_VAULT_ADDRESS=0x...   # Vault address for multi-user trading

# xAI Grok API
XAI_API_KEY=your-xai-api-key-here
```

⚠️ **Security Warning**: Never commit private keys or API keys to version control!

## Usage

### Dry Run (Recommended First)

Run the bot in analysis-only mode to see what Grok would recommend:

```bash
python hypergrok_trader.py --dry-run
```

### Live Trading

⚠️ **Warning**: Live trading involves real financial risk. Start with small amounts.

```bash
python hypergrok_trader.py --live
```

The bot will:
1. Connect to Hyperliquid API and fetch market data
2. Analyze current positions and market conditions
3. Query xAI Grok for trading recommendations
4. Execute trades based on AI analysis (in live mode)
5. Log all activities

### Stop Trading

Press `Ctrl+C` to stop the bot gracefully.

## Architecture

### Core Components

- **HyperliquidAPI**: Handles all interactions with Hyperliquid exchange
- **GrokIntegration**: Manages communication with xAI Grok API
- **HyperGrokTrader**: Main orchestrator class managing the trading loop

### Trading Logic

The bot uses a market sentiment-driven strategy:

1. **Data Collection**: Fetches market data and current positions
2. **AI Analysis**: Grok analyzes market conditions and provides recommendations
3. **Risk Assessment**: Evaluates position sizing and risk limits
4. **Order Execution**: Places limit orders based on recommendations

### Risk Management

- Maximum position size as fraction of portfolio (default: 10%)
- Risk per trade limit (default: 2%)
- Adjustable check intervals
- Comprehensive error handling and logging

## Configuration

Edit the `HyperGrokTrader` class parameters for customization:

```python
self.max_position = 0.1      # Max position size (10% of portfolio)
self.risk_per_trade = 0.02   # Max risk per trade (2%)
self.check_interval = 60     # Analysis interval in seconds
```

## Logging

All activities are logged to `hypergrok.log` and console output. Log levels:
- INFO: General operations and AI recommendations
- WARNING: Non-critical issues
- ERROR: API errors and failures
- CRITICAL: Critical failures requiring immediate attention

## Backtesting

For backtesting strategies:

```python
# Use historical data instead of live market data
# Implement backtesting framework in future version
```

## API Reference

### HyperliquidAPI Class

- `get_user_state(user_address)`: Get positions and balances
- `get_perpetuals_metadata()`: Get all available perpetual contracts
- `place_order(coin, is_buy, sz, limit_px, ...)`: Place limit orders
- `get_open_orders(user_address)`: Get current open orders

### GrokIntegration Class

- `analyze_market(market_data, positions)`: Get AI market analysis

## Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies and derivatives involves significant financial risk. Always:

- Start with small amounts
- Use testnet for initial testing
- Never risk more than you can afford to lose
- Understand the risks of automated trading

The authors are not responsible for any financial losses incurred through the use of this software.

## License

MIT License - see LICENSE file for details.
