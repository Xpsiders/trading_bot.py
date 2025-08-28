# AI Trading Bot with Telegram Integration

A Python-based trading bot that analyzes market charts, predicts trends, and provides trading recommendations with 2% risk management. Integrated with Telegram for easy access and control.

## Features

- **Market Analysis**: Technical analysis using multiple indicators (SMA, RSI, MACD, Bollinger Bands)
- **Risk Management**: Implements 2% risk per trade with automatic position sizing
- **Multi-Asset Support**: Works with stocks, forex pairs, and cryptocurrencies
- **Real-Time Data**: Fetches current market prices from Yahoo Finance
- **Telegram Integration**: Full control via Telegram bot commands
- **Chart Generation**: Visual technical analysis with matplotlib charts
- **Portfolio Tracking**: Basic portfolio management and trade history

## Supported Markets

- **Stocks**: AAPL, MSFT, TSLA, etc.
- **Forex**: EUR/USD, GBP/JPY, USD/JPY, etc.
- **Cryptocurrencies**: BTC-USD, ETH-USD, XRP-USD, etc

**Project Structure**
├── trading_bot.py      # Main bot application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in repo)
├── trading_bot.db     # Database file (created automatically)
└── README.md          # This file

**Disclaimer**
This trading bot is for educational purposes only. Trading financial instruments involves significant risk and is not suitable for all investors. Past performance is not indicative of future results. The authors are not responsible for any financial losses incurred while using this software. All provided information is  simulated.
