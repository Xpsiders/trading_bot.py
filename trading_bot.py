import pandas as pd
import numpy as np
import logging
import sqlite3
import io
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, risk_per_trade=0.02):
        self.risk_per_trade = risk_per_trade
        self.portfolio_value = 10000  # Default starting portfolio
        self.setup_database()
        self.api_connected = False
        
        # Try to connect to Binance if keys are available
        if os.getenv('BINANCE_API_KEY') and os.getenv('BINANCE_API_SECRET'):
            try:
                from binance.client import Client
                self.client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
                self.api_connected = True
                logger.info("Connected to Binance API")
            except:
                logger.warning("Binance API keys found but connection failed")
    
    def setup_database(self):
        self.conn = sqlite3.connect('trading_bot.db')
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS trades
                     (id INTEGER PRIMARY KEY, symbol TEXT, action TEXT, 
                     price REAL, quantity REAL, timestamp DATETIME, 
                     stop_loss REAL, take_profit REAL)''')
        self.conn.commit()
    
    def get_market_data(self, symbol, source="simulated"):
        """Get market data from various sources"""
        if source == "simulated":
            return self.get_simulated_data(symbol)
        elif source == "alphavantage" and os.getenv('ALPHAVANTAGE_API_KEY'):
            return self.get_alphavantage_data(symbol)
        elif source == "twelvedata" and os.getenv('TWELVEDATA_API_KEY'):
            return self.get_twelvedata_data(symbol)
        else:
            # Fall back to simulated data
            return self.get_simulated_data(symbol)
    
    def get_simulated_data(self, symbol):
        """Generate simulated market data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        np.random.seed(hash(symbol) % 10000)  # Seed based on symbol for consistency
        
        # Start with a random price based on symbol
        base_price = 100 + (hash(symbol) % 100)
        
        # Generate random walk
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * (1 + returns).cumprod()
        
        # Generate OHLCV data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.lognormal(10, 1, len(dates))
        })
        
        return data
    
    def get_alphavantage_data(self, symbol):
        """Get real market data from Alpha Vantage (stocks, forex)"""
        api_key = os.getenv('ALPHAVANTAGE_API_KEY')
        if not api_key:
            return self.get_simulated_data(symbol)
        
        # Determine if it's forex or stock
        if len(symbol) == 6 and '/' in symbol:
            # Forex pair like USD/EUR
            from_currency, to_currency = symbol.split('/')
            url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_currency}&to_symbol={to_currency}&apikey={api_key}&outputsize=compact"
        else:
            # Stock symbol
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=compact"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            # Parse response based on type
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
            elif 'Time Series FX (Daily)' in data:
                time_series = data['Time Series FX (Daily)']
            else:
                return self.get_simulated_data(symbol)
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df = df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            })
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            return df.tail(100)  # Return last 100 days
            
        except:
            return self.get_simulated_data(symbol)
    
    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        # Simple Moving Averages
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        data['SMA_50'] = data['close'].rolling(window=50).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = data['close'].ewm(span=12, adjust=False).mean()
        exp26 = data['close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp12 - exp26
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_hist'] = data['MACD'] - data['MACD_signal']
        
        # Bollinger Bands
        data['BB_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
        data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        
        return data
    
    def generate_signal(self, data):
        """Generate trading signal based on indicators"""
        latest = data.iloc[-1]
        
        # Count bullish and bearish signals
        bullish_signals = 0
        bearish_signals = 0
        
        # Trend signals
        if latest['SMA_20'] > latest['SMA_50']:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # RSI signals
        if latest['RSI'] < 30:
            bullish_signals += 1
        elif latest['RSI'] > 70:
            bearish_signals += 1
        
        # MACD signals
        if latest['MACD'] > latest['MACD_signal']:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Bollinger Bands signals
        if latest['close'] < latest['BB_lower']:
            bullish_signals += 1
        elif latest['close'] > latest['BB_upper']:
            bearish_signals += 1
        
        current_price = latest['close']
        
        # Determine action
        if bullish_signals >= 3:
            action = "BUY"
            stop_loss = current_price * 0.98
            take_profit = current_price * 1.04
        elif bearish_signals >= 3:
            action = "SELL"
            stop_loss = current_price * 1.02
            take_profit = current_price * 0.96
        else:
            action = "HOLD"
            stop_loss = None
            take_profit = None
        
        return action, stop_loss, take_profit, current_price
    
    def calculate_position_size(self, entry_price, stop_loss_price):
        risk_amount = self.portfolio_value * self.risk_per_trade
        price_difference = abs(entry_price - stop_loss_price)
        position_size = risk_amount / price_difference
        return position_size
    
    def analyze_symbol(self, symbol):
        """Main analysis function"""
        # Get market data (try Alpha Vantage first, then simulated)
        data = self.get_alphavantage_data(symbol)
        if data is None:
            data = self.get_simulated_data(symbol)
        
        # Calculate indicators
        data_with_indicators = self.calculate_indicators(data)
        
        # Generate signal
        action, stop_loss, take_profit, current_price = self.generate_signal(data_with_indicators)
        
        # Calculate position size if applicable
        position_size = None
        if action != "HOLD":
            position_size = self.calculate_position_size(current_price, stop_loss)
        
        return action, stop_loss, take_profit, current_price, position_size, data_with_indicators
    
    def generate_chart(self, data, symbol):
        """Generate technical analysis chart"""
        plt.figure(figsize=(10, 8))
        
        # Price chart
        plt.subplot(3, 1, 1)
        plt.plot(data.index, data['close'], label='Price')
        plt.plot(data.index, data['SMA_20'], label='SMA 20')
        plt.plot(data.index, data['SMA_50'], label='SMA 50')
        plt.plot(data.index, data['BB_upper'], label='BB Upper', linestyle='--', alpha=0.5)
        plt.plot(data.index, data['BB_lower'], label='BB Lower', linestyle='--', alpha=0.5)
        plt.title(f'{symbol} Price Chart')
        plt.legend()
        plt.grid(True)
        
        # RSI
        plt.subplot(3, 1, 2)
        plt.plot(data.index, data['RSI'], label='RSI', color='purple')
        plt.axhline(70, linestyle='--', alpha=0.5, color='red')
        plt.axhline(30, linestyle='--', alpha=0.5, color='green')
        plt.title('RSI')
        plt.legend()
        plt.grid(True)
        
        # MACD
        plt.subplot(3, 1, 3)
        plt.plot(data.index, data['MACD'], label='MACD', color='blue')
        plt.plot(data.index, data['MACD_signal'], label='Signal', color='red')
        plt.bar(data.index, data['MACD_hist'], label='Histogram', alpha=0.3)
        plt.title('MACD')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf

# Initialize trading bot
TRADING_BOT = TradingBot()

# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    user = update.effective_user
    keyboard = [['/analyze AAPL', '/analyze MSFT'], 
                ['/analyze EUR/USD', '/analyze GBP/USD'],
                ['/portfolio', '/help']]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_html(
        rf"Hi {user.mention_html()}! I'm your AI Trading Bot. "
        "I can analyze market trends for stocks, forex, and cryptocurrencies.",
        reply_markup=reply_markup
    )

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Analyze a symbol."""
    if not context.args:
        await update.message.reply_text('Please specify a symbol. Examples:\n- Stocks: /analyze AAPL\n- Forex: /analyze EUR/USD\n- Crypto: /analyze BTC/USD')
        return
    
    symbol = context.args[0].upper()
    await update.message.reply_text(f'Analyzing {symbol}, please wait...')
    
    # Get trading signal
    action, stop_loss, take_profit, current_price, position_size, data = TRADING_BOT.analyze_symbol(symbol)
    
    # Generate chart
    chart = TRADING_BOT.generate_chart(data, symbol)
    
    # Prepare response message
    message = (
        f"Analysis for {symbol}:\n"
        f"Current Price: ${current_price:.2f}\n"
        f"Recommended Action: {action}\n"
    )
    
    if action != "HOLD":
        message += (
            f"Stop Loss: ${stop_loss:.2f}\n"
            f"Take Profit: ${take_profit:.2f}\n"
            f"Position Size: {position_size:.2f} units\n"
            f"Risk: 2% of portfolio (${TRADING_BOT.portfolio_value * 0.02:.2f})"
        )
    
    # Send message and chart
    await update.message.reply_photo(photo=chart, caption=message)

async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show portfolio information."""
    await update.message.reply_text(f"Current Portfolio Value: ${TRADING_BOT.portfolio_value:.2f}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = (
        "Available commands:\n"
        "/start - Start the bot\n"
        "/analyze [symbol] - Analyze a trading symbol\n"
        "  Examples:\n"
        "  - Stocks: /analyze AAPL, /analyze MSFT\n"
        "  - Forex: /analyze EUR/USD, /analyze GBP/USD\n"
        "  - Crypto: /analyze BTC/USD, /analyze ETH/USD\n"
        "/portfolio - Show portfolio value\n"
        "/help - Show this help message\n\n"
        "Note: This bot uses simulated data by default. To use real market data:\n"
        "1. Get a free API key from Alpha Vantage (https://www.alphavantage.com/)\n"
        "2. Set ALPHAVANTAGE_API_KEY in your .env file"
    )
    await update.message.reply_text(help_text)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors caused by Updates."""
    logger.error(f"Update {update} caused error {context.error}")

def main():
    """Start the bot."""
    # Get the Telegram bot token from environment variable
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set")
        print("Please set your Telegram bot token in the .env file")
        return
    
    # Create the Application
    application = Application.builder().token(token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("portfolio", portfolio_command))
    application.add_handler(CommandHandler("help", help_command))
    
    # Add error handler
    application.add_error_handler(error_handler)

    # Run the bot until interrupted
    print("Bot is running...")
    application.run_polling()

if __name__ == '__main__':
    main()