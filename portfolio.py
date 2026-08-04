import math
import logging
import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import ta

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class PortfolioManager:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        
    def fetch_universe_data(self, tickers: List[str], lookback_days: int = 300) -> Dict[str, pd.DataFrame]:
        """Fetch daily historical bar data once for all target tickers and enrich with technical indicators."""
        logging.info(f"Fetching {lookback_days} days of historical data for {len(tickers)} tickers...")
        
        end_date = datetime.datetime.now(datetime.timezone.utc)
        start_date = end_date - datetime.timedelta(days=lookback_days)
        
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
            feed='sip'
        )
        
        try:
            bars = self.data_client.get_stock_bars(request_params)
            df_all = bars.df.reset_index()
        except Exception as e:
            logging.error(f"Error fetching market data from Alpaca: {e}")
            return {}

        data_by_ticker = {}
        for symbol in tickers:
            df_symbol = df_all[df_all['symbol'] == symbol].copy()
            if df_symbol.empty or len(df_symbol) < 200:
                logging.warning(f"Insufficient historical data for {symbol}. Skipping.")
                continue
                
            df_symbol = df_symbol.sort_values('timestamp').reset_index(drop=True)
            
            # Enrich Indicators
            df_symbol['atr_14'] = ta.volatility.average_true_range(df_symbol['high'], df_symbol['low'], df_symbol['close'], window=14)
            df_symbol['sma_200'] = ta.trend.sma_indicator(df_symbol['close'], window=200)
            df_symbol['sma_50'] = ta.trend.sma_indicator(df_symbol['close'], window=50)
            df_symbol['sma_20'] = ta.trend.sma_indicator(df_symbol['close'], window=20)
            df_symbol['vol_sma_20'] = ta.trend.sma_indicator(df_symbol['volume'], window=20)
            df_symbol['rsi_14'] = ta.momentum.rsi(df_symbol['close'], window=14)
            df_symbol['rsi_5'] = ta.momentum.rsi(df_symbol['close'], window=5)
            df_symbol['adx_14'] = ta.trend.adx(df_symbol['high'], df_symbol['low'], df_symbol['close'], window=14)
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df_symbol['close'], window=20, window_dev=2)
            df_symbol['bb_lower'] = bb.bollinger_lband()
            df_symbol['bb_middle'] = bb.bollinger_mavg()
            
            data_by_ticker[symbol] = df_symbol
            
        return data_by_ticker

    def process_and_execute_signals(self, raw_signals: List[Dict[str, Any]]):
        """Deduplicate signals, enforce portfolio risk limits, calculate ATR sizing, and execute bracket orders."""
        if not raw_signals:
            logging.info("No strategy signals generated today.")
            return

        account = self.trading_client.get_account()
        portfolio_equity = float(account.equity)
        buying_power = float(account.buying_power)
        
        # Get active positions and open orders for deduplication
        existing_positions = {p.symbol for p in self.trading_client.get_all_positions()}
        open_orders = self.trading_client.get_orders()
        pending_symbols = {o.symbol for o in open_orders}
        
        processed_symbols = set()
        today_str = datetime.datetime.now().strftime("%Y%m%d")

        logging.info(f"Account Equity: ${portfolio_equity:,.2f} | Available Buying Power: ${buying_power:,.2f}")

        for signal in raw_signals:
            symbol = signal['symbol']
            strategy_name = signal['strategy_name']
            
            # 1. Deduplication checks
            if symbol in existing_positions or symbol in pending_symbols or symbol in processed_symbols:
                logging.info(f"[{strategy_name}] Skipping {symbol}: Active position or pending order already exists.")
                continue

            close_price = signal['close_price']
            atr_14 = signal['atr_14']
            risk_pct = signal['risk_pct']
            atr_mult_stop = signal['atr_mult_stop']
            atr_mult_target = signal.get('atr_mult_target')

            if pd.isna(atr_14) or atr_14 <= 0:
                logging.warning(f"[{strategy_name}] Invalid ATR for {symbol}. Skipping.")
                continue

            # 2. Volatility-Based ATR Position Sizing Math
            stop_distance = atr_14 * atr_mult_stop
            risk_dollars = portfolio_equity * risk_pct
            
            # Whole shares only (Floor function)
            shares = math.floor(risk_dollars / stop_distance)
            notional_cost = shares * close_price

            if shares <= 0 or notional_cost > buying_power:
                logging.warning(f"[{strategy_name}] Insufficient capital/buying power for {shares} shares of {symbol}.")
                continue

            # 3. Calculate Bracket Legs
            stop_loss_price = round(close_price - stop_distance, 2)
            take_profit_price = round(close_price + (atr_14 * atr_mult_target), 2) if atr_mult_target else None

            # 4. Deterministic Idempotency Key
            client_order_id = f"{strategy_name}_{symbol}_{today_str}"

            # 5. Build Alpaca Bracket Request
            stop_loss_req = StopLossRequest(stop_price=stop_loss_price)
            take_profit_req = TakeProfitRequest(limit_price=take_profit_price) if take_profit_price else None

            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                client_order_id=client_order_id,
                order_class='bracket',
                stop_loss=stop_loss_req,
                take_profit=take_profit_req
            )

            try:
                logging.info(f"[{strategy_name}] Submitting BRACKET ORDER: BUY {shares} shares of {symbol} at ~${close_price} "
                             f"(Stop: ${stop_loss_price}, Target: ${take_profit_price or 'None'})")
                self.trading_client.submit_order(order_req)
                
                # Update local trackers to prevent double allocation in the same run
                processed_symbols.add(symbol)
                buying_power -= notional_cost
            except Exception as e:
                logging.error(f"Failed to execute bracket order for {symbol}: {e}")
