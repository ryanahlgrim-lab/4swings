import os
import logging
from typing import List, Dict, Any
from portfolio import PortfolioManager

# Define standard universe of high-volume, liquid assets
UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "SPY", "QQQ", "IWM"]

def strategy_1_breakout(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Strategy 1: 20-Day Breakout Strategy (Aggressive)"""
    signals = []
    spy_df = data.get("SPY")
    
    # Macro Filter: SPY > 200 SMA
    if spy_df is None or spy_df.iloc[-1]['close'] <= spy_df.iloc[-1]['sma_200']:
        logging.info("[Strategy 1 - Breakout] Macro filter active (SPY <= 200 SMA). Skipping signals.")
        return signals

    for symbol, df in data.items():
        if symbol == "SPY": continue
        if len(df) < 21: continue

        bar_T = df.iloc[-1]      # Today's closing bar
        bars_T_minus_20 = df.iloc[-21:-1] # Historical lookback T-20 to T-1 (Excludes bar T)

        max_20_high = bars_T_minus_20['high'].max()
        vol_sma_20 = bar_T['vol_sma_20']

        # Signal Logic: Close > 20-day high AND Volume > 1.5x 20-day volume average
        if bar_T['close'] > max_20_high and bar_T['volume'] > (1.5 * vol_sma_20):
            signals.append({
                'symbol': symbol,
                'strategy_name': "S1_Breakout",
                'close_price': bar_T['close'],
                'atr_14': bar_T['atr_14'],
                'risk_pct': 0.0075,        # 0.75% portfolio risk
                'atr_mult_stop': 2.0,       # Stop: 2.0 x ATR
                'atr_mult_target': 6.0      # Target: 6.0 x ATR (3:1 R:R)
            })
    return signals


def strategy_2_gap_and_close(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Strategy 2: Gap & Close Momentum Strategy (Aggressive)"""
    signals = []
    for symbol, df in data.items():
        if len(df) < 2: continue

        bar_T = df.iloc[-1]
        bar_prior = df.iloc[-2]

        day_range = bar_T['high'] - bar_T['low']
        if day_range <= 0: continue

        top_third_threshold = bar_T['low'] + (0.67 * day_range)
        gap_pct = (bar_T['open'] - bar_prior['close']) / bar_prior['close']

        # Signal Logic: Open gapped >= 4% AND Close in top 33% of range AND RSI(14) > 60
        if gap_pct >= 0.04 and bar_T['close'] >= top_third_threshold and bar_T['rsi_14'] > 60:
            signals.append({
                'symbol': symbol,
                'strategy_name': "S2_GapClose",
                'close_price': bar_T['close'],
                'atr_14': bar_T['atr_14'],
                'risk_pct': 0.0075,        # 0.75% portfolio risk
                'atr_mult_stop': 2.0,       # Stop: 2.0 x ATR
                'atr_mult_target': 5.0      # Trailing target exit
            })
    return signals


def strategy_3_trend_pullback(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Strategy 3: Trend-Following Pullback Strategy (Moderate)"""
    signals = []
    for symbol, df in data.items():
        if len(df) < 200: continue

        bar_T = df.iloc[-1]
        
        # Distance from 50 SMA
        dist_from_50_sma = abs(bar_T['close'] - bar_T['sma_50']) / bar_T['sma_50']

        # Signal Logic: Price > 200 SMA AND RSI(5) < 35 AND Price within 1.5% of 50 SMA
        if bar_T['close'] > bar_T['sma_200'] and bar_T['rsi_5'] < 35 and dist_from_50_sma <= 0.015:
            signals.append({
                'symbol': symbol,
                'strategy_name': "S3_Pullback",
                'close_price': bar_T['close'],
                'atr_14': bar_T['atr_14'],
                'risk_pct': 0.0050,        # 0.50% portfolio risk
                'atr_mult_stop': 1.5,       # Stop: 1.5 x ATR
                'atr_mult_target': 4.5      # Exit when RSI(5) > 60 or Target hit
            })
    return signals


def strategy_4_mean_reversion(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Strategy 4: Bollinger Band Mean Reversion Strategy (Least Aggressive)"""
    signals = []
    for symbol, df in data.items():
        if len(df) < 20: continue

        bar_T = df.iloc[-1]

        # Signal Logic: Close < Lower Bollinger Band AND ADX(14) < 25 (non-trending market)
        if bar_T['close'] < bar_T['bb_lower'] and bar_T['adx_14'] < 25:
            signals.append({
                'symbol': symbol,
                'strategy_name': "S4_MeanRevert",
                'close_price': bar_T['close'],
                'atr_14': bar_T['atr_14'],
                'risk_pct': 0.0050,        # 0.50% portfolio risk
                'atr_mult_stop': 2.5,       # Stop: Wide 2.5 x ATR
                'atr_mult_target': 2.5      # Target: Reversion to Middle Band (20 SMA)
            })
    return signals


def main():
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("Missing Alpaca API credentials. Ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are set.")

    # Initialize Portfolio Manager
    manager = PortfolioManager(api_key=api_key, secret_key=secret_key, paper=True)

    # Step 1: Fetch and enrich market data ONCE for all universe assets
    universe_data = manager.fetch_universe_data(UNIVERSE)

    if not universe_data:
        logging.error("Failed to load market data. Exiting execution cycle.")
        return

    # Step 2: Run all 4 strategies in isolated try blocks
    raw_signals = []
    strategies = [
        strategy_1_breakout,
        strategy_2_gap_and_close,
        strategy_3_trend_pullback,
        strategy_4_mean_reversion
    ]

    for strat_func in strategies:
        try:
            strat_signals = strat_func(universe_data)
            logging.info(f"[{strat_func.__name__}] Generated {len(strat_signals)} raw signals.")
            raw_signals.extend(strat_signals)
        except Exception as e:
            logging.error(f"Error executing strategy {strat_func.__name__}: {e}")

    # Step 3: Global Portfolio Layer processes signals, applies risk/sizing, and places orders
    manager.process_and_execute_signals(raw_signals)


if __name__ == "__main__":
    main()
