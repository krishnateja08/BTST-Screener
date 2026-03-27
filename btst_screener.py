"""
============================================================
  BTST (Buy Today Sell Tomorrow) Stock Screener
  India — Nifty 100 (NSE)  |  USA — S&P 500 Top 100 (NYSE/NASDAQ)
============================================================
Requirements:
    pip install yfinance pandas pandas-ta requests tabulate colorama

Usage:
    python btst_screener.py              # scans both markets (BTST + ORB)
    python btst_screener.py --india      # India only
    python btst_screener.py --usa        # USA only
    python btst_screener.py --no-orb     # skip ORB intraday scan
    python btst_screener.py --backtest   # replay past CSV picks (last 30 days)
    python btst_screener.py --backtest --days 60   # extend backtest window
    python btst_screener.py --backtest --india      # India backtest only

Output:
    btst_report_YYYY-MM-DD.html    (combined HTML with BTST + ORB tabs)
    btst_india_YYYY-MM-DD.csv
    btst_usa_YYYY-MM-DD.csv
    orb_india_YYYY-MM-DD.csv
    orb_usa_YYYY-MM-DD.csv
============================================================
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import warnings
import sys
import json
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo          # stdlib — Python 3.9+
from tabulate import tabulate
from colorama import Fore, Style, init
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
init(autoreset=True)

# ══════════════════════════════════════════════════════════
# SYMBOL LISTS
# ══════════════════════════════════════════════════════════

NIFTY100_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFY.NS", "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "ADANIENT.NS",
    "NESTLEIND.NS", "JSWSTEEL.NS", "POWERGRID.NS", "NTPC.NS",
    "ONGC.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "TECHM.NS", "HCLTECH.NS",
    "TATASTEEL.NS", "HINDALCO.NS", "GRASIM.NS", "DRREDDY.NS", "CIPLA.NS",
    "BPCL.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
    "TMCV.NS", "BRITANNIA.NS", "TMPV.NS", "ADANIPORTS.NS", "SBILIFE.NS", "HDFCLIFE.NS",
    "BAJAJ-AUTO.NS", "M&M.NS", "VEDL.NS", "SIEMENS.NS", "PIDILITIND.NS",
    "DABUR.NS", "MARICO.NS", "HAVELLS.NS", "BERGEPAINT.NS", "LUPIN.NS",
    "TORNTPHARM.NS", "MUTHOOTFIN.NS", "SHREECEM.NS", "AMBUJACEM.NS", "GAIL.NS",
    "IOC.NS", "INDUSINDBK.NS", "BANDHANBNK.NS", "PNB.NS", "CANBK.NS",
    "BANKBARODA.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "HDFCAMC.NS", "NAUKRI.NS",
    "DMART.NS", "ETERNAL.NS", "PAYTM.NS", "IRCTC.NS", "HAL.NS",
    "BEL.NS", "BHEL.NS", "DLF.NS", "GODREJCP.NS", "COLPAL.NS",
    "PGHH.NS", "CONCOR.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "TATAPOWER.NS",
    "NHPC.NS", "RECLTD.NS", "PFC.NS", "INDIANB.NS", "UNIONBANK.NS",
    "LICHSGFIN.NS", "CHOLAFIN.NS", "SBICARD.NS", "ICICIPRULI.NS", "HINDPETRO.NS",
    "PETRONET.NS", "IGL.NS", "MGL.NS", "SAIL.NS", "NMDC.NS",
]

# S&P 500 Top 100 by market cap (Yahoo Finance tickers)
SP500_TOP100_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "AVGO", "JPM",
    "LLY", "UNH", "XOM", "V", "MA", "COST", "HD", "PG", "JNJ", "ABBV",
    "WMT", "NFLX", "BAC", "CRM", "ORCL", "CVX", "MRK", "KO", "AMD", "PEP",
    "TMO", "ADBE", "ACN", "LIN", "MCD", "CSCO", "WFC", "ABT", "IBM", "GE",
    "DHR", "AXP", "CAT", "ISRG", "INTU", "NOW", "QCOM", "VZ", "GS", "RTX",
    "SPGI", "BKNG", "AMGN", "TXN", "MS", "BLK", "PFE", "SYK", "AMAT", "NEE",
    "PLD", "UNP", "HON", "LOW", "T", "CMCSA", "UBER", "ETN", "DE", "BSX",
    "ADP", "VRTX", "PANW", "CB", "TJX", "BMY", "GILD", "SO", "SCHW", "ZTS",
    "MMC", "LRCX", "ADI", "BA", "MO", "ELV", "DUK", "CI", "MDLZ", "APH",
    "PH", "ICE", "SHW", "KLAC", "MCO", "AON", "CME", "REGN", "CEG", "CL",
]

# ── Index / VIX symbols ────────────────────────────────────
INDIA_INDEX  = "^NSEI"
INDIA_VIX    = "^INDIAVIX"
USA_INDEX    = "^GSPC"      # S&P 500
USA_VIX      = "^VIX"       # CBOE VIX

# ── Sector index / ETF maps ────────────────────────────────
# Maps each stock symbol → its sector benchmark ticker
INDIA_SECTOR_MAP: dict[str, str] = {
    # Banking → Nifty Bank
    **{s: "^NSEBANK" for s in [
        "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS",
        "INDUSINDBK.NS","BANDHANBNK.NS","PNB.NS","CANBK.NS","BANKBARODA.NS",
        "FEDERALBNK.NS","IDFCFIRSTB.NS","INDIANB.NS","UNIONBANK.NS",
    ]},
    # IT → Nifty IT
    **{s: "^CNXIT" for s in [
        "TCS.NS","INFY.NS","WIPRO.NS","TECHM.NS","HCLTECH.NS","NAUKRI.NS",
    ]},
    # Pharma → Nifty Pharma
    **{s: "^CNXPHARMA" for s in [
        "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","LUPIN.NS",
        "TORNTPHARM.NS","APOLLOHOSP.NS",
    ]},
    # Auto → Nifty Auto
    **{s: "^CNXAUTO" for s in [
        "MARUTI.NS","TMCV.NS","TMPV.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS",
        "M&M.NS","EICHERMOT.NS",
    ]},
    # Metal → Nifty Metal
    **{s: "^CNXMETAL" for s in [
        "JSWSTEEL.NS","TATASTEEL.NS","HINDALCO.NS","VEDL.NS","SAIL.NS","NMDC.NS",
    ]},
    # FMCG → Nifty FMCG
    **{s: "^CNXFMCG" for s in [
        "HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","TATACONSUM.NS","BRITANNIA.NS",
        "DABUR.NS","MARICO.NS","COLPAL.NS","GODREJCP.NS","PGHH.NS",
    ]},
    # Energy / Oil & Gas → Nifty Energy
    **{s: "^CNXENERGY" for s in [
        "RELIANCE.NS","ONGC.NS","BPCL.NS","IOC.NS","HINDPETRO.NS",
        "GAIL.NS","PETRONET.NS","IGL.NS","MGL.NS","TATAPOWER.NS",
        "ADANIGREEN.NS","ADANIPOWER.NS","COALINDIA.NS","NHPC.NS",
        "RECLTD.NS","PFC.NS","NTPC.NS","POWERGRID.NS",
    ]},
    # Finance (non-bank) → Nifty Financial Services
    **{s: "^CNXFINANCE" for s in [
        "BAJFINANCE.NS","BAJAJFINSV.NS","HDFCAMC.NS","MUTHOOTFIN.NS",
        "LICHSGFIN.NS","CHOLAFIN.NS","SBICARD.NS","ICICIPRULI.NS",
        "SBILIFE.NS","HDFCLIFE.NS",
    ]},
    # Infra / Capital Goods → Nifty Infra
    **{s: "^CNXINFRA" for s in [
        "LT.NS","SIEMENS.NS","HAVELLS.NS","BEL.NS","BHEL.NS","HAL.NS",
        "CONCOR.NS","ADANIPORTS.NS","DLF.NS",
    ]},
    # Cement / Materials
    **{s: "^CNXCMDT" for s in [
        "ULTRACEMCO.NS","SHREECEM.NS","AMBUJACEM.NS","GRASIM.NS",
        "ASIANPAINT.NS","BERGER.NS","PIDILITIND.NS",
    ]},
}

USA_SECTOR_MAP: dict[str, str] = {
    # Technology → XLK
    **{s: "XLK" for s in [
        "AAPL","MSFT","NVDA","AVGO","AMD","ADBE","CRM","QCOM","INTU",
        "NOW","PANW","AMAT","LRCX","ADI","KLAC","IBM","ACN","APH",
    ]},
    # Financials → XLF
    **{s: "XLF" for s in [
        "JPM","BAC","WFC","GS","MS","V","MA","AXP","BLK","SCHW",
        "SPGI","CB","MMC","ICE","CME","MCO","AON","BRK-B",
    ]},
    # Healthcare → XLV
    **{s: "XLV" for s in [
        "UNH","LLY","JNJ","ABBV","MRK","TMO","DHR","ISRG","BMY",
        "AMGN","GILD","VRTX","PFE","SYK","ZTS","ELV","CI","BSX",
    ]},
    # Energy → XLE
    **{s: "XLE" for s in ["XOM","CVX"]},
    # Consumer Discretionary → XLY
    **{s: "XLY" for s in [
        "AMZN","TSLA","HD","MCD","LOW","TJX","BKNG","UBER",
    ]},
    # Consumer Staples → XLP
    **{s: "XLP" for s in [
        "WMT","PG","KO","PEP","COST","MDLZ","MO","CL",
    ]},
    # Industrials → XLI
    **{s: "XLI" for s in [
        "GE","CAT","HON","UNP","RTX","ETN","DE","ADP","BA","PH",
    ]},
    # Communication Services → XLC
    **{s: "XLC" for s in [
        "GOOGL","META","NFLX","CMCSA","T","VZ",
    ]},
    # Utilities → XLU
    **{s: "XLU" for s in ["NEE","SO","DUK","CEG"]},
    # Real Estate → XLRE
    **{s: "XLRE" for s in ["PLD"]},
    # Materials → XLB
    **{s: "XLB" for s in ["LIN","SHW"]},
}

SECTOR_BONUS_PTS = 7   # bonus added when stock's sector index is green on the day

LOOKBACK_DAYS  = 365          # extended to 1 year for 52-week high calculation
AVG_VOL_PERIOD = 10

WEIGHTS = {
    "volume_surge":   20,
    "rsi_zone":       15,
    "macd_bullish":   15,
    "above_ema":      15,
    "price_breakout": 15,
    "near_52w_high":  10,
    "adx_trend":      10,
    "sector_bonus":    7,   # sector index green today
    "candle_pattern": 10,   # Morning Star=10, Engulfing=8, Hammer=6
    "rel_strength":    5,   # stock beats index % change today
    "weekly_confirm":  8,   # close > weekly EMA20 (multi-timeframe)
    "gap_up":          8,   # gap-up open that held by close (+5 mild / +8 strong)
}

IST = ZoneInfo("Asia/Kolkata")
EST = ZoneInfo("America/New_York")

# ══════════════════════════════════════════════════════════
# ORB (Opening Range Breakout) CONFIG  — intraday 5-min
# ══════════════════════════════════════════════════════════
# Opening Range = first ORB_BARS × 5-min candles after market open
#   India: 9:15–9:30 AM IST  (3 bars)
#   USA  : 9:30–9:45 AM EST  (3 bars)

ORB_BARS = 3          # number of 5-min bars defining the opening range

ORB_WEIGHTS = {
    "breakout_strength": 25,   # how far price is above ORB high
    "volume_surge":      20,   # current bar vol vs ORB avg vol
    "rsi_5m":            15,   # RSI on 5m >= 55
    "adx_5m":            10,   # ADX on 5m >= 25
    "orb_range_tight":   10,   # tight ORB = higher conviction
    "open_candle_bull":   8,   # first bar of day was green
    "sector_bonus":        7,  # sector index green (shared with BTST)
}
# Max ORB score ~95 pts


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

def hdr(msg):
    print(f"\n{Fore.CYAN}{'─'*62}\n  {msg}\n{'─'*62}{Style.RESET_ALL}")


def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _load_prev_scores(prefix: str, date_str: str) -> dict[str, float]:
    """
    Load the most recent previous day's top CSV to get prior BTST scores.
    Walks back up to 4 days (handles weekends / holidays).
    Returns {Symbol: BTST_Score} or {} if no file found.
    """
    base = datetime.strptime(date_str, "%Y-%m-%d")
    for days_back in range(1, 5):
        prev = (base - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            df = pd.read_csv(f"btst_{prefix}_{prev}.csv")
            if "Symbol" in df.columns and "BTST_Score" in df.columns:
                return dict(zip(df["Symbol"].astype(str), df["BTST_Score"].astype(float)))
        except FileNotFoundError:
            continue
        except Exception:
            break
    return {}


# ══════════════════════════════════════════════════════════
# MARKET HEALTH CHECK
# ══════════════════════════════════════════════════════════

def check_market(market: str = "india") -> tuple[bool, float, float]:
    """
    Returns (is_safe, index_chg_pct, vix_value)
    market: 'india' | 'usa'
    """
    idx_sym = INDIA_INDEX if market == "india" else USA_INDEX
    vix_sym = INDIA_VIX   if market == "india" else USA_VIX
    label   = "Nifty 50"  if market == "india" else "S&P 500"
    vix_thr = 20          if market == "india" else 20

    hdr(f"Market Health — {label.upper()}")

    try:
        combined = yf.download(
            [idx_sym, vix_sym],
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
        )
        if isinstance(combined.columns, pd.MultiIndex):
            idx = combined[idx_sym].dropna()
            vix = combined[vix_sym].dropna()
        else:
            idx = combined.dropna()
            vix = pd.DataFrame()
    except Exception:
        print(f"  {Fore.YELLOW}⚠  Could not fetch market data{Style.RESET_ALL}")
        return True, 0.0, 0.0

    if idx.empty:
        return True, 0.0, 0.0

    close = float(idx["Close"].iloc[-1])
    prev  = float(idx["Close"].iloc[-2]) if len(idx) >= 2 else close
    chg   = round((close - prev) / prev * 100, 2)
    vix_v = float(vix["Close"].iloc[-1]) if not vix.empty else 0.0

    bullish  = chg >= -0.5
    vix_safe = vix_v < vix_thr if vix_v > 0 else True
    safe     = bullish and vix_safe

    col = Fore.GREEN if bullish else Fore.RED
    print(f"  {label} Change : {col}{chg:+.2f}%{Style.RESET_ALL}  (Close: {close:,.2f})")
    print(f"  VIX            : {'🟢' if vix_safe else '🔴'} {vix_v:.2f}  "
          f"({'Safe' if vix_safe else 'HIGH — caution!'})")
    if not bullish:
        print(f"  {Fore.RED}⚠  Market weak today — higher BTST risk{Style.RESET_ALL}")
    if not vix_safe:
        print(f"  {Fore.RED}⚠  VIX > {vix_thr} — avoid overnight positions{Style.RESET_ALL}")

    return safe, chg, vix_v


# ══════════════════════════════════════════════════════════
# BATCH DOWNLOAD — all tickers in one request
# ══════════════════════════════════════════════════════════

def _batch_download(symbols: list) -> dict:
    """
    Download 1 year of OHLCV for all symbols in a single yf.download() call.
    Returns dict {symbol: DataFrame}.  1 year needed for 52-week high check.
    """
    raw = yf.download(
        symbols,
        period="1y",
        interval="1d",
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )
    result = {}
    for sym in symbols:
        try:
            df = raw[sym].dropna() if isinstance(raw.columns, pd.MultiIndex) else raw.dropna()
            if len(df) >= 20:
                result[sym] = df
        except Exception:
            pass
    return result


# ══════════════════════════════════════════════════════════
# SECTOR PERFORMANCE  — fetch all relevant sector indices/ETFs
# ══════════════════════════════════════════════════════════

def fetch_sector_perf(market: str) -> dict[str, bool]:
    """
    Returns {sector_ticker: is_up_today} for all sector benchmarks used
    in the given market.  'is_up_today' = True if today's close > prev close.
    """
    sector_map = INDIA_SECTOR_MAP if market == "india" else USA_SECTOR_MAP
    tickers    = list(set(sector_map.values()))

    print(f"  📊  Fetching sector data ({len(tickers)} indices/ETFs) …", flush=True)
    try:
        raw = yf.download(
            tickers,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return {}

    result: dict[str, bool] = {}
    for ticker in tickers:
        try:
            df = raw[ticker].dropna() if isinstance(raw.columns, pd.MultiIndex) else raw.dropna()
            if len(df) >= 2:
                chg = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-2])
                result[ticker] = chg > 0
        except Exception:
            pass

    up_count = sum(1 for v in result.values() if v)
    print(f"  ✅  Sectors: {up_count}/{len(result)} green today.")
    return result


# ══════════════════════════════════════════════════════════
# SCORE A SINGLE STOCK (from pre-downloaded DataFrame)
# ══════════════════════════════════════════════════════════

def score_stock_from_df(symbol: str, df: pd.DataFrame,
                        sector_bonus: float = 0.0,
                        index_chg: float = 0.0) -> dict | None:
    try:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close  = float(df["Close"].iloc[-1])
        high   = float(df["High"].iloc[-1])
        low    = float(df["Low"].iloc[-1])
        volume = float(df["Volume"].iloc[-1])

        # ── Volume surge ──────────────────────────────────────────
        avg_vol   = df["Volume"].iloc[-AVG_VOL_PERIOD-1:-1].mean()
        vol_ratio = volume / avg_vol if avg_vol > 0 else 0
        s_vol     = (WEIGHTS["volume_surge"] if vol_ratio >= 1.5 else
                     WEIGHTS["volume_surge"] * 0.5 if vol_ratio >= 1.2 else 0)

        # ── RSI ───────────────────────────────────────────────────
        rsi_s = ta.rsi(df["Close"], length=14)
        rsi   = float(rsi_s.iloc[-1]) if rsi_s is not None else 50
        s_rsi = (WEIGHTS["rsi_zone"] if 55 <= rsi <= 75 else
                 WEIGHTS["rsi_zone"] * 0.5 if 50 <= rsi < 55 else 0)

        # ── MACD ──────────────────────────────────────────────────
        macd_df   = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        s_macd    = 0
        macd_hist = None
        if macd_df is not None and not macd_df.empty:
            hcol = [c for c in macd_df.columns if "MACDh" in c]
            if hcol:
                macd_hist = float(macd_df[hcol[0]].iloc[-1])
                prev_h    = float(macd_df[hcol[0]].iloc[-2])
                s_macd    = (WEIGHTS["macd_bullish"] if macd_hist > 0 and prev_h <= 0
                             else WEIGHTS["macd_bullish"] * 0.7 if macd_hist > 0 else 0)

        # ── EMA ───────────────────────────────────────────────────
        ema20 = float(ta.ema(df["Close"], length=20).iloc[-1])
        ema50 = float(ta.ema(df["Close"], length=50).iloc[-1])
        s_ema = (WEIGHTS["above_ema"] if close > ema20 and close > ema50
                 else WEIGHTS["above_ema"] * 0.5 if close > ema20 else 0)

        # ── Price breakout (intraday range position) ──────────────
        rng   = high - low
        pos   = (close - low) / rng if rng > 0 else 0
        s_brk = (WEIGHTS["price_breakout"] if pos >= 0.90
                 else WEIGHTS["price_breakout"] * 0.6 if pos >= 0.75 else 0)

        # ── ADX ───────────────────────────────────────────────────
        adx_df  = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        adx_val = 0.0
        s_adx   = 0
        if adx_df is not None and not adx_df.empty:
            ac = [c for c in adx_df.columns if c.startswith("ADX_")]
            if ac:
                adx_val = float(adx_df[ac[0]].iloc[-1])
                s_adx   = (WEIGHTS["adx_trend"] if adx_val >= 25
                           else WEIGHTS["adx_trend"] * 0.5 if adx_val >= 20 else 0)

        # ── 52-Week High Proximity ────────────────────────────────
        w52_high   = float(df["High"].max())
        proximity  = close / w52_high if w52_high > 0 else 0
        near_52w   = proximity >= 0.95
        s_52w      = (WEIGHTS["near_52w_high"] if proximity >= 0.95 else
                      WEIGHTS["near_52w_high"] * 0.5 if proximity >= 0.90 else 0)

        # ── ATR (used for penalty + SL/Target) ───────────────────
        atr_s   = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        atr_val = float(atr_s.iloc[-1]) if atr_s is not None and not atr_s.empty else (rng * 0.5)

        # ── Day change ────────────────────────────────────────────
        prev_close = float(df["Close"].iloc[-2])
        day_chg    = (close - prev_close) / prev_close * 100

        # ── Candlestick Pattern Detection ────────────────────────
        # Uses today ([-1]), yesterday ([-2]), day before ([-3])
        candle_name = ""
        s_candle    = 0
        try:
            o0 = float(df["Open"].iloc[-1])   # today open
            o1 = float(df["Open"].iloc[-2])   # yesterday open
            c1 = prev_close                    # yesterday close
            o2 = float(df["Open"].iloc[-3])   # 2 days ago open
            c2 = float(df["Close"].iloc[-3])  # 2 days ago close
            body0 = abs(close - o0)
            body1 = abs(c1 - o1)
            body2 = abs(c2 - o2)
            lower_shadow0 = min(o0, close) - low
            upper_shadow0 = high - max(o0, close)

            # Morning Star (3-candle, strongest reversal)
            # Day-2: big red | Day-1: small body indecision | Today: big green > midpoint of day-2
            morning_star = (
                c2 < o2 and                    # day-2 red (bearish)
                body1 <= body2 * 0.4 and       # day-1 small body (indecision/doji)
                close > o0 and                 # today green
                close > (o2 + c2) / 2 and     # closes above midpoint of day-2
                body0 >= body2 * 0.5           # today's body substantial
            )
            # Bullish Engulfing: prev red candle fully engulfed by today's green candle
            bullish_engulfing = (
                c1 < o1 and       # yesterday red
                close > o0 and    # today green
                o0 <= c1 and      # today open at/below yesterday close
                close >= o1       # today close at/above yesterday open
            )
            # Hammer: small body in upper portion, long lower wick, appears after decline
            hammer = (
                rng > 0 and
                body0 <= rng * 0.35 and             # small body
                lower_shadow0 >= 2.0 * body0 and    # long lower wick
                upper_shadow0 <= body0 * 0.6 and    # tiny upper wick
                close > o0                           # green preferred
            )

            if morning_star:
                s_candle, candle_name = WEIGHTS["candle_pattern"],       "Morning Star"
            elif bullish_engulfing:
                s_candle, candle_name = WEIGHTS["candle_pattern"] * 0.8, "Engulfing"
            elif hammer:
                s_candle, candle_name = WEIGHTS["candle_pattern"] * 0.6, "Hammer"
        except Exception:
            pass   # fewer than 3 rows or missing Open — just skip pattern

        # ── Relative Strength vs Index ────────────────────────────
        # +5 pts if stock outperformed the broader index today
        s_rs = WEIGHTS["rel_strength"] if day_chg > index_chg else 0.0

        # ── Multi-Timeframe Confirmation (weekly) ─────────────────
        # Resample daily → weekly, compute weekly EMA20.
        # +8 pts if daily close is above weekly EMA20 (trend alignment).
        s_mtf        = 0.0
        weekly_align = False
        try:
            weekly_close = df["Close"].resample("W").last().dropna()
            if len(weekly_close) >= 20:
                wema20       = float(ta.ema(weekly_close, length=20).iloc[-1])
                weekly_align = close > wema20
                s_mtf        = WEIGHTS["weekly_confirm"] if weekly_align else 0.0
        except Exception:
            pass

        # ── ATR-based dynamic penalty ─────────────────────────────
        total   = s_vol + s_rsi + s_macd + s_ema + s_brk + s_52w + s_adx
        atr_pct = (atr_val / close * 100) if close > 0 else 4.0
        if day_chg > max(1.5 * atr_pct, 4.0):
            total *= 0.6

        # ── Gap-up and hold ───────────────────────────────────────
        # Gap-up: today's open > yesterday's close → bullish overnight demand
        # Held: close did not fill the gap (close stays above prev_close)
        # Strong: gap ≥1% AND close held AND range position ≥60% (buyers in control)
        # Mild  : gap ≥0.5% AND close held
        gap_pct  = 0.0
        gap_held = False
        s_gap    = 0.0
        try:
            open0   = float(df["Open"].iloc[-1])
            gap_pct = (open0 - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
            gap_held = close > prev_close   # gap not filled by end of day
            if gap_pct >= 1.0 and gap_held and pos >= 0.60:
                s_gap = WEIGHTS["gap_up"]          # strong gap-and-hold: +8
            elif gap_pct >= 0.5 and gap_held:
                s_gap = WEIGHTS["gap_up"] * 0.625  # mild gap-and-hold: +5
        except Exception:
            pass

        # ── Additive bonuses (applied after penalty) ─────────────
        total += sector_bonus + s_candle + s_rs + s_mtf + s_gap

        # ── Stop-Loss and Target ──────────────────────────────────
        stop_loss = round(max(low, close - atr_val), 2)
        target    = round(close + 1.5 * atr_val, 2)
        risk      = close - stop_loss
        reward    = target - close
        rr_ratio  = round(reward / risk, 2) if risk > 0 else 0.0

        clean_sym = (symbol.replace(".NS", "")
                           .replace("-", ".")
                           .replace("BRK.B", "BRK-B"))

        return {
            "Symbol":        clean_sym,
            "Close":         round(close, 2),
            "Change%":       round(day_chg, 2),
            "Volume_Ratio":  round(vol_ratio, 2),
            "RSI":           round(rsi, 1),
            "MACD_Hist":     round(macd_hist, 4) if macd_hist is not None else None,
            "EMA20":         round(ema20, 2),
            "EMA50":         round(ema50, 2),
            "ADX":           round(adx_val, 1),
            "Range_Pos%":    round(pos * 100, 1),
            "ATR":           round(atr_val, 2),
            "52W_High":      round(w52_high, 2),
            "Near_52W_High": near_52w,
            "Candle":        candle_name,
            "RS_Beat":       day_chg > index_chg,
            "Weekly_Align":  weekly_align,
            "Gap_Up":        gap_pct >= 0.5 and gap_held,  # True = gap-up held
            "Gap_Pct":       round(gap_pct, 2),
            "Sector_Align":  sector_bonus > 0,
            "Stop_Loss":     stop_loss,
            "Target":        target,
            "RR_Ratio":      rr_ratio,
            "BTST_Score":    round(total, 1),
        }
    except Exception as e:
        print(f"  {Fore.YELLOW}⚠  Skipping {symbol}: {e}{Style.RESET_ALL}")
        return None


# ══════════════════════════════════════════════════════════
# ORB SCORE — single stock, live 5-min intraday data
# ══════════════════════════════════════════════════════════

def score_orb_stock(symbol: str, sector_bonus: float = 0.0) -> dict | None:
    """
    Download today's 5-min bars, identify the Opening Range (first ORB_BARS bars),
    and score bullish breakouts above the ORB High.
    Returns None if no breakout or insufficient data.
    """
    try:
        # ── Fetch intraday 5-min data (2 days to guarantee today's bars) ──
        raw = yf.download(symbol, period="2d", interval="5m",
                          progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna()
        if len(raw) < ORB_BARS + 2:
            return None

        # ── Localise index to market timezone ────────────────────────
        is_india = symbol.endswith(".NS")
        tz       = IST if is_india else EST
        currency = "₹" if is_india else "$"

        if raw.index.tzinfo is None:
            raw.index = raw.index.tz_localize("UTC")
        raw.index = raw.index.tz_convert(tz)

        # ── Keep only today's bars ────────────────────────────────────
        today_date = datetime.now(tz=tz).date()
        df = raw[raw.index.date == today_date].copy()
        if len(df) < ORB_BARS + 1:
            return None          # market not open long enough yet

        # ── Opening Range (first ORB_BARS bars) ──────────────────────
        orb     = df.iloc[:ORB_BARS]
        orb_high  = float(orb["High"].max())
        orb_low   = float(orb["Low"].min())
        orb_range = orb_high - orb_low
        orb_range_pct = (orb_range / orb_high * 100) if orb_high > 0 else 0.0

        # ── Current (latest) bar ─────────────────────────────────────
        cur_bar = df.iloc[-1]
        price   = float(cur_bar["Close"])
        cur_vol = float(cur_bar["Volume"])

        # ── Breakout condition: price must be ABOVE ORB High ─────────
        if price <= orb_high:
            return None          # not yet broken out (or broken down)

        # ── 1. Breakout strength ──────────────────────────────────────
        brk_pct = (price - orb_high) / orb_high * 100
        s_brk   = (ORB_WEIGHTS["breakout_strength"]        if brk_pct >= 1.0 else
                   ORB_WEIGHTS["breakout_strength"] * 0.72 if brk_pct >= 0.5 else
                   ORB_WEIGHTS["breakout_strength"] * 0.48)

        # ── 2. Volume surge (current bar vs ORB avg) ─────────────────
        orb_avg_vol = float(orb["Volume"].mean())
        vol_ratio   = cur_vol / orb_avg_vol if orb_avg_vol > 0 else 1.0
        s_vol = (ORB_WEIGHTS["volume_surge"]        if vol_ratio >= 2.0 else
                 ORB_WEIGHTS["volume_surge"] * 0.70 if vol_ratio >= 1.5 else
                 ORB_WEIGHTS["volume_surge"] * 0.40 if vol_ratio >= 1.2 else 0)

        # ── 3. RSI on 5-min bars ──────────────────────────────────────
        rsi_s = ta.rsi(df["Close"], length=14)
        rsi   = float(rsi_s.iloc[-1]) if rsi_s is not None and not rsi_s.empty else 50.0
        s_rsi = (ORB_WEIGHTS["rsi_5m"]        if rsi >= 55 else
                 ORB_WEIGHTS["rsi_5m"] * 0.53 if rsi >= 50 else 0)

        # ── 4. ADX on 5-min bars ──────────────────────────────────────
        adx_df  = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        adx_val = 0.0
        s_adx   = 0
        if adx_df is not None and not adx_df.empty:
            ac = [c for c in adx_df.columns if c.startswith("ADX_")]
            if ac:
                adx_val = float(adx_df[ac[0]].iloc[-1])
                s_adx   = (ORB_WEIGHTS["adx_5m"]        if adx_val >= 25 else
                           ORB_WEIGHTS["adx_5m"] * 0.50 if adx_val >= 20 else 0)

        # ── 5. ORB range quality (tighter = cleaner breakout) ────────
        s_range = (ORB_WEIGHTS["orb_range_tight"]        if orb_range_pct <= 1.0 else
                   ORB_WEIGHTS["orb_range_tight"] * 0.70 if orb_range_pct <= 1.5 else
                   ORB_WEIGHTS["orb_range_tight"] * 0.40 if orb_range_pct <= 2.0 else 0)

        # ── 6. First bar of the day was bullish ───────────────────────
        first = df.iloc[0]
        s_candle = (ORB_WEIGHTS["open_candle_bull"]
                    if float(first["Close"]) > float(first["Open"]) else 0)

        # ── 7. Sector alignment bonus (passed in) ─────────────────────
        total = s_brk + s_vol + s_rsi + s_adx + s_range + s_candle + sector_bonus

        # ── ATR for stop-loss ─────────────────────────────────────────
        atr_s   = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        atr_val = (float(atr_s.iloc[-1])
                   if atr_s is not None and not atr_s.empty else orb_range)

        # ── Stop Loss and Target ──────────────────────────────────────
        # SL  = ORB Low (clean invalidation level)  OR  price – ATR, whichever is tighter
        # Tgt = ORB High + 1.5 × ORB Range  (classic first extension)
        stop_loss = round(max(orb_low, price - atr_val), 2)
        target    = round(orb_high + 1.5 * orb_range, 2)
        risk      = price - stop_loss
        reward    = target - price
        rr_ratio  = round(reward / risk, 2) if risk > 0 else 0.0

        clean_sym = (symbol.replace(".NS", "")
                           .replace("-", ".")
                           .replace("BRK.B", "BRK-B"))

        return {
            "Symbol":       clean_sym,
            "Price":        round(price, 2),
            "ORB_High":     round(orb_high, 2),
            "ORB_Low":      round(orb_low, 2),
            "ORB_Range%":   round(orb_range_pct, 2),
            "Brk_Pct":      round(brk_pct, 2),
            "Vol_Ratio":    round(vol_ratio, 2),
            "RSI_5m":       round(rsi, 1),
            "ADX_5m":       round(adx_val, 1),
            "Sector_Align": sector_bonus > 0,
            "Stop_Loss":    stop_loss,
            "Target":       target,
            "RR_Ratio":     rr_ratio,
            "ORB_Score":    round(total, 1),
        }
    except Exception as e:
        print(f"  {Fore.YELLOW}⚠  ORB skip {symbol}: {e}{Style.RESET_ALL}")
        return None


# ══════════════════════════════════════════════════════════
# RUN SCREENER  — batch download + parallel scoring
# ══════════════════════════════════════════════════════════

def run_screener(symbols: list, label: str,
                 index_chg: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """
    Returns (results_df, sector_perf) so the caller can pass sector_perf
    to run_orb_screener and avoid a duplicate fetch.
    """
    hdr(f"Scanning {len(symbols)} {label} stocks (batch + parallel) …")

    market     = "india" if any(s.endswith(".NS") for s in symbols) else "usa"
    sector_map = INDIA_SECTOR_MAP if market == "india" else USA_SECTOR_MAP

    # Step 1+2 in PARALLEL: batch OHLCV download + sector perf fetch overlap
    print(f"  📥  Fetching 1-year OHLCV + sector data in parallel …", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pre:
        f_cache  = pre.submit(_batch_download, symbols)
        f_sector = pre.submit(fetch_sector_perf, market)
        cache       = f_cache.result()
        sector_perf = f_sector.result()
    print(f"  ✅  Downloaded {len(cache)}/{len(symbols)}. Scoring …", flush=True)

    def _sector_bonus(sym: str) -> float:
        sec_ticker = sector_map.get(sym)
        return float(SECTOR_BONUS_PTS) if sec_ticker and sector_perf.get(sec_ticker, False) else 0.0

    # Step 3: score all stocks in parallel (TA calcs are CPU-bound)
    results = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {
            pool.submit(score_stock_from_df, sym, df, _sector_bonus(sym), index_chg): sym
            for sym, df in cache.items()
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            print(f"  ⚙️  Scoring [{done:>3}/{len(cache)}]", end="\r", flush=True)
            r = future.result()
            if r:
                results.append(r)

    print(" " * 60, end="\r")
    print(f"  ✅  Scored {len(results)} stocks successfully.")
    return pd.DataFrame(results), sector_perf


# ══════════════════════════════════════════════════════════
# BATCH INTRADAY DOWNLOAD — all tickers in ONE 5-min call
# ══════════════════════════════════════════════════════════

def _batch_download_intraday(symbols: list) -> dict:
    """
    Download today's 5-min bars for ALL symbols in a single yf.download() call.
    Returns {symbol: DataFrame} with today's bars only, tz-aware index.
    """
    is_india = any(s.endswith(".NS") for s in symbols)
    tz = IST if is_india else EST

    print(f"  📥  Batch-fetching 5-min bars for {len(symbols)} tickers …", flush=True)
    try:
        raw = yf.download(
            symbols,
            period="2d",
            interval="5m",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        print(f"  ⚠  Intraday batch download failed: {e}")
        return {}

    today_date = datetime.now(tz=tz).date()
    result = {}
    for sym in symbols:
        try:
            df = raw[sym].dropna() if isinstance(raw.columns, pd.MultiIndex) else raw.dropna()
            if df.empty:
                continue
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert(tz)
            df = df[df.index.date == today_date]
            if len(df) >= ORB_BARS + 1:
                result[sym] = df
        except Exception:
            pass

    print(f"  ✅  Intraday data: {len(result)}/{len(symbols)} symbols with enough bars.")
    return result


def score_orb_stock_from_df(symbol: str, df: pd.DataFrame, sector_bonus: float = 0.0) -> dict | None:
    """
    Score ORB for a single stock using pre-downloaded 5-min DataFrame.
    Same logic as score_orb_stock() but no network I/O.
    """
    try:
        is_india  = symbol.endswith(".NS")
        orb       = df.iloc[:ORB_BARS]
        orb_high  = float(orb["High"].max())
        orb_low   = float(orb["Low"].min())
        orb_range = orb_high - orb_low
        orb_range_pct = (orb_range / orb_high * 100) if orb_high > 0 else 0.0

        cur_bar = df.iloc[-1]
        price   = float(cur_bar["Close"])
        cur_vol = float(cur_bar["Volume"])

        if price <= orb_high:
            return None   # no breakout

        brk_pct = (price - orb_high) / orb_high * 100
        s_brk   = (ORB_WEIGHTS["breakout_strength"]        if brk_pct >= 1.0 else
                   ORB_WEIGHTS["breakout_strength"] * 0.72 if brk_pct >= 0.5 else
                   ORB_WEIGHTS["breakout_strength"] * 0.48)

        orb_avg_vol = float(orb["Volume"].mean())
        vol_ratio   = cur_vol / orb_avg_vol if orb_avg_vol > 0 else 1.0
        s_vol = (ORB_WEIGHTS["volume_surge"]        if vol_ratio >= 2.0 else
                 ORB_WEIGHTS["volume_surge"] * 0.70 if vol_ratio >= 1.5 else
                 ORB_WEIGHTS["volume_surge"] * 0.40 if vol_ratio >= 1.2 else 0)

        rsi_s = ta.rsi(df["Close"], length=14)
        rsi   = float(rsi_s.iloc[-1]) if rsi_s is not None and not rsi_s.empty else 50.0
        s_rsi = (ORB_WEIGHTS["rsi_5m"]        if rsi >= 55 else
                 ORB_WEIGHTS["rsi_5m"] * 0.53 if rsi >= 50 else 0)

        adx_df  = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        adx_val = 0.0
        s_adx   = 0
        if adx_df is not None and not adx_df.empty:
            ac = [c for c in adx_df.columns if c.startswith("ADX_")]
            if ac:
                adx_val = float(adx_df[ac[0]].iloc[-1])
                s_adx   = (ORB_WEIGHTS["adx_5m"]        if adx_val >= 25 else
                           ORB_WEIGHTS["adx_5m"] * 0.50 if adx_val >= 20 else 0)

        s_range = (ORB_WEIGHTS["orb_range_tight"]        if orb_range_pct <= 1.0 else
                   ORB_WEIGHTS["orb_range_tight"] * 0.70 if orb_range_pct <= 1.5 else
                   ORB_WEIGHTS["orb_range_tight"] * 0.40 if orb_range_pct <= 2.0 else 0)

        first    = df.iloc[0]
        s_candle = (ORB_WEIGHTS["open_candle_bull"]
                    if float(first["Close"]) > float(first["Open"]) else 0)

        total = s_brk + s_vol + s_rsi + s_adx + s_range + s_candle + sector_bonus

        atr_s   = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        atr_val = (float(atr_s.iloc[-1])
                   if atr_s is not None and not atr_s.empty else orb_range)

        stop_loss = round(max(orb_low, price - atr_val), 2)
        target    = round(orb_high + 1.5 * orb_range, 2)
        risk      = price - stop_loss
        reward    = target - price
        rr_ratio  = round(reward / risk, 2) if risk > 0 else 0.0

        clean_sym = (symbol.replace(".NS", "")
                           .replace("-", ".")
                           .replace("BRK.B", "BRK-B"))

        return {
            "Symbol":       clean_sym,
            "Price":        round(price, 2),
            "ORB_High":     round(orb_high, 2),
            "ORB_Low":      round(orb_low, 2),
            "ORB_Range%":   round(orb_range_pct, 2),
            "Brk_Pct":      round(brk_pct, 2),
            "Vol_Ratio":    round(vol_ratio, 2),
            "RSI_5m":       round(rsi, 1),
            "ADX_5m":       round(adx_val, 1),
            "Sector_Align": sector_bonus > 0,
            "Stop_Loss":    stop_loss,
            "Target":       target,
            "RR_Ratio":     rr_ratio,
            "ORB_Score":    round(total, 1),
        }
    except Exception as e:
        print(f"  {Fore.YELLOW}⚠  ORB skip {symbol}: {e}{Style.RESET_ALL}")
        return None


# ══════════════════════════════════════════════════════════
# RUN ORB SCREENER  — batch intraday download + parallel score
# ══════════════════════════════════════════════════════════

def run_orb_screener(symbols: list, label: str,
                     sector_perf: dict | None = None) -> pd.DataFrame:
    hdr(f"ORB Scan — {len(symbols)} {label} stocks (5-min bars) …")

    market     = "india" if any(s.endswith(".NS") for s in symbols) else "usa"
    sector_map = INDIA_SECTOR_MAP if market == "india" else USA_SECTOR_MAP
    if sector_perf is None:          # fallback: fetch if not passed in
        sector_perf = fetch_sector_perf(market)
    else:
        print(f"  ♻️  Reusing sector perf from BTST scan (skipping re-fetch).")

    def _sector_bonus(sym: str) -> float:
        sec = sector_map.get(sym)
        return float(SECTOR_BONUS_PTS) if sec and sector_perf.get(sec, False) else 0.0

    # One batch intraday download instead of N individual downloads
    intraday_cache = _batch_download_intraday(symbols)

    results = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {
            pool.submit(score_orb_stock_from_df, sym, df, _sector_bonus(sym)): sym
            for sym, df in intraday_cache.items()
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            print(f"  ⚙️  ORB [{done:>3}/{len(intraday_cache)}]", end="\r", flush=True)
            r = future.result()
            if r:
                results.append(r)

    print(" " * 60, end="\r")
    breakouts = len(results)
    print(f"  ✅  ORB: {breakouts} confirmed breakout(s) found out of {len(symbols)} scanned.")
    return pd.DataFrame(results)


def filter_and_rank_orb(df: pd.DataFrame) -> pd.DataFrame:
    """Filter ORB candidates: valid R:R, decent volume, momentum; rank by score."""
    if df.empty:
        return df
    f = df[
        (df["RSI_5m"]   >= 50) &
        (df["Vol_Ratio"] >= 1.2) &
        (df["RR_Ratio"]  >= 1.5)
    ].copy()
    f.sort_values("ORB_Score", ascending=False, inplace=True)
    return f.head(15)


# ══════════════════════════════════════════════════════════
# FILTER & RANK
# ══════════════════════════════════════════════════════════

def filter_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    f = df[
        (df["RSI"] >= 50) &
        (df["Volume_Ratio"] >= 1.2) &
        (df["Change%"] > 0) &
        (df["Change%"] <= 5.0) &
        (df["Close"] > df["EMA20"])
    ].copy()
    f.sort_values("BTST_Score", ascending=False, inplace=True)
    return f.head(15)


# ══════════════════════════════════════════════════════════
# CONSOLE PRINT REPORT
# ══════════════════════════════════════════════════════════

def print_report(df: pd.DataFrame, label: str, market_ok: bool, idx_chg: float):
    tz   = IST if "INDIA" in label.upper() else EST
    now  = datetime.now(tz=tz)
    tzn  = "IST" if "INDIA" in label.upper() else "EST"
    hdr(f"{label} BTST Report  |  {now.strftime('%d-%b-%Y %I:%M %p')} {tzn}")

    col = Fore.GREEN if market_ok else Fore.RED
    print(f"  Market: {col}{'BULLISH' if market_ok else 'CAUTION'} ({idx_chg:+.2f}%){Style.RESET_ALL}")

    if df.empty:
        print(f"\n  {Fore.YELLOW}No strong candidates found.{Style.RESET_ALL}")
        return

    cols = ["Symbol", "Close", "Change%", "Volume_Ratio", "RSI", "ADX",
            "Range_Pos%", "Near_52W_High", "Gap_Up", "Candle", "RS_Beat",
            "Weekly_Align", "Sector_Align", "Stop_Loss", "Target", "RR_Ratio", "BTST_Score"]
    available = [c for c in cols if c in df.columns]
    disp = df[available].reset_index(drop=True)
    disp.index += 1
    print(f"\n{Fore.CYAN}  TOP BTST CANDIDATES{Style.RESET_ALL}\n")
    print(tabulate(disp, headers="keys", tablefmt="fancy_grid", floatfmt=".2f", showindex=True))


# ══════════════════════════════════════════════════════════
# SAVE CSV
# ══════════════════════════════════════════════════════════

def save_csv(top_df: pd.DataFrame, full_df: pd.DataFrame, prefix: str, date_str: str):
    top_df.to_csv(f"btst_{prefix}_{date_str}.csv", index=False)
    full_df.sort_values("BTST_Score", ascending=False).to_csv(
        f"btst_{prefix}_full_{date_str}.csv", index=False)
    print(f"  💾  {prefix.upper()} CSVs saved.")


def save_meta(prefix: str, date_str: str, ok: bool, chg: float, vix: float):
    """Save market health metadata so --html-only can reconstruct the report."""
    meta = {"ok": ok, "chg": round(chg, 4), "vix": round(vix, 4)}
    with open(f"btst_{prefix}_meta_{date_str}.json", "w") as f:
        json.dump(meta, f)
    print(f"  📝  {prefix.upper()} metadata saved.")


def load_meta(prefix: str, date_str: str) -> tuple[bool, float, float]:
    """Load saved market health metadata. Returns (ok, chg, vix) or safe defaults."""
    try:
        with open(f"btst_{prefix}_meta_{date_str}.json") as f:
            m = json.load(f)
        return bool(m.get("ok", True)), float(m.get("chg", 0.0)), float(m.get("vix", 0.0))
    except Exception:
        return True, 0.0, 0.0


# ══════════════════════════════════════════════════════════
# HTML TABLE ROWS BUILDER
# ══════════════════════════════════════════════════════════

def _rows(df: pd.DataFrame, currency: str = "₹",
          prev_scores: dict | None = None) -> str:
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    rows   = ""
    prev_scores = prev_scores or {}

    for rank, (_, row) in enumerate(df.iterrows(), 1):
        medal  = medals.get(rank, f"#{rank}")
        chg_c  = "#00e676" if row["Change%"] >= 0 else "#ff5252"
        chg_a  = "▲" if row["Change%"] >= 0 else "▼"

        vol_cls = "bg" if row["Volume_Ratio"] >= 1.5 else "by" if row["Volume_Ratio"] >= 1.2 else "br"
        rsi_cls = "bg" if 55 <= row["RSI"] <= 75 else "by" if row["RSI"] >= 50 else "br"

        sc  = row["BTST_Score"]
        bc  = "#00e676" if sc >= 70 else "#ffca28" if sc >= 50 else "#ff5252"
        pct = min(sc / 130 * 100, 100)   # new max ≈130 pts

        # ── 52W high badge ───────────────────────────────────────
        near52  = row.get("Near_52W_High", False)
        badge52 = '<span class="badge bg">🚀 52W</span>' if near52 else '<span class="badge br">—</span>'

        # ── Sector alignment — colour-coded TD background ────────
        sec_align = row.get("Sector_Align", False)
        sec_bg    = "rgba(0,230,118,0.10)" if sec_align else "rgba(255,82,82,0.08)"
        sec_txt   = "#00e676"              if sec_align else "#ff5252"
        sec_label = "✅ Green"             if sec_align else "❌ Red"
        badge_sec = (f'<td style="background:{sec_bg};text-align:center">'
                     f'<span style="color:{sec_txt};font-weight:700;font-size:.72rem">'
                     f'{sec_label}</span></td>')

        # ── Gap-up badge ─────────────────────────────────────────
        gap_up  = row.get("Gap_Up", False)
        gap_pct_val = row.get("Gap_Pct", 0.0)
        if gap_up and gap_pct_val >= 1.0:
            badge_gap = f'<span class="badge bg">⬆ {gap_pct_val:+.1f}%</span>'
        elif gap_up:
            badge_gap = f'<span class="badge by">⬆ {gap_pct_val:+.1f}%</span>'
        else:
            badge_gap = '<span class="badge br" style="color:var(--muted)">—</span>'

        # ── Candlestick pattern badge ────────────────────────────
        candle = str(row.get("Candle", "")) if row.get("Candle") else ""
        if candle == "Morning Star":
            badge_candle = '<span class="badge bg">⭐ M-Star</span>'
        elif candle == "Engulfing":
            badge_candle = '<span class="badge bg">🕯 Engulf</span>'
        elif candle == "Hammer":
            badge_candle = '<span class="badge by">🔨 Hammer</span>'
        else:
            badge_candle = '<span class="badge br" style="color:var(--muted)">—</span>'

        # ── Relative Strength badge ──────────────────────────────
        rs_beat = row.get("RS_Beat", False)
        badge_rs = ('<span class="badge bg">📈 RS+</span>' if rs_beat
                    else '<span class="badge br" style="color:var(--muted)">—</span>')

        # ── Weekly MTF badge ─────────────────────────────────────
        w_align   = row.get("Weekly_Align", False)
        badge_mtf = ('<span class="badge bg">✅ W</span>' if w_align
                     else '<span class="badge br" style="color:var(--muted)">—</span>')

        # ── Score trend arrow ────────────────────────────────────
        sym = str(row.get("Symbol", ""))
        prev_sc = prev_scores.get(sym)
        if prev_sc is not None:
            diff = sc - prev_sc
            if diff >= 2:
                arrow = f'<span style="color:#00e676;font-size:.7rem"> ▲{diff:+.0f}</span>'
            elif diff <= -2:
                arrow = f'<span style="color:#ff5252;font-size:.7rem"> ▼{diff:+.0f}</span>'
            else:
                arrow = '<span style="color:var(--muted);font-size:.7rem"> ●</span>'
        else:
            arrow = ""

        # ── Stop-loss / Target / R:R ─────────────────────────────
        sl     = row.get("Stop_Loss", 0)
        tgt    = row.get("Target", 0)
        rr     = row.get("RR_Ratio", 0)
        rr_col = "#00e676" if rr >= 2 else "#ffca28" if rr >= 1.5 else "#ff5252"

        rows += f"""
        <tr>
          <td class="rnk">{medal}</td>
          <td class="sym">{sym}</td>
          <td class="num">{currency}{row['Close']:,.2f}</td>
          <td><span style="color:{chg_c};font-weight:700">{chg_a} {abs(row['Change%']):.2f}%</span></td>
          <td><span class="badge {vol_cls}">{row['Volume_Ratio']:.2f}x</span></td>
          <td><span class="badge {rsi_cls}">{row['RSI']:.1f}</span></td>
          <td class="num">{row['ADX']:.1f}</td>
          <td class="num">{row['Range_Pos%']:.1f}%</td>
          <td>{badge52}</td>
          <td>{badge_gap}</td>
          <td>{badge_candle}</td>
          <td>{badge_rs}</td>
          <td>{badge_mtf}</td>
          {badge_sec}
          <td class="num" style="color:#ff5252">{currency}{sl:,.2f}</td>
          <td class="num" style="color:#00e676">{currency}{tgt:,.2f}</td>
          <td class="num" style="color:{rr_col};font-weight:700">{rr:.1f}x</td>
          <td>
            <div class="bw">
              <div class="bt"><div class="b" style="width:{pct:.0f}%;background:{bc}"></div></div>
              <span class="bl">{sc:.1f}{arrow}</span>
            </div>
          </td>
        </tr>"""
    return rows


def _rows_orb(df: pd.DataFrame, currency: str = "₹") -> str:
    """Build HTML table rows for ORB candidates."""
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    rows   = ""
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        medal   = medals.get(rank, f"#{rank}")
        sc      = row["ORB_Score"]
        bc      = "#00e676" if sc >= 60 else "#ffca28" if sc >= 40 else "#ff5252"
        pct     = min(sc / 95 * 100, 100)

        vol_cls = "bg" if row["Vol_Ratio"] >= 2.0 else "by" if row["Vol_Ratio"] >= 1.5 else "br"
        rsi_cls = "bg" if row["RSI_5m"] >= 55    else "by" if row["RSI_5m"] >= 50    else "br"

        sec_align = row.get("Sector_Align", False)
        sec_bg    = "rgba(0,230,118,0.10)" if sec_align else "rgba(255,82,82,0.08)"
        sec_txt   = "#00e676" if sec_align else "#ff5252"
        sec_label = "✅ Green" if sec_align else "❌ Red"

        rr     = row.get("RR_Ratio", 0)
        rr_col = "#00e676" if rr >= 2 else "#ffca28" if rr >= 1.5 else "#ff5252"
        sl     = row.get("Stop_Loss", 0)
        tgt    = row.get("Target", 0)

        brk_pct = row.get("Brk_Pct", 0)
        brk_col = "#00e676" if brk_pct >= 1.0 else "#ffca28"

        rows += f"""
        <tr>
          <td class="rnk">{medal}</td>
          <td class="sym">{row['Symbol']}</td>
          <td class="num">{currency}{row['Price']:,.2f}</td>
          <td class="num" style="color:var(--green);font-weight:700">{currency}{row['ORB_High']:,.2f}</td>
          <td class="num" style="color:var(--red)">{currency}{row['ORB_Low']:,.2f}</td>
          <td class="num">{row['ORB_Range%']:.2f}%</td>
          <td><span style="color:{brk_col};font-weight:700">+{brk_pct:.2f}%</span></td>
          <td><span class="badge {vol_cls}">{row['Vol_Ratio']:.2f}x</span></td>
          <td><span class="badge {rsi_cls}">{row['RSI_5m']:.1f}</span></td>
          <td class="num">{row['ADX_5m']:.1f}</td>
          <td style="background:{sec_bg};text-align:center">
            <span style="color:{sec_txt};font-weight:700;font-size:.72rem">{sec_label}</span>
          </td>
          <td class="num" style="color:#ff5252">{currency}{sl:,.2f}</td>
          <td class="num" style="color:#00e676">{currency}{tgt:,.2f}</td>
          <td class="num" style="color:{rr_col};font-weight:700">{rr:.1f}x</td>
          <td>
            <div class="bw">
              <div class="bt"><div class="b" style="width:{pct:.0f}%;background:{bc}"></div></div>
              <span class="bl">{sc:.1f}</span>
            </div>
          </td>
        </tr>"""
    return rows


def _summary_cards(top_df: pd.DataFrame, total_scanned: int, tab_id: str) -> str:
    if top_df.empty or "BTST_Score" not in top_df.columns:
        strong = moderate = weak = 0
    else:
        strong   = len(top_df[top_df["BTST_Score"] >= 70])
        moderate = len(top_df[(top_df["BTST_Score"] >= 50) & (top_df["BTST_Score"] < 70)])
        weak     = len(top_df[top_df["BTST_Score"] < 50])
    return f"""
    <div class="cards" id="cards-{tab_id}">
      <div class="card cg"><span class="card-ico">🟢</span><div class="card-val" style="color:var(--green)">{strong}</div><div class="card-lbl">Strong Picks (≥70)</div></div>
      <div class="card cy"><span class="card-ico">🟡</span><div class="card-val" style="color:var(--yellow)">{moderate}</div><div class="card-lbl">Moderate (50–69)</div></div>
      <div class="card cr"><span class="card-ico">🔴</span><div class="card-val" style="color:var(--red)">{weak}</div><div class="card-lbl">Weak (&lt;50)</div></div>
      <div class="card cb"><span class="card-ico">📋</span><div class="card-val" style="color:var(--blue)">{len(top_df)}</div><div class="card-lbl">Total Candidates</div></div>
      <div class="card cm"><span class="card-ico">🔍</span><div class="card-val" style="color:var(--text)">{total_scanned}</div><div class="card-lbl">Stocks Scanned</div></div>
    </div>"""


# ══════════════════════════════════════════════════════════
# GENERATE COMBINED HTML REPORT
# ══════════════════════════════════════════════════════════

def generate_html_report(
    india_top, india_full, india_ok, india_chg, india_vix,
    usa_top,   usa_full,   usa_ok,   usa_chg,   usa_vix,
    date_str: str,
    orb_india_df: pd.DataFrame | None = None,
    orb_usa_df:   pd.DataFrame | None = None,
):
    now_ist  = datetime.now(tz=IST)
    now_est  = datetime.now(tz=EST)
    time_ist = now_ist.strftime("%d %b %Y, %I:%M %p IST")
    time_est = now_est.strftime("%d %b %Y, %I:%M %p EST")
    html_file = f"btst_report_{date_str}.html"

    india_rows = _rows(india_top, "₹", _load_prev_scores("india", date_str)) if not india_top.empty else "<tr><td colspan='18' style='text-align:center;color:var(--muted);padding:30px'>No candidates found today</td></tr>"
    usa_rows   = _rows(usa_top,   "$", _load_prev_scores("usa",   date_str)) if not usa_top.empty   else "<tr><td colspan='18' style='text-align:center;color:var(--muted);padding:30px'>No candidates found today</td></tr>"

    # ── ORB rows ──────────────────────────────────────────────
    _orb_empty = "<tr><td colspan='15' style='text-align:center;color:var(--muted);padding:30px'>No ORB breakouts detected — market may not be open yet, or no confirmed breakouts this session.</td></tr>"
    orb_india  = orb_india_df if orb_india_df is not None else pd.DataFrame()
    orb_usa    = orb_usa_df   if orb_usa_df   is not None else pd.DataFrame()
    orb_india_rows = _rows_orb(orb_india, "₹") if not orb_india.empty else _orb_empty
    orb_usa_rows   = _rows_orb(orb_usa,   "$") if not orb_usa.empty   else _orb_empty
    orb_india_count = len(orb_india)
    orb_usa_count   = len(orb_usa)

    india_cards = _summary_cards(india_top, len(india_full), "india")
    usa_cards   = _summary_cards(usa_top,   len(usa_full),   "usa")

    india_m_col = "#00e676" if india_ok else "#ff5252"
    usa_m_col   = "#00e676" if usa_ok   else "#ff5252"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTST Screener — {date_str}</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:     #080c10; --surf:  #0d1117; --surf2: #161b22;
    --border: #21262d; --green: #00e676; --yellow:#ffca28;
    --red:    #ff5252; --blue:  #40c4ff; --text:  #e6edf3;
    --muted:  #7d8590; --r:     12px;
    --mono: 'Space Mono',monospace; --sans: 'Syne',sans-serif;
  }}
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  html{{scroll-behavior:smooth}}
  body{{background:var(--bg);color:var(--text);font-family:var(--sans);padding-bottom:60px;-webkit-font-smoothing:antialiased}}

  /* HEADER */
  .header{{background:linear-gradient(135deg,#0d1117 0%,#0a1628 55%,#0d1117 100%);border-bottom:1px solid var(--border);padding:clamp(18px,4vw,36px) clamp(14px,5vw,48px) clamp(16px,3vw,28px);position:relative;overflow:hidden}}
  .header::before{{content:'';position:absolute;top:-80px;right:-80px;width:clamp(160px,28vw,300px);height:clamp(160px,28vw,300px);background:radial-gradient(circle,rgba(0,230,118,.07) 0%,transparent 70%);pointer-events:none}}
  .header::after{{content:'';position:absolute;bottom:-70px;left:25%;width:200px;height:200px;background:radial-gradient(circle,rgba(64,196,255,.05) 0%,transparent 70%);pointer-events:none}}
  .header-top{{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap}}
  .logo h1{{font-size:clamp(1.35rem,4vw,2.2rem);font-weight:800;letter-spacing:-.5px;line-height:1.1}}
  .logo h1 span{{color:var(--green)}}
  .logo p{{font-family:var(--mono);font-size:clamp(.58rem,1.3vw,.72rem);color:var(--muted);margin-top:5px;letter-spacing:1.5px;text-transform:uppercase}}
  .ts{{font-family:var(--mono);font-size:clamp(.6rem,1.2vw,.72rem);color:var(--muted);text-align:right;line-height:1.7;flex-shrink:0}}
  .ts-time{{display:block;font-size:clamp(.82rem,2vw,.95rem);font-weight:700;color:var(--blue);margin-bottom:2px;white-space:nowrap}}
  .tz-tag{{display:inline-block;background:rgba(64,196,255,.12);color:var(--blue);border:1px solid rgba(64,196,255,.3);border-radius:4px;font-family:var(--mono);font-size:.58rem;padding:1px 6px;letter-spacing:1px;vertical-align:middle;margin-left:4px}}

  /* PILLS */
  .pills{{display:flex;gap:8px;margin-top:20px;flex-wrap:wrap}}
  .pill{{display:flex;align-items:center;gap:7px;background:var(--surf2);border:1px solid var(--border);border-radius:999px;padding:5px 13px;font-family:var(--mono);font-size:clamp(.58rem,1.2vw,.72rem);color:var(--muted);white-space:nowrap}}
  .dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
  .dot.live{{animation:blink 2s infinite}}
  @keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.25}}}}

  /* MARKET PILLS — per-tab, shown/hidden via JS */
  .market-pills{{display:none;flex-wrap:wrap;gap:8px;margin-top:16px}}
  .market-pills.active{{display:flex}}

  /* CONTENT */
  .content{{padding:clamp(20px,4vw,36px) clamp(14px,5vw,48px)}}

  /* ── TOGGLE ── */
  .toggle-wrap{{display:flex;gap:0;margin-bottom:30px;background:var(--surf2);border:1px solid var(--border);border-radius:999px;padding:4px;width:fit-content}}
  .tab-btn{{
    display:flex;align-items:center;gap:8px;
    background:transparent;border:none;border-radius:999px;
    padding:9px 22px;cursor:pointer;
    font-family:var(--mono);font-size:clamp(.68rem,1.4vw,.78rem);
    font-weight:700;color:var(--muted);
    transition:all .25s ease;white-space:nowrap;
  }}
  .tab-btn .flag{{font-size:1rem}}
  .tab-btn.active{{background:var(--surf);color:var(--text);box-shadow:0 2px 8px rgba(0,0,0,.4)}}
  .tab-btn.active.india-btn{{color:var(--green)}}
  .tab-btn.active.usa-btn{{color:var(--blue)}}
  .tab-btn:hover:not(.active){{color:var(--text)}}

  /* TAB PANELS */
  .tab-panel{{display:none;animation:fadeUp .35s ease}}
  .tab-panel.active{{display:block}}

  /* CARDS */
  .cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:32px}}
  @media(max-width:860px){{.cards{{grid-template-columns:repeat(3,1fr)}}}}
  @media(max-width:520px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}
  .card{{background:var(--surf);border:1px solid var(--border);border-radius:var(--r);padding:clamp(13px,2.5vw,22px) clamp(12px,2vw,20px) clamp(11px,2vw,18px);position:relative;overflow:hidden;transition:transform .2s,box-shadow .2s;animation:fadeUp .4s ease both}}
  .card:hover{{transform:translateY(-3px);box-shadow:0 10px 28px rgba(0,0,0,.45)}}
  .card::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:3px;border-radius:0 0 var(--r) var(--r)}}
  .card.cg::after{{background:var(--green)}} .card.cy::after{{background:var(--yellow)}}
  .card.cr::after{{background:var(--red)}}   .card.cb::after{{background:var(--blue)}}
  .card.cm::after{{background:var(--muted)}}
  .card-ico{{font-size:1.15rem;margin-bottom:8px;display:block}}
  .card-val{{font-family:var(--mono);line-height:1;font-size:clamp(1.5rem,3.5vw,2.4rem);font-weight:700}}
  .card-lbl{{font-size:clamp(.58rem,1.2vw,.7rem);color:var(--muted);text-transform:uppercase;letter-spacing:.9px;margin-top:7px;line-height:1.4}}
  .card:nth-child(1){{animation-delay:.04s}} .card:nth-child(2){{animation-delay:.08s}}
  .card:nth-child(3){{animation-delay:.12s}} .card:nth-child(4){{animation-delay:.16s}}
  .card:nth-child(5){{animation-delay:.20s}}

  /* SECTION HEADER */
  .sh{{display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap}}
  .sh-title{{font-size:clamp(.85rem,2vw,1rem);font-weight:700;white-space:nowrap}}
  .sh-line{{flex:1;height:1px;background:var(--border);min-width:16px}}
  .sh-sub{{font-family:var(--mono);font-size:clamp(.58rem,1.2vw,.68rem);color:var(--muted);white-space:nowrap}}

  /* TABLE */
  .scroll-hint{{display:none;font-family:var(--mono);font-size:.62rem;color:var(--muted);text-align:right;margin-bottom:6px}}
  @media(max-width:680px){{.scroll-hint{{display:block}}}}
  .tw{{overflow-x:auto;border-radius:var(--r);border:1px solid var(--border);-webkit-overflow-scrolling:touch}}
  table{{width:100%;border-collapse:collapse;font-size:clamp(.76rem,1.5vw,.875rem)}}
  thead tr{{background:var(--surf2)}}
  th{{font-family:var(--mono);font-size:clamp(.56rem,1.1vw,.63rem);text-transform:uppercase;letter-spacing:.9px;color:var(--muted);padding:clamp(10px,1.5vw,14px) clamp(10px,1.5vw,16px);text-align:left;white-space:nowrap;border-bottom:1px solid var(--border)}}
  tbody tr{{background:var(--surf);border-bottom:1px solid var(--border);transition:background .15s;animation:fadeUp .3s ease both}}
  tbody tr:hover{{background:var(--surf2)}}
  tbody tr:last-child{{border-bottom:none}}
  td{{padding:clamp(10px,1.5vw,13px) clamp(10px,1.5vw,16px);white-space:nowrap}}
  td.rnk{{font-size:1.05rem;text-align:center;width:46px}}
  td.sym{{font-family:var(--mono);font-weight:700;font-size:clamp(.76rem,1.5vw,.88rem);color:var(--blue);letter-spacing:.5px}}
  td.num{{font-family:var(--mono);font-size:clamp(.72rem,1.4vw,.82rem)}}
  tbody tr:nth-child(1){{animation-delay:.05s}} tbody tr:nth-child(2){{animation-delay:.09s}}
  tbody tr:nth-child(3){{animation-delay:.13s}} tbody tr:nth-child(4){{animation-delay:.17s}}
  tbody tr:nth-child(5){{animation-delay:.21s}} tbody tr:nth-child(6){{animation-delay:.25s}}
  tbody tr:nth-child(7){{animation-delay:.29s}} tbody tr:nth-child(8){{animation-delay:.33s}}
  tbody tr:nth-child(9){{animation-delay:.37s}} tbody tr:nth-child(10){{animation-delay:.41s}}

  /* BADGES */
  .badge{{display:inline-block;padding:3px 9px;border-radius:999px;font-family:var(--mono);font-size:clamp(.6rem,1.1vw,.7rem);font-weight:700}}
  .bg{{background:rgba(0,230,118,.12);color:var(--green);border:1px solid rgba(0,230,118,.3)}}
  .by{{background:rgba(255,202,40,.12);color:var(--yellow);border:1px solid rgba(255,202,40,.3)}}
  .br{{background:rgba(255,82,82,.12);color:var(--red);border:1px solid rgba(255,82,82,.3)}}

  /* SCORE BAR */
  .bw{{display:flex;align-items:center;gap:9px;min-width:115px}}
  .bt{{flex:1;height:6px;background:var(--border);border-radius:99px;overflow:hidden}}
  .b{{height:100%;border-radius:99px}}
  .bl{{font-family:var(--mono);font-size:clamp(.68rem,1.3vw,.78rem);font-weight:700;min-width:30px;text-align:right}}

  /* LEGEND */
  .legend{{display:flex;flex-wrap:wrap;gap:10px 18px;margin-top:18px;padding:clamp(12px,2vw,18px) clamp(12px,2vw,20px);background:var(--surf);border:1px solid var(--border);border-radius:var(--r)}}
  .li{{display:flex;align-items:center;gap:7px;font-family:var(--mono);font-size:clamp(.58rem,1.1vw,.68rem);color:var(--muted)}}
  .ld{{width:9px;height:9px;border-radius:2px;flex-shrink:0}}

  /* PARAMS GRID */
  .pg{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:28px}}
  @media(max-width:780px){{.pg{{grid-template-columns:repeat(2,1fr)}}}}
  @media(max-width:460px){{.pg{{grid-template-columns:1fr}}}}
  .pc{{background:var(--surf);border:1px solid var(--border);border-radius:10px;padding:clamp(11px,2vw,16px)}}
  .pn{{font-family:var(--mono);font-size:clamp(.6rem,1.2vw,.72rem);color:var(--blue);text-transform:uppercase;letter-spacing:.9px;margin-bottom:5px;display:flex;justify-content:space-between;align-items:center}}
  .pw{{font-family:var(--mono);font-size:clamp(.56rem,1.1vw,.65rem);color:var(--green);font-weight:700}}
  .pd{{font-size:clamp(.7rem,1.3vw,.78rem);color:var(--muted);line-height:1.55}}

  /* DISCLAIMER */
  .disc{{margin-top:26px;padding:clamp(12px,2vw,16px) clamp(13px,2vw,20px);background:rgba(255,202,40,.04);border:1px solid rgba(255,202,40,.18);border-radius:8px;font-size:clamp(.68rem,1.3vw,.74rem);color:var(--muted);line-height:1.75}}
  .disc strong{{color:var(--yellow)}}

  /* FOOTER */
  .footer{{text-align:center;padding:clamp(18px,3vw,28px) clamp(14px,5vw,48px) 0;font-family:var(--mono);font-size:clamp(.56rem,1.1vw,.65rem);color:var(--muted);border-top:1px solid var(--border);margin-top:40px;line-height:2.2}}

  @keyframes fadeUp{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:translateY(0)}}}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-top">
    <div class="logo">
      <h1>BTST <span>Screener</span></h1>
      <p>India — Nifty 100 &nbsp;·&nbsp; USA — S&P 500 Top 100 &nbsp;·&nbsp; Buy Today Sell Tomorrow</p>
    </div>
    <div class="ts">
      <span class="ts-time" id="hdr-time">—</span>
      <span id="hdr-date">—</span> · Auto-generated
    </div>
  </div>

  <!-- India status pills -->
  <div class="market-pills active" id="pills-india">
    <div class="pill"><div class="dot live" style="background:{india_m_col}"></div>Market:&nbsp;<strong style="color:{india_m_col}">{'BULLISH' if india_ok else 'CAUTION'}</strong></div>
    <div class="pill"><div class="dot" style="background:#40c4ff"></div>Nifty 50:&nbsp;<strong style="color:#40c4ff">{'+' if india_chg>=0 else ''}{india_chg:.2f}%</strong></div>
    <div class="pill"><div class="dot" style="background:{'#00e676' if india_vix<20 else '#ff5252'}"></div>India VIX:&nbsp;<strong style="color:{'#00e676' if india_vix<20 else '#ff5252'}">{india_vix:.2f}</strong></div>
    <div class="pill"><div class="dot" style="background:#7d8590"></div>Scanned:&nbsp;<strong style="color:#e6edf3">{len(india_full)}</strong></div>
    <div class="pill"><div class="dot" style="background:#7d8590"></div>Candidates:&nbsp;<strong style="color:#e6edf3">{len(india_top)}</strong></div>
  </div>

  <!-- USA status pills -->
  <div class="market-pills" id="pills-usa">
    <div class="pill"><div class="dot live" style="background:{usa_m_col}"></div>Market:&nbsp;<strong style="color:{usa_m_col}">{'BULLISH' if usa_ok else 'CAUTION'}</strong></div>
    <div class="pill"><div class="dot" style="background:#40c4ff"></div>S&amp;P 500:&nbsp;<strong style="color:#40c4ff">{'+' if usa_chg>=0 else ''}{usa_chg:.2f}%</strong></div>
    <div class="pill"><div class="dot" style="background:{'#00e676' if usa_vix<20 else '#ff5252'}"></div>CBOE VIX:&nbsp;<strong style="color:{'#00e676' if usa_vix<20 else '#ff5252'}">{usa_vix:.2f}</strong></div>
    <div class="pill"><div class="dot" style="background:#7d8590"></div>Scanned:&nbsp;<strong style="color:#e6edf3">{len(usa_full)}</strong></div>
    <div class="pill"><div class="dot" style="background:#7d8590"></div>Candidates:&nbsp;<strong style="color:#e6edf3">{len(usa_top)}</strong></div>
  </div>

  <!-- ORB status pills -->
  <div class="market-pills" id="pills-orb">
    <div class="pill"><div class="dot live" style="background:#f9a825"></div>Mode:&nbsp;<strong style="color:#f9a825">INTRADAY</strong></div>
    <div class="pill"><div class="dot" style="background:#f9a825"></div>Strategy:&nbsp;<strong style="color:#e6edf3">Opening Range Breakout</strong></div>
    <div class="pill"><div class="dot" style="background:#7d8590"></div>Interval:&nbsp;<strong style="color:#e6edf3">5-min bars</strong></div>
    <div class="pill"><div class="dot" style="background:#7d8590"></div>ORB Window:&nbsp;<strong style="color:#e6edf3">First 15 min</strong></div>
    <div class="pill"><div class="dot" style="background:#7d8590"></div>🇮🇳 Breakouts:&nbsp;<strong style="color:#e6edf3">{orb_india_count}</strong></div>
    <div class="pill"><div class="dot" style="background:#7d8590"></div>🇺🇸 Breakouts:&nbsp;<strong style="color:#e6edf3">{orb_usa_count}</strong></div>
  </div>
</div>

<!-- CONTENT -->
<div class="content">

  <!-- TOGGLE -->
  <div class="toggle-wrap">
    <button class="tab-btn india-btn active" onclick="switchTab('india')">
      <span class="flag">🇮🇳</span> India &nbsp;<span style="font-size:.65rem;opacity:.6">NIFTY 100</span>
    </button>
    <button class="tab-btn usa-btn" onclick="switchTab('usa')">
      <span class="flag">🇺🇸</span> USA &nbsp;<span style="font-size:.65rem;opacity:.6">S&amp;P 500 TOP 100</span>
    </button>
    <button class="tab-btn orb-btn" onclick="switchTab('orb')">
      <span class="flag">📊</span> ORB &nbsp;<span style="font-size:.65rem;opacity:.6">INTRADAY 5-MIN</span>
    </button>
  </div>

  <!-- INDIA PANEL -->
  <div class="tab-panel active" id="panel-india">
    {india_cards}
    <div class="sh">
      <div class="sh-title">🎯 Top BTST Candidates — India</div>
      <div class="sh-line"></div>
      <div class="sh-sub">Entry window: 3:00–3:20 PM IST</div>
    </div>
    <p class="scroll-hint">← swipe to see all columns</p>
    <div class="tw">
      <table>
        <thead><tr><th>#</th><th>Symbol</th><th>Close (₹)</th><th>Change</th><th>Vol Ratio</th><th>RSI</th><th>ADX</th><th>Range Pos</th><th>52W High</th><th>Gap</th><th>Candle</th><th>RS</th><th>Weekly</th><th>Sector</th><th>Stop Loss</th><th>Target</th><th>R:R</th><th>BTST Score</th></tr></thead>
        <tbody>{india_rows}</tbody>
      </table>
    </div>
    {_legend()}
  </div>

  <!-- USA PANEL -->
  <div class="tab-panel" id="panel-usa">
    {usa_cards}
    <div class="sh">
      <div class="sh-title">🎯 Top BTST Candidates — USA</div>
      <div class="sh-line"></div>
      <div class="sh-sub">Entry window: 3:30–4:00 PM EST</div>
    </div>
    <p class="scroll-hint">← swipe to see all columns</p>
    <div class="tw">
      <table>
        <thead><tr><th>#</th><th>Symbol</th><th>Close ($)</th><th>Change</th><th>Vol Ratio</th><th>RSI</th><th>ADX</th><th>Range Pos</th><th>52W High</th><th>Gap</th><th>Candle</th><th>RS</th><th>Weekly</th><th>Sector</th><th>Stop Loss</th><th>Target</th><th>R:R</th><th>BTST Score</th></tr></thead>
        <tbody>{usa_rows}</tbody>
      </table>
    </div>
    {_legend()}
  </div>

  <!-- ORB PANEL -->
  <style>
    .tab-btn.active.orb-btn{{color:#f9a825}}
    .orb-sub{{margin:20px 0 10px;padding:10px 14px;background:rgba(249,168,37,.05);border:1px solid rgba(249,168,37,.18);border-radius:8px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}}
    .orb-sub-title{{font-family:var(--mono);font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#f9a825}}
    .orb-chip{{font-family:var(--mono);font-size:.65rem;padding:2px 8px;border-radius:4px;background:rgba(249,168,37,.12);border:1px solid rgba(249,168,37,.25);color:#f9a825}}
    .orb-info{{font-size:.72rem;color:var(--muted);line-height:1.5;margin-top:8px;padding:9px 13px;background:var(--surf);border:1px solid var(--border);border-radius:7px}}
    .orb-info strong{{color:var(--text)}}
  </style>

  <div class="tab-panel" id="panel-orb">

    <!-- ORB explainer banner -->
    <div class="orb-info">
      <strong>⚡ Opening Range Breakout (ORB) — Intraday Strategy</strong><br>
      Opening Range = first 15 min of trading (3 × 5-min bars).
      &nbsp;·&nbsp; <strong>Buy</strong> when price breaks above ORB High with volume confirmation.
      &nbsp;·&nbsp; <strong>Target</strong> = ORB High + 1.5 × ORB Range.
      &nbsp;·&nbsp; <strong>Stop Loss</strong> = ORB Low (or price − ATR, whichever is tighter).
      &nbsp;·&nbsp; Entry window: <strong>9:30–10:30 AM IST / EST</strong> &nbsp;·&nbsp; Exit by: <strong>3:00 PM IST / 3:30 PM EST</strong>.
    </div>

    <!-- India ORB -->
    <div class="orb-sub" style="margin-top:16px">
      <span class="orb-sub-title">🇮🇳 India ORB</span>
      <span class="orb-chip">Nifty 100 · 5-min</span>
      <span class="orb-chip">{orb_india_count} breakout(s)</span>
      <span style="font-family:var(--mono);font-size:.62rem;color:var(--muted);margin-left:auto">ORB Window: 9:15–9:30 AM IST</span>
    </div>
    <p class="scroll-hint">← swipe to see all columns</p>
    <div class="tw">
      <table>
        <thead><tr>
          <th>#</th><th>Symbol</th><th>Price (₹)</th>
          <th>ORB High</th><th>ORB Low</th><th>ORB Range%</th>
          <th>Brk Above</th><th>Vol Ratio</th><th>RSI 5m</th><th>ADX 5m</th>
          <th>Sector</th><th>Stop Loss</th><th>Target</th><th>R:R</th><th>ORB Score</th>
        </tr></thead>
        <tbody>{orb_india_rows}</tbody>
      </table>
    </div>

    <!-- USA ORB -->
    <div class="orb-sub" style="margin-top:24px">
      <span class="orb-sub-title">🇺🇸 USA ORB</span>
      <span class="orb-chip">S&amp;P 500 · 5-min</span>
      <span class="orb-chip">{orb_usa_count} breakout(s)</span>
      <span style="font-family:var(--mono);font-size:.62rem;color:var(--muted);margin-left:auto">ORB Window: 9:30–9:45 AM EST</span>
    </div>
    <p class="scroll-hint">← swipe to see all columns</p>
    <div class="tw">
      <table>
        <thead><tr>
          <th>#</th><th>Symbol</th><th>Price ($)</th>
          <th>ORB High</th><th>ORB Low</th><th>ORB Range%</th>
          <th>Brk Above</th><th>Vol Ratio</th><th>RSI 5m</th><th>ADX 5m</th>
          <th>Sector</th><th>Stop Loss</th><th>Target</th><th>R:R</th><th>ORB Score</th>
        </tr></thead>
        <tbody>{orb_usa_rows}</tbody>
      </table>
    </div>

    <!-- ORB legend -->
    <div class="legend" style="margin-top:16px">
      <div class="li"><div class="ld" style="background:#f9a825"></div>ORB Score ≥60 — Strong breakout signal</div>
      <div class="li"><div class="ld" style="background:var(--yellow)"></div>Score 40–59 — Moderate breakout</div>
      <div class="li"><div class="ld" style="background:var(--red)"></div>Score &lt;40 — Weak / avoid</div>
      <div class="li"><div class="ld" style="background:var(--green)"></div>ORB High — Resistance turned support on breakout</div>
      <div class="li"><div class="ld" style="background:var(--red)"></div>ORB Low — Invalidation / stop level</div>
      <div class="li"><div class="ld" style="background:var(--green)"></div>Brk Above — % price is above ORB High</div>
      <div class="li"><div class="ld" style="background:var(--blue)"></div>Vol Ratio ≥2× = Strong institutional buying</div>
      <div class="li"><div class="ld" style="background:var(--muted)"></div>ADX 5m &gt;25 = Intraday trend confirmed</div>
      <div class="li"><div class="ld" style="background:var(--muted)"></div>Tight ORB Range (&lt;1%) = Cleaner, more reliable breakout</div>
    </div>

  </div>

  <div class="sh" style="margin-top:36px">
    <div class="sh-title">⚙️ Scoring Parameters</div>
    <div class="sh-line"></div>
    <div class="sh-sub">Max score ≈ 138 pts · base 100 + sector (7) + candle (10) + RS (5) + weekly MTF (8) + gap-up (8)</div>
  </div>

  <!-- ── Compact 4-col scoring params ── -->
  <style>
    .sp-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:16px}}
    @media(max-width:900px){{.sp-grid{{grid-template-columns:repeat(2,1fr)}}}}
    @media(max-width:480px){{.sp-grid{{grid-template-columns:1fr}}}}
    .sp-section{{grid-column:1/-1;font-family:var(--mono);font-size:.58rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);padding:2px 0 5px;border-bottom:1px solid var(--border);margin-top:6px}}
    .sp-card{{background:var(--surf);border:1px solid var(--border);border-radius:8px;padding:9px 11px;position:relative;overflow:hidden;transition:border-color .15s}}
    .sp-card:hover{{border-color:#2a3245}}
    .sp-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--sp-accent,transparent);opacity:.75}}
    .sp-card.sg{{--sp-accent:var(--green)}} .sp-card.sr{{--sp-accent:var(--red)}} .sp-card.sc{{--sp-accent:var(--blue)}} .sp-card.sy{{--sp-accent:var(--yellow)}}
    .sp-top{{display:flex;align-items:flex-start;justify-content:space-between;gap:5px;margin-bottom:3px}}
    .sp-name{{font-family:var(--mono);font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#dde6f8}}
    .sp-badge{{font-family:var(--mono);font-size:.68rem;font-weight:700;padding:1px 6px;border-radius:4px;white-space:nowrap;flex-shrink:0}}
    .sp-badge.sg{{color:var(--green);background:rgba(0,230,118,.12)}} .sp-badge.sr{{color:var(--red);background:rgba(255,82,82,.12)}}
    .sp-badge.sc{{color:var(--blue);background:rgba(64,196,255,.12)}} .sp-badge.sy{{color:var(--yellow);background:rgba(255,202,40,.1)}}
    .sp-desc{{font-size:.72rem;color:var(--muted);line-height:1.4;margin-bottom:4px}}
    .sp-tags{{display:flex;flex-wrap:wrap;gap:3px}}
    .sp-tag{{font-family:var(--mono);font-size:.58rem;padding:1px 5px;border-radius:3px;background:var(--surf2);border:1px solid var(--border);color:var(--text);white-space:nowrap}}
    .sp-tag.tg{{color:var(--green);border-color:rgba(0,230,118,.25);background:rgba(0,230,118,.06)}}
    .sp-tag.tr{{color:var(--red);border-color:rgba(255,82,82,.25);background:rgba(255,82,82,.06)}}
    .sp-tag.tc{{color:var(--blue);border-color:rgba(64,196,255,.2);background:rgba(64,196,255,.06)}}
    .sp-formula{{font-family:var(--mono);font-size:.62rem;color:var(--blue);background:rgba(64,196,255,.1);border:1px solid rgba(64,196,255,.2);padding:2px 7px;border-radius:4px;display:inline-block;margin-top:4px}}
    .sp-footer{{margin-top:10px;padding:9px 13px;background:var(--surf);border:1px solid var(--border);border-radius:8px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}}
    .sp-footer-lbl{{font-family:var(--mono);font-size:.58rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-right:4px}}
    .sp-chip{{font-family:var(--mono);font-size:.65rem;font-weight:600;padding:2px 8px;border-radius:5px;border:1px solid}}
    .sp-chip.sg{{color:var(--green);background:rgba(0,230,118,.1);border-color:rgba(0,230,118,.2)}}
    .sp-chip.sr{{color:var(--red);background:rgba(255,82,82,.1);border-color:rgba(255,82,82,.2)}}
    .sp-chip.sc{{color:var(--blue);background:rgba(64,196,255,.1);border-color:rgba(64,196,255,.2)}}
    .sp-sep{{color:var(--border);margin:0 1px}}
  </style>

  <div class="sp-grid">

    <!-- MOMENTUM & TREND -->
    <div class="sp-section">📈 Momentum &amp; Trend</div>

    <div class="sp-card sg">
      <div class="sp-top"><span class="sp-name">Volume Surge</span><span class="sp-badge sg">20 PTS</span></div>
      <div class="sp-desc">Confirms institutional participation.</div>
      <div class="sp-tags"><span class="sp-tag tg">Vol &gt;1.5× 10-day avg</span></div>
    </div>

    <div class="sp-card sg">
      <div class="sp-top"><span class="sp-name">EMA Alignment</span><span class="sp-badge sg">15 PTS</span></div>
      <div class="sp-desc">Confirms bullish structure.</div>
      <div class="sp-tags"><span class="sp-tag tg">Above 20 EMA</span><span class="sp-tag tg">Above 50 EMA</span></div>
    </div>

    <div class="sp-card sg">
      <div class="sp-top"><span class="sp-name">RSI Zone</span><span class="sp-badge sg">15 PTS</span></div>
      <div class="sp-desc">Strong momentum without being overbought.</div>
      <div class="sp-tags"><span class="sp-tag tg">RSI 55 – 75</span></div>
    </div>

    <div class="sp-card sg">
      <div class="sp-top"><span class="sp-name">MACD Signal</span><span class="sp-badge sg">15 PTS</span></div>
      <div class="sp-desc">Signals continuation.</div>
      <div class="sp-tags"><span class="sp-tag tg">+ve histogram</span><span class="sp-tag tg">Fresh bull crossover</span></div>
    </div>

    <div class="sp-card sg">
      <div class="sp-top"><span class="sp-name">ADX Trend</span><span class="sp-badge sg">10 PTS</span></div>
      <div class="sp-desc">Reduces overnight whipsaw risk.</div>
      <div class="sp-tags"><span class="sp-tag tg">ADX &gt; 25</span></div>
    </div>

    <div class="sp-card sg">
      <div class="sp-top"><span class="sp-name">52-Week High</span><span class="sp-badge sg">10 PTS</span></div>
      <div class="sp-desc">Proximity to breakout zone.</div>
      <div class="sp-tags"><span class="sp-tag tg">Within 5% → Full</span><span class="sp-tag">Within 10% → Half</span></div>
    </div>

    <!-- PRICE ACTION -->
    <div class="sp-section">🕯 Price Action &amp; Structure</div>

    <div class="sp-card sg">
      <div class="sp-top"><span class="sp-name">Price Breakout</span><span class="sp-badge sg">15 PTS</span></div>
      <div class="sp-desc">Buyer dominance at close.</div>
      <div class="sp-tags"><span class="sp-tag tg">Close in top 5–10% of range</span></div>
    </div>

    <div class="sp-card sg">
      <div class="sp-top"><span class="sp-name">Candlestick Pattern</span><span class="sp-badge sg">+6–10 PTS</span></div>
      <div class="sp-desc">High-confidence reversal/continuation candles.</div>
      <div class="sp-tags"><span class="sp-tag tg">M-Star +10</span><span class="sp-tag tg">Engulfing +8</span><span class="sp-tag tg">Hammer +6</span></div>
    </div>

    <div class="sp-card sg">
      <div class="sp-top"><span class="sp-name">Gap-Up &amp; Hold</span><span class="sp-badge sg">+5–8 PTS</span></div>
      <div class="sp-desc">Open gapped above prior close and held.</div>
      <div class="sp-tags"><span class="sp-tag tg">≥1% gap + pos ≥60% → +8</span><span class="sp-tag">≥0.5% held → +5</span></div>
    </div>

    <div class="sp-card sg">
      <div class="sp-top"><span class="sp-name">Relative Strength</span><span class="sp-badge sg">+5 PTS</span></div>
      <div class="sp-desc">Outperformance vs the broader index.</div>
      <div class="sp-tags"><span class="sp-tag tg">Daily gain &gt; Index</span></div>
    </div>

    <!-- CONFIRMATION -->
    <div class="sp-section">🔗 Confirmation &amp; Alignment</div>

    <div class="sp-card sy">
      <div class="sp-top"><span class="sp-name">Sector Alignment</span><span class="sp-badge sy">+7 PTS</span></div>
      <div class="sp-desc">Bonus when sector index is also green on the day.</div>
      <div class="sp-tags"><span class="sp-tag">e.g. Nifty Bank, XLK</span></div>
    </div>

    <div class="sp-card sy">
      <div class="sp-top"><span class="sp-name">Weekly MTF Confirm</span><span class="sp-badge sy">+8 PTS</span></div>
      <div class="sp-desc">Weekly &amp; daily trends aligned. Reduces reversal risk.</div>
      <div class="sp-tags"><span class="sp-tag">Daily close &gt; Weekly EMA20</span></div>
    </div>

    <!-- RISK -->
    <div class="sp-section">⚠ Risk &amp; Dynamic Levels</div>

    <div class="sp-card sr">
      <div class="sp-top"><span class="sp-name">ATR Penalty</span><span class="sp-badge sr">−40%</span></div>
      <div class="sp-desc">Avoids chasing overextended stocks. Triggered if day's move exceeds 1.5× ATR.</div>
      <div class="sp-tags"><span class="sp-tag tr">Move &gt; 1.5× ATR → penalty</span></div>
    </div>

    <div class="sp-card sc">
      <div class="sp-top"><span class="sp-name">Stop Loss</span><span class="sp-badge sc">DYNAMIC</span></div>
      <div class="sp-desc">Limits overnight gap risk. Tighter of the two values.</div>
      <div class="sp-formula">max( Today's Low,  Close − 1×ATR )</div>
    </div>

    <div class="sp-card sc">
      <div class="sp-top"><span class="sp-name">Target</span><span class="sp-badge sc">DYNAMIC</span></div>
      <div class="sp-desc">Respects each stock's typical daily volatility range.</div>
      <div class="sp-formula">Close + 1.5× ATR</div>
    </div>

  </div>

  <!-- Score breakdown footer -->
  <div class="sp-footer">
    <span class="sp-footer-lbl">Score Breakdown</span>
    <span class="sp-chip sg">Base 100</span><span class="sp-sep">+</span>
    <span class="sp-chip sg">Sector +7</span><span class="sp-sep">+</span>
    <span class="sp-chip sg">Candle +10</span><span class="sp-sep">+</span>
    <span class="sp-chip sg">RS +5</span><span class="sp-sep">+</span>
    <span class="sp-chip sg">Weekly MTF +8</span><span class="sp-sep">+</span>
    <span class="sp-chip sg">Gap-Up +8</span><span class="sp-sep">=</span>
    <span class="sp-chip sg" style="font-size:.72rem;padding:3px 10px">Max 138 PTS</span>
    <span class="sp-chip sr" style="margin-left:auto">ATR Penalty −40%</span>
    <span class="sp-chip sc">SL: Dynamic</span>
    <span class="sp-chip sc">Target: Dynamic</span>
  </div>

  <!-- DISCLAIMER -->
  <div class="disc">
    <strong>⚠ Disclaimer:</strong> This report is for educational and research purposes only.
    It does <strong>not</strong> constitute financial advice or any recommendation to buy or sell securities.
    Equity trading involves significant risk. Past patterns do not guarantee future performance.
    For Indian markets, consult a <strong>SEBI-registered advisor</strong>.
    For US markets, consult a <strong>FINRA/SEC-registered advisor</strong> before placing any trades.
  </div>
</div>

<!-- FOOTER -->
<div class="footer">
  Generated by BTST Screener · Python + yfinance + pandas-ta
  &nbsp;|&nbsp; India: {time_ist} &nbsp;·&nbsp; USA: {time_est}
  <br>
  Parameters: Vol Surge (20) · RSI (15) · MACD (15) · EMA (15) · Breakout (15) · ADX (10) · 52W (10) · Gap (+8) · Sector (+7) · Candle (+10) · RS (+5) · Weekly MTF (+8)
</div>

<script>
  // Timestamps per BTST tab; ORB shows both market times
  const times = {{
    india: {{ time: "{now_ist.strftime('%I:%M %p')}", tz: "IST", date: "{now_ist.strftime('%d %b %Y')}" }},
    usa:   {{ time: "{now_est.strftime('%I:%M %p')}", tz: "EST", date: "{now_est.strftime('%d %b %Y')}" }},
    orb:   {{ time: "{now_ist.strftime('%I:%M %p')}", tz: "IST/EST", date: "{now_ist.strftime('%d %b %Y')}" }}
  }};

  function updateTime(tab) {{
    const t = times[tab];
    document.getElementById('hdr-time').innerHTML =
      t.time + ' <span class="tz-tag">' + t.tz + '</span>';
    document.getElementById('hdr-date').textContent = t.date;
  }}

  function switchTab(tab) {{
    // panels
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.getElementById('panel-' + tab).classList.add('active');
    // buttons
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('.' + tab + '-btn').classList.add('active');
    // pills — only show pills row that exists for this tab
    document.querySelectorAll('.market-pills').forEach(p => p.classList.remove('active'));
    const pillEl = document.getElementById('pills-' + tab);
    if (pillEl) pillEl.classList.add('active');
    // time
    updateTime(tab);
  }}

  // init
  updateTime('india');
</script>
</body>
</html>"""

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  🌐  HTML Report saved → {Fore.CYAN}{html_file}{Style.RESET_ALL}")

    # Also save as index.html → GitHub Pages serves it at the root URL
    # e.g. https://krishnateja08.github.io/BTST-Screener/ (no filename needed)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  🔗  index.html updated → root URL now serves latest report")

    return html_file


def _legend() -> str:
    return """
    <div class="legend">
      <div class="li"><div class="ld" style="background:var(--green)"></div>Score ≥70 — Strong Signal</div>
      <div class="li"><div class="ld" style="background:var(--yellow)"></div>Score 50–69 — Moderate</div>
      <div class="li"><div class="ld" style="background:var(--red)"></div>Score &lt;50 — Avoid</div>
      <div class="li"><div class="ld" style="background:var(--blue)"></div>Vol Ratio &gt;1.5× = Conviction buying</div>
      <div class="li"><div class="ld" style="background:var(--muted)"></div>RSI 55–75 = Ideal momentum zone</div>
      <div class="li"><div class="ld" style="background:var(--muted)"></div>ADX &gt;25 = Confirmed trend</div>
      <div class="li"><div class="ld" style="background:var(--muted)"></div>Range Pos &gt;90% = Closing near high</div>
      <div class="li"><div class="ld" style="background:var(--green)"></div>⬆ = Gap-up open that held by close</div>
      <div class="li"><div class="ld" style="background:var(--green)"></div>⭐/🕯/🔨 = Candle pattern detected</div>
      <div class="li"><div class="ld" style="background:var(--green)"></div>RS+ = Stock beat index today</div>
      <div class="li"><div class="ld" style="background:var(--green)"></div>✅ W = Above weekly EMA20</div>
      <div class="li"><div class="ld" style="background:var(--muted)"></div>▲/▼ in score = vs previous session</div>
    </div>"""


# ══════════════════════════════════════════════════════════
# BACKTEST — replay past CSV picks against next-day actuals
# ══════════════════════════════════════════════════════════

def run_backtest(prefix: str, days: int = 30):
    """
    For each past btst_{prefix}_YYYY-MM-DD.csv found in the last `days` calendar days:
      - Load picks: Symbol, Close (entry), Stop_Loss, Target, BTST_Score
      - Download next trading day's OHLC via yfinance
      - Classify each pick: WIN (High ≥ Target), LOSS (Low ≤ SL or gap-down open ≤ SL),
        NEUTRAL (neither touched)
    Prints hit rate, score-stratified stats, score↔return correlation,
    top-5 wins / worst-5 losses, and saves a backtest CSV.
    """
    tz_now   = datetime.now(tz=IST if prefix == "india" else EST)
    today    = tz_now.date()
    sym_sfx  = ".NS" if prefix == "india" else ""

    hdr(f"BACKTEST — {prefix.upper()} | Scanning last {days} calendar days")

    # ── 1. Collect all available past CSVs ───────────────────
    all_rows: list[dict] = []
    dates_found: list[str] = []

    for d in range(1, days + 1):
        date      = today - timedelta(days=d)
        date_str  = date.strftime("%Y-%m-%d")
        csv_path  = f"btst_{prefix}_{date_str}.csv"
        try:
            df = pd.read_csv(csv_path)
            if df.empty or "Symbol" not in df.columns:
                continue
            for _, row in df.iterrows():
                all_rows.append({
                    "date_str": date_str,
                    "Symbol":   str(row.get("Symbol", "")),
                    "Entry":    float(row.get("Close", 0)),
                    "SL":       float(row.get("Stop_Loss", 0)),
                    "Target":   float(row.get("Target", 0)),
                    "Score":    float(row.get("BTST_Score", 0)),
                })
            dates_found.append(date_str)
        except FileNotFoundError:
            continue
        except Exception:
            continue

    if not all_rows:
        print(f"\n  {Fore.YELLOW}No past CSVs found in the last {days} days.")
        print(f"  Run the screener daily for a few sessions first, then backtest.{Style.RESET_ALL}")
        return

    print(f"  📂  Found {len(dates_found)} past session(s), "
          f"{len(all_rows)} total pick(s) to evaluate.")

    # ── 2. Batch-download price history for all unique symbols ─
    raw_syms  = list({r["Symbol"] for r in all_rows if r["Symbol"]})
    dl_syms   = [s + sym_sfx for s in raw_syms]

    print(f"  📥  Downloading 1-year history for {len(dl_syms)} symbols …", flush=True)
    cache_raw = _batch_download(dl_syms)

    # Normalise keys → clean symbol (same logic as score_stock_from_df)
    norm_cache: dict[str, pd.DataFrame] = {}
    for sym, df in cache_raw.items():
        clean = sym.replace(".NS", "").replace("-", ".").replace("BRK.B", "BRK-B")
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df_copy.index)
        norm_cache[clean] = df_copy

    print(f"  ✅  History loaded for {len(norm_cache)} symbols. Evaluating picks …")

    # ── 3. Evaluate each pick against the next trading day ────
    results: list[dict] = []

    for r in all_rows:
        sym   = r["Symbol"]
        entry = r["Entry"]
        sl    = r["SL"]
        tgt   = r["Target"]
        score = r["Score"]
        pick_date = pd.Timestamp(r["date_str"])

        if sym not in norm_cache or entry <= 0 or sl <= 0 or tgt <= 0:
            continue

        df_sym     = norm_cache[sym]
        future     = df_sym[df_sym.index > pick_date]
        if future.empty:
            continue          # no next-day data yet (picked today)

        nxt        = future.iloc[0]
        nxt_open   = float(nxt.get("Open",  nxt["Close"]))
        nxt_high   = float(nxt["High"])
        nxt_low    = float(nxt["Low"])
        nxt_close  = float(nxt["Close"])
        nxt_date   = future.index[0].strftime("%Y-%m-%d")

        # Gap-down through SL at open → instant loss
        if nxt_open <= sl:
            outcome = "LOSS"
        elif nxt_high >= tgt:
            outcome = "WIN"
        elif nxt_low <= sl:
            outcome = "LOSS"
        else:
            outcome = "NEUTRAL"

        actual_chg = (nxt_close - entry) / entry * 100 if entry > 0 else 0.0

        results.append({
            "Date":       r["date_str"],
            "Symbol":     sym,
            "Score":      round(score, 1),
            "Entry":      round(entry, 2),
            "SL":         round(sl,    2),
            "Target":     round(tgt,   2),
            "Next_Open":  round(nxt_open,  2),
            "Next_High":  round(nxt_high,  2),
            "Next_Low":   round(nxt_low,   2),
            "Next_Close": round(nxt_close, 2),
            "Actual_%":   round(actual_chg, 2),
            "Outcome":    outcome,
            "Next_Date":  nxt_date,
        })

    if not results:
        print(f"  {Fore.YELLOW}No picks could be evaluated "
              f"(next-day data unavailable — try again tomorrow).{Style.RESET_ALL}")
        return

    res_df = pd.DataFrame(results)

    # ── 4. Aggregate stats ────────────────────────────────────
    total   = len(res_df)
    wins    = (res_df["Outcome"] == "WIN").sum()
    losses  = (res_df["Outcome"] == "LOSS").sum()
    neutral = (res_df["Outcome"] == "NEUTRAL").sum()
    hit_rt  = wins / total * 100 if total else 0.0
    avg_chg = res_df["Actual_%"].mean()

    # Score-stratified: ≥70 vs <70
    hs      = res_df[res_df["Score"] >= 70]
    hs_hits = (hs["Outcome"] == "WIN").sum()
    hs_rt   = hs_hits / len(hs) * 100 if len(hs) else 0.0

    ls      = res_df[res_df["Score"] < 70]
    ls_hits = (ls["Outcome"] == "WIN").sum()
    ls_rt   = ls_hits / len(ls) * 100 if len(ls) else 0.0

    # Score ↔ return correlation
    corr = (res_df[["Score", "Actual_%"]].corr().iloc[0, 1]
            if len(res_df) >= 5 else float("nan"))

    hdr(f"BACKTEST RESULTS — {prefix.upper()}")
    print(f"  Sessions analysed  : {len(dates_found)}  ({dates_found[-1]} → {dates_found[0]})")
    print(f"  Total picks        : {total}")
    print()

    w_col = Fore.GREEN if hit_rt >= 55 else Fore.YELLOW if hit_rt >= 45 else Fore.RED
    print(f"  {'Wins  (Target hit)':<22}: {Fore.GREEN}{wins:>4}{Style.RESET_ALL}")
    print(f"  {'Losses (SL hit)':<22}: {Fore.RED}{losses:>4}{Style.RESET_ALL}")
    print(f"  {'Neutral (neither)':<22}: {Fore.YELLOW}{neutral:>4}{Style.RESET_ALL}")
    print(f"  {'Overall Hit Rate':<22}: {w_col}{hit_rt:>6.1f}%{Style.RESET_ALL}")
    c_col = Fore.GREEN if avg_chg > 0 else Fore.RED
    print(f"  {'Avg Next-Day Chg':<22}: {c_col}{avg_chg:>+6.2f}%{Style.RESET_ALL}")
    print()

    # Score-stratified table
    strat_rows = [
        ["Score ≥ 70", len(hs), hs_hits, f"{hs_rt:.1f}%",
         f"{hs['Actual_%'].mean():+.2f}%" if len(hs) else "—"],
        ["Score < 70", len(ls), ls_hits, f"{ls_rt:.1f}%",
         f"{ls['Actual_%'].mean():+.2f}%" if len(ls) else "—"],
    ]
    print(tabulate(strat_rows,
                   headers=["Tier", "Picks", "Wins", "Hit Rate", "Avg Chg"],
                   tablefmt="simple"))
    print()

    if not pd.isna(corr):
        corr_col = Fore.GREEN if corr > 0.15 else Fore.YELLOW if corr > 0 else Fore.RED
        corr_lbl = ("✅ Score predicts returns"  if corr > 0.15 else
                    "↔ Weak positive link"       if corr > 0    else
                    "⚠ Score not yet predictive")
        print(f"  Score ↔ Return corr : {corr_col}{corr:+.3f}  {corr_lbl}{Style.RESET_ALL}")
        print()

    # Top-5 wins
    top_wins = (res_df[res_df["Outcome"] == "WIN"]
                .nlargest(5, "Actual_%")
                [["Date", "Symbol", "Score", "Entry", "Target", "Actual_%"]]
                .reset_index(drop=True))
    if not top_wins.empty:
        print(f"  {Fore.GREEN}── Top 5 Winning Picks ──{Style.RESET_ALL}")
        print(tabulate(top_wins,
                       headers=["Date", "Symbol", "Score", "Entry", "Target", "Actual %"],
                       tablefmt="simple", floatfmt=".2f"))
        print()

    # Worst-5 losses
    worst = (res_df[res_df["Outcome"] == "LOSS"]
             .nsmallest(5, "Actual_%")
             [["Date", "Symbol", "Score", "Entry", "SL", "Actual_%"]]
             .reset_index(drop=True))
    if not worst.empty:
        print(f"  {Fore.RED}── Worst 5 Losses ──{Style.RESET_ALL}")
        print(tabulate(worst,
                       headers=["Date", "Symbol", "Score", "Entry", "SL", "Actual %"],
                       tablefmt="simple", floatfmt=".2f"))
        print()

    # Save backtest CSV
    out_path = f"btst_{prefix}_backtest_{today}.csv"
    res_df.to_csv(out_path, index=False)
    print(f"  💾  Full backtest results → {Fore.CYAN}{out_path}{Style.RESET_ALL}")


# ══════════════════════════════════════════════════════════
# DISCLAIMER
# ══════════════════════════════════════════════════════════

def print_disclaimer():
    print(f"\n{Fore.YELLOW}{'─'*62}")
    print("  ⚠  DISCLAIMER: Educational/research purposes only.")
    print("  Not financial advice. Always do your own due diligence.")
    print(f"{'─'*62}{Style.RESET_ALL}\n")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def _scan_india(date_str: str, run_orb: bool = True):
    """Run India BTST scan (+ optional ORB scan). Returns (ok, chg, vix, top_df, full_df, orb_df)."""
    ok, chg, vix = check_market("india")
    full, sector_perf = run_screener(NIFTY100_SYMBOLS, "Nifty 100", chg)
    top  = pd.DataFrame()
    if not full.empty:
        top = filter_and_rank(full)
        print_report(top, "INDIA", ok, chg)
        save_csv(top, full, "india", date_str)
        save_meta("india", date_str, ok, chg, vix)
    # ORB scan — reuse sector_perf from BTST scan (no duplicate fetch)
    orb_top = pd.DataFrame()
    if run_orb:
        orb_raw = run_orb_screener(NIFTY100_SYMBOLS, "Nifty 100", sector_perf=sector_perf)
        orb_top = filter_and_rank_orb(orb_raw)
        if not orb_top.empty:
            orb_top.to_csv(f"orb_india_{date_str}.csv", index=False)
            print(f"  💾  ORB India CSV saved → orb_india_{date_str}.csv")
    return ok, chg, vix, top, full, orb_top


def _scan_usa(date_str: str, run_orb: bool = True):
    """Run USA BTST scan (+ optional ORB scan). Returns (ok, chg, vix, top_df, full_df, orb_df)."""
    ok, chg, vix = check_market("usa")
    full, sector_perf = run_screener(SP500_TOP100_SYMBOLS, "S&P 500 Top 100", chg)
    top  = pd.DataFrame()
    if not full.empty:
        top = filter_and_rank(full)
        print_report(top, "USA", ok, chg)
        save_csv(top, full, "usa", date_str)
        save_meta("usa", date_str, ok, chg, vix)
    # ORB scan — reuse sector_perf from BTST scan (no duplicate fetch)
    orb_top = pd.DataFrame()
    if run_orb:
        orb_raw = run_orb_screener(SP500_TOP100_SYMBOLS, "S&P 500 Top 100", sector_perf=sector_perf)
        orb_top = filter_and_rank_orb(orb_raw)
        if not orb_top.empty:
            orb_top.to_csv(f"orb_usa_{date_str}.csv", index=False)
            print(f"  💾  ORB USA CSV saved → orb_usa_{date_str}.csv")
    return ok, chg, vix, top, full, orb_top


def main():
    parser = argparse.ArgumentParser(description="BTST Screener — India + USA")
    parser.add_argument("--india",     action="store_true", help="Scan India only")
    parser.add_argument("--usa",       action="store_true", help="Scan USA only")
    parser.add_argument("--no-orb",    action="store_true", help="Skip ORB intraday scan")
    parser.add_argument("--html-only", action="store_true",
                        help="Skip scanning — read existing CSVs and regenerate HTML report")
    parser.add_argument("--backtest",  action="store_true",
                        help="Replay past CSV picks against next-day actuals")
    parser.add_argument("--days",      type=int, default=30,
                        help="Calendar days to look back for backtest (default: 30)")
    args      = parser.parse_args()
    run_india = not args.usa   or args.india
    run_usa   = not args.india or args.usa
    do_orb    = not args.no_orb   # True by default; False when --no-orb passed

    print(f"\n{Fore.CYAN}{'='*62}")
    print("   BTST SCREENER  |  India (Nifty 100)  +  USA (S&P 500 Top 100)")
    print(f"{'='*62}{Style.RESET_ALL}")

    IST_NOW  = datetime.now(tz=IST)
    date_str = IST_NOW.strftime("%Y-%m-%d")

    # ── BACKTEST MODE ──────────────────────────────────────
    if args.backtest:
        if run_india:
            run_backtest("india", args.days)
        if run_usa:
            run_backtest("usa", args.days)
        print_disclaimer()
        return

    # ── HTML-ONLY MODE (used by CI commit job) ─────────────
    if args.html_only:
        hdr("HTML-ONLY MODE — reading saved CSVs + metadata")

        def _load_csv(prefix):
            top_path  = f"btst_{prefix}_{date_str}.csv"
            full_path = f"btst_{prefix}_full_{date_str}.csv"
            try:
                top  = pd.read_csv(top_path)
                full = pd.read_csv(full_path)
                print(f"  ✅  Loaded {prefix.upper()} CSVs ({len(top)} top / {len(full)} full)")
                return top, full
            except FileNotFoundError:
                print(f"  ⚠️   {prefix.upper()} CSVs not found for {date_str} — using empty")
                return pd.DataFrame(), pd.DataFrame()

        def _load_orb_csv(prefix):
            try:
                df = pd.read_csv(f"orb_{prefix}_{date_str}.csv")
                print(f"  ✅  Loaded ORB {prefix.upper()} CSV ({len(df)} picks)")
                return df
            except FileNotFoundError:
                return pd.DataFrame()

        india_top,  india_full = _load_csv("india")
        usa_top,    usa_full   = _load_csv("usa")
        india_ok,   india_chg, india_vix = load_meta("india", date_str)
        usa_ok,     usa_chg,   usa_vix   = load_meta("usa",   date_str)
        orb_india = _load_orb_csv("india")
        orb_usa   = _load_orb_csv("usa")

        generate_html_report(
            india_top, india_full, india_ok, india_chg, india_vix,
            usa_top,   usa_full,   usa_ok,   usa_chg,   usa_vix,
            date_str,
            orb_india_df=orb_india,
            orb_usa_df=orb_usa,
        )
        print_disclaimer()
        return

    # ── LIVE SCAN MODE ─────────────────────────────────────
    india_ok, india_chg, india_vix = True, 0.0, 0.0
    india_full = india_top = pd.DataFrame()
    usa_ok,    usa_chg,   usa_vix   = True, 0.0, 0.0
    usa_full   = usa_top  = pd.DataFrame()
    orb_india  = orb_usa  = pd.DataFrame()

    if run_india and run_usa:
        hdr("Running India + USA scans in PARALLEL")
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_india = pool.submit(_scan_india, date_str, do_orb)
            f_usa   = pool.submit(_scan_usa,   date_str, do_orb)
            india_ok, india_chg, india_vix, india_top, india_full, orb_india = f_india.result()
            usa_ok,   usa_chg,   usa_vix,   usa_top,   usa_full,   orb_usa   = f_usa.result()

    elif run_india:
        india_ok, india_chg, india_vix, india_top, india_full, orb_india = _scan_india(date_str, do_orb)

    elif run_usa:
        usa_ok, usa_chg, usa_vix, usa_top, usa_full, orb_usa = _scan_usa(date_str, do_orb)

    # ── HTML ───────────────────────────────────────────────
    if run_india and run_usa:
        generate_html_report(
            india_top, india_full, india_ok, india_chg, india_vix,
            usa_top,   usa_full,   usa_ok,   usa_chg,   usa_vix,
            date_str,
            orb_india_df=orb_india,
            orb_usa_df=orb_usa,
        )

    print_disclaimer()


if __name__ == "__main__":
    main()
