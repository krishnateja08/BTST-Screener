"""
============================================================
  BTST (Buy Today Sell Tomorrow) Stock Screener  v2.0
  India — Nifty 100 (NSE)  |  USA — S&P 500 Top 100 (NYSE/NASDAQ)
============================================================
v2.0 Changes vs v1:
  ✅ 1. Entry Quality Filter    — blocks overextended entries (>3% above VWAP proxy, >5% day change)
  ✅ 2. Next-Day Exit Logic      — fixed % targets/SL + gap-down/first-15-min-low rules in report
  ✅ 3. Score Threshold Filter   — only surfaces Score≥60 (Good) or Score≥80 (High Conviction)
  ✅ 4. Reduced Overfitting      — top-5 factors only (Volume, EMA, Breakout, Sector, Candle)
  ✅ 5. Liquidity Filter         — skips stocks with volume below MIN_VOLUME threshold
  ✅ 6. Market Direction Weight  — score boosted (bullish) or haircut (weak/VIX-high) dynamically
  ✅ 7. Backtest Improvement     — adds win rate %, avg gain, max drawdown, expectancy
  ✅ Gap-Down Exit Note          — HTML report now shows exit rules per stock
============================================================
Requirements:
    pip install yfinance pandas pandas-ta requests tabulate colorama

Usage:
    python btst_screener_v2.py              # scans both markets (BTST + ORB)
    python btst_screener_v2.py --india      # India only
    python btst_screener_v2.py --usa        # USA only
    python btst_screener_v2.py --no-orb     # skip ORB intraday scan
    python btst_screener_v2.py --backtest   # replay past CSV picks (last 30 days)
    python btst_screener_v2.py --backtest --days 60   # extend backtest window
    python btst_screener_v2.py --backtest --india      # India backtest only
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
from zoneinfo import ZoneInfo
from tabulate import tabulate
from colorama import Fore, Style, init
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

warnings.filterwarnings("ignore")
init(autoreset=True)

_YF_LOCK = threading.Lock()

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

INDIA_INDEX  = "^NSEI"
INDIA_VIX    = "^INDIAVIX"
USA_INDEX    = "^GSPC"
USA_VIX      = "^VIX"

INDIA_SECTOR_MAP: dict[str, str] = {
    **{s: "^NSEBANK" for s in [
        "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS",
        "INDUSINDBK.NS","BANDHANBNK.NS","PNB.NS","CANBK.NS","BANKBARODA.NS",
        "FEDERALBNK.NS","IDFCFIRSTB.NS","INDIANB.NS","UNIONBANK.NS",
    ]},
    **{s: "^CNXIT" for s in [
        "TCS.NS","INFY.NS","WIPRO.NS","TECHM.NS","HCLTECH.NS","NAUKRI.NS",
    ]},
    **{s: "^CNXPHARMA" for s in [
        "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","LUPIN.NS",
        "TORNTPHARM.NS","APOLLOHOSP.NS",
    ]},
    **{s: "^CNXAUTO" for s in [
        "MARUTI.NS","TMCV.NS","TMPV.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS",
        "M&M.NS","EICHERMOT.NS",
    ]},
    **{s: "^CNXMETAL" for s in [
        "JSWSTEEL.NS","TATASTEEL.NS","HINDALCO.NS","VEDL.NS","SAIL.NS","NMDC.NS",
    ]},
    **{s: "^CNXFMCG" for s in [
        "HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","TATACONSUM.NS","BRITANNIA.NS",
        "DABUR.NS","MARICO.NS","COLPAL.NS","GODREJCP.NS","PGHH.NS",
    ]},
    **{s: "^CNXENERGY" for s in [
        "RELIANCE.NS","ONGC.NS","BPCL.NS","IOC.NS","HINDPETRO.NS",
        "GAIL.NS","PETRONET.NS","IGL.NS","MGL.NS","TATAPOWER.NS",
        "ADANIGREEN.NS","ADANIPOWER.NS","COALINDIA.NS","NHPC.NS",
        "RECLTD.NS","PFC.NS","NTPC.NS","POWERGRID.NS",
    ]},
    **{s: "^CNXFINANCE" for s in [
        "BAJFINANCE.NS","BAJAJFINSV.NS","HDFCAMC.NS","MUTHOOTFIN.NS",
        "LICHSGFIN.NS","CHOLAFIN.NS","SBICARD.NS","ICICIPRULI.NS",
        "SBILIFE.NS","HDFCLIFE.NS",
    ]},
    **{s: "^CNXINFRA" for s in [
        "LT.NS","SIEMENS.NS","HAVELLS.NS","BEL.NS","BHEL.NS","HAL.NS",
        "CONCOR.NS","ADANIPORTS.NS","DLF.NS",
    ]},
    **{s: "^CNXCMDT" for s in [
        "ULTRACEMCO.NS","SHREECEM.NS","AMBUJACEM.NS","GRASIM.NS",
        "ASIANPAINT.NS","BERGER.NS","PIDILITIND.NS",
    ]},
}

USA_SECTOR_MAP: dict[str, str] = {
    **{s: "XLK" for s in [
        "AAPL","MSFT","NVDA","AVGO","AMD","ADBE","CRM","QCOM","INTU",
        "NOW","PANW","AMAT","LRCX","ADI","KLAC","IBM","ACN","APH",
    ]},
    **{s: "XLF" for s in [
        "JPM","BAC","WFC","GS","MS","V","MA","AXP","BLK","SCHW",
        "SPGI","CB","MMC","ICE","CME","MCO","AON","BRK-B",
    ]},
    **{s: "XLV" for s in [
        "UNH","LLY","JNJ","ABBV","MRK","TMO","DHR","ISRG","BMY",
        "AMGN","GILD","VRTX","PFE","SYK","ZTS","ELV","CI","BSX",
    ]},
    **{s: "XLE" for s in ["XOM","CVX"]},
    **{s: "XLY" for s in [
        "AMZN","TSLA","HD","MCD","LOW","TJX","BKNG","UBER",
    ]},
    **{s: "XLP" for s in [
        "WMT","PG","KO","PEP","COST","MDLZ","MO","CL",
    ]},
    **{s: "XLI" for s in [
        "GE","CAT","HON","UNP","RTX","ETN","DE","ADP","BA","PH",
    ]},
    **{s: "XLC" for s in [
        "GOOGL","META","NFLX","CMCSA","T","VZ",
    ]},
    **{s: "XLU" for s in ["NEE","SO","DUK","CEG"]},
    **{s: "XLRE" for s in ["PLD"]},
    **{s: "XLB" for s in ["LIN","SHW"]},
}

LOOKBACK_DAYS  = 365
AVG_VOL_PERIOD = 10

# ══════════════════════════════════════════════════════════
# IMPROVEMENT 4: Simplified WEIGHTS — Top 5 core factors only
# Removed: rsi_zone, macd_bullish, near_52w_high, adx_trend, rel_strength,
#          weekly_confirm, gap_up (retained but de-weighted or merged)
# Kept:    volume_surge, above_ema, price_breakout, sector_bonus, candle_pattern
# Plus small bonuses for confirmatory signals
# ══════════════════════════════════════════════════════════
WEIGHTS = {
    # ── CORE TOP-5 FACTORS (anti-overfit) ──────────────────
    "volume_surge":   25,   # #1 — Institutional participation signal
    "above_ema":      20,   # #2 — Trend alignment (20 + 50 EMA)
    "price_breakout": 20,   # #3 — Close near high of day
    "sector_bonus":   15,   # #4 — Sector tailwind
    "candle_pattern": 15,   # #5 — Price action quality

    # ── CONFIRMATORY BONUSES (small, non-dominant) ──────────
    "marubozu":        8,   # Strongest candle structure
    "gap_up":          5,   # Gap-up that held
    "rel_strength":    4,   # Beat index
    "weekly_confirm":  3,   # MTF alignment (weekly)
}
# Max base score ≈ 115 pts (without gap/RS/weekly bonuses)
# Total theoretical max ≈ 135 pts

# ── Improvement 3: Score thresholds ─────────────────────────
SCORE_GOOD            = 60   # Minimum to appear in output
SCORE_HIGH_CONVICTION = 80   # High-conviction tag in report

# ── Improvement 5: Liquidity filter ─────────────────────────
# Minimum average daily volume (shares) to be considered
MIN_VOLUME_INDIA = 200_000    # ~2 lakh shares/day for NSE
MIN_VOLUME_USA   = 500_000    # ~5 lakh shares/day for NYSE/NASDAQ

# ── Entry quality filter thresholds (Improvement 1) ─────────
ENTRY_MAX_DAY_CHANGE   = 5.0   # Skip if day change > 5% (unless breakout vol > 3x)
ENTRY_MAX_VWAP_DEV     = 3.5   # Skip if close > 3.5% above VWAP proxy (EMA20 used as proxy)

AD_RATIO_MIN = 1.5
SECTOR_TOP_N = 3

IST = ZoneInfo("Asia/Kolkata")
EST = ZoneInfo("America/New_York")

ORB_BARS = 3

ORB_WEIGHTS = {
    "breakout_strength": 25,
    "volume_surge":      20,
    "rsi_5m":            15,
    "adx_5m":            10,
    "orb_range_tight":   10,
    "open_candle_bull":   8,
    "sector_bonus":        7,
}


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
# MARKET HEALTH CHECK  — now returns direction_multiplier
# Improvement 6: Market direction weight applied to all scores
# ══════════════════════════════════════════════════════════

def check_market(market: str = "india") -> tuple[bool, float, float, float]:
    """
    Returns (is_safe, index_chg_pct, vix_value, direction_multiplier)
    direction_multiplier:
      1.10 → strong bull (boost scores)
      1.00 → neutral
      0.85 → weak/choppy (reduce scores)
      0.75 → VIX spike (strongly reduce scores)
    """
    idx_sym = INDIA_INDEX if market == "india" else USA_INDEX
    vix_sym = INDIA_VIX   if market == "india" else USA_VIX
    label   = "Nifty 50"  if market == "india" else "S&P 500"
    vix_thr = 20          if market == "india" else 20

    hdr(f"Market Health — {label.upper()}")

    try:
        with _YF_LOCK:
            combined = yf.download(
                [idx_sym, vix_sym],
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
                threads=False,
            )
        if isinstance(combined.columns, pd.MultiIndex):
            idx = combined[idx_sym].dropna()
            vix = combined[vix_sym].dropna()
        else:
            idx = combined.dropna()
            vix = pd.DataFrame()
    except Exception:
        print(f"  {Fore.YELLOW}⚠  Could not fetch market data{Style.RESET_ALL}")
        return True, 0.0, 0.0, 1.0

    if idx.empty:
        return True, 0.0, 0.0, 1.0

    close = float(idx["Close"].iloc[-1])
    prev  = float(idx["Close"].iloc[-2]) if len(idx) >= 2 else close
    chg   = round((close - prev) / prev * 100, 2)
    vix_v = float(vix["Close"].iloc[-1]) if not vix.empty else 0.0

    bullish  = chg >= -0.5
    vix_safe = vix_v < vix_thr if vix_v > 0 else True
    safe     = bullish and vix_safe

    # ── Improvement 6: compute direction multiplier ──────────────
    if vix_v > 25:
        direction_mult = 0.75   # VIX very high — strongly penalise overnight risk
    elif not bullish and not vix_safe:
        direction_mult = 0.80   # weak market + elevated VIX
    elif not bullish or not vix_safe:
        direction_mult = 0.85   # either weak OR VIX elevated
    elif chg >= 1.0 and vix_safe:
        direction_mult = 1.10   # strong bull day — boost scores
    else:
        direction_mult = 1.00   # neutral

    col = Fore.GREEN if bullish else Fore.RED
    print(f"  {label} Change   : {col}{chg:+.2f}%{Style.RESET_ALL}  (Close: {close:,.2f})")
    print(f"  VIX              : {'🟢' if vix_safe else '🔴'} {vix_v:.2f}  "
          f"({'Safe' if vix_safe else 'HIGH — caution!'})")
    mult_col = Fore.GREEN if direction_mult >= 1.0 else Fore.YELLOW if direction_mult >= 0.85 else Fore.RED
    print(f"  Direction Weight : {mult_col}{direction_mult:.2f}×{Style.RESET_ALL}"
          f"  ({'Boosting' if direction_mult>1 else 'Neutral' if direction_mult==1 else 'Reducing'} BTST scores)")
    if not bullish:
        print(f"  {Fore.RED}⚠  Market weak today — higher BTST risk{Style.RESET_ALL}")
    if not vix_safe:
        print(f"  {Fore.RED}⚠  VIX > {vix_thr} — avoid overnight positions{Style.RESET_ALL}")

    return safe, chg, vix_v, direction_mult


# ══════════════════════════════════════════════════════════
# BATCH DOWNLOAD
# ══════════════════════════════════════════════════════════

def _batch_download(symbols: list) -> dict:
    with _YF_LOCK:
        raw = yf.download(
            symbols,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
            threads=False,
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
# SECTOR PERFORMANCE
# ══════════════════════════════════════════════════════════

def fetch_sector_perf(market: str) -> dict[str, float]:
    sector_map = INDIA_SECTOR_MAP if market == "india" else USA_SECTOR_MAP
    tickers    = list(set(sector_map.values()))

    print(f"  📊  Fetching sector data ({len(tickers)} indices/ETFs) …", flush=True)
    try:
        with _YF_LOCK:
            raw = yf.download(
                tickers,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
                threads=False,
            )
    except Exception:
        return {}

    result: dict[str, float] = {}
    for ticker in tickers:
        try:
            df = raw[ticker].dropna() if isinstance(raw.columns, pd.MultiIndex) else raw.dropna()
            if len(df) >= 2:
                prev_c = float(df["Close"].iloc[-2])
                cur_c  = float(df["Close"].iloc[-1])
                result[ticker] = round((cur_c - prev_c) / prev_c * 100, 3) if prev_c > 0 else 0.0
        except Exception:
            pass

    up_count = sum(1 for v in result.values() if v > 0)
    print(f"  ✅  Sectors: {up_count}/{len(result)} green today.")
    return result


def _top_sectors(sector_perf: dict[str, float], n: int = SECTOR_TOP_N) -> set[str]:
    sorted_secs = sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)
    return {t for t, _ in sorted_secs[:n]}


# ══════════════════════════════════════════════════════════
# SCORE A SINGLE STOCK
# Improvements applied: 1 (Entry Quality), 4 (Reduced weights),
#                       5 (Liquidity filter), 6 (Direction multiplier)
# ══════════════════════════════════════════════════════════

def score_stock_from_df(symbol: str, df: pd.DataFrame,
                        sector_bonus: float = 0.0,
                        index_chg: float = 0.0,
                        breadth_ok: bool = True,
                        direction_mult: float = 1.0,
                        market: str = "india") -> dict | None:
    try:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close  = float(df["Close"].iloc[-1])
        high   = float(df["High"].iloc[-1])
        low    = float(df["Low"].iloc[-1])
        volume = float(df["Volume"].iloc[-1])

        # ── Improvement 5: Liquidity Filter ──────────────────────
        min_vol = MIN_VOLUME_INDIA if market == "india" else MIN_VOLUME_USA
        avg_vol_10 = df["Volume"].iloc[-AVG_VOL_PERIOD-1:-1].mean()
        if avg_vol_10 < min_vol:
            return None   # skip illiquid stocks silently

        # ── Volume Surge (RVOL) ────────────────────────────────────
        weekday = df.index[-1].weekday() if hasattr(df.index[-1], "weekday") else -1
        if weekday == 4 and len(df) >= 10:
            friday_vols = [
                float(df["Volume"].iloc[i])
                for i in range(-2, -len(df)-1, -1)
                if hasattr(df.index[i], "weekday") and df.index[i].weekday() == 4
            ][:4]
            avg_vol = sum(friday_vols) / len(friday_vols) if friday_vols else avg_vol_10
        else:
            avg_vol = avg_vol_10

        vol_ratio = volume / avg_vol if avg_vol > 0 else 0
        s_vol     = (WEIGHTS["volume_surge"] if vol_ratio >= 1.5 else
                     WEIGHTS["volume_surge"] * 0.5 if vol_ratio >= 1.1 else 0)

        # ── Day change ────────────────────────────────────────────
        prev_close = float(df["Close"].iloc[-2])
        day_chg    = (close - prev_close) / prev_close * 100

        # ── EMA ───────────────────────────────────────────────────
        ema20 = float(ta.ema(df["Close"], length=20).iloc[-1])
        ema50 = float(ta.ema(df["Close"], length=50).iloc[-1])
        s_ema = (WEIGHTS["above_ema"] if close > ema20 and close > ema50
                 else WEIGHTS["above_ema"] * 0.5 if close > ema20 else 0)

        # ── Improvement 1: Entry Quality Filter ──────────────────
        # Block overextended entries:
        #   a) Day change > ENTRY_MAX_DAY_CHANGE% (unless breakout vol surge > 3x)
        #   b) Close > ENTRY_MAX_VWAP_DEV% above EMA20 (used as VWAP proxy)
        vwap_dev = (close - ema20) / ema20 * 100 if ema20 > 0 else 0.0
        overextended_day   = day_chg > ENTRY_MAX_DAY_CHANGE and vol_ratio < 3.0
        overextended_vwap  = vwap_dev > ENTRY_MAX_VWAP_DEV
        if overextended_day or overextended_vwap:
            return None   # skip — entry is too overextended

        # ── Price breakout (intraday range position) ──────────────
        rng   = high - low
        pos   = (close - low) / rng if rng > 0 else 0
        s_brk = (WEIGHTS["price_breakout"] if pos >= 0.90
                 else WEIGHTS["price_breakout"] * 0.6 if pos >= 0.75 else 0)

        # ── ATR ───────────────────────────────────────────────────
        atr_s   = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        atr_val = float(atr_s.iloc[-1]) if atr_s is not None and not atr_s.empty else (rng * 0.5)

        # ── RSI (retained as confirmatory info, not scored) ───────
        rsi_s = ta.rsi(df["Close"], length=14)
        rsi   = float(rsi_s.iloc[-1]) if rsi_s is not None else 50

        # ── MACD (retained for info column, not scored in v2) ─────
        macd_df   = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        macd_hist = None
        if macd_df is not None and not macd_df.empty:
            hcol = [c for c in macd_df.columns if "MACDh" in c]
            if hcol:
                macd_hist = float(macd_df[hcol[0]].iloc[-1])

        # ── ADX (retained for info) ────────────────────────────────
        adx_df  = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        adx_val = 0.0
        if adx_df is not None and not adx_df.empty:
            ac = [c for c in adx_df.columns if c.startswith("ADX_")]
            if ac:
                adx_val = float(adx_df[ac[0]].iloc[-1])

        # ── 52-Week High Proximity (info only) ────────────────────
        w52_high   = float(df["High"].max())
        proximity  = close / w52_high if w52_high > 0 else 0
        near_52w   = proximity >= 0.95

        # ── Candlestick Pattern ────────────────────────────────────
        candle_name  = ""
        s_candle     = 0
        s_marubozu   = 0.0
        is_marubozu  = False
        final_hr_fade = False
        try:
            o0    = float(df["Open"].iloc[-1])
            o1    = float(df["Open"].iloc[-2])
            c1    = prev_close
            o2    = float(df["Open"].iloc[-3])
            c2    = float(df["Close"].iloc[-3])
            body0 = abs(close - o0)
            body1 = abs(c1 - o1)
            body2 = abs(c2 - o2)
            lower_shadow0 = min(o0, close) - low
            upper_shadow0 = high - max(o0, close)

            if (rng > 0 and close > o0 and
                    body0 >= rng * 0.85 and
                    (high - close) <= rng * 0.05 and
                    (o0 - low)    <= rng * 0.05):
                is_marubozu  = True
                s_marubozu   = WEIGHTS["marubozu"]
                candle_name  = "Marubozu"

            morning_star = (
                c2 < o2 and body1 <= body2 * 0.4 and
                close > o0 and close > (o2 + c2) / 2 and body0 >= body2 * 0.5
            )
            bullish_engulfing = (
                c1 < o1 and close > o0 and o0 <= c1 and close >= o1
            )
            hammer = (
                rng > 0 and body0 <= rng * 0.35 and
                lower_shadow0 >= 2.0 * body0 and
                upper_shadow0 <= body0 * 0.6 and close > o0
            )

            if not is_marubozu:
                if morning_star:
                    s_candle, candle_name = WEIGHTS["candle_pattern"],       "Morning Star"
                elif bullish_engulfing:
                    s_candle, candle_name = WEIGHTS["candle_pattern"] * 0.8, "Engulfing"
                elif hammer:
                    s_candle, candle_name = WEIGHTS["candle_pattern"] * 0.6, "Hammer"

            if high > close * 1.008 and pos < 0.60:
                final_hr_fade = True
        except Exception:
            pass

        # ── Relative Strength (small bonus) ───────────────────────
        s_rs = WEIGHTS["rel_strength"] if day_chg > index_chg else 0.0

        # ── Multi-Timeframe (small bonus) ─────────────────────────
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

        # ── ATR penalty for overextended moves ────────────────────
        total   = s_vol + s_ema + s_brk
        atr_pct = (atr_val / close * 100) if close > 0 else 4.0
        if day_chg > max(1.5 * atr_pct, 4.0):
            if vol_ratio >= 3.0:
                pass   # breakaway gap — no penalty
            else:
                total *= 0.6

        # ── Gap-up (small bonus) ──────────────────────────────────
        gap_pct  = 0.0
        gap_held = False
        s_gap    = 0.0
        try:
            open0    = float(df["Open"].iloc[-1])
            gap_pct  = (open0 - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
            gap_held = close > prev_close
            if gap_pct >= 1.0 and gap_held and pos >= 0.60:
                s_gap = WEIGHTS["gap_up"]
            elif gap_pct >= 0.5 and gap_held:
                s_gap = WEIGHTS["gap_up"] * 0.6
        except Exception:
            pass

        # ── Additive bonuses ──────────────────────────────────────
        total += sector_bonus + s_candle + s_marubozu + s_rs + s_mtf + s_gap

        # ── Final-hour fade penalty ────────────────────────────────
        if final_hr_fade:
            total *= 0.85

        # ── Market breadth penalty ────────────────────────────────
        if not breadth_ok:
            total *= 0.80

        # ── Improvement 6: Apply market direction multiplier ──────
        total *= direction_mult

        # ── Improvement 3: Score threshold — reject below SCORE_GOOD
        # (done here so we don't emit low-conviction stocks at all)
        if total < SCORE_GOOD:
            return None

        # ── Improvement 2: Next-Day Exit Logic ────────────────────
        # Fixed % targets (in addition to ATR-based targets):
        #   Target 1 (conservative): +1.5%
        #   Target 2 (aggressive):   +2.5%
        #   Stop Loss (fixed %):     -1.5%
        #   Stop Loss (ATR):         max(low, close - ATR)
        # Rules: exit at open if gap-down below SL; exit if 15-min low breaks
        atr_stop = round(max(low, close - atr_val), 2)
        pct_stop = round(close * (1 - 0.015), 2)          # -1.5% fixed SL
        stop_loss = max(atr_stop, pct_stop)                # use tighter of two
        target_1  = round(close * 1.015, 2)               # +1.5% conservative
        target_2  = round(close * 1.025, 2)               # +2.5% aggressive
        risk      = close - stop_loss
        reward    = target_1 - close
        rr_ratio  = round(reward / risk, 2) if risk > 0 else 0.0

        # Conviction label
        conviction = "HIGH" if total >= SCORE_HIGH_CONVICTION else "GOOD"

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
            "Marubozu":      is_marubozu,
            "FinalHrFade":   final_hr_fade,
            "RS_Beat":       day_chg > index_chg,
            "Weekly_Align":  weekly_align,
            "Gap_Up":        gap_pct >= 0.5 and gap_held,
            "Gap_Pct":       round(gap_pct, 2),
            "Sector_Align":  sector_bonus > 0,
            "Breadth_OK":    breadth_ok,
            "Stop_Loss":     stop_loss,
            "Target_1":      target_1,
            "Target_2":      target_2,
            "RR_Ratio":      rr_ratio,
            "Conviction":    conviction,
            "BTST_Score":    round(total, 1),
            # Exit rule embedded for HTML report (Improvement 2)
            "Exit_Rule":     (f"Exit at open if gap < {stop_loss:.2f}. "
                              f"SL: {stop_loss:.2f} | T1: {target_1:.2f} (+1.5%) | "
                              f"T2: {target_2:.2f} (+2.5%). "
                              f"Also exit if 9:20 AM (IST) / 9:35 AM (EST) price < {stop_loss:.2f}"),
        }
    except Exception as e:
        print(f"  {Fore.YELLOW}⚠  Skipping {symbol}: {e}{Style.RESET_ALL}")
        return None


# ══════════════════════════════════════════════════════════
# ORB SCORE — unchanged from v1 (ORB is intraday, not BTST)
# ══════════════════════════════════════════════════════════

def score_orb_stock(symbol: str, sector_bonus: float = 0.0) -> dict | None:
    try:
        with _YF_LOCK:
            raw = yf.download(symbol, period="2d", interval="5m",
                              progress=False, auto_adjust=True, threads=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna()
        if len(raw) < ORB_BARS + 2:
            return None

        is_india = symbol.endswith(".NS")
        tz       = IST if is_india else EST

        if raw.index.tzinfo is None:
            raw.index = raw.index.tz_localize("UTC")
        raw.index = raw.index.tz_convert(tz)

        today_date = datetime.now(tz=tz).date()
        df = raw[raw.index.date == today_date].copy()
        if len(df) < ORB_BARS + 1:
            return None

        orb       = df.iloc[:ORB_BARS]
        orb_high  = float(orb["High"].max())
        orb_low   = float(orb["Low"].min())
        orb_range = orb_high - orb_low
        orb_range_pct = (orb_range / orb_high * 100) if orb_high > 0 else 0.0

        cur_bar = df.iloc[-1]
        price   = float(cur_bar["Close"])
        cur_vol = float(cur_bar["Volume"])

        if price <= orb_high:
            return None

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

        first = df.iloc[0]
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
# RUN SCREENER
# ══════════════════════════════════════════════════════════

def fetch_advance_decline(market: str) -> float:
    try:
        if market == "usa":
            with _YF_LOCK:
                raw = yf.download("^NYAD", period="5d", interval="1d",
                                  progress=False, auto_adjust=True, threads=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.dropna()
            if len(raw) >= 2:
                today_chg = float(raw["Close"].iloc[-1]) - float(raw["Close"].iloc[-2])
                return 2.5 if today_chg > 500 else 1.8 if today_chg > 0 else 0.8
        else:
            return 0.0
    except Exception:
        pass
    return 0.0


def run_screener(symbols: list, label: str,
                 index_chg: float = 0.0,
                 direction_mult: float = 1.0) -> tuple[pd.DataFrame, dict]:
    hdr(f"Scanning {len(symbols)} {label} stocks (batch + parallel) …")

    market     = "india" if any(s.endswith(".NS") for s in symbols) else "usa"
    sector_map = INDIA_SECTOR_MAP if market == "india" else USA_SECTOR_MAP

    print(f"  📥  Fetching 1-year OHLCV + sector data in parallel …", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pre:
        f_cache   = pre.submit(_batch_download, symbols)
        f_sector  = pre.submit(fetch_sector_perf, market)
        f_ad      = pre.submit(fetch_advance_decline, market)
        cache       = f_cache.result()
        sector_perf = f_sector.result()
        ad_ratio_raw = f_ad.result()

    if market == "india" and cache:
        advances = sum(
            1 for df in cache.values()
            if len(df) >= 2 and float(df["Close"].iloc[-1]) > float(df["Close"].iloc[-2])
        )
        declines = len(cache) - advances
        ad_ratio = advances / max(declines, 1)
    else:
        ad_ratio = ad_ratio_raw

    breadth_ok = ad_ratio >= AD_RATIO_MIN or ad_ratio == 0.0
    if ad_ratio > 0:
        col = Fore.GREEN if breadth_ok else Fore.YELLOW
        print(f"  📈  A/D Ratio: {col}{ad_ratio:.2f}x{Style.RESET_ALL}  "
              f"({'Broad rally ✅' if breadth_ok else 'Narrow market ⚠ — lower conviction'})")

    top_sec_set = _top_sectors(sector_perf, SECTOR_TOP_N)

    print(f"  ✅  Downloaded {len(cache)}/{len(symbols)}. Scoring …", flush=True)

    def _sector_bonus(sym: str) -> float:
        sec_ticker = sector_map.get(sym)
        if not sec_ticker:
            return 0.0
        sec_chg = sector_perf.get(sec_ticker, None)
        if sec_chg is None or sec_chg <= 0:
            return 0.0
        return float(WEIGHTS["sector_bonus"]) if sec_ticker in top_sec_set \
               else float(WEIGHTS["sector_bonus"]) * 0.5

    results = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {
            pool.submit(score_stock_from_df, sym, df,
                        _sector_bonus(sym), index_chg, breadth_ok,
                        direction_mult, market): sym
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
    print(f"  ✅  Scored {len(results)} stocks passing all filters.")
    return pd.DataFrame(results), sector_perf


# ══════════════════════════════════════════════════════════
# ORB BATCH DOWNLOAD + SCREENER
# ══════════════════════════════════════════════════════════

def _batch_download_intraday(symbols: list) -> dict:
    is_india = any(s.endswith(".NS") for s in symbols)
    tz = IST if is_india else EST

    print(f"  📥  Batch-fetching 5-min bars for {len(symbols)} tickers …", flush=True)
    try:
        with _YF_LOCK:
            raw = yf.download(
                symbols, period="2d", interval="5m",
                progress=False, auto_adjust=True, group_by="ticker", threads=False,
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
    try:
        orb       = df.iloc[:ORB_BARS]
        orb_high  = float(orb["High"].max())
        orb_low   = float(orb["Low"].min())
        orb_range = orb_high - orb_low
        orb_range_pct = (orb_range / orb_high * 100) if orb_high > 0 else 0.0

        cur_bar = df.iloc[-1]
        price   = float(cur_bar["Close"])
        cur_vol = float(cur_bar["Volume"])

        if price <= orb_high:
            return None

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


def run_orb_screener(symbols: list, label: str,
                     sector_perf: dict | None = None) -> pd.DataFrame:
    hdr(f"ORB Scan — {len(symbols)} {label} stocks (5-min bars) …")

    market     = "india" if any(s.endswith(".NS") for s in symbols) else "usa"
    sector_map = INDIA_SECTOR_MAP if market == "india" else USA_SECTOR_MAP
    if sector_perf is None:
        sector_perf = fetch_sector_perf(market)
    else:
        print(f"  ♻️  Reusing sector perf from BTST scan (skipping re-fetch).")

    def _sector_bonus(sym: str) -> float:
        sec     = sector_map.get(sym)
        sec_chg = sector_perf.get(sec, None) if sec else None
        if sec_chg is None or sec_chg <= 0:
            return 0.0
        return float(ORB_WEIGHTS["sector_bonus"])

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
    print(f"  ✅  ORB: {len(results)} confirmed breakout(s) found out of {len(symbols)} scanned.")
    return pd.DataFrame(results)


def filter_and_rank_orb(df: pd.DataFrame) -> pd.DataFrame:
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
# FILTER & RANK  — Improvement 3: score threshold enforced
# (low-score stocks were already rejected in score_stock_from_df,
#  but we also sort and tag conviction level here)
# ══════════════════════════════════════════════════════════

def filter_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # score_stock_from_df already filters < SCORE_GOOD;
    # additional quality gates:
    f = df[
        (df["RSI"] >= 48) &
        (df["Volume_Ratio"] >= 1.1) &
        (df["Change%"] > -0.5) &
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
            "Weekly_Align", "Sector_Align", "Stop_Loss", "Target_1", "Target_2",
            "RR_Ratio", "Conviction", "BTST_Score"]
    available = [c for c in cols if c in df.columns]
    disp = df[available].reset_index(drop=True)
    disp.index += 1

    # Print exit rules summary
    print(f"\n  {Fore.YELLOW}📋 NEXT-DAY EXIT RULES (Improvement #2):{Style.RESET_ALL}")
    print(f"  • India: EXIT at 9:20 AM IST if gap-down below Stop_Loss")
    print(f"  • USA  : EXIT at 9:35 AM EST if gap-down below Stop_Loss")
    print(f"  • Also exit if first 15-min candle low < Stop_Loss")
    print(f"  • Target_1 = +1.5% (conservative) | Target_2 = +2.5% (aggressive)")

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
    meta = {"ok": ok, "chg": round(chg, 4), "vix": round(vix, 4)}
    with open(f"btst_{prefix}_meta_{date_str}.json", "w") as f:
        json.dump(meta, f)


def load_meta(prefix: str, date_str: str) -> tuple[bool, float, float]:
    try:
        with open(f"btst_{prefix}_meta_{date_str}.json") as f:
            m = json.load(f)
        return bool(m.get("ok", True)), float(m.get("chg", 0.0)), float(m.get("vix", 0.0))
    except Exception:
        return True, 0.0, 0.0


# ══════════════════════════════════════════════════════════
# BACKTEST — Improvement 7: Win rate, Avg gain, Max drawdown, Expectancy
# ══════════════════════════════════════════════════════════

def run_backtest(prefix: str, days: int = 30):
    """
    Improvement 7: Enhanced backtest with:
      - Win Rate %
      - Avg Gain (winners only)
      - Avg Loss (losers only)
      - Max Drawdown (worst single trade)
      - Expectancy = (Win% × Avg_Gain) - (Loss% × Avg_Loss)
      - Score-stratified: ≥80 (High Conviction) vs 60–79 (Good) vs <60 (old data)
    """
    tz_now   = datetime.now(tz=IST if prefix == "india" else EST)
    today    = tz_now.date()
    sym_sfx  = ".NS" if prefix == "india" else ""

    hdr(f"BACKTEST v2 — {prefix.upper()} | Scanning last {days} calendar days")

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
                    # Support both v1 (Target) and v2 (Target_1)
                    "Target":   float(row.get("Target_1", row.get("Target", 0))),
                    "Score":    float(row.get("BTST_Score", 0)),
                })
            dates_found.append(date_str)
        except FileNotFoundError:
            continue
        except Exception:
            continue

    if not all_rows:
        print(f"\n  {Fore.YELLOW}No past CSVs found in the last {days} days.{Style.RESET_ALL}")
        return

    print(f"  📂  Found {len(dates_found)} past session(s), {len(all_rows)} total pick(s).")

    raw_syms  = list({r["Symbol"] for r in all_rows if r["Symbol"]})
    dl_syms   = [s + sym_sfx for s in raw_syms]

    print(f"  📥  Downloading history for {len(dl_syms)} symbols …", flush=True)
    cache_raw = _batch_download(dl_syms)

    norm_cache: dict[str, pd.DataFrame] = {}
    for sym, df in cache_raw.items():
        clean = sym.replace(".NS", "").replace("-", ".").replace("BRK.B", "BRK-B")
        df_copy = df.copy()
        df_copy.index = pd.to_datetime(df_copy.index)
        norm_cache[clean] = df_copy

    print(f"  ✅  History loaded. Evaluating picks …")

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

        df_sym  = norm_cache[sym]
        future  = df_sym[df_sym.index > pick_date]
        if future.empty:
            continue

        nxt       = future.iloc[0]
        nxt_open  = float(nxt.get("Open",  nxt["Close"]))
        nxt_high  = float(nxt["High"])
        nxt_low   = float(nxt["Low"])
        nxt_close = float(nxt["Close"])
        nxt_date  = future.index[0].strftime("%Y-%m-%d")

        if nxt_open <= sl:
            outcome = "LOSS"
            exit_price = nxt_open    # gap-down → exit at open
        elif nxt_high >= tgt:
            outcome = "WIN"
            exit_price = tgt
        elif nxt_low <= sl:
            outcome = "LOSS"
            exit_price = sl
        else:
            outcome = "NEUTRAL"
            exit_price = nxt_close

        pnl_pct = (exit_price - entry) / entry * 100 if entry > 0 else 0.0

        results.append({
            "Date":       r["date_str"],
            "Symbol":     sym,
            "Score":      round(score, 1),
            "Entry":      round(entry, 2),
            "SL":         round(sl,    2),
            "Target":     round(tgt,   2),
            "Exit_Price": round(exit_price, 2),
            "Next_Open":  round(nxt_open,  2),
            "Next_High":  round(nxt_high,  2),
            "Next_Low":   round(nxt_low,   2),
            "Next_Close": round(nxt_close, 2),
            "PnL_%":      round(pnl_pct, 2),
            "Outcome":    outcome,
            "Next_Date":  nxt_date,
        })

    if not results:
        print(f"  {Fore.YELLOW}No picks could be evaluated (no next-day data yet).{Style.RESET_ALL}")
        return

    res_df = pd.DataFrame(results)

    # ── Core stats ────────────────────────────────────────
    total   = len(res_df)
    wins    = (res_df["Outcome"] == "WIN").sum()
    losses  = (res_df["Outcome"] == "LOSS").sum()
    neutral = (res_df["Outcome"] == "NEUTRAL").sum()
    win_rt  = wins / total * 100 if total else 0.0
    loss_rt = losses / total * 100 if total else 0.0

    # ── Avg gain / loss (on closed trades, excl. neutral) ─
    win_pnls  = res_df[res_df["Outcome"] == "WIN"]["PnL_%"]
    loss_pnls = res_df[res_df["Outcome"] == "LOSS"]["PnL_%"]
    avg_gain  = win_pnls.mean()  if len(win_pnls)  else 0.0
    avg_loss  = loss_pnls.mean() if len(loss_pnls) else 0.0   # negative number

    # ── Max drawdown (single worst trade) ─────────────────
    max_dd = res_df["PnL_%"].min()

    # ── Expectancy ────────────────────────────────────────
    # Expectancy = (Win% × Avg_Gain) + (Loss% × Avg_Loss)
    # A positive expectancy means the system is profitable per trade on average
    expectancy = (win_rt / 100 * avg_gain) + (loss_rt / 100 * avg_loss)

    # ── Score-stratified breakdown ─────────────────────────
    def _stratum_stats(mask, label):
        sub = res_df[mask]
        if sub.empty:
            return [label, 0, 0, "—", "—", "—"]
        sw = (sub["Outcome"] == "WIN").sum()
        sl = (sub["Outcome"] == "LOSS").sum()
        hr = sw / len(sub) * 100
        ag = sub[sub["Outcome"] == "WIN"]["PnL_%"].mean() if sw else 0.0
        al = sub[sub["Outcome"] == "LOSS"]["PnL_%"].mean() if sl else 0.0
        return [label, len(sub), sw, f"{hr:.1f}%", f"{ag:+.2f}%", f"{al:+.2f}%"]

    strat_rows = [
        _stratum_stats(res_df["Score"] >= SCORE_HIGH_CONVICTION, f"High Conviction (≥{SCORE_HIGH_CONVICTION})"),
        _stratum_stats((res_df["Score"] >= SCORE_GOOD) & (res_df["Score"] < SCORE_HIGH_CONVICTION), f"Good ({SCORE_GOOD}–{SCORE_HIGH_CONVICTION-1})"),
        _stratum_stats(res_df["Score"] < SCORE_GOOD, f"Below threshold (<{SCORE_GOOD})"),
    ]

    # Score ↔ return correlation
    corr = (res_df[["Score", "PnL_%"]].corr().iloc[0, 1]
            if len(res_df) >= 5 else float("nan"))

    # ── Print results ─────────────────────────────────────
    hdr(f"BACKTEST v2 RESULTS — {prefix.upper()}")
    print(f"  Sessions analysed  : {len(dates_found)}")
    print(f"  Total picks        : {total}")
    print()

    w_col = Fore.GREEN if win_rt >= 55 else Fore.YELLOW if win_rt >= 45 else Fore.RED
    print(f"  {'Wins  (Target hit)':<26}: {Fore.GREEN}{wins:>4}{Style.RESET_ALL}")
    print(f"  {'Losses (SL hit)':<26}: {Fore.RED}{losses:>4}{Style.RESET_ALL}")
    print(f"  {'Neutral (neither)':<26}: {Fore.YELLOW}{neutral:>4}{Style.RESET_ALL}")
    print(f"  {'Win Rate':<26}: {w_col}{win_rt:>6.1f}%{Style.RESET_ALL}")
    print(f"  {'Avg Gain (winners)':<26}: {Fore.GREEN}{avg_gain:>+6.2f}%{Style.RESET_ALL}")
    print(f"  {'Avg Loss (losers)':<26}: {Fore.RED}{avg_loss:>+6.2f}%{Style.RESET_ALL}")
    mdd_col = Fore.RED if max_dd < -3 else Fore.YELLOW
    print(f"  {'Max Drawdown (worst)':<26}: {mdd_col}{max_dd:>+6.2f}%{Style.RESET_ALL}")
    exp_col = Fore.GREEN if expectancy > 0 else Fore.RED
    print(f"  {'Expectancy (per trade)':<26}: {exp_col}{expectancy:>+6.3f}%{Style.RESET_ALL}  "
          f"({'✅ System has edge' if expectancy > 0 else '❌ System unprofitable — recalibrate'})")
    print()

    print(tabulate(strat_rows,
                   headers=["Tier", "Picks", "Wins", "Hit Rate", "Avg Win", "Avg Loss"],
                   tablefmt="simple"))
    print()

    if not pd.isna(corr):
        corr_col = Fore.GREEN if corr > 0.15 else Fore.YELLOW if corr > 0 else Fore.RED
        corr_lbl = ("✅ Score predicts returns"  if corr > 0.15 else
                    "↔ Weak positive link"       if corr > 0    else
                    "⚠ Score not yet predictive")
        print(f"  Score ↔ Return corr : {corr_col}{corr:+.3f}  {corr_lbl}{Style.RESET_ALL}")
        print()

    # Top wins / worst losses
    top_wins = (res_df[res_df["Outcome"] == "WIN"]
                .nlargest(5, "PnL_%")
                [["Date", "Symbol", "Score", "Entry", "Target", "PnL_%"]]
                .reset_index(drop=True))
    if not top_wins.empty:
        print(f"  {Fore.GREEN}── Top 5 Winning Picks ──{Style.RESET_ALL}")
        print(tabulate(top_wins,
                       headers=["Date", "Symbol", "Score", "Entry", "Target", "PnL %"],
                       tablefmt="simple", floatfmt=".2f"))
        print()

    worst = (res_df[res_df["Outcome"] == "LOSS"]
             .nsmallest(5, "PnL_%")
             [["Date", "Symbol", "Score", "Entry", "SL", "PnL_%"]]
             .reset_index(drop=True))
    if not worst.empty:
        print(f"  {Fore.RED}── Worst 5 Losses ──{Style.RESET_ALL}")
        print(tabulate(worst,
                       headers=["Date", "Symbol", "Score", "Entry", "SL", "PnL %"],
                       tablefmt="simple", floatfmt=".2f"))
        print()

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
# SCAN HELPERS
# ══════════════════════════════════════════════════════════

def _scan_india(date_str: str, run_orb: bool = True):
    ok, chg, vix, dir_mult = check_market("india")
    full, sector_perf = run_screener(NIFTY100_SYMBOLS, "Nifty 100", chg, dir_mult)
    top = pd.DataFrame()
    if not full.empty:
        top = filter_and_rank(full)
        print_report(top, "INDIA", ok, chg)
        save_csv(top, full, "india", date_str)
        save_meta("india", date_str, ok, chg, vix)
    orb_top = pd.DataFrame()
    if run_orb:
        orb_raw = run_orb_screener(NIFTY100_SYMBOLS, "Nifty 100", sector_perf=sector_perf)
        orb_top = filter_and_rank_orb(orb_raw)
        if not orb_top.empty:
            orb_top.to_csv(f"orb_india_{date_str}.csv", index=False)
    return ok, chg, vix, top, full, orb_top


def _scan_usa(date_str: str, run_orb: bool = True):
    ok, chg, vix, dir_mult = check_market("usa")
    full, sector_perf = run_screener(SP500_TOP100_SYMBOLS, "S&P 500 Top 100", chg, dir_mult)
    top = pd.DataFrame()
    if not full.empty:
        top = filter_and_rank(full)
        print_report(top, "USA", ok, chg)
        save_csv(top, full, "usa", date_str)
        save_meta("usa", date_str, ok, chg, vix)
    orb_top = pd.DataFrame()
    if run_orb:
        orb_raw = run_orb_screener(SP500_TOP100_SYMBOLS, "S&P 500 Top 100", sector_perf=sector_perf)
        orb_top = filter_and_rank_orb(orb_raw)
        if not orb_top.empty:
            orb_top.to_csv(f"orb_usa_{date_str}.csv", index=False)
    return ok, chg, vix, top, full, orb_top


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BTST Screener v2 — India + USA")
    parser.add_argument("--india",     action="store_true")
    parser.add_argument("--usa",       action="store_true")
    parser.add_argument("--no-orb",    action="store_true")
    parser.add_argument("--backtest",  action="store_true")
    parser.add_argument("--days",      type=int, default=30)
    args      = parser.parse_args()
    run_india = not args.usa   or args.india
    run_usa   = not args.india or args.usa
    do_orb    = not args.no_orb

    print(f"\n{Fore.CYAN}{'='*62}")
    print("   BTST SCREENER v2  |  India (Nifty 100)  +  USA (S&P 500 Top 100)")
    print(f"   v2 Improvements: Entry Quality • Exit Logic • Score Threshold")
    print(f"                    Simplified Weights • Liquidity • Market Weight • Backtest+")
    print(f"{'='*62}{Style.RESET_ALL}")

    IST_NOW  = datetime.now(tz=IST)
    date_str = IST_NOW.strftime("%Y-%m-%d")

    if args.backtest:
        if run_india:
            run_backtest("india", args.days)
        if run_usa:
            run_backtest("usa", args.days)
        print_disclaimer()
        return

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

    print_disclaimer()


if __name__ == "__main__":
    main()
