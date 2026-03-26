"""
============================================================
  BTST (Buy Today Sell Tomorrow) Stock Screener
  India — Nifty 100 (NSE)  |  USA — S&P 500 Top 100 (NYSE/NASDAQ)
============================================================
Requirements:
    pip install yfinance pandas pandas-ta requests tabulate colorama

Usage:
    python btst_screener.py              # scans both markets
    python btst_screener.py --india      # India only
    python btst_screener.py --usa        # USA only

Output:
    btst_report_YYYY-MM-DD.html    (combined HTML with toggle)
    btst_india_YYYY-MM-DD.csv
    btst_usa_YYYY-MM-DD.csv
============================================================
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import warnings
import sys
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo          # stdlib — Python 3.9+
from tabulate import tabulate
from colorama import Fore, Style, init

warnings.filterwarnings("ignore")
init(autoreset=True)

# ══════════════════════════════════════════════════════════
# SYMBOL LISTS
# ══════════════════════════════════════════════════════════

NIFTY100_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFOSYS.NS", "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "ADANIENT.NS",
    "NESTLEIND.NS", "JSWSTEEL.NS", "TATAMOTORS.NS", "POWERGRID.NS", "NTPC.NS",
    "ONGC.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "TECHM.NS", "HCLTECH.NS",
    "TATASTEEL.NS", "HINDALCO.NS", "GRASIM.NS", "DRREDDY.NS", "CIPLA.NS",
    "BPCL.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
    "TATACONSUM.NS", "BRITANNIA.NS", "ADANIPORTS.NS", "SBILIFE.NS", "HDFCLIFE.NS",
    "BAJAJ-AUTO.NS", "M&M.NS", "VEDL.NS", "SIEMENS.NS", "PIDILITIND.NS",
    "DABUR.NS", "MARICO.NS", "HAVELLS.NS", "BERGEPAINT.NS", "LUPIN.NS",
    "TORNTPHARM.NS", "MUTHOOTFIN.NS", "SHREECEM.NS", "AMBUJACEM.NS", "GAIL.NS",
    "IOC.NS", "INDUSINDBK.NS", "BANDHANBNK.NS", "PNB.NS", "CANBK.NS",
    "BANKBARODA.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "HDFCAMC.NS", "NAUKRI.NS",
    "DMART.NS", "ZOMATO.NS", "PAYTM.NS", "IRCTC.NS", "HAL.NS",
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

LOOKBACK_DAYS  = 60
AVG_VOL_PERIOD = 10

WEIGHTS = {
    "volume_surge":   20,
    "rsi_zone":       15,
    "macd_bullish":   15,
    "above_ema":      15,
    "price_breakout": 15,
    "delivery_pct":   10,   # placeholder
    "adx_trend":      10,
}

IST = ZoneInfo("Asia/Kolkata")
EST = ZoneInfo("America/New_York")


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

def hdr(msg):
    print(f"\n{Fore.CYAN}{'─'*62}\n  {msg}\n{'─'*62}{Style.RESET_ALL}")


def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


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
        idx = _flatten(yf.download(idx_sym, period="5d", interval="1d", progress=False))
        vix = _flatten(yf.download(vix_sym, period="5d", interval="1d", progress=False))
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
# SCORE A SINGLE STOCK
# ══════════════════════════════════════════════════════════

def score_stock(symbol: str) -> dict | None:
    try:
        df = yf.download(symbol, period=f"{LOOKBACK_DAYS}d", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 20:
            return None
        df = _flatten(df).dropna()

        close  = float(df["Close"].iloc[-1])
        high   = float(df["High"].iloc[-1])
        low    = float(df["Low"].iloc[-1])
        volume = float(df["Volume"].iloc[-1])

        # Volume surge
        avg_vol   = df["Volume"].iloc[-AVG_VOL_PERIOD-1:-1].mean()
        vol_ratio = volume / avg_vol if avg_vol > 0 else 0
        s_vol     = (WEIGHTS["volume_surge"] if vol_ratio >= 1.5 else
                     WEIGHTS["volume_surge"] * 0.5 if vol_ratio >= 1.2 else 0)

        # RSI
        rsi_s = ta.rsi(df["Close"], length=14)
        rsi   = float(rsi_s.iloc[-1]) if rsi_s is not None else 50
        s_rsi = (WEIGHTS["rsi_zone"] if 55 <= rsi <= 75 else
                 WEIGHTS["rsi_zone"] * 0.5 if 50 <= rsi < 55 else 0)

        # MACD
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

        # EMA
        ema20 = float(ta.ema(df["Close"], length=20).iloc[-1])
        ema50 = float(ta.ema(df["Close"], length=50).iloc[-1])
        s_ema = (WEIGHTS["above_ema"] if close > ema20 and close > ema50
                 else WEIGHTS["above_ema"] * 0.5 if close > ema20 else 0)

        # Price breakout
        rng      = high - low
        pos      = (close - low) / rng if rng > 0 else 0
        s_brk    = (WEIGHTS["price_breakout"] if pos >= 0.90
                    else WEIGHTS["price_breakout"] * 0.6 if pos >= 0.75 else 0)

        # ADX
        adx_df  = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        adx_val = 0.0
        s_adx   = 0
        if adx_df is not None and not adx_df.empty:
            ac = [c for c in adx_df.columns if c.startswith("ADX_")]
            if ac:
                adx_val = float(adx_df[ac[0]].iloc[-1])
                s_adx   = (WEIGHTS["adx_trend"] if adx_val >= 25
                            else WEIGHTS["adx_trend"] * 0.5 if adx_val >= 20 else 0)

        prev_close = float(df["Close"].iloc[-2])
        day_chg    = (close - prev_close) / prev_close * 100

        total = s_vol + s_rsi + s_macd + s_ema + s_brk + s_adx
        if day_chg > 4.0:
            total *= 0.6

        clean_sym = (symbol.replace(".NS", "")
                           .replace("-", ".")
                           .replace("BRK.B", "BRK-B"))

        return {
            "Symbol":       clean_sym,
            "Close":        round(close, 2),
            "Change%":      round(day_chg, 2),
            "Volume_Ratio": round(vol_ratio, 2),
            "RSI":          round(rsi, 1),
            "MACD_Hist":    round(macd_hist, 4) if macd_hist is not None else None,
            "EMA20":        round(ema20, 2),
            "EMA50":        round(ema50, 2),
            "ADX":          round(adx_val, 1),
            "Range_Pos%":   round(pos * 100, 1),
            "BTST_Score":   round(total, 1),
        }
    except Exception as e:
        print(f"  {Fore.YELLOW}⚠  Skipping {symbol}: {e}{Style.RESET_ALL}")
        return None


# ══════════════════════════════════════════════════════════
# RUN SCREENER
# ══════════════════════════════════════════════════════════

def run_screener(symbols: list, label: str) -> pd.DataFrame:
    hdr(f"Scanning {len(symbols)} {label} stocks …")
    results, total = [], len(symbols)
    for i, sym in enumerate(symbols, 1):
        print(f"  [{i:>3}/{total}] {sym:<22}", end="\r")
        r = score_stock(sym)
        if r:
            results.append(r)
    print(" " * 60, end="\r")
    print(f"  ✅  Scanned {len(results)} stocks successfully.")
    return pd.DataFrame(results)


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

    cols = ["Symbol", "Close", "Change%", "Volume_Ratio", "RSI", "ADX", "Range_Pos%", "BTST_Score"]
    disp = df[cols].reset_index(drop=True)
    disp.index += 1
    print(f"\n{Fore.CYAN}  TOP BTST CANDIDATES{Style.RESET_ALL}\n")
    print(tabulate(disp, headers="keys", tablefmt="fancy_grid", floatfmt=".2f", showindex=True))


# ══════════════════════════════════════════════════════════
# SAVE CSV
# ══════════════════════════════════════════════════════════

def save_csv(top_df: pd.DataFrame, full_df: pd.DataFrame, prefix: str, date_str: str):
    top_df.to_csv(f"btst_{prefix}_{date_str}.csv", index=False)
    if not full_df.empty and "BTST_Score" in full_df.columns:
        full_df.sort_values("BTST_Score", ascending=False).to_csv(
            f"btst_{prefix}_full_{date_str}.csv", index=False)
    else:
        full_df.to_csv(f"btst_{prefix}_full_{date_str}.csv", index=False)
    print(f"  💾  {prefix.upper()} CSVs saved.")


# ══════════════════════════════════════════════════════════
# HTML TABLE ROWS BUILDER
# ══════════════════════════════════════════════════════════

def _rows(df: pd.DataFrame, currency: str = "₹") -> str:
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    rows   = ""
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        medal  = medals.get(rank, f"#{rank}")
        chg_c  = "#00e676" if row["Change%"] >= 0 else "#ff5252"
        chg_a  = "▲" if row["Change%"] >= 0 else "▼"

        vol_cls = "bg" if row["Volume_Ratio"] >= 1.5 else "by" if row["Volume_Ratio"] >= 1.2 else "br"
        rsi_cls = "bg" if 55 <= row["RSI"] <= 75 else "by" if row["RSI"] >= 50 else "br"

        sc  = row["BTST_Score"]
        bc  = "#00e676" if sc >= 70 else "#ffca28" if sc >= 50 else "#ff5252"
        pct = min(sc, 100)

        rows += f"""
        <tr>
          <td class="rnk">{medal}</td>
          <td class="sym">{row['Symbol']}</td>
          <td class="num">{currency}{row['Close']:,.2f}</td>
          <td><span style="color:{chg_c};font-weight:700">{chg_a} {abs(row['Change%']):.2f}%</span></td>
          <td><span class="badge {vol_cls}">{row['Volume_Ratio']:.2f}x</span></td>
          <td><span class="badge {rsi_cls}">{row['RSI']:.1f}</span></td>
          <td class="num">{row['ADX']:.1f}</td>
          <td class="num">{row['Range_Pos%']:.1f}%</td>
          <td>
            <div class="bw">
              <div class="bt"><div class="b" style="width:{pct}%;background:{bc}"></div></div>
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
):
    now_ist  = datetime.now(tz=IST)
    now_est  = datetime.now(tz=EST)
    time_ist = now_ist.strftime("%d %b %Y, %I:%M %p IST")
    time_est = now_est.strftime("%d %b %Y, %I:%M %p EST")
    html_file = f"btst_report_{date_str}.html"

    india_rows = _rows(india_top, "₹") if not india_top.empty else "<tr><td colspan='9' style='text-align:center;color:var(--muted);padding:30px'>No candidates found today</td></tr>"
    usa_rows   = _rows(usa_top,   "$") if not usa_top.empty   else "<tr><td colspan='9' style='text-align:center;color:var(--muted);padding:30px'>No candidates found today</td></tr>"

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
        <thead><tr><th>#</th><th>Symbol</th><th>Close (₹)</th><th>Change</th><th>Vol Ratio</th><th>RSI</th><th>ADX</th><th>Range Pos</th><th>BTST Score</th></tr></thead>
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
        <thead><tr><th>#</th><th>Symbol</th><th>Close ($)</th><th>Change</th><th>Vol Ratio</th><th>RSI</th><th>ADX</th><th>Range Pos</th><th>BTST Score</th></tr></thead>
        <tbody>{usa_rows}</tbody>
      </table>
    </div>
    {_legend()}
  </div>

  <!-- SCORING PARAMS (shared) -->
  <div class="sh" style="margin-top:36px">
    <div class="sh-title">⚙️ Scoring Parameters</div>
    <div class="sh-line"></div>
    <div class="sh-sub">Max score = 100 pts · applies to both markets</div>
  </div>
  <div class="pg">
    <div class="pc"><div class="pn">Volume Surge<span class="pw">20 pts</span></div><div class="pd">Volume &gt;1.5× 10-day average confirms institutional participation.</div></div>
    <div class="pc"><div class="pn">RSI Zone<span class="pw">15 pts</span></div><div class="pd">RSI 55–75 signals strong momentum without being overbought.</div></div>
    <div class="pc"><div class="pn">MACD Signal<span class="pw">15 pts</span></div><div class="pd">Positive histogram or fresh bullish crossover signals continuation.</div></div>
    <div class="pc"><div class="pn">EMA Alignment<span class="pw">15 pts</span></div><div class="pd">Price above both 20 EMA and 50 EMA confirms bullish structure.</div></div>
    <div class="pc"><div class="pn">Price Breakout<span class="pw">15 pts</span></div><div class="pd">Closing in top 5–10% of day's range shows buyer dominance at close.</div></div>
    <div class="pc"><div class="pn">ADX Trend<span class="pw">10 pts</span></div><div class="pd">ADX &gt;25 confirms strong trend, reducing overnight whipsaw risk.</div></div>
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
  Parameters: Volume Surge (20) · RSI Zone (15) · MACD Signal (15) · EMA Alignment (15) · Price Breakout (15) · ADX Trend (10)
</div>

<script>
  // Set correct timestamp based on active tab
  const times = {{
    india: {{ time: "{now_ist.strftime('%I:%M %p')}", tz: "IST", date: "{now_ist.strftime('%d %b %Y')}" }},
    usa:   {{ time: "{now_est.strftime('%I:%M %p')}", tz: "EST", date: "{now_est.strftime('%d %b %Y')}" }}
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
    // pills
    document.querySelectorAll('.market-pills').forEach(p => p.classList.remove('active'));
    document.getElementById('pills-' + tab).classList.add('active');
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
    </div>"""


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

def main():
    parser = argparse.ArgumentParser(description="BTST Screener — India + USA")
    parser.add_argument("--india", action="store_true", help="Scan India only")
    parser.add_argument("--usa",   action="store_true", help="Scan USA only")
    args   = parser.parse_args()
    run_india = not args.usa   or args.india
    run_usa   = not args.india or args.usa

    IST_NOW  = datetime.now(tz=IST)
    date_str = IST_NOW.strftime("%Y-%m-%d")

    print(f"\n{Fore.CYAN}{'='*62}")
    print("   BTST SCREENER  |  India (Nifty 100)  +  USA (S&P 500 Top 100)")
    print(f"{'='*62}{Style.RESET_ALL}")

    # ── INDIA ──────────────────────────────────────────────
    india_ok, india_chg, india_vix = True, 0.0, 0.0
    india_full = india_top = pd.DataFrame()

    if run_india:
        india_ok, india_chg, india_vix = check_market("india")
        india_full = run_screener(NIFTY100_SYMBOLS, "Nifty 100")
        if not india_full.empty:
            india_top = filter_and_rank(india_full)
            print_report(india_top, "INDIA", india_ok, india_chg)
            save_csv(india_top, india_full, "india", date_str)

    # ── USA ────────────────────────────────────────────────
    usa_ok, usa_chg, usa_vix = True, 0.0, 0.0
    usa_full = usa_top = pd.DataFrame()

    if run_usa:
        usa_ok, usa_chg, usa_vix = check_market("usa")
        usa_full = run_screener(SP500_TOP100_SYMBOLS, "S&P 500 Top 100")
        if not usa_full.empty:
            usa_top = filter_and_rank(usa_full)
            print_report(usa_top, "USA", usa_ok, usa_chg)
            save_csv(usa_top, usa_full, "usa", date_str)

    # ── HTML ───────────────────────────────────────────────
    generate_html_report(
        india_top, india_full, india_ok, india_chg, india_vix,
        usa_top,   usa_full,   usa_ok,   usa_chg,   usa_vix,
        date_str,
    )

    print_disclaimer()


if __name__ == "__main__":
    main()
