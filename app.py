import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time, math, re
from datetime import datetime, timedelta, date
from google.oauth2.service_account import Credentials

# Rerun-shim (Streamlit 1.30+)
try:
    _rerun = st.rerun
except AttributeError:
    _rerun = st.experimental_rerun

st.set_page_config(page_title="Relative Yield – utdelningsportfölj", layout="wide")

# ── Secrets / Google Sheets ────────────────────────────────────────────────
SHEET_URL  = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"  # <- vi skriver till Blad1
BACKUP_PREFIX = "Backup_"

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
credentials = Credentials.from_service_account_info(
    st.secrets["GOOGLE_CREDENTIALS"], scopes=scope
)
client = gspread.authorize(credentials)

# ── Kolumnschema ──────────────────────────────────────────────────────────
COLUMNS = [
    "Ticker","Bolagsnamn","Aktuell kurs","Valuta","Kategori",
    "Direktavkastning (%)","Utdelning/år","Utdelning/år (manuell)","Lås utdelning",
    "Frekvens/år","Utdelningsfrekvens","Payment-lag (dagar)","Ex-Date","Nästa utbetalning (est)",
    "Antal aktier","GAV",
    "Portföljandel (%)","Årlig utdelning (SEK)","Kurs (SEK)","Utdelningstillväxt (%)",
    "Utdelningskälla","Senaste uppdatering","Källa",
    "Marknadsvärde (SEK)","Insatt (SEK)","Orealiserad P/L (SEK)","Orealiserad P/L (%)"
]

# Vilka kolumner ska vara numeriska i Sheets
NUMERIC_COLS = [
    "Aktuell kurs","Direktavkastning (%)","Utdelning/år","Utdelning/år (manuell)",
    "Frekvens/år","Payment-lag (dagar)","Antal aktier","GAV",
    "Portföljandel (%)","Årlig utdelning (SEK)","Kurs (SEK)","Utdelningstillväxt (%)",
    "Marknadsvärde (SEK)","Insatt (SEK)","Orealiserad P/L (SEK)","Orealiserad P/L (%)"
]

# ── Robust numerik: in & ut ur Sheets ─────────────────────────────────────
_decimal_rgx = re.compile(r"[^0-9\-\.\,]")

def _parse_float(x) -> float:
    """Robust parser: accepterar 1 234,56 / '14.47.02' / '09.30' / NaN etc."""
    if x is None:
        return 0.0
    if isinstance(x, (int, float, np.number)):
        try:
            if np.isfinite(x):
                return float(x)
            return 0.0
        except Exception:
            return 0.0
    s = str(x).strip()
    if s == "" or s.lower() in ("nan","none","inf","-inf"):
        return 0.0
    s = s.replace(" ", "")
    s = _decimal_rgx.sub("", s)          # ta bort konstiga tecken
    if s.count(",") > 0 and s.count(".") == 0:
        s = s.replace(",", ".")          # komma → punkt
    if s.count(".") > 1:
        # tolka tidsliknande "14.47.02" som "14.47"
        first = s.split(".")[0]
        rest  = s.split(".")[1]
        s = f"{first}.{rest}"
    try:
        v = float(s)
        if np.isfinite(v):
            return v
        return 0.0
    except Exception:
        return 0.0

def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in NUMERIC_COLS:
        if c in d.columns:
            d[c] = d[c].apply(_parse_float)
    return d

def sanitize_for_sheets(df: pd.DataFrame) -> pd.DataFrame:
    """Gör DataFrame helt 'Sheets-safe': korrekta numeriska typer, NaN→0,
    inga oändligheter, strängar trimmas."""
    d = df.copy()
    # säkerställ alla kolumner finns
    for c in COLUMNS:
        if c not in d.columns:
            d[c] = "" if c not in NUMERIC_COLS else 0.0
    # numeriska kolumner till float, NaN/inf -> 0
    d = coerce_numeric_columns(d)
    for c in NUMERIC_COLS:
        d[c] = d[c].astype(float)
        d[c] = d[c].replace([np.inf, -np.inf, np.nan], 0.0)
    # textkolumner trimmas
    for c in d.columns:
        if c not in NUMERIC_COLS:
            d[c] = d[c].astype(str).fillna("").str.strip()
    # håll kolumnordning
    return d[COLUMNS].copy()

# ── Google Sheets helpers (med RAW-skrivning & backup) ─────────────────────
def _open_sheet():
    return client.open_by_url(SHEET_URL)

def skapa_koppling():
    return _open_sheet().worksheet(SHEET_NAME)

def backup_sheet():
    try:
        sh = _open_sheet()
        title = f"{BACKUP_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M')}"
        if any(ws.title == title for ws in sh.worksheets()):
            return
        ws = skapa_koppling()
        data = [ws.row_values(i) for i in range(1, ws.row_count + 1)]
        # skapa backup-flik och skriv HEAD + DATA
        b = sh.add_worksheet(title=title, rows=max(len(data), 2), cols=max(len(data[0]) if data else 2, 2))
        if data:
            b.update(data, value_input_option="RAW")
    except Exception:
        pass  # backup får inte stoppa appen

def spara_data(df: pd.DataFrame):
    d = sanitize_for_sheets(df)
    # skydd: spara aldrig helt tomt med befintligt ark
    if d["Ticker"].astype(str).str.strip().eq("").all():
        st.error("Avbröt sparning: inga tickers (skydd mot att råka tömma arket).")
        return
    ws = skapa_koppling()
    # skriv i ett svep med RAW så Google inte formaterar om
    values = [d.columns.tolist()] + d.values.tolist()
    ws.clear()
    ws.update(values, value_input_option="RAW")

def hamta_data() -> pd.DataFrame:
    try:
        ws = skapa_koppling()
        data = ws.get_all_records()
        raw = pd.DataFrame(data)
        if raw.empty:
            return säkerställ_kolumner(pd.DataFrame())
        # Säkerställ kolumner + numerik coercion
        d = säkerställ_kolumner(raw)
        d = coerce_numeric_columns(d)
        return d
    except Exception as e:
        st.warning(f"Kunde inte läsa Google Sheet just nu: {e}")
        return säkerställ_kolumner(pd.DataFrame())

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        d = pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in d.columns:
            d[c] = "" if c not in NUMERIC_COLS else 0.0
    # typer
    d["Ticker"] = d["Ticker"].astype(str).str.strip().str.upper()
    d["Valuta"] = d["Valuta"].astype(str).str.strip().str.upper()
    d["Kategori"] = d["Kategori"].astype(str).replace({"": "QUALITY"})
    # booleans
    if "Lås utdelning" in d.columns:
        d["Lås utdelning"] = d["Lås utdelning"].apply(lambda x: bool(x) if pd.notna(x) else False)
    else:
        d["Lås utdelning"] = False
    # numerik
    d = coerce_numeric_columns(d)
    return d[COLUMNS].copy()

# ── FX (standardvärden + sidopanelstyrning) ───────────────────────────────
DEF_FX = {"USDSEK": 9.60, "NOKSEK": 0.94, "CADSEK": 6.95, "EURSEK": 11.10}
for k, v in DEF_FX.items():
    st.session_state.setdefault(k, v)

def fx_for(cur: str) -> float:
    m = {
        "USD": st.session_state.get("USDSEK", DEF_FX["USDSEK"]),
        "EUR": st.session_state.get("EURSEK", DEF_FX["EURSEK"]),
        "CAD": st.session_state.get("CADSEK", DEF_FX["CADSEK"]),
        "NOK": st.session_state.get("NOKSEK", DEF_FX["NOKSEK"]),
        "SEK": 1.0
    }
    return float(m.get((cur or "SEK").upper(), 1.0))

# ── Yahoo Finance-hämtning (pris, valuta, utdelning, frekvens, ex-date) ───
def hamta_yahoo_data(ticker: str):
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.get_info() or {}
        except Exception:
            try:
                info = t.info or {}
            except Exception:
                info = {}

        # Pris
        price = None
        try:
            price = t.fast_info.get("last_price")
        except Exception:
            pass
        if price in (None, ""):
            price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price in (None, ""):
            try:
                h = t.history(period="5d")
                if not h.empty:
                    price = float(h["Close"].iloc[-1])
            except Exception:
                price = None
        price = _parse_float(price)

        # Valuta
        currency = (info.get("currency") or "").upper()
        if not currency:
            try:
                currency = (t.fast_info.get("currency") or "").upper()
            except Exception:
                currency = "SEK"

        # Utdelningshistorik → 12m-summa + frekvens
        div_rate = 0.0
        freq = 0
        freq_text = "Oregelbunden"
        ex_date_str = ""
        try:
            divs = t.dividends
            if divs is not None and not divs.empty:
                cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
                last12 = divs[divs.index >= cutoff]
                div_rate = _parse_float(last12.sum()) if not last12.empty else 0.0
                cnt = int(last12.shape[0]) if not last12.empty else 0
                if cnt >= 10:
                    freq, freq_text = 12, "Månads"
                elif cnt >= 3:
                    freq, freq_text = 4, "Kvartals"
                elif cnt == 2:
                    freq, freq_text = 2, "Halvårs"
                elif cnt == 1:
                    freq, freq_text = 1, "Års"
                else:
                    freq, freq_text = 0, "Oregelbunden"
                ex_date_str = pd.to_datetime(divs.index.max()).strftime("%Y-%m-%d")
        except Exception:
            pass

        # fallback rates
        if div_rate == 0.0:
            for k in ("forwardAnnualDividendRate", "trailingAnnualDividendRate"):
                try:
                    v = info.get(k)
                    if v not in (None, "", 0):
                        div_rate = _parse_float(v)
                        if div_rate > 0:
                            break
                except Exception:
                    pass

        # ex-date fallback
        if not ex_date_str:
            try:
                ts = info.get("exDividendDate")
                if ts not in (None, "", 0):
                    ex_date_str = pd.to_datetime(int(ts), unit="s", utc=True).strftime("%Y-%m-%d")
            except Exception:
                ex_date_str = ""

        return {
            "kurs": price,
            "valuta": currency,
            "utdelning": div_rate,
            "frekvens": int(freq),
            "frekvens_text": freq_text,
            "ex_date": ex_date_str,
            "källa": "Yahoo",
            "uppdaterad": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    except Exception as e:
        st.warning(f"Kunde inte hämta Yahoo-data för {ticker}: {e}")
        return None

# ── Beräkningar ────────────────────────────────────────────────────────────
def beräkna_allt(df: pd.DataFrame) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()

    # Effektiv utdelning (manuell låsning prioriteras)
    use_manual = (d["Lås utdelning"] == True) & (d["Utdelning/år (manuell)"] > 0)
    d["Utdelning/år_eff"] = d["Utdelning/år"].where(~use_manual, d["Utdelning/år (manuell)"])
    d["Utdelning/år_eff"] = d["Utdelning/år_eff"].apply(_parse_float)

    # Källa-indikator
    d["Utdelningskälla"] = np.where(use_manual, "Manuell 🔒", "Yahoo")

    # Kurs (SEK)
    d["Aktuell kurs"] = d["Aktuell kurs"].apply(_parse_float)
    d["Kurs (SEK)"]    = (d["Aktuell kurs"] * d["Valuta"].map(fx_for)).round(6)

    # Antal, GAV (lokal valuta)
    d["Antal aktier"] = d["Antal aktier"].apply(_parse_float)
    d["GAV"]          = d["GAV"].apply(_parse_float)

    # Utdelning / DA
    d["Årlig utdelning (SEK)"] = (d["Antal aktier"] * d["Utdelning/år_eff"] * d["Valuta"].map(fx_for)).round(2)
    ok = (d["Aktuell kurs"] > 0) & (d["Utdelning/år_eff"] > 0)
    d["Direktavkastning (%)"] = 0.0
    d.loc[ok, "Direktavkastning (%)"] = (100.0 * d.loc[ok,"Utdelning/år_eff"] / d.loc[ok,"Aktuell kurs"]).round(2)

    # Marknadsvärde / Portföljandel / Insatt & P/L
    d["Marknadsvärde (SEK)"] = (d["Antal aktier"] * d["Kurs (SEK)"]).round(2)
    tot_mv = float(d["Marknadsvärde (SEK)"].sum()) or 1.0
    d["Portföljandel (%)"]   = (100.0 * d["Marknadsvärde (SEK)"] / tot_mv).round(2)
    d["Insatt (SEK)"]        = (d["Antal aktier"] * d["GAV"] * d["Valuta"].map(fx_for)).round(2)
    d["Orealiserad P/L (SEK)"] = (d["Marknadsvärde (SEK)"] - d["Insatt (SEK)"]).round(2)
    d["Orealiserad P/L (%)"]   = np.where(d["Insatt (SEK)"] > 0,
                                          (100.0 * d["Orealiserad P/L (SEK)"] / d["Insatt (SEK)"]).round(2),
                                          0.0)

    # Defaults för tidskolumner
    d["Frekvens/år"] = d["Frekvens/år"].replace(0, 4).apply(_parse_float).clip(lower=1)
    d["Payment-lag (dagar)"] = d["Payment-lag (dagar)"].replace(0, 30).apply(_parse_float).clip(lower=0)
    return d

def _gen_next_payment(ex_date_str, freq_per_year, payment_lag_days):
    ts = pd.to_datetime(ex_date_str, errors="coerce")
    if pd.isna(ts): return ""
    exd = ts.date()
    try: freq = int(float(freq_per_year))
    except: freq = 4
    freq = max(freq, 1)
    try: lag = int(float(payment_lag_days))
    except: lag = 30
    step_days = max(1, int(round(365.0 / freq)))
    while exd < date.today():
        exd = exd + timedelta(days=step_days)
    pay_date = exd + timedelta(days=lag)
    return pay_date.strftime("%Y-%m-%d")

def uppdatera_nästa_utd(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Nästa utbetalning (est)"] = [
        _gen_next_payment(d.at[i,"Ex-Date"], d.at[i,"Frekvens/år"], d.at[i,"Payment-lag (dagar)"])
        for i in d.index
    ]
    return d

# ── Regler ─────────────────────────────────────────────────────────────────
GLOBAL_MAX_NAME_DEFAULT = 12.0  # standard max per bolag
MAX_CAT = {
    "QUALITY": 40.0, "REIT": 25.0, "mREIT": 10.0, "BDC": 20.0, "Shipping": 25.0,
    "Bank": 20.0, "Telecom": 20.0, "Finance": 20.0, "Tech": 25.0, "Other": 10.0
}
CATEGORY_CHOICES = list(MAX_CAT.keys())

def get_cat_max(cat: str) -> float:
    return float(MAX_CAT.get((cat or "QUALITY").strip(), 100.0))

# ── Trading avgifter ───────────────────────────────────────────────────────
MIN_COURTAGE_RATE = 0.0025
MIN_COURTAGE_SEK  = 1.0
FX_FEE_RATE       = 0.0025

def is_foreign(ccy: str) -> bool:
    return (ccy or "SEK").upper() != "SEK"

def calc_fees(order_value_sek: float, foreign: bool):
    courtage = max(MIN_COURTAGE_RATE * order_value_sek, MIN_COURTAGE_SEK)
    fx_fee   = (FX_FEE_RATE * order_value_sek) if foreign else 0.0
    total    = round(courtage + fx_fee, 2)
    return round(courtage,2), round(fx_fee,2), total

# ── Köpförslag (enligt tidigare logik) ─────────────────────────────────────
def _cap_shares_by_weight_limit(Vi, T, price_sek, max_pct):
    if price_sek <= 0: return 0
    m = max_pct/100.0; numer = m*T - Vi; denom = (1.0 - m) * price_sek
    if denom <= 0: return 0
    return int(max(0, math.floor(numer/denom)))

def _cap_shares_by_category(C, T, price_sek, cat_max_pct):
    if price_sek <= 0: return 0
    M = cat_max_pct/100.0; numer = M*T - C; denom = (1.0 - M) * price_sek
    if denom <= 0: return 0
    return int(max(0, math.floor(numer/denom)))

def suggest_buys(df: pd.DataFrame, cash_sek: float, per_trade_budget: float,
                 w_val=0.5, w_under=0.35, w_time=0.15, topk=5,
                 global_max_pct: float = None) -> pd.DataFrame:
    d = uppdatera_nästa_utd(beräkna_allt(df).copy())
    if d.empty:
        return pd.DataFrame(columns=["Ticker","Kategori","Poäng","DA %","Vikt %","Nästa utb","Föreslagna st","Kostnad ca","Motivering"])
    T = float(d["Marknadsvärde (SEK)"].sum())
    d["Kategori"] = d["Kategori"].astype(str).replace({"": "QUALITY"})
    cat_values = d.groupby("Kategori", as_index=False)["Marknadsvärde (SEK)"].sum().set_index("Kategori")["Marknadsvärde (SEK)"].to_dict()

    da = d["Direktavkastning (%)"].astype(float)
    da_score = (da.clip(lower=0, upper=15) / 15.0) * 100.0
    max_name = GLOBAL_MAX_NAME_DEFAULT if global_max_pct is None else float(global_max_pct)
    under = (max_name - d["Portföljandel (%)"]).clip(lower=0)
    under_score = (under / max_name) * 100.0

    def _days_to(s):
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt): return 9999
        return max(0, (dt.date() - date.today()).days)
    days = d["Nästa utbetalning (est)"].apply(_days_to)
    time_score = ((90 - days.clip(upper=90)) / 90.0).clip(lower=0) * 100.0

    totw = max(1e-9, (w_val + w_under + w_time))
    total_score = (w_val/totw)*da_score + (w_under/totw)*under_score + (w_time/totw)*time_score
    order = total_score.sort_values(ascending=False).index

    rows, used = [], 0.0
    for i in order:
        tkr = d.at[i,"Ticker"]
        price = float(d.at[i,"Kurs (SEK)"])
        if price <= 0: continue
        cat = str(d.at[i,"Kategori"]) or "QUALITY"
        Vi = float(d.at[i,"Marknadsvärde (SEK)"]); C = float(cat_values.get(cat, 0.0))
        foreign = (d.at[i,"Valuta"] or "SEK").upper() != "SEK"

        n_name_cap = _cap_shares_by_weight_limit(Vi, T, price, max_name)
        n_cat_cap  = _cap_shares_by_category(C, T, price, get_cat_max(cat))

        # budget per trade ≈ 500kr (avrundad uppåt)
        target_gross = max(1.0, float(per_trade_budget))
        n_budget = max(0, math.ceil(target_gross / price))
        n = min(n_name_cap, n_cat_cap, n_budget)
        if n <= 0: continue

        gross = price * n
        c, fx, tot = calc_fees(gross, foreign)
        cost = round(gross + tot, 2)
        if used + cost > cash_sek + 1e-9:
            continue

        rows.append({
            "Ticker": tkr, "Kategori": cat, "Poäng": round(float(total_score.at[i]),1),
            "DA %": round(float(da.at[i]),2), "Vikt %": float(d.at[i,"Portföljandel (%)"]),
            "Nästa utb": d.at[i,"Nästa utbetalning (est)"], "Föreslagna st": int(n),
            "Kostnad ca": cost,
            "Motivering": f"Undervikt vs {max_name:.0f}% & kategori≤{get_cat_max(cat):.0f}%"
        })
        used += cost; Vi += gross; C += gross; T += gross; cat_values[cat] = C
        if used >= cash_sek - 1e-9 or len(rows) >= topk:
            break

    cols = ["Ticker","Kategori","Poäng","DA %","Vikt %","Nästa utb","Föreslagna st","Kostnad ca","Motivering"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)

# ── Sidopanel (FX + max per bolag) ─────────────────────────────────────────
def sidopanel():
    st.sidebar.header("⚙️ Inställningar")
    st.sidebar.markdown("**Växelkurser (SEK)**")
    colA, colB = st.sidebar.columns(2)
    with colA:
        usd = st.number_input("USD/SEK", min_value=0.0, value=float(st.session_state["USDSEK"]), step=0.01, format="%.4f")
        eur = st.number_input("EUR/SEK", min_value=0.0, value=float(st.session_state["EURSEK"]), step=0.01, format="%.4f")
    with colB:
        cad = st.number_input("CAD/SEK", min_value=0.0, value=float(st.session_state["CADSEK"]), step=0.01, format="%.4f")
        nok = st.number_input("NOK/SEK", min_value=0.0, value=float(st.session_state["NOKSEK"]), step=0.01, format="%.4f")
    st.session_state["USDSEK"], st.session_state["EURSEK"], st.session_state["CADSEK"], st.session_state["NOKSEK"] = usd, eur, cad, nok

    st.sidebar.markdown("---")
    if "global_max_name" not in st.session_state:
        st.session_state["global_max_name"] = GLOBAL_MAX_NAME_DEFAULT
    st.session_state["global_max_name"] = st.sidebar.number_input("Max per bolag (%)", min_value=1.0, max_value=40.0,
                                                                  value=float(st.session_state["global_max_name"]), step=0.5)

# ── Lägg till / uppdatera bolag ───────────────────────────────────────────
def lagg_till_eller_uppdatera(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("➕ Lägg till / ✏ Uppdatera bolag")
    tickers = ["Ny"] + sorted(df["Ticker"].unique().tolist())
    val = st.selectbox("Välj bolag", options=tickers)

    if val == "Ny":
        ticker = st.text_input("Ticker").strip().upper()
        antal = st.number_input("Antal aktier", min_value=0.0, value=0.0, step=1.0)
        gav   = st.number_input("GAV (i bolagets valuta)", min_value=0.0, value=0.0, step=0.01)
        kategori = st.selectbox("Kategori", options=CATEGORY_CHOICES, index=CATEGORY_CHOICES.index("QUALITY"))
        lås = st.checkbox("Lås utdelning (använd manuell om >0)", value=False)
        man_utd = st.number_input("Utdelning/år (manuell)", min_value=0.0, value=0.0, step=0.01)
    else:
        rad = df[df["Ticker"] == val].iloc[0]
        ticker = st.text_input("Ticker", value=rad["Ticker"]).strip().upper()
        antal  = st.number_input("Antal aktier", min_value=0.0, value=float(rad["Antal aktier"]), step=1.0)
        gav    = st.number_input("GAV (i bolagets valuta)", min_value=0.0, value=float(rad["GAV"]), step=0.01)
        kategori = st.selectbox("Kategori", options=CATEGORY_CHOICES,
                                index=CATEGORY_CHOICES.index(rad.get("Kategori","QUALITY")) if rad.get("Kategori","QUALITY") in CATEGORY_CHOICES else 0)
        lås = st.checkbox("Lås utdelning (använd manuell om >0)", value=bool(rad.get("Lås utdelning", False)))
        man_utd = st.number_input("Utdelning/år (manuell)", min_value=0.0, value=float(rad.get("Utdelning/år (manuell)", 0.0)), step=0.01)

    if st.button("💾 Spara bolag"):
        if not ticker:
            st.error("Ticker måste anges.")
            return df
        data_y = hamta_yahoo_data(ticker)
        if (df["Ticker"] == ticker).any():
            m = df["Ticker"] == ticker
        else:
            df = pd.concat([df, pd.DataFrame([{"Ticker": ticker}])], ignore_index=True)
            m = df["Ticker"] == ticker

        df.loc[m, "Antal aktier"] = _parse_float(antal)
        df.loc[m, "GAV"] = _parse_float(gav)
        df.loc[m, "Kategori"] = kategori
        df.loc[m, "Utdelning/år (manuell)"] = _parse_float(man_utd)
        df.loc[m, "Lås utdelning"] = bool(lås)

        if data_y:
            if _parse_float(data_y.get("utdelning")) > 0 and not lås:
                df.loc[m, "Utdelning/år"] = _parse_float(data_y["utdelning"])
            df.loc[m, "Aktuell kurs"] = _parse_float(data_y.get("kurs"))
            if data_y.get("valuta"): df.loc[m, "Valuta"] = data_y["valuta"]
            if data_y.get("frekvens", 0) > 0: df.loc[m, "Frekvens/år"] = int(data_y["frekvens"])
            if data_y.get("frekvens_text"): df.loc[m, "Utdelningsfrekvens"] = data_y["frekvens_text"]
            if data_y.get("ex_date"): df.loc[m, "Ex-Date"] = data_y["ex_date"]
            df.loc[m, "Källa"] = "Yahoo"
            if data_y.get("uppdaterad"): df.loc[m, "Senaste uppdatering"] = data_y["uppdaterad"]

        d = uppdatera_nästa_utd(beräkna_allt(df))
        spara_data(d)
        st.success(f"{ticker} sparad!")
        return d
    return df

# ── Uppdatera enskilt / massuppdatera ─────────────────────────────────────
def uppdatera_bolag(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🔄 Uppdatera enskilt bolag")
    if df.empty:
        st.info("Ingen data att uppdatera."); return df
    val = st.selectbox("Välj bolag", options=sorted(df["Ticker"].tolist()))
    if st.button("Uppdatera från Yahoo"):
        data_y = hamta_yahoo_data(val)
        if data_y:
            m = df["Ticker"] == val
            if _parse_float(data_y.get("utdelning")) > 0 and not bool(df.loc[m,"Lås utdelning"].iloc[0]):
                df.loc[m, "Utdelning/år"] = _parse_float(data_y["utdelning"])
            df.loc[m, "Aktuell kurs"] = _parse_float(data_y.get("kurs"))
            if data_y.get("valuta"): df.loc[m, "Valuta"] = data_y["valuta"]
            if data_y.get("frekvens",0) > 0: df.loc[m, "Frekvens/år"] = int(data_y["frekvens"])
            if data_y.get("frekvens_text"): df.loc[m, "Utdelningsfrekvens"] = data_y["frekvens_text"]
            if data_y.get("ex_date"): df.loc[m, "Ex-Date"] = data_y["ex_date"]
            df.loc[m, "Källa"] = "Yahoo"
            if data_y.get("uppdaterad"): df.loc[m, "Senaste uppdatering"] = data_y["uppdaterad"]
            d = uppdatera_nästa_utd(beräkna_allt(df))
            spara_data(d); st.success(f"{val} uppdaterad!")
            return d
        else:
            st.warning("Kunde inte hämta data.")
    return df

def massuppdatera(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("⏩ Massuppdatera alla (med progress)")
    if df.empty:
        st.info("Ingen data att uppdatera."); return df
    if st.button("Starta massuppdatering"):
        prog = st.progress(0, text="Hämtar från Yahoo…")
        total = len(df)
        for i, tkr in enumerate(df["Ticker"].tolist(), start=1):
            data_y = hamta_yahoo_data(tkr)
            if data_y:
                m = df["Ticker"] == tkr
                if _parse_float(data_y.get("utdelning")) > 0 and not bool(df.loc[m,"Lås utdelning"].iloc[0]):
                    df.loc[m, "Utdelning/år"] = _parse_float(data_y["utdelning"])
                df.loc[m, "Aktuell kurs"] = _parse_float(data_y.get("kurs"))
                if data_y.get("valuta"): df.loc[m, "Valuta"] = data_y["valuta"]
                if data_y.get("frekvens",0) > 0: df.loc[m, "Frekvens/år"] = int(data_y["frekvens"])
                if data_y.get("frekvens_text"): df.loc[m, "Utdelningsfrekvens"] = data_y["frekvens_text"]
                if data_y.get("ex_date"): df.loc[m, "Ex-Date"] = data_y["ex_date"]
                df.loc[m, "Källa"] = "Yahoo"
                if data_y.get("uppdaterad"): df.loc[m, "Senaste uppdatering"] = data_y["uppdaterad"]
            prog.progress(i/total, text=f"{tkr} ({i}/{total})")
            time.sleep(1.0)
        d = uppdatera_nästa_utd(beräkna_allt(df))
        spara_data(d); st.success("Massuppdatering klar!")
        return d
    return df

# ── Portföljvy ────────────────────────────────────────────────────────────
def page_portfolio(df: pd.DataFrame):
    st.subheader("📦 Portföljöversikt")
    d = uppdatera_nästa_utd(beräkna_allt(df).copy())
    tot_mv  = float(d["Marknadsvärde (SEK)"].sum())
    tot_ins = float(d["Insatt (SEK)"].sum())
    tot_pl  = float(d["Orealiserad P/L (SEK)"].sum())
    tot_div = float(d["Årlig utdelning (SEK)"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portföljvärde (SEK)", f"{round(tot_mv,2):,}".replace(",", " "))
    c2.metric("Insatt (SEK)", f"{round(tot_ins,2):,}".replace(",", " "))
    c3.metric("Orealiserad P/L (SEK)", f"{round(tot_pl,2):,}".replace(",", " "),
              delta=f"{(0 if tot_ins==0 else (100*tot_pl/tot_ins)):.2f}%")
    c4.metric("Årlig utdelning (SEK)", f"{round(tot_div,2):,}".replace(",", " "))

    show_cols = [
        "Ticker","Bolagsnamn","Valuta","Kategori","Aktuell kurs","Kurs (SEK)",
        "Antal aktier","GAV","Insatt (SEK)","Marknadsvärde (SEK)",
        "Orealiserad P/L (SEK)","Orealiserad P/L (%)",
        "Direktavkastning (%)","Utdelning/år","Utdelning/år (manuell)","Lås utdelning","Utdelningskälla",
        "Utdelningsfrekvens","Frekvens/år","Årlig utdelning (SEK)",
        "Ex-Date","Nästa utbetalning (est)","Portföljandel (%)","Senaste uppdatering"
    ]
    st.dataframe(d[show_cols], use_container_width=True, hide_index=True)

# ── Köpförslag ────────────────────────────────────────────────────────────
def page_buy(df: pd.DataFrame):
    st.subheader("🎯 Köpförslag")
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        cash = st.number_input("Tillgänglig kassa (SEK)", min_value=0.0, value=500.0, step=100.0)
    with c2:
        per_trade = st.number_input("Mål per köp (SEK)", min_value=100.0, value=500.0, step=50.0)
    with c3:
        w_under = st.slider("Vikt undervikt", 0.0, 1.0, 0.35, 0.05)
    with c4:
        w_time = st.slider("Vikt timing", 0.0, 1.0, 0.15, 0.05)
    w_val = 1.0 - (w_under + w_time)
    if st.button("Beräkna"):
        sug = suggest_buys(df, cash, per_trade, w_val=w_val, w_under=w_under, w_time=w_time,
                           topk=10, global_max_pct=st.session_state.get("global_max_name", GLOBAL_MAX_NAME_DEFAULT))
        if sug.empty:
            st.info("Inga kandidater ryms inom regler och kassa.")
        else:
            st.dataframe(sug, use_container_width=True)

# ── Spara ─────────────────────────────────────────────────────────────────
def page_save(df: pd.DataFrame):
    st.subheader("💾 Spara till Google Sheets")
    preview = sanitize_for_sheets(uppdatera_nästa_utd(beräkna_allt(df)))
    st.write("Antal rader som sparas:", len(preview))
    st.dataframe(preview, use_container_width=True)
    if st.button("✅ Bekräfta och spara (med backup)"):
        backup_sheet()
        spara_data(preview)
        st.success("Sparat.")

# ── Main/router ────────────────────────────────────────────────────────────
def main():
    st.title("Relative Yield – utdelningsportfölj")

    # Ladda (säker) data en gång
    if "working_df" not in st.session_state:
        st.session_state["working_df"] = hamta_data()
    base = säkerställ_kolumner(st.session_state["working_df"])

    sidopanel()

    page = st.sidebar.radio(
        "Meny",
        [
            "➕ Lägg till / ✏ Uppdatera bolag",
            "🔄 Uppdatera EN",
            "⏩ Massuppdatera alla",
            "📦 Portföljöversikt",
            "🎯 Köpförslag",
            "💾 Spara",
        ],
        index=3
    )

    if page == "➕ Lägg till / ✏ Uppdatera bolag":
        base = lagg_till_eller_uppdatera(base)
    elif page == "🔄 Uppdatera EN":
        base = uppdatera_bolag(base)
    elif page == "⏩ Massuppdatera alla":
        base = massuppdatera(base)
    elif page == "📦 Portföljöversikt":
        page_portfolio(base)
    elif page == "🎯 Köpförslag":
        page_buy(base)
    elif page == "💾 Spara":
        page_save(base)

    # håll i session
    st.session_state["working_df"] = säkerställ_kolumner(base)

if __name__ == "__main__":
    main()
