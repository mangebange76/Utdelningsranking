import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time
import math
from datetime import datetime, timedelta, date
from google.oauth2.service_account import Credentials

# ── Streamlit rerun shim ─────────────────────────────────────────────────────
try:
    _rerun = st.rerun
except AttributeError:
    _rerun = st.experimental_rerun

st.set_page_config(page_title="Utdelningsranking", layout="wide")

# ── Google Sheets Setup ─────────────────────────────────────────────────────
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Bolag"

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
credentials = Credentials.from_service_account_info(
    st.secrets["GOOGLE_CREDENTIALS"], scopes=scope
)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def spara_data(df: pd.DataFrame):
    ws = skapa_koppling()
    ws.clear()
    ws.update([df.columns.tolist()] + df.astype(str).values.tolist(), value_input_option="USER_ENTERED")

def hamta_data():
    try:
        ws = skapa_koppling()
        data = ws.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Kunde inte läsa Google Sheet just nu: {e}")
        return pd.DataFrame()

def migrate_sheet_columns():
    raw = hamta_data()
    fixed = säkerställ_kolumner(raw)
    try:
        if raw.empty:
            spara_data(fixed)
        elif list(raw.columns) != list(fixed.columns) or raw.shape[1] != fixed.shape[1]:
            spara_data(fixed)
    except Exception as e:
        st.warning(f"Kunde inte synka kolumnlayout mot Google Sheet: {e}")
    return fixed

# ── Standard FX-kurser (kan ändras i sidopanelen) ────────────────────────────
DEF = {"USDSEK": 9.60, "NOKSEK": 0.94, "CADSEK": 6.95, "EURSEK": 11.10}
for k, v in DEF.items():
    if k not in st.session_state:
        st.session_state[k] = v

def fx_for(cur: str) -> float:
    if pd.isna(cur):
        return 1.0
    c = str(cur).strip().upper()
    m = {
        "USD": st.session_state.get("USDSEK", DEF["USDSEK"]),
        "EUR": st.session_state.get("EURSEK", DEF["EURSEK"]),
        "CAD": st.session_state.get("CADSEK", DEF["CADSEK"]),
        "NOK": st.session_state.get("NOKSEK", DEF["NOKSEK"]),
        "SEK": 1.0
    }
    return float(m.get(c, 1.0))

# ── Kolumnschema ────────────────────────────────────────────────────────────
COLUMNS = [
    "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Kategori",
    "Direktavkastning (%)", "Utdelning/år", "Utdelning/år (manuell)", "Lås utdelning",
    "Frekvens/år", "Utdelningsfrekvens",
    "Payment-lag (dagar)", "Ex-Date", "Nästa utbetalning (est)",
    "Antal aktier", "GAV", "Portföljandel (%)", "Årlig utdelning (SEK)",
    "Kurs (SEK)", "Utdelningstillväxt (%)", "Utdelningskälla",
    "Senaste uppdatering", "Källa"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        d = pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in d.columns:
            d[c] = ""
    # datatyper / defaults
    d["Ticker"] = d["Ticker"].astype(str).str.strip().str.upper()
    d["Valuta"] = d["Valuta"].astype(str).str.strip().str.upper()
    d["Kategori"] = d["Kategori"].astype(str).replace({"": "QUALITY"})
    num_cols = ["Aktuell kurs","Utdelning/år","Utdelning/år (manuell)","Frekvens/år","Payment-lag (dagar)","Antal aktier","GAV"]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    if "Lås utdelning" in d.columns:
        d["Lås utdelning"] = d["Lås utdelning"].apply(lambda x: bool(x) if pd.notna(x) else False)
    else:
        d["Lås utdelning"] = False
    d["Utdelningskälla"] = d.get("Utdelningskälla", "Yahoo")
    return d[COLUMNS].copy()

# ── Yahoo Finance-hämtning ──────────────────────────────────────────────────
def hamta_yahoo_data(ticker: str):
    """Hämtar kurs, valuta, utdelning, frekvens m.m. från Yahoo Finance"""
    try:
        yf_tkr = yf.Ticker(ticker)
        info = yf_tkr.info

        kurs = info.get("currentPrice") or info.get("regularMarketPrice")
        valuta = info.get("currency", "")
        utdelning = info.get("dividendRate", 0.0)
        utd_frek = info.get("dividendFrequency", None)

        if not utd_frek:
            # Gissa utdelningsfrekvens
            try:
                cal = yf_tkr.dividends
                if len(cal) >= 4:
                    utd_frek = 4
                elif len(cal) == 1:
                    utd_frek = 1
                elif len(cal) == 2:
                    utd_frek = 2
                else:
                    utd_frek = 0
            except:
                utd_frek = 0

        return {
            "kurs": kurs,
            "valuta": valuta,
            "utdelning": utdelning,
            "frekvens": utd_frek,
            "källa": "Yahoo"
        }
    except Exception as e:
        st.warning(f"Kunde inte hämta Yahoo-data för {ticker}: {e}")
        return None

# ── Beräkningar ────────────────────────────────────────────────────────────
def beräkna_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Omvandla kurs till SEK
    d["Kurs (SEK)"] = d.apply(lambda r: r["Aktuell kurs"] * fx_for(r["Valuta"]), axis=1)

    # Bestäm utdelning/år
    def choose_div(row):
        if row.get("Lås utdelning"):
            return row.get("Utdelning/år (manuell)", 0.0)
        if pd.notna(row.get("Utdelning/år (manuell)")) and row["Utdelning/år (manuell)"] > 0:
            return row["Utdelning/år (manuell)"]
        return row.get("Utdelning/år", 0.0)

    d["Utdelning_final"] = d.apply(choose_div, axis=1)

    # Direktavkastning (%)
    d["Direktavkastning (%)"] = d.apply(lambda r: (r["Utdelning_final"] / r["Aktuell kurs"] * 100) if r["Aktuell kurs"] > 0 else 0, axis=1)

    # Årlig utdelning (SEK)
    d["Årlig utdelning (SEK)"] = d.apply(lambda r: r["Utdelning_final"] * r["Antal aktier"] * fx_for(r["Valuta"]), axis=1)

    # Portföljandel (%)
    tot_val = d.apply(lambda r: r["Kurs (SEK)"] * r["Antal aktier"], axis=1).sum()
    d["Marknadsvärde (SEK)"] = d.apply(lambda r: r["Kurs (SEK)"] * r["Antal aktier"], axis=1)
    d["Portföljandel (%)"] = d["Marknadsvärde (SEK)"].apply(lambda v: (v / tot_val * 100) if tot_val > 0 else 0)

    return d

# ── Lägg till / uppdatera bolag ────────────────────────────────────────────
def lagg_till_eller_uppdatera(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("➕ Lägg till / ✏ Uppdatera bolag")
    tickers = ["Ny"] + sorted(df["Ticker"].unique().tolist())
    val = st.selectbox("Välj bolag", options=tickers)

    if val == "Ny":
        ticker = st.text_input("Ticker").strip().upper()
        antal = st.number_input("Antal aktier", min_value=0, value=0, step=1)
        gav = st.number_input("GAV", min_value=0.0, value=0.0, step=0.01)
        kategori = st.selectbox("Kategori", options=["BDC", "REIT", "Shipping", "Övrigt"])
        man_utd = st.number_input("Utdelning/år (manuell)", min_value=0.0, step=0.01)
    else:
        rad = df[df["Ticker"] == val].iloc[0]
        ticker = st.text_input("Ticker", value=rad["Ticker"]).strip().upper()
        antal = st.number_input("Antal aktier", min_value=0, value=int(rad["Antal aktier"]), step=1)
        gav = st.number_input("GAV", min_value=0.0, value=float(rad["GAV"]), step=0.01)
        kategori = st.selectbox("Kategori", options=["BDC", "REIT", "Shipping", "Övrigt"], index=["BDC","REIT","Shipping","Övrigt"].index(rad.get("Kategori", "Övrigt")))
        man_utd = st.number_input("Utdelning/år (manuell)", min_value=0.0, value=float(rad.get("Utdelning/år (manuell)", 0.0)), step=0.01)

    if st.button("💾 Spara bolag"):
        data_y = hamta_yahoo_data(ticker)
        if data_y:
            akt_kurs = data_y["kurs"] or 0.0
            valuta = data_y["valuta"] or ""
            utd_år = data_y["utdelning"] or 0.0
            frekvens = data_y["frekvens"] or 0
            källa = data_y["källa"]
        else:
            akt_kurs = 0.0
            valuta = ""
            utd_år = 0.0
            frekvens = 0
            källa = "Manuell"

        new_row = {
            "Ticker": ticker,
            "Antal aktier": antal,
            "GAV": gav,
            "Aktuell kurs": akt_kurs,
            "Valuta": valuta,
            "Utdelning/år": utd_år,
            "Utdelning/år (manuell)": man_utd,
            "Utdelningsfrekvens": frekvens,
            "Kategori": kategori,
            "Källa utdelning": källa
        }

        if val == "Ny":
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df.loc[df["Ticker"] == val, new_row.keys()] = new_row.values()

        df = beräkna_kolumner(df)
        spara_data(df)
        st.success(f"{ticker} sparad!")

    return df

# ── Uppdatera enskilt bolag från Yahoo ─────────────────────────────────────
def uppdatera_bolag(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🔄 Uppdatera enskilt bolag")
    if df.empty:
        st.info("Ingen data att uppdatera.")
        return df

    val = st.selectbox("Välj bolag", options=sorted(df["Ticker"].unique().tolist()))
    if st.button("Uppdatera från Yahoo"):
        data_y = hamta_yahoo_data(val)
        if data_y:
            df.loc[df["Ticker"] == val, "Aktuell kurs"] = data_y["kurs"]
            df.loc[df["Ticker"] == val, "Valuta"] = data_y["valuta"]
            df.loc[df["Ticker"] == val, "Utdelning/år"] = data_y["utdelning"]
            df.loc[df["Ticker"] == val, "Utdelningsfrekvens"] = data_y["frekvens"]
            df.loc[df["Ticker"] == val, "Källa utdelning"] = data_y["källa"]
            df = beräkna_kolumner(df)
            spara_data(df)
            st.success(f"{val} uppdaterad!")
        else:
            st.warning(f"Kunde inte hämta data för {val}")

    return df

# ── Massuppdatera alla bolag ──────────────────────────────────────────────
def massuppdatera(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("⏩ Massuppdatera alla bolag från Yahoo")
    if df.empty:
        st.info("Ingen data att uppdatera.")
        return df

    if st.button("Starta massuppdatering"):
        for i, ticker in enumerate(df["Ticker"].tolist(), start=1):
            st.write(f"Uppdaterar {ticker} ({i}/{len(df)})...")
            data_y = hamta_yahoo_data(ticker)
            if data_y:
                df.loc[df["Ticker"] == ticker, "Aktuell kurs"] = data_y["kurs"]
                df.loc[df["Ticker"] == ticker, "Valuta"] = data_y["valuta"]
                df.loc[df["Ticker"] == ticker, "Utdelning/år"] = data_y["utdelning"]
                df.loc[df["Ticker"] == ticker, "Utdelningsfrekvens"] = data_y["frekvens"]
                df.loc[df["Ticker"] == ticker, "Källa utdelning"] = data_y["källa"]
                df = beräkna_kolumner(df)
                time.sleep(1)

        spara_data(df)
        st.success("Massuppdatering klar!")

    return df

# ── Regler (max per innehav & kategori) ─────────────────────────────────────
GLOBAL_MAX_NAME = 12.0  # % per enskilt bolag

MAX_CAT = {
    "QUALITY": 40.0,
    "REIT": 25.0,
    "mREIT": 10.0,
    "BDC": 15.0,
    "MLP": 20.0,
    "Shipping": 20.0,
    "Telecom": 20.0,
    "Tobacco": 20.0,
    "Utility": 20.0,
    "Tech": 25.0,
    "Bank": 20.0,
    "Industrial": 20.0,
    "Energy": 25.0,
    "Finance": 20.0,
    "Other": 10.0,
}
CATEGORY_CHOICES = list(MAX_CAT.keys())

def get_cat_max(cat: str) -> float:
    return float(MAX_CAT.get(str(cat or "").strip() or "QUALITY", 100.0))

# ── Beräkna fulla kolumner (med källa-indikator) ───────────────────────────
def beräkna(df: pd.DataFrame) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()

    # Välj utdelning (manuell låst vinner)
    use_manual = (d["Lås utdelning"] == True) & (d["Utdelning/år (manuell)"] > 0)
    d["Utdelning/år_eff"] = pd.to_numeric(d["Utdelning/år"], errors="coerce").fillna(0.0)
    d.loc[use_manual, "Utdelning/år_eff"] = pd.to_numeric(d["Utdelning/år (manuell)"], errors="coerce").fillna(0.0)

    # Indikator: källa
    try:
        d["Utdelningskälla"] = [
            "Manuell 🔒" if (bool(d.at[i, "Lås utdelning"]) and float(d.at[i, "Utdelning/år (manuell)"]) > 0.0) else "Yahoo"
            for i in d.index
        ]
    except Exception:
        d["Utdelningskälla"] = "Yahoo"

    # Valuta & kurs
    d["Aktuell kurs"] = pd.to_numeric(d["Aktuell kurs"], errors="coerce").fillna(0.0)
    rates = d["Valuta"].apply(fx_for).astype(float)
    d["Kurs (SEK)"] = (d["Aktuell kurs"] * rates).round(6)

    # Storheter
    d["Antal aktier"] = pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0)
    d["Årlig utdelning (SEK)"] = (d["Antal aktier"] * d["Utdelning/år_eff"] * rates).round(2)

    d["Direktavkastning (%)"] = 0.0
    ok = (d["Aktuell kurs"] > 0) & (d["Utdelning/år_eff"] > 0)
    d.loc[ok, "Direktavkastning (%)"] = (100.0 * d.loc[ok, "Utdelning/år_eff"] / d.loc[ok, "Aktuell kurs"]).round(2)

    mv = (d["Antal aktier"] * d["Kurs (SEK)"]).round(2)
    tot_mv = float(mv.sum()) if mv.sum() else 1.0
    d["Marknadsvärde (SEK)"] = mv
    d["Portföljandel (%)"] = (100.0 * mv / tot_mv).round(2)

    # Frekvens & nästa (fallbacks)
    d["Frekvens/år"] = pd.to_numeric(d["Frekvens/år"], errors="coerce").fillna(0.0).replace(0, 4)
    d["Payment-lag (dagar)"] = pd.to_numeric(d["Payment-lag (dagar)"], errors="coerce").fillna(0.0).replace(0, 30)

    def next_pay(ex_date_str, freq_per_year, payment_lag_days):
        ts = pd.to_datetime(ex_date_str, errors="coerce")
        if pd.isna(ts):
            return ""
        exd = ts.date()
        try: freq = int(float(freq_per_year))
        except: freq = 4
        try: lag = int(float(payment_lag_days))
        except: lag = 30
        freq = max(freq, 1)
        step_days = max(1, int(round(365.0 / freq)))
        today_d = date.today()
        while exd < today_d:
            exd = exd + timedelta(days=step_days)
        pay_date = exd + timedelta(days=lag)
        return pay_date.strftime("%Y-%m-%d")

    d["Nästa utbetalning (est)"] = [
        next_pay(d.at[i, "Ex-Date"], d.at[i, "Frekvens/år"], d.at[i, "Payment-lag (dagar)"]) for i in d.index
    ]
    return d

# ── Trim-förslag (>12 %) ───────────────────────────────────────────────────
def trim_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    d = beräkna(df).copy()
    if d.empty:
        return pd.DataFrame(columns=["Ticker","Vikt (%)","Kurs (SEK)","Föreslagen sälj (st)","Nettolikvid ca (SEK)"])

    T = float(d["Marknadsvärde (SEK)"].sum())
    if T <= 0:
        return pd.DataFrame(columns=["Ticker","Vikt (%)","Kurs (SEK)","Föreslagen sälj (st)","Nettolikvid ca (SEK)"])

    rows = []
    for _, r in d.iterrows():
        V = float(r["Marknadsvärde (SEK)"])
        w = 100.0 * V / T if T else 0.0
        if w <= GLOBAL_MAX_NAME + 1e-9:
            continue
        price = float(pd.to_numeric(r["Kurs (SEK)"], errors="coerce") or 0.0)
        qty   = float(pd.to_numeric(r["Antal aktier"], errors="coerce") or 0.0)
        if price <= 0 or qty <= 0:
            continue
        # n >= (V - mT) / ((1-m)P)
        n_min = (V - (GLOBAL_MAX_NAME/100.0)*T) / ((1.0 - GLOBAL_MAX_NAME/100.0) * price)
        n = max(0, math.ceil(n_min))
        n = int(min(n, qty))
        if n > 0:
            gross = round(price * n, 2)
            foreign = str(r.get("Valuta","SEK")).upper() != "SEK"
            fee_court, fee_fx, fee_tot = calc_fees(gross, foreign=True if foreign else False)
            net = round(gross - fee_tot, 2)
            rows.append({
                "Ticker": r["Ticker"],
                "Vikt (%)": round(w,2),
                "Kurs (SEK)": round(price,2),
                "Föreslagen sälj (st)": n,
                "Nettolikvid ca (SEK)": net,
                "Kommentar": f"Ner till {GLOBAL_MAX_NAME:.0f}%"
            })
    return pd.DataFrame(rows)

# ── Köpförslag (respekterar 12 % & kategori-tak & kassa) ───────────────────
def _max_affordable_shares(price_sek: float, cash_sek: float, foreign: bool) -> int:
    if price_sek <= 0 or cash_sek <= 0:
        return 0
    n_hi = int(cash_sek // price_sek)
    if n_hi <= 0:
        return 0
    for n in range(n_hi, 0, -1):
        gross = price_sek * n
        c, fx, tot = calc_fees(gross, foreign)
        if gross + tot <= cash_sek + 1e-9:
            return n
    return 0

def _cap_shares_by_weight_limit(Vi: float, T: float, price_sek: float, max_pct: float) -> int:
    if price_sek <= 0:
        return 0
    m = max_pct / 100.0
    numer = m * T - Vi
    denom = (1.0 - m) * price_sek
    if denom <= 0:
        return 0
    return int(max(0, math.floor(numer / denom)))

def _cap_shares_by_category(C: float, T: float, price_sek: float, cat_max_pct: float) -> int:
    if price_sek <= 0:
        return 0
    M = cat_max_pct / 100.0
    numer = M * T - C
    denom = (1.0 - M) * price_sek
    if denom <= 0:
        return 0
    return int(max(0, math.floor(numer / denom)))

def suggest_buys(df: pd.DataFrame, cash_sek: float, w_val: float=0.5, w_under: float=0.35, w_time: float=0.15, topk: int=5) -> pd.DataFrame:
    d = beräkna(df).copy()
    if d.empty or cash_sek <= 0:
        return pd.DataFrame(columns=["Ticker","Kategori","Poäng","DA %","Vikt %","Nästa utb","Föreslagna st","Kostnad ca","Motivering"])

    T = float(d["Marknadsvärde (SEK)"].sum())
    d["Kategori"] = d["Kategori"].astype(str).replace({"": "QUALITY"})

    # Kategori-summor nu
    cat_values = d.groupby("Kategori", as_index=False)["Marknadsvärde (SEK)"].sum().set_index("Kategori")["Marknadsvärde (SEK)"].to_dict()

    # Delpoäng
    da = pd.to_numeric(d["Direktavkastning (%)"], errors="coerce").fillna(0.0)
    da_score = (da.clip(lower=0, upper=15) / 15.0) * 100.0
    under = (GLOBAL_MAX_NAME - d["Portföljandel (%)"]).clip(lower=0)
    under_score = (under / GLOBAL_MAX_NAME) * 100.0

    def _days_to(date_str: str) -> int:
        try:
            dt = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(dt): return 9999
            return max(0, (dt.date() - date.today()).days)
        except Exception:
            return 9999

    days = d["Nästa utbetalning (est)"].apply(_days_to)
    time_score = ((90 - days.clip(upper=90)) / 90.0).clip(lower=0) * 100.0

    # Normalisera vikter
    totw = max(1e-9, (w_val + w_under + w_time))
    w_val, w_under, w_time = w_val/totw, w_under/totw, w_time/totw
    total_score = (w_val*da_score + w_under*under_score + w_time*time_score)

    order = total_score.sort_values(ascending=False).index
    rows, used = [], 0.0

    for i in order:
        tkr = d.at[i,"Ticker"]
        price = float(pd.to_numeric(d.at[i,"Kurs (SEK)"], errors="coerce") or 0.0)
        if price <= 0:
            continue
        cat = str(d.at[i,"Kategori"]) if str(d.at[i,"Kategori"]).strip() else "QUALITY"
        Vi = float(d.at[i,"Marknadsvärde (SEK)"])
        C  = float(cat_values.get(cat, 0.0))
        foreign = str(d.at[i,"Valuta"]).upper() != "SEK"

        n_name_cap = _cap_shares_by_weight_limit(Vi, T, price, GLOBAL_MAX_NAME)
        n_cat_cap  = _cap_shares_by_category(C, T, price, get_cat_max(cat))
        remaining_cash = max(0.0, cash_sek - used)
        n_cash_cap = _max_affordable_shares(price, remaining_cash, foreign)

        n = min(n_name_cap, n_cat_cap, n_cash_cap)
        if n <= 0:
            continue

        gross = price * n
        c, fx, tot = calc_fees(gross, foreign)
        cost = round(gross + tot, 2)
        rows.append({
            "Ticker": tkr,
            "Kategori": cat,
            "Poäng": round(float(total_score.at[i]), 1),
            "DA %": round(float(da.at[i]), 2),
            "Vikt %": float(d.at[i,"Portföljandel (%)"]),
            "Nästa utb": d.at[i,"Nästa utbetalning (est)"],
            "Föreslagna st": int(n),
            "Kostnad ca": cost,
            "Motivering": f"Undervikt vs {GLOBAL_MAX_NAME:.0f}% & kategori≤{get_cat_max(cat):.0f}%"
        })

        # uppdatera state
        used += cost
        Vi += gross
        C  += gross
        T  += gross
        cat_values[cat] = C

        if used >= cash_sek - 1e-9 or len(rows) >= topk:
            break

    cols = ["Ticker","Kategori","Poäng","DA %","Vikt %","Nästa utb","Föreslagna st","Kostnad ca","Motivering"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)

# ── Portföljöversikt (visa & redigera) ─────────────────────────────────────
def portfolj_oversikt(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("📦 Portföljöversikt")
    d = beräkna(df).copy()

    d["Insatt (SEK)"] = (pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0) * pd.to_numeric(d["GAV"], errors="coerce").fillna(0.0)).round(2)
    d["Orealiserad P/L (SEK)"] = (d["Marknadsvärde (SEK)"] - d["Insatt (SEK)"]).round(2)
    d["Orealiserad P/L (%)"] = (100.0 * d["Orealiserad P/L (SEK)"] / d["Insatt (SEK)"].replace({0: pd.NA})).fillna(0.0).round(2)

    tot_mv, tot_ins = float(d["Marknadsvärde (SEK)"].sum()), float(d["Insatt (SEK)"].sum())
    tot_pl, tot_div = float(d["Orealiserad P/L (SEK)"].sum()), float(d["Årlig utdelning (SEK)"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portföljvärde", f"{round(tot_mv,2):,}".replace(",", " "))
    c2.metric("Insatt kapital", f"{round(tot_ins,2):,}".replace(",", " "))
    c3.metric("Orealiserad P/L", f"{round(tot_pl,2):,}".replace(",", " "), delta=f"{(0 if tot_ins==0 else (100*tot_pl/tot_ins)):.2f}%")
    c4.metric("Årsutdelning", f"{round(tot_div,2):,}".replace(",", " "))

    locked_count = int(((d["Lås utdelning"] == True) & (d["Utdelning/år (manuell)"] > 0)).sum())
    if locked_count > 0:
        st.caption(f"🔒 {locked_count} bolag använder **manuellt låst** utdelning just nu.")

    show_cols = [
        "Ticker","Bolagsnamn","Valuta","Kategori","Aktuell kurs","Kurs (SEK)",
        "Antal aktier","GAV","Insatt (SEK)","Marknadsvärde (SEK)",
        "Orealiserad P/L (SEK)","Orealiserad P/L (%)",
        "Direktavkastning (%)","Utdelning/år","Utdelning/år (manuell)","Lås utdelning","Utdelningskälla",
        "Utdelningsfrekvens","Frekvens/år","Årlig utdelning (SEK)",
        "Ex-Date","Nästa utbetalning (est)","Portföljandel (%)","Senaste uppdatering"
    ]
    edit_cols = ["Antal aktier","GAV","Frekvens/år","Payment-lag (dagar)","Kategori","Utdelning/år (manuell)","Lås utdelning"]

    editor = st.data_editor(
        d[show_cols], hide_index=True, num_rows="dynamic", use_container_width=True,
        column_config={
            "Kategori": st.column_config.SelectboxColumn("Kategori", options=CATEGORY_CHOICES, default="QUALITY", required=True),
            "Utdelningskälla": st.column_config.TextColumn("Utdelningskälla", help="“Manuell 🔒” när lås+manuell > 0, annars Yahoo.", disabled=True)
        }
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("💾 Spara ändringar (in-memory)"):
            base = säkerställ_kolumner(st.session_state.get("working_df", d))
            for _, r in editor.iterrows():
                t = str(r["Ticker"]).upper().strip()
                if not t:
                    continue
                m = base["Ticker"].astype(str).str.upper() == t
                if not m.any():
                    continue
                for c in edit_cols:
                    base.loc[m, c] = r[c]
                lås = bool(r.get("Lås utdelning", False))
                try:
                    vals = hamta_yahoo_data(t)
                    if vals:
                        # skriv inte över utd med 0 och respektera lås
                        new_div = float(vals.get("utdelning") or 0.0)
                        if new_div > 0 and not lås:
                            base.loc[m, "Utdelning/år"] = new_div
                        base.loc[m, "Aktuell kurs"] = vals.get("kurs") or base.loc[m, "Aktuell kurs"]
                        base.loc[m, "Valuta"] = (vals.get("valuta") or base.loc[m, "Valuta"])
                        # enkel label för frekvens
                        f = int(vals.get("frekvens") or 0)
                        if f > 0:
                            base.loc[m, "Frekvens/år"] = f
                            base.loc[m, "Utdelningsfrekvens"] = ("Månads" if f==12 else "Kvartals" if f==4 else "Halvårs" if f==2 else "Års" if f==1 else base.loc[m,"Utdelningsfrekvens"])
                        base.loc[m, "Källa"] = "Yahoo"
                except Exception as e:
                    st.warning(f"{t}: kunde inte hämta Yahoo-data ({e}). Sparar ändå manuellt.")
            st.session_state["working_df"] = beräkna(base)
            st.success("Ändringar sparade (in-memory) med färsk Yahoo.")
            _rerun()

    with colB:
        if st.button("💾 Spara ändringar till Google Sheets NU"):
            base = säkerställ_kolumner(st.session_state.get("working_df", d))
            for _, r in editor.iterrows():
                t = str(r["Ticker"]).upper().strip()
                if not t:
                    continue
                m = base["Ticker"].astype(str).str.upper() == t
                if not m.any():
                    continue
                for c in edit_cols:
                    base.loc[m, c] = r[c]
                lås = bool(r.get("Lås utdelning", False))
                try:
                    vals = hamta_yahoo_data(t)
                    if vals:
                        new_div = float(vals.get("utdelning") or 0.0)
                        if new_div > 0 and not lås:
                            base.loc[m, "Utdelning/år"] = new_div
                        base.loc[m, "Aktuell kurs"] = vals.get("kurs") or base.loc[m, "Aktuell kurs"]
                        base.loc[m, "Valuta"] = (vals.get("valuta") or base.loc[m, "Valuta"])
                        f = int(vals.get("frekvens") or 0)
                        if f > 0:
                            base.loc[m, "Frekvens/år"] = f
                            base.loc[m, "Utdelningsfrekvens"] = ("Månads" if f==12 else "Kvartals" if f==4 else "Halvårs" if f==2 else "Års" if f==1 else base.loc[m,"Utdelningsfrekvens"])
                        base.loc[m, "Källa"] = "Yahoo"
                except Exception as e:
                    st.warning(f"{t}: kunde inte hämta Yahoo-data ({e}). Sparar ändå manuellt.")
            base = beräkna(base)
            spara_data(base)
            st.session_state["working_df"] = base
            st.success("Ändringar sparade till Google Sheets (med Yahoo-uppdatering).")
            _rerun()

    # Regler & vikter
    with st.expander("📏 Regler & vikter"):
        if "Marknadsvärde (SEK)" not in d.columns:
            d["Marknadsvärde (SEK)"] = (
                pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0)
                * pd.to_numeric(d["Kurs (SEK)"], errors="coerce").fillna(0.0)
            ).round(2)

        if "Kategori" not in d.columns:
            d["Kategori"] = "QUALITY"
        cat_df = d.groupby("Kategori", as_index=False)["Marknadsvärde (SEK)"].sum()
        T = float(cat_df["Marknadsvärde (SEK)"].sum()) if not cat_df.empty else 0.0
        if T > 0:
            cat_df["Nu (%)"] = (100.0 * cat_df["Marknadsvärde (SEK)"] / T).round(2)

        max_df = pd.DataFrame([{"Kategori": k, "Max (%)": v} for k, v in MAX_CAT.items()])
        merged = pd.merge(
            max_df,
            (cat_df[["Kategori","Nu (%)"]] if "Nu (%)" in cat_df.columns else pd.DataFrame(columns=["Kategori","Nu (%)"])),
            on="Kategori",
            how="left"
        ).fillna({"Nu (%)": 0.0})
        st.dataframe(merged.sort_values("Kategori"), use_container_width=True)

        trims = trim_suggestions(d)
        if not trims.empty:
            st.warning("Följande innehav ligger över 12% – förslag att skala ned:")
            st.dataframe(trims, use_container_width=True)

    return d

# ── Köpförslag-sida ────────────────────────────────────────────────────────
def page_buy_suggestions(df: pd.DataFrame):
    st.subheader("🎯 Köpförslag (tar hänsyn till 12%/kategori & kassa)")
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        cash = st.number_input("Tillgänglig kassa (SEK)", min_value=0.0, value=5000.0, step=100.0)
    with c2:
        w_val = st.slider("Vikt: Värdering (DA)", 0.0, 1.0, 0.50, 0.05)
    with c3:
        w_under = st.slider("Vikt: Undervikt mot 12%", 0.0, 1.0, 0.35, 0.05)
    with c4:
        w_time = st.slider("Vikt: Timing (nära utdelning)", 0.0, 1.0, 0.15, 0.05)

    totw = max(1e-9, (w_val + w_under + w_time))
    w_val, w_under, w_time = w_val/totw, w_under/totw, w_time/totw

    if st.button("Beräkna köpförslag"):
        sug = suggest_buys(df, cash_sek=cash, w_val=w_val, w_under=w_under, w_time=w_time, topk=5)
        if sug.empty:
            st.info("Ingen kandidat ryms i kassan eller skulle passera 12%/kategori-tak.")
        else:
            st.dataframe(sug, use_container_width=True)
            st.caption("Poäng = DA + undervikt mot 12% + hur nära nästa utdelning.")

# ── Trading (Köp/Sälj) med avgifter & transaktionslogg ─────────────────────
MIN_COURTAGE_RATE = 0.0025
MIN_COURTAGE_SEK  = 1.0
FX_FEE_RATE       = 0.0025

def is_foreign(ccy: str) -> bool:
    return str(ccy or "").upper() != "SEK"

def calc_fees(order_value_sek: float, foreign: bool):
    courtage = max(MIN_COURTAGE_RATE * order_value_sek, MIN_COURTAGE_SEK)
    fx_fee   = (FX_FEE_RATE * order_value_sek) if foreign else 0.0
    total    = round(courtage + fx_fee, 2)
    return round(courtage,2), round(fx_fee,2), total

TX_SHEET = "Transaktioner"

def ensure_tx_sheet():
    sh = client.open_by_url(SHEET_URL)
    try:
        return sh.worksheet(TX_SHEET)
    except gspread.WorksheetNotFound:
        ws_tx = sh.add_worksheet(title=TX_SHEET, rows=1, cols=14)
        ws_tx.update([[
            "Tid","Typ","Ticker","Antal","Pris (lokal)","Valuta","FX",
            "Pris (SEK)","Belopp (SEK)","Courtage (SEK)","FX-avgift (SEK)","Tot.avgifter (SEK)","Kommentar"
        ]])
        return ws_tx

def save_pending_transactions():
    if "pending_txs" not in st.session_state or not st.session_state["pending_txs"]:
        return
    ws_tx = ensure_tx_sheet()
    rows = st.session_state["pending_txs"]
    values = [[r.get("Tid"), r.get("Typ"), r.get("Ticker"), r.get("Antal"), r.get("Pris (lokal)"), r.get("Valuta"), r.get("FX"),
               r.get("Pris (SEK)"), r.get("Belopp (SEK)"), r.get("Courtage (SEK)"), r.get("FX-avgift (SEK)"), r.get("Tot.avgifter (SEK)"), r.get("Kommentar")] for r in rows]
    ws_tx.append_rows(values, value_input_option="USER_ENTERED")
    st.session_state["pending_txs"] = []

def block_trading(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🛒 Köp / 📤 Sälj (avgifter, in-memory)")
    if df.empty:
        st.info("Lägg till minst en ticker först.")
        return df

    tickers = df["Ticker"].astype(str).tolist()
    tkr = st.selectbox("Ticker", options=tickers)
    side = st.radio("Typ", ["KÖP", "SÄLJ"], horizontal=True)
    qty  = st.number_input("Antal", min_value=1, value=10, step=1)
    px_local = st.number_input("Pris (lokal valuta)", min_value=0.0, value=10.0)

    ccy_default = df.loc[df["Ticker"]==tkr, "Valuta"].iloc[0] if (df["Ticker"]==tkr).any() else "SEK"
    ccy = st.selectbox(
        "Valuta",
        options=["SEK","USD","EUR","CAD","NOK"],
        index=(["SEK","USD","EUR","CAD","NOK"].index(str(ccy_default).upper())
               if str(ccy_default).upper() in ["SEK","USD","EUR","CAD","NOK"] else 0)
    )

    fx_rate = fx_for(ccy)
    px_sek  = round(px_local * fx_rate, 6)
    gross   = round(px_sek * qty, 2)
    fee_court, fee_fx, fee_tot = calc_fees(gross, is_foreign(ccy))
    net = round(gross + fee_tot, 2) if side == "KÖP" else round(gross - fee_tot, 2)

    st.caption(
        f"Pris (SEK): **{px_sek}** | Brutto: **{gross} SEK** | "
        f"Courtage: **{fee_court}** | FX-avgift: **{fee_fx}** | "
        f"{'Totalkostnad' if side=='KÖP' else 'Nettolikvid'}: **{net} SEK**"
    )

    # Kontroll mot 12%-regeln innan lägg i minnet
    if st.button("Kontrollera mot 12%-regeln", key="check_rules_btn2"):
        base = säkerställ_kolumner(st.session_state["working_df"]).copy()
        if not (base["Ticker"] == tkr).any():
            st.error("Ticker finns inte i portföljen ännu. Lägg till under '➕ Lägg till bolag'.")
            return df

        i = base.index[base["Ticker"] == tkr][0]
        sim = base.copy()
        if side == "KÖP":
            sim.at[i, "Antal aktier"] = float(pd.to_numeric(sim.at[i,"Antal aktier"], errors="coerce") or 0.0) + qty
        else:
            new_q = float(pd.to_numeric(sim.at[i,"Antal aktier"], errors="coerce") or 0.0) - qty
            if new_q < 0:
                st.error("Sälj ger negativt antal. Minska antal.")
                return df
            sim.at[i, "Antal aktier"] = new_q

        d_sim = beräkna(sim)
        mv_sim = (
            pd.to_numeric(d_sim["Antal aktier"], errors="coerce").fillna(0.0)
            * pd.to_numeric(d_sim["Kurs (SEK)"], errors="coerce").fillna(0.0)
        )
        tot_mv_sim = float(mv_sim.sum()) if mv_sim.sum() else 1.0
        w_after = float(100.0 * float(mv_sim.loc[d_sim["Ticker"]==tkr].sum()) / tot_mv_sim)

        if side == "KÖP" and w_after > GLOBAL_MAX_NAME + 1e-9:
            st.error(f"KÖP skulle ge vikt {w_after:.2f}% > max {GLOBAL_MAX_NAME:.2f}% – blockerat.")
        else:
            st.success("OK enligt reglerna.")

    # Lägg order i minnet (och uppdatera antal/GAV)
    if st.button("Lägg order i minnet"):
        base = säkerställ_kolumner(st.session_state["working_df"]).copy()
        if not (base["Ticker"] == tkr).any():
            st.error("Ticker saknas i portföljen. Lägg till under '➕ Lägg till bolag' först.")
            return df
        i = base.index[base["Ticker"] == tkr][0]

        # snabb efter-affärsviktkontroll
        sim = base.copy()
        if side == "KÖP":
            sim.at[i, "Antal aktier"] = float(pd.to_numeric(sim.at[i,"Antal aktier"], errors="coerce") or 0.0) + qty
        else:
            new_q = float(pd.to_numeric(sim.at[i,"Antal aktier"], errors="coerce") or 0.0) - qty
            if new_q < 0:
                st.error("Sälj ger negativt antal.")
                return df
            sim.at[i, "Antal aktier"] = new_q

        d_chk = beräkna(sim)
        mv_chk = (
            pd.to_numeric(d_chk["Antal aktier"], errors="coerce").fillna(0.0)
            * pd.to_numeric(d_chk["Kurs (SEK)"], errors="coerce").fillna(0.0)
        )
        tot_mv_chk = float(mv_chk.sum()) if mv_chk.sum() else 1.0
        w_after = float(100.0 * float(mv_chk.loc[d_chk["Ticker"]==tkr].sum()) / tot_mv_chk)

        if side == "KÖP" and w_after > GLOBAL_MAX_NAME + 1e-9:
            st.error(f"Order stoppad: {tkr} skulle väga {w_after:.2f}% > {GLOBAL_MAX_NAME:.2f}%.")
            return df

        # uppdatera antal & GAV / eller minska antal vid sälj
        if side == "KÖP":
            old_qty = float(pd.to_numeric(base.at[i,"Antal aktier"], errors="coerce") or 0.0)
            old_gav = float(pd.to_numeric(base.at[i,"GAV"], errors="coerce") or 0.0)
            new_qty = old_qty + qty
            new_gav = 0.0 if new_qty == 0 else round(((old_gav * old_qty) + (gross + fee_tot)) / new_qty, 6)
            base.at[i,"Antal aktier"] = new_qty
            base.at[i,"GAV"] = new_gav
        else:
            old_qty = float(pd.to_numeric(base.at[i,"Antal aktier"], errors="coerce") or 0.0)
            if qty > old_qty:
                st.error(f"Du kan inte sälja {qty} st – du äger {int(old_qty)}.")
                return df
            new_qty = old_qty - qty
            base.at[i,"Antal aktier"] = new_qty
            if new_qty == 0:
                base.at[i,"GAV"] = 0.0

        # logga transaktion i minnet
        if "pending_txs" not in st.session_state:
            st.session_state["pending_txs"] = []
        st.session_state["pending_txs"].append({
            "Tid": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Typ": side, "Ticker": tkr, "Antal": int(qty),
            "Pris (lokal)": float(px_local), "Valuta": ccy, "FX": float(fx_rate),
            "Pris (SEK)": float(px_sek), "Belopp (SEK)": float(gross),
            "Courtage (SEK)": float(fee_court), "FX-avgift (SEK)": float(fee_fx),
            "Tot.avgifter (SEK)": float(fee_tot), "Kommentar": "in-memory"
        })

        st.session_state["working_df"] = beräkna(base)
        st.success(f"{side} registrerad i minnet. Gå till '💾 Spara' för att skriva till Google Sheets.")
        return st.session_state["working_df"]

    # Visa ej sparade transaktioner (om några)
    if "pending_txs" in st.session_state and st.session_state["pending_txs"]:
        st.info(f"Ej sparade transaktioner: {len(st.session_state['pending_txs'])} st")
        st.dataframe(pd.DataFrame(st.session_state["pending_txs"]), use_container_width=True)

    return df

# ── Sidopanel (FX mm.) ─────────────────────────────────────────────────────
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

    st.session_state["USDSEK"] = usd
    st.session_state["EURSEK"] = eur
    st.session_state["CADSEK"] = cad
    st.session_state["NOKSEK"] = nok

    if st.sidebar.button("↩︎ Återställ FX till standard"):
        for k, v in DEF.items():
            st.session_state[k] = v
        _rerun()

    st.sidebar.markdown("---")
    # snabb-uppdatera EN ticker
    one_ticker = st.sidebar.text_input("Uppdatera EN ticker (Yahoo)", placeholder="t.ex. VICI").strip().upper()
    if st.sidebar.button("🔄 Uppdatera EN"):
        if one_ticker:
            base = säkerställ_kolumner(st.session_state.get("working_df", pd.DataFrame()))
            if one_ticker not in base["Ticker"].tolist():
                base = pd.concat([base, pd.DataFrame([{"Ticker": one_ticker, "Kategori": "QUALITY"}])], ignore_index=True)
            vals = hamta_yahoo_data(one_ticker)
            if vals:
                m = base["Ticker"] == one_ticker
                new_div = float(vals.get("utdelning") or 0.0)
                lås = bool(base.loc[m, "Lås utdelning"].iloc[0]) if m.any() else False
                if new_div > 0 and not lås:
                    base.loc[m, "Utdelning/år"] = new_div
                base.loc[m, "Aktuell kurs"] = vals.get("kurs") or base.loc[m, "Aktuell kurs"]
                base.loc[m, "Valuta"] = vals.get("valuta") or base.loc[m, "Valuta"]
                f = int(vals.get("frekvens") or 0)
                if f > 0:
                    base.loc[m, "Frekvens/år"] = f
                    base.loc[m, "Utdelningsfrekvens"] = ("Månads" if f==12 else "Kvartals" if f==4 else "Halvårs" if f==2 else "Års" if f==1 else base.loc[m,"Utdelningsfrekvens"])
                base.loc[m, "Källa"] = "Yahoo"
                st.session_state["working_df"] = beräkna(base)
                st.sidebar.success(f"{one_ticker} uppdaterad.")
            else:
                st.sidebar.warning("Kunde inte hämta data.")

# ── Utdelningskalender / prognos ───────────────────────────────────────────
def _gen_payment_dates(first_ex_date: str, freq_per_year: float, payment_lag_days: float, months_ahead: int = 12):
    ts = pd.to_datetime(first_ex_date, errors="coerce")
    if pd.isna(ts):
        return []
    exd = ts.date()
    try: freq = int(float(freq_per_year))
    except: freq = 4
    freq = max(freq, 1)
    try: lag = int(float(payment_lag_days))
    except: lag = 30
    lag = max(lag, 0)

    step_days = max(1, int(round(365.0 / freq)))
    today_d = date.today()
    horizon = today_d + timedelta(days=int(round(months_ahead * 30.44)))

    while exd < today_d:
        exd = exd + timedelta(days=step_days)

    dates = []
    pay = exd + timedelta(days=lag)
    while pay <= horizon:
        dates.append(pay)
        exd = exd + timedelta(days=step_days)
        pay = exd + timedelta(days=lag)
    return dates

def prognos_kalender(df: pd.DataFrame, months_ahead: int = 12):
    d = beräkna(df).copy()
    if d.empty:
        return pd.DataFrame(columns=["Månad","Utdelning (SEK)"]), pd.DataFrame()

    rows = []
    for _, r in d.iterrows():
        try:
            per_share_local = float(r.get("Utdelning/år_eff", 0.0)) / max(1.0, float(r.get("Frekvens/år", 4.0)))
            qty = float(r.get("Antal aktier", 0.0))
            fx = fx_for(r.get("Valuta", "SEK"))
            per_payment_sek = per_share_local * fx * qty
            if per_payment_sek <= 0:
                continue
            pays = _gen_payment_dates(r.get("Ex-Date",""), r.get("Frekvens/år",4), r.get("Payment-lag (dagar)",30), months_ahead)
            for p in pays:
                rows.append({"Datum": p, "Ticker": r["Ticker"], "Belopp (SEK)": round(per_payment_sek, 2)})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["Månad","Utdelning (SEK)"]), pd.DataFrame()

    cal = pd.DataFrame(rows)
    cal["Månad"] = cal["Datum"].apply(lambda d: f"{d.year}-{str(d.month).zfill(2)}")
    monthly = cal.groupby("Månad", as_index=False)["Belopp (SEK)"].sum().rename(columns={"Belopp (SEK)":"Utdelning (SEK)"})
    monthly = monthly.sort_values("Månad")
    return monthly, cal

def page_calendar(df: pd.DataFrame):
    st.subheader("📅 Utdelningskalender")
    months = st.selectbox("Prognoshorisont", options=[12, 24, 36], index=0)
    monthly, cal = prognos_kalender(df, months_ahead=months)

    if monthly.empty:
        st.info("Ingen prognos – saknar Ex-Date/frekvens/utdelningsdata.")
        return

    st.write(f"**Månadsvis prognos ({months} mån) i SEK:**")
    st.dataframe(monthly, use_container_width=True)
    st.bar_chart(monthly.set_index("Månad")["Utdelning (SEK)"])

    if not cal.empty:
        with st.expander("Detaljerade kommande betalningar per ticker"):
            st.dataframe(cal.sort_values("Datum"), use_container_width=True)

# ── Spara-sida ─────────────────────────────────────────────────────────────
def page_save_now(df: pd.DataFrame):
    st.subheader("💾 Spara till Google Sheets")
    preview = beräkna( säkerställ_kolumner(df) )
    st.write("Antal rader som sparas:", len(preview))
    st.dataframe(
        preview[["Ticker","Bolagsnamn","Valuta","Kategori","Antal aktier","GAV","Aktuell kurs","Utdelning/år","Kurs (SEK)","Årlig utdelning (SEK)"]],
        use_container_width=True
    )

    if st.button("✅ Bekräfta och spara"):
        if preview["Ticker"].astype(str).str.strip().eq("").all():
            st.error("Inget att spara: inga tickers i tabellen.")
            return df
        spara_data(preview)
        try:
            save_pending_transactions()
        except Exception:
            pass
        st.success("Data (och ev. transaktioner) sparade till Google Sheets!")
    return preview

# ── Meny: Lägg till / Uppdatera / Massuppdatera (Del 3-funktioner) ─────────
def page_add_or_update():
    base = säkerställ_kolumner(st.session_state.get("working_df", pd.DataFrame()))
    st.session_state["working_df"] = lagg_till_eller_uppdatera(base)
    return st.session_state["working_df"]

def page_update_single():
    base = säkerställ_kolumner(st.session_state.get("working_df", pd.DataFrame()))
    st.session_state["working_df"] = uppdatera_bolag(base)
    return st.session_state["working_df"]

def page_update_all():
    base = säkerställ_kolumner(st.session_state.get("working_df", pd.DataFrame()))
    st.session_state["working_df"] = massuppdatera(base)
    return st.session_state["working_df"]

# ── Main (router/meny) ─────────────────────────────────────────────────────
def main():
    st.title("Relative Yield – utdelningsportfölj")

    # Initiera arbetskopia från Google Sheets
    if "working_df" not in st.session_state:
        try:
            st.session_state["working_df"] = migrate_sheet_columns()
        except Exception:
            st.session_state["working_df"] = säkerställ_kolumner(pd.DataFrame())

    base = säkerställ_kolumner(st.session_state["working_df"])

    # Sidopanel
    sidopanel()
    st.sidebar.caption(f"📄 Rader i databasen: {len(base)}")

    # Meny
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Meny",
        [
            "➕ Lägg till / ✏ Uppdatera bolag",
            "🔄 Uppdatera EN",
            "⏩ Massuppdatera alla",
            "📦 Portföljöversikt",
            "🛒 Köp/Sälj",
            "🎯 Köpförslag",
            "📅 Utdelningskalender",
            "💾 Spara",
        ],
        index=0
    )

    # Router
    if page == "➕ Lägg till / ✏ Uppdatera bolag":
        base = page_add_or_update()
    elif page == "🔄 Uppdatera EN":
        base = page_update_single()
    elif page == "⏩ Massuppdatera alla":
        base = page_update_all()
    elif page == "📦 Portföljöversikt":
        base = portfolj_oversikt(base)
    elif page == "🛒 Köp/Sälj":
        base = block_trading(base)
    elif page == "🎯 Köpförslag":
        page_buy_suggestions(base)
    elif page == "📅 Utdelningskalender":
        page_calendar(base)
    elif page == "💾 Spara":
        page_save_now(base)

    # Uppdatera state
    st.session_state["working_df"] = säkerställ_kolumner(base)

# Entrypoint
if __name__ == "__main__":
    main()
