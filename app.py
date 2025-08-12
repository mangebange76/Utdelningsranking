import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time, math
from datetime import datetime, timedelta, date
from google.oauth2.service_account import Credentials

# --- Streamlit rerun shim ---
try:
    _rerun = st.rerun
except AttributeError:
    _rerun = st.experimental_rerun

st.set_page_config(page_title="Utdelningsranking", layout="wide")

# --- Secrets / Google Sheets ---
SHEET_URL  = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"   # <- fliken i ditt Google Sheet

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]
credentials = Credentials.from_service_account_info(
    st.secrets["GOOGLE_CREDENTIALS"], scopes=scope
)
client = gspread.authorize(credentials)

def _open_sheet():
    return client.open_by_url(SHEET_URL)

def skapa_koppling():
    return _open_sheet().worksheet(SHEET_NAME)

def hamta_data():
    try:
        ws = skapa_koppling()
        rows = ws.get_all_records()
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Kunde inte läsa Google Sheet just nu: {e}")
        return pd.DataFrame()

def spara_data(df: pd.DataFrame):
    ws = skapa_koppling()
    ws.clear()
    ws.update([df.columns.tolist()] + df.astype(str).values.tolist(),
              value_input_option="USER_ENTERED")

# --- FX defaults (kan ändras i sidopanelen) ---
DEF_FX = {"USDSEK": 9.60, "NOKSEK": 0.94, "CADSEK": 6.95, "EURSEK": 11.10}
for k, v in DEF_FX.items():
    if k not in st.session_state:
        st.session_state[k] = v

def fx_for(cur: str) -> float:
    if pd.isna(cur): return 1.0
    c = str(cur).strip().upper()
    return float({
        "SEK": 1.0,
        "USD": st.session_state.get("USDSEK", DEF_FX["USDSEK"]),
        "EUR": st.session_state.get("EURSEK", DEF_FX["EURSEK"]),
        "CAD": st.session_state.get("CADSEK", DEF_FX["CADSEK"]),
        "NOK": st.session_state.get("NOKSEK", DEF_FX["NOKSEK"]),
    }.get(c, 1.0))

# --- Kolumnschema ---
# OBS: "GAV" = GAV i aktiens egen valuta
COLUMNS = [
    "Ticker","Bolagsnamn","Aktuell kurs","Valuta","Kategori",
    "Direktavkastning (%)","Utdelning/år","Utdelning/år (manuell)","Lås utdelning",
    "Frekvens/år","Utdelningsfrekvens","Payment-lag (dagar)","Ex-Date","Nästa utbetalning (est)",
    "Antal aktier","GAV",
    "Portföljandel (%)","Årlig utdelning (SEK)","Kurs (SEK)","Utdelningstillväxt (%)",
    "Utdelningskälla","Senaste uppdatering","Källa",
    "Marknadsvärde (SEK)","Insatt (SEK)","Orealiserad P/L (SEK)","Orealiserad P/L (%)"
]

CATEGORY_CHOICES = ["Shipping","mREIT","REIT","Bank","BDC","Telecom","Finance","Tech","Other"]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty: d = pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in d.columns: d[c] = ""
    d["Ticker"]   = d["Ticker"].astype(str).str.strip().str.upper()
    d["Valuta"]   = d["Valuta"].astype(str).str.strip().str.upper()
    d["Kategori"] = d["Kategori"].astype(str).replace({"": "Other"})
    for c in ["Aktuell kurs","Utdelning/år","Utdelning/år (manuell)","Frekvens/år","Payment-lag (dagar)","Antal aktier","GAV"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["Lås utdelning"] = d.get("Lås utdelning", False).apply(lambda x: bool(x) if pd.notna(x) else False)
    return d[COLUMNS].copy()

def migrate_sheet_columns():
    raw = hamta_data()
    fixed = säkerställ_kolumner(raw)
    try:
        if raw.empty or (list(raw.columns)!=list(fixed.columns) or raw.shape[1]!=fixed.shape[1]):
            spara_data(fixed)
    except Exception as e:
        st.warning(f"Kunde inte synka kolumnlayout mot Google Sheet: {e}")
    return fixed

SETTINGS_SHEET = "Settings"
DEFAULT_TARGETS = {
    "Shipping": 30.0, "mREIT": 10.0, "REIT": 20.0, "Bank": 10.0,
    "BDC": 15.0, "Telecom": 10.0, "Finance": 10.0, "Tech": 5.0, "Other": 0.0,
}

def ensure_settings_ws():
    sh = _open_sheet()
    try:
        return sh.worksheet(SETTINGS_SHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=SETTINGS_SHEET, rows=500, cols=4)
        rows = [
            ["Nyckel","Kategori/Ticker","Värde","Kommentar"],
            ["IGNORE_EMPTY_CATEGORIES","", "1","1=ignorera kategorier utan ägda aktier"],
            ["NAME_MAX_DEFAULT","", "7.0","Max % per enskilt bolag (default)"],
        ]
        for cat in CATEGORY_CHOICES:
            rows.append(["TARGET", cat, str(DEFAULT_TARGETS.get(cat, 0.0)), "Tak % / mål %"])
        ws.update(rows, value_input_option="USER_ENTERED")
        return ws

def load_settings():
    try:
        ws = ensure_settings_ws()
        rows = ws.get_all_values()
        if not rows or len(rows) < 2:
            return DEFAULT_TARGETS.copy(), True, 7.0, {}
        header = rows[0]; data = rows[1:]
        df = pd.DataFrame(data, columns=header)
        df["Värde"] = pd.to_numeric(df["Värde"], errors="coerce").fillna(0.0)

        ignore_empty = bool(df.loc[df["Nyckel"]=="IGNORE_EMPTY_CATEGORIES","Värde"].max() > 0.0)
        tgt_df = df[df["Nyckel"]=="TARGET"]
        targets = {str(r["Kategori/Ticker"]): float(r["Värde"]) for _, r in tgt_df.iterrows()
                   if str(r["Kategori/Ticker"]) in CATEGORY_CHOICES} or DEFAULT_TARGETS.copy()
        name_default = float(df.loc[df["Nyckel"]=="NAME_MAX_DEFAULT","Värde"].max() or 7.0)
        ov_df = df[df["Nyckel"]=="NAME_MAX_OVERRIDE"]
        name_overrides = {str(r["Kategori/Ticker"]).upper(): float(r["Värde"]) for _, r in ov_df.iterrows()
                          if str(r["Kategori/Ticker"]).strip()}
        return targets, ignore_empty, name_default, name_overrides
    except Exception:
        return DEFAULT_TARGETS.copy(), True, 7.0, {}

def save_settings(targets: dict, ignore_empty: bool, name_default: float, name_overrides: dict):
    ws = ensure_settings_ws()
    rows = ws.get_all_values()
    header = rows[0] if rows else ["Nyckel","Kategori/Ticker","Värde","Kommentar"]
    df = pd.DataFrame(rows[1:], columns=header) if len(rows) > 1 else pd.DataFrame(columns=header)

    def upsert(nyckel, key, value, comment=""):
        mask = (df["Nyckel"]==nyckel) & (df["Kategori/Ticker"]==key)
        if mask.any():
            df.loc[mask, "Värde"] = str(float(value))
            if comment: df.loc[mask, "Kommentar"] = comment
        else:
            df.loc[len(df)] = [nyckel, key, str(float(value)), comment]

    upsert("IGNORE_EMPTY_CATEGORIES","", 1 if ignore_empty else 0, "1=ignorera kategorier utan ägda aktier")
    upsert("NAME_MAX_DEFAULT","", name_default, "Max % per enskilt bolag (default)")
    for cat in CATEGORY_CHOICES:
        upsert("TARGET", cat, float(targets.get(cat, 0.0)), "Tak % / mål %")

    keep = [k.upper() for k in name_overrides.keys()]
    if not df.empty:
        df = df[~((df["Nyckel"]=="NAME_MAX_OVERRIDE") & (~df["Kategori/Ticker"].str.upper().isin(keep)))]
    for tkr, pct in name_overrides.items():
        if str(tkr).strip():
            upsert("NAME_MAX_OVERRIDE", str(tkr).upper(), float(pct), "Override max % för ticker")

    out = pd.concat([pd.DataFrame([header]), df], ignore_index=True)
    ws.clear(); ws.update(out.values.tolist(), value_input_option="USER_ENTERED")

def name_max_for(ticker: str) -> float:
    _, _, name_default, name_overrides = load_settings()
    return float(name_overrides.get(str(ticker or "").upper(), name_default))

def hamta_yahoo_data(ticker: str):
    try:
        t = yf.Ticker(ticker)
        info = {}
        try: info = t.get_info() or {}
        except Exception:
            try: info = t.info or {}
            except Exception: info = {}

        price = None
        try: price = t.fast_info.get("last_price")
        except Exception: pass
        if price in (None, ""):
            price = info.get("regularMarketPrice") or info.get("currentPrice")
        if price in (None, ""):
            try:
                h = t.history(period="5d")
                if not h.empty: price = float(h["Close"].iloc[-1])
            except Exception: price = 0.0
        price = float(price or 0.0)

        currency = (info.get("currency") or "").upper()
        if not currency:
            try: currency = (t.fast_info.get("currency") or "").upper()
            except Exception: currency = "SEK"

        div_rate = 0.0; freq = 0; freq_text = "Oregelbunden"; ex_date_str = ""
        try:
            divs = t.dividends
            if divs is not None and not divs.empty:
                cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
                last12 = divs[divs.index >= cutoff]
                div_rate = float(last12.sum()) if not last12.empty else 0.0
                cnt = int(last12.shape[0]) if not last12.empty else 0
                if cnt >= 10: freq, freq_text = 12, "Månads"
                elif cnt >= 3: freq, freq_text = 4, "Kvartals"
                elif cnt == 2: freq, freq_text = 2, "Halvårs"
                elif cnt == 1: freq, freq_text = 1, "Års"
                else: freq, freq_text = 0, "Oregelbunden"
                ex_date_str = pd.to_datetime(divs.index.max()).strftime("%Y-%m-%d")
        except Exception:
            pass

        if div_rate == 0.0:
            for k in ("forwardAnnualDividendRate","trailingAnnualDividendRate"):
                try:
                    v = info.get(k)
                    if v not in (None, "", 0): div_rate = float(v); break
                except Exception: pass

        if not ex_date_str:
            try:
                ts = info.get("exDividendDate")
                if ts not in (None, "", 0):
                    ex_date_str = pd.to_datetime(int(ts), unit="s", utc=True).strftime("%Y-%m-%d")
            except Exception: ex_date_str = ""

        return dict(kurs=price, valuta=currency, utdelning=div_rate, frekvens=freq,
                    frekvens_text=freq_text, ex_date=ex_date_str, källa="Yahoo",
                    uppdaterad=datetime.now().strftime("%Y-%m-%d %H:%M"))
    except Exception as e:
        st.warning(f"Kunde inte hämta Yahoo-data för {ticker}: {e}")
        return None

def beräkna(df: pd.DataFrame) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()

    use_manual = (d["Lås utdelning"] == True) & (pd.to_numeric(d["Utdelning/år (manuell)"], errors="coerce") > 0)
    d["Utdelning/år_eff"] = pd.to_numeric(d["Utdelning/år"], errors="coerce").fillna(0.0)
    d.loc[use_manual, "Utdelning/år_eff"] = pd.to_numeric(d["Utdelning/år (manuell)"], errors="coerce").fillna(0.0)
    d["Utdelningskälla"] = ["Manuell 🔒" if um else "Yahoo" for um in use_manual]

    d["Aktuell kurs"] = pd.to_numeric(d["Aktuell kurs"], errors="coerce").fillna(0.0)
    rates = d["Valuta"].apply(fx_for).astype(float)
    d["Kurs (SEK)"] = (d["Aktuell kurs"] * rates).round(6)

    d["Antal aktier"] = pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0)
    d["GAV"]          = pd.to_numeric(d["GAV"], errors="coerce").fillna(0.0)

    d["Insatt (SEK)"]          = (d["Antal aktier"] * d["GAV"] * rates).round(2)
    d["Marknadsvärde (SEK)"]   = (d["Antal aktier"] * d["Kurs (SEK)"]).round(2)
    d["Orealiserad P/L (SEK)"] = (d["Marknadsvärde (SEK)"] - d["Insatt (SEK)"]).round(2)
    d["Orealiserad P/L (%)"]   = (100.0 * d["Orealiserad P/L (SEK)"] / d["Insatt (SEK)"].replace({0: pd.NA})).fillna(0.0).round(2)

    ok = (d["Aktuell kurs"] > 0) & (d["Utdelning/år_eff"] > 0)
    d["Direktavkastning (%)"] = 0.0
    d.loc[ok, "Direktavkastning (%)"] = (100.0 * d.loc[ok, "Utdelning/år_eff"] / d.loc[ok, "Aktuell kurs"]).round(2)
    d["Årlig utdelning (SEK)"] = (d["Antal aktier"] * d["Utdelning/år_eff"] * rates).round(2)

    tot_mv = float(d["Marknadsvärde (SEK)"].sum()) or 1.0
    d["Portföljandel (%)"] = (100.0 * d["Marknadsvärde (SEK)"] / tot_mv).round(2)

    d["Frekvens/år"] = pd.to_numeric(d["Frekvens/år"], errors="coerce").fillna(0.0).replace(0, 4)
    d["Payment-lag (dagar)"] = pd.to_numeric(d["Payment-lag (dagar)"], errors="coerce").fillna(0.0).replace(0, 30)

    def next_pay(ex_date_str, freq_per_year, payment_lag_days):
        ts = pd.to_datetime(ex_date_str, errors="coerce")
        if pd.isna(ts): return ""
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

# --- Avgifter ---
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

# --- Transaktionslogg (plan) till Google Sheets ---
TX_SHEET = "Transaktioner"

def ensure_tx_sheet():
    sh = _open_sheet()
    try:
        return sh.worksheet(TX_SHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=TX_SHEET, rows=1000, cols=10)
        ws.update([[
            "Tid","Källa","Steg","Ticker","Kategori","Köpta st",
            "Pris/aktie (SEK)","Kostnad ca","Kvar kassa","Notering"
        ]], value_input_option="USER_ENTERED")
        return ws

def save_plan_to_sheet(steps_records: list, note: str = ""):
    if not steps_records:
        return False, "Ingen plan i minnet."
    try:
        ws = ensure_tx_sheet()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        values = []
        for r in steps_records:
            values.append([
                now, "Planerare",
                r.get("Steg",""), r.get("Ticker",""), r.get("Kategori",""),
                r.get("Köpta st",""), r.get("Pris/aktie (SEK)",""), r.get("Kostnad ca",""),
                r.get("Kvar kassa",""), note or ""
            ])
        ws.append_rows(values, value_input_option="USER_ENTERED")
        return True, f"Sparade {len(values)} rader till fliken '{TX_SHEET}'."
    except Exception as e:
        return False, f"Kunde inte spara logg: {e}"

# --- Trim (> per-bolagstak) ---
def trim_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    d = beräkna(df).copy()
    if d.empty: return pd.DataFrame(columns=["Ticker","Vikt (%)","Kurs (SEK)","Föreslagen sälj (st)","Nettolikvid ca (SEK)"])
    T = float(d["Marknadsvärde (SEK)"].sum()) or 0.0
    if T <= 0: return pd.DataFrame(columns=["Ticker","Vikt (%)","Kurs (SEK)","Föreslagen sälj (st)","Nettolikvid ca (SEK)"])

    rows = []
    for _, r in d.iterrows():
        price = float(pd.to_numeric(r["Kurs (SEK)"], errors="coerce") or 0.0)
        qty   = float(pd.to_numeric(r["Antal aktier"], errors="coerce") or 0.0)
        V     = float(pd.to_numeric(r["Marknadsvärde (SEK)"], errors="coerce") or 0.0)
        if price <= 0 or qty <= 0: continue
        w = 100.0 * V / T if T else 0.0
        cap_pct = name_max_for(r["Ticker"])
        if w <= cap_pct + 1e-9: continue

        n_min = (V - (cap_pct/100.0)*T) / ((1.0 - cap_pct/100.0) * price)
        n = max(0, math.ceil(n_min)); n = int(min(n, qty))
        if n > 0:
            gross = round(price * n, 2)
            fee_court, fee_fx, fee_tot = calc_fees(gross, foreign=is_foreign(r.get("Valuta","SEK")))
            net = round(gross - fee_tot, 2)
            rows.append({"Ticker": r["Ticker"], "Vikt (%)": round(w,2), "Kurs (SEK)": round(price,2),
                         "Föreslagen sälj (st)": n, "Nettolikvid ca (SEK)": net,
                         "Kommentar": f"Ner till {cap_pct:.0f}%"})
    return pd.DataFrame(rows)

# --- Hjälpare för tak ---
def _cap_by_name(Vi: float, T: float, price_sek: float, cap_pct: float) -> int:
    if price_sek <= 0: return 0
    m = cap_pct / 100.0
    numer = m*T - Vi
    denom = (1.0 - m) * price_sek
    if denom <= 0: return 0
    return int(max(0, math.floor(numer/denom)))

def _cap_by_category(C: float, T: float, price_sek: float, cat_cap_pct: float) -> int:
    if price_sek <= 0: return 0
    M = cat_cap_pct / 100.0
    numer = M*T - C
    denom = (1.0 - M) * price_sek
    if denom <= 0: return 0
    return int(max(0, math.floor(numer/denom)))

# --- Köpförslag (engångslista) ---
def suggest_buys(df: pd.DataFrame, cash_sek: float, w_val: float=0.5, w_under: float=0.35, w_time: float=0.15, topk: int=10):
    targets, ignore_empty, name_default, _ov = load_settings()
    d = beräkna(df).copy()
    if d.empty:
        return pd.DataFrame(columns=["Ticker","Kategori","Poäng","DA %","Vikt %","Nästa utb","Föreslagna st","Kostnad ca","Motivering"]), pd.DataFrame()

    T = float(d["Marknadsvärde (SEK)"].sum())
    d["Kategori"] = d["Kategori"].astype(str).replace({"": "Other"})
    cat_mv = d.groupby("Kategori", as_index=False)["Marknadsvärde (SEK)"].sum().set_index("Kategori")["Marknadsvärde (SEK)"].to_dict()
    active_cats = {k for k,v in cat_mv.items() if v > 0} if ignore_empty else set(CATEGORY_CHOICES)

    da = pd.to_numeric(d["Direktavkastning (%)"], errors="coerce").fillna(0.0)
    da_score = (da.clip(lower=0, upper=15) / 15.0) * 100.0

    under = []
    for i, r in d.iterrows():
        cap_pct = name_max_for(r["Ticker"])
        under.append(max(0.0, cap_pct - float(r["Portföljandel (%)"] or 0.0)))
    under = pd.Series(under, index=d.index)
    under_score = (under / max(1.0, name_default)).clip(lower=0, upper=1) * 100.0

    def _days_to(date_str: str) -> int:
        try:
            dt = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(dt): return 9999
            return max(0, (dt.date() - date.today()).days)
        except Exception:
            return 9999
    days = d["Nästa utbetalning (est)"].apply(_days_to)
    time_score = ((90 - days.clip(upper=90)) / 90.0).clip(lower=0) * 100.0

    totw = max(1e-9, (w_val + w_under + w_time))
    w_val, w_under, w_time = w_val/totw, w_under/totw, w_time/totw
    total_score = (w_val*da_score + w_under*under_score + w_time*time_score)

    diag_rows = []; order = total_score.sort_values(ascending=False).index
    rows, used = [], 0.0
    for i in order:
        tkr = d.at[i,"Ticker"]; cat = str(d.at[i,"Kategori"]) or "Other"
        if cat not in active_cats: 
            diag_rows.append([tkr, f"kat. '{cat}' inaktiv"]); continue
        price = float(pd.to_numeric(d.at[i,"Kurs (SEK)"], errors="coerce") or 0.0)
        if price <= 0: diag_rows.append([tkr, "saknar pris"]); continue

        Vi = float(d.at[i,"Marknadsvärde (SEK)"])
        C  = float(cat_mv.get(cat, 0.0))
        cat_cap = float(targets.get(cat, 0.0))
        name_cap = name_max_for(tkr)

        n_name  = _cap_by_name(Vi, T, price, name_cap)
        n_cat   = _cap_by_category(C, T, price, cat_cap)
        n_cash  = max(0, int((cash_sek - used) // price))
        n       = min(n_name, n_cat, n_cash)

        if n <= 0:
            reason = []
            if n_name <= 0: reason.append(f"bolagstak {name_cap:.0f}%")
            if n_cat  <= 0: reason.append(f"kategoritak {cat_cap:.0f}%")
            if n_cash <= 0: reason.append("kassa")
            diag_rows.append([tkr, " & ".join(reason) if reason else "okänd begränsning"])
            continue

        gross = price * n
        c, fx, tot = calc_fees(gross, is_foreign(d.at[i,"Valuta"]))
        cost = round(gross + tot, 2)
        rows.append({
            "Ticker": tkr, "Kategori": cat, "Poäng": round(float(total_score.at[i]),1),
            "DA %": round(float(da.at[i]),2), "Vikt %": float(d.at[i,"Portföljandel (%)"]),
            "Nästa utb": d.at[i,"Nästa utbetalning (est)"], "Föreslagna st": int(n),
            "Kostnad ca": cost, "Motivering": f"Bolag ≤ {name_cap:.0f}% & {cat} ≤ {cat_cap:.0f}%"
        })
        used += cost; Vi += gross; C += gross; T += gross; cat_mv[cat] = C
        if used >= cash_sek - 1e-9 or len(rows) >= topk: break

    diag = pd.DataFrame(diag_rows, columns=["Ticker","Skäl"]).sort_values("Ticker") if diag_rows else pd.DataFrame(columns=["Ticker","Skäl"])
    cols = ["Ticker","Kategori","Poäng","DA %","Vikt %","Nästa utb","Föreslagna st","Kostnad ca","Motivering"]
    return (pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols), diag)

# --- Planerare: iterativ 500 kr-block ---
def planera_blockkop(df: pd.DataFrame, cash_sek: float, block_size: float = 500.0,
                     w_val: float=0.5, w_under: float=0.35, w_time: float=0.15,
                     max_steps: int = 200):
    targets, ignore_empty, name_default, _ov = load_settings()
    d = beräkna(df).copy()
    if d.empty or cash_sek <= 0:
        return pd.DataFrame(columns=["Steg","Ticker","Kategori","Köpta st","Pris/aktie (SEK)","Courtage","FX-avgift","Tot.avgifter","Kostnad ca","Kvar kassa"]), pd.DataFrame()

    def score_series(dd: pd.DataFrame):
        da = pd.to_numeric(dd["Direktavkastning (%)"], errors="coerce").fillna(0.0)
        da_score = (da.clip(lower=0, upper=15) / 15.0) * 100.0
        under = []
        for _, r in dd.iterrows():
            cap_pct = name_max_for(r["Ticker"])
            under.append(max(0.0, cap_pct - float(r["Portföljandel (%)"] or 0.0)))
        under = pd.Series(under, index=dd.index)
        under_score = (under / max(1.0, name_default)).clip(lower=0, upper=1) * 100.0
        def _days_to(date_str: str) -> int:
            try:
                dt = pd.to_datetime(date_str, errors="coerce")
                if pd.isna(dt): return 9999
                return max(0, (dt.date() - date.today()).days)
            except Exception: return 9999
        days = dd["Nästa utbetalning (est)"].apply(_days_to)
        time_score = ((90 - days.clip(upper=90)) / 90.0).clip(lower=0) * 100.0
        totw = max(1e-9, (w_val + w_under + w_time))
        return (w_val/totw)*da_score + (w_under/totw)*under_score + (w_time/totw)*time_score

    steps = []; used = 0.0
    for step in range(1, max_steps+1):
        d = beräkna(d)
        T = float(d["Marknadsvärde (SEK)"].sum())

        cat_mv = d.groupby("Kategori", as_index=False)["Marknadsvärde (SEK)"].sum().set_index("Kategori")["Marknadsvärde (SEK)"].to_dict()
        targets_now, ignore_empty, _, _ = load_settings()
        active_cats = {k for k,v in cat_mv.items() if v > 0} if ignore_empty else set(CATEGORY_CHOICES)
        if not active_cats: active_cats = set(CATEGORY_CHOICES)

        scores = score_series(d).sort_values(ascending=False)
        chosen = None
        for i in scores.index:
            tkr = d.at[i,"Ticker"]; cat = str(d.at[i,"Kategori"]) or "Other"
            if cat not in active_cats: continue
            price = float(pd.to_numeric(d.at[i,"Kurs (SEK)"], errors="coerce") or 0.0)
            if price <= 0: continue
            n_block = max(1, math.ceil(block_size / price))

            Vi = float(d.at[i,"Marknadsvärde (SEK)"])
            C  = float(cat_mv.get(cat, 0.0))
            name_cap = name_max_for(tkr)
            cat_cap  = float(targets_now.get(cat, 0.0))

            n_name = _cap_by_name(Vi, T, price, name_cap)
            n_cat  = _cap_by_category(C, T, price, cat_cap)
            n = min(n_block, n_name, n_cat)
            if n <= 0: continue

            gross = price * n
            c_fee, fx_fee, tot_fee = calc_fees(gross, is_foreign(d.at[i,"Valuta"]))
            cost = round(gross + tot_fee, 2)

            if used + cost > cash_sek + 1e-9:
                n_try = max(0, int((cash_sek - used) // price))
                n_try = min(n_try, n_name, n_cat)
                if n_try <= 0: continue
                n = n_try
                gross = price * n
                c_fee, fx_fee, tot_fee = calc_fees(gross, is_foreign(d.at[i,"Valuta"]))
                cost = round(gross + tot_fee, 2)

            idx = d.index[d["Ticker"]==tkr][0]
            d.at[idx, "Antal aktier"] = float(d.at[idx, "Antal aktier"] or 0.0) + n
            used += cost
            steps.append({
                "Steg": step, "Ticker": tkr, "Kategori": cat,
                "Köpta st": int(n), "Pris/aktie (SEK)": round(price,2),
                "Courtage": float(c_fee), "FX-avgift": float(fx_fee), "Tot.avgifter": float(tot_fee),
                "Kostnad ca": cost, "Kvar kassa": round(max(0.0, cash_sek - used), 2)
            })
            chosen = True
            break

        if not chosen or used >= cash_sek - 1e-9:
            break

    step_df = pd.DataFrame(steps)
    if step_df.empty:
        return pd.DataFrame(columns=["Steg","Ticker","Kategori","Köpta st","Pris/aktie (SEK)","Courtage","FX-avgift","Tot.avgifter","Kostnad ca","Kvar kassa"]), pd.DataFrame()
    summary = step_df.groupby(["Ticker","Kategori"], as_index=False).agg({"Köpta st":"sum","Kostnad ca":"sum"})
    return step_df, summary

# --- Lägg till / uppdatera bolag ---
def lagg_till_eller_uppdatera(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("➕ Lägg till / ✏ Uppdatera bolag")
    d = säkerställ_kolumner(df).copy()

    tickers = ["Ny"] + sorted(d["Ticker"].unique().tolist())
    val = st.selectbox("Välj bolag", options=tickers)

    if val == "Ny":
        ticker = st.text_input("Ticker").strip().upper()
        antal  = st.number_input("Antal aktier", min_value=0, value=0, step=1)
        gav    = st.number_input("GAV (i aktiens valuta)", min_value=0.0, value=0.0, step=0.01)
        kategori = st.selectbox("Kategori", options=CATEGORY_CHOICES, index=CATEGORY_CHOICES.index("Other"))
        lås = st.checkbox("Lås utdelning (använd manuell om angiven)", value=False)
        man_utd = st.number_input("Utdelning/år (manuell, lokal valuta)", min_value=0.0, step=0.01)
    else:
        r = d[d["Ticker"]==val].iloc[0]
        ticker = st.text_input("Ticker", value=r["Ticker"]).strip().upper()
        antal  = st.number_input("Antal aktier", min_value=0, value=int(float(r["Antal aktier"] or 0)), step=1)
        gav    = st.number_input("GAV (i aktiens valuta)", min_value=0.0, value=float(r["GAV"] or 0.0), step=0.01)
        kategori = st.selectbox("Kategori", options=CATEGORY_CHOICES,
                                index=CATEGORY_CHOICES.index(r.get("Kategori","Other")) if r.get("Kategori","Other") in CATEGORY_CHOICES else CATEGORY_CHOICES.index("Other"))
        lås = st.checkbox("Lås utdelning (använd manuell om angiven)", value=bool(r.get("Lås utdelning", False)))
        man_utd = st.number_input("Utdelning/år (manuell, lokal valuta)", min_value=0.0, value=float(r.get("Utdelning/år (manuell)",0.0)), step=0.01)

    c1, c2 = st.columns(2)
    with c1:
        fetch = st.button("🌐 Hämta data från Yahoo")
    with c2:
        save  = st.button("💾 Spara bolag till Google Sheets")

    if fetch and ticker:
        vals = hamta_yahoo_data(ticker)
        if vals:
            st.info(f"Hämtat: kurs {vals['kurs']} {vals['valuta']}, utd/år {vals['utdelning']}, frekvens {vals['frekvens_text']}, ex-date {vals['ex_date']}")

    if save:
        if not ticker:
            st.error("Ticker måste anges."); return d
        # upsert rad
        if (d["Ticker"]==ticker).any():
            m = d["Ticker"]==ticker
        else:
            d = pd.concat([d, pd.DataFrame([{"Ticker": ticker}])], ignore_index=True)
            m = d["Ticker"]==ticker

        d.loc[m,"Antal aktier"] = float(antal)
        d.loc[m,"GAV"] = float(gav)
        d.loc[m,"Kategori"] = kategori
        d.loc[m,"Lås utdelning"] = bool(lås)
        d.loc[m,"Utdelning/år (manuell)"] = float(man_utd)

        vals = hamta_yahoo_data(ticker)
        if vals:
            d.loc[m,"Aktuell kurs"] = vals.get("kurs") or d.loc[m,"Aktuell kurs"]
            if vals.get("valuta"): d.loc[m,"Valuta"] = vals["valuta"]
            if not lås and float(vals.get("utdelning") or 0.0) > 0:
                d.loc[m,"Utdelning/år"] = float(vals["utdelning"])
            f = int(vals.get("frekvens") or 0)
            ft = vals.get("frekvens_text") or ""
            xd = vals.get("ex_date") or ""
            if f > 0: d.loc[m,"Frekvens/år"] = f
            if ft:    d.loc[m,"Utdelningsfrekvens"] = ft
            if xd:    d.loc[m,"Ex-Date"] = xd
            d.loc[m,"Källa"] = "Yahoo"
            d.loc[m,"Senaste uppdatering"] = vals.get("uppdaterad","")

        d = beräkna(d)
        spara_data(d)
        st.success(f"{ticker} sparad.")
    return d

# --- Portföljöversikt ---
def portfolj_oversikt(df: pd.DataFrame) -> pd.DataFrame:
    d = beräkna(df).copy()
    st.subheader("📦 Portföljöversikt")

    tot_mv  = float(d["Marknadsvärde (SEK)"].sum())
    tot_ins = float(d["Insatt (SEK)"].sum())
    tot_pl  = float(d["Orealiserad P/L (SEK)"].sum())
    tot_div = float(d["Årlig utdelning (SEK)"].sum())

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Portföljvärde", f"{round(tot_mv,2):,}".replace(",", " "))
    c2.metric("Insatt kapital", f"{round(tot_ins,2):,}".replace(",", " "))
    c3.metric("Orealiserad P/L", f"{round(tot_pl,2):,}".replace(",", " "), delta=f"{(0 if tot_ins==0 else (100*tot_pl/tot_ins)):.2f}%")
    c4.metric("Årsutdelning", f"{round(tot_div,2):,}".replace(",", " "))

    st.dataframe(d[[
        "Ticker","Bolagsnamn","Valuta","Kategori",
        "Antal aktier","GAV","Aktuell kurs","Kurs (SEK)",
        "Insatt (SEK)","Marknadsvärde (SEK)","Orealiserad P/L (SEK)","Orealiserad P/L (%)",
        "Direktavkastning (%)","Utdelning/år","Utdelning/år (manuell)","Lås utdelning","Utdelningskälla",
        "Utdelningsfrekvens","Frekvens/år","Årlig utdelning (SEK)",
        "Ex-Date","Nästa utbetalning (est)","Portföljandel (%)","Senaste uppdatering"
    ]], use_container_width=True, hide_index=True)

    trims = trim_suggestions(d)
    if not trims.empty:
        st.warning("Innehav över per-bolagstak – förslag att skala ned:")
        st.dataframe(trims, use_container_width=True)
    return d

# --- Köpförslag (engång) ---
def page_buy_suggestions(df: pd.DataFrame):
    st.subheader("🎯 Köpförslag (engångsberäkning)")
    c1,c2,c3,c4 = st.columns([1,1,1,1])
    with c1:
        cash = st.number_input("Tillgänglig kassa (SEK)", min_value=0.0, value=500.0, step=100.0)
    with c2:
        w_val   = st.slider("Vikt: Värdering (DA)", 0.0, 1.0, 0.50, 0.05)
    with c3:
        w_under = st.slider("Vikt: Undervikt vs bolagstak", 0.0, 1.0, 0.35, 0.05)
    with c4:
        w_time  = st.slider("Vikt: Timing (nära utdelning)", 0.0, 1.0, 0.15, 0.05)

    if st.button("Beräkna köpförslag"):
        sug, diag = suggest_buys(df, cash_sek=cash, w_val=w_val, w_under=w_under, w_time=w_time, topk=10)
        if sug.empty:
            st.info("Inga köpförslag som klarar reglerna just nu.")
        else:
            st.dataframe(sug, use_container_width=True)
        with st.expander("Varför filtrerades vissa bort? (diagnostik)"):
            st.dataframe(diag, use_container_width=True)

# --- Planerare 500 kr-block (UI + minne + spara logg) ---
def page_planner(df: pd.DataFrame):
    st.subheader("🧩 Planerare – köp i 500 kr‑block (iterativ)")
    c1,c2,c3,c4 = st.columns([1,1,1,1])
    with c1:
        cash = st.number_input("Tillgänglig kassa (SEK)", min_value=0.0, value=2000.0, step=100.0)
    with c2:
        block = st.number_input("Blockstorlek per köp (SEK)", min_value=100.0, value=500.0, step=50.0)
    with c3:
        w_under = st.slider("Vikt: Undervikt", 0.0, 1.0, 0.35, 0.05)
    with c4:
        w_val = st.slider("Vikt: Värdering (DA)", 0.0, 1.0, 0.50, 0.05)
    w_time = st.slider("Vikt: Timing (nära utdelning)", 0.0, 1.0, 0.15, 0.05)

    if st.button("Planera köpsekvens"):
        steps, summary = planera_blockkop(df, cash_sek=cash, block_size=block, w_val=w_val, w_under=w_under, w_time=w_time)
        if steps.empty:
            st.info("Ingen plan kunde skapas med givna regler och kassa.")
            return
        st.write("**Steg-för-steg:**")
        st.dataframe(steps, use_container_width=True)
        st.write("**Summering per ticker:**")
        st.dataframe(summary, use_container_width=True)

        if "pending_plan" not in st.session_state:
            st.session_state["pending_plan"] = []
        colA, colB, colC = st.columns([1,1,1])
        with colA:
            if st.button("➕ Lägg denna plan i minnet"):
                st.session_state["pending_plan"] = steps.to_dict(orient="records")
                st.success("Planen ligger i minnet. Du kan rensa den när som helst.")
        with colB:
            if st.button("💾 Spara plan som logg i Google Sheets"):
                ok, msg = save_plan_to_sheet(steps.to_dict(orient="records"))
                (st.success if ok else st.error)(msg)
        with colC:
            st.text_input("Notering (valfritt)", key="plan_note", placeholder="t.ex. Testkörning")

    if "pending_plan" in st.session_state and st.session_state["pending_plan"]:
        st.info(f"Plan i minnet: {len(st.session_state['pending_plan'])} steg")
        st.dataframe(pd.DataFrame(st.session_state["pending_plan"]), use_container_width=True)
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("💾 Spara MINNES‑plan som logg"):
                ok, msg = save_plan_to_sheet(st.session_state["pending_plan"], note=st.session_state.get("plan_note",""))
                (st.success if ok else st.error)(msg)
        with c2:
            if st.button("🗑️ Rensa plan‑minne"):
                st.session_state["pending_plan"] = []
                st.success("Planminnet rensat.")

# --- Uppdatera EN ---
def uppdatera_enskild(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🔄 Uppdatera EN ticker från Yahoo")
    d = säkerställ_kolumner(df).copy()
    if d.empty:
        st.info("Ingen data i portföljen ännu."); return d

    tickers = sorted(d["Ticker"].astype(str).unique().tolist())
    tkr = st.selectbox("Välj ticker", options=tickers)

    if st.button("Hämta & uppdatera"):
        m = d["Ticker"] == tkr
        vals = hamta_yahoo_data(tkr)
        if not vals:
            st.warning("Kunde inte hämta Yahoo-data."); 
            return d

        locked = bool(d.loc[m, "Lås utdelning"].iloc[0])
        new_div = float(vals.get("utdelning") or 0.0)

        d.loc[m, "Aktuell kurs"] = vals.get("kurs") or d.loc[m, "Aktuell kurs"]
        if vals.get("valuta"): d.loc[m, "Valuta"] = vals["valuta"]
        if new_div > 0 and not locked:
            d.loc[m, "Utdelning/år"] = new_div

        f  = int(vals.get("frekvens") or 0)
        ft = vals.get("frekvens_text") or ""
        xd = vals.get("ex_date") or ""
        if f  > 0: d.loc[m, "Frekvens/år"] = f
        if ft:     d.loc[m, "Utdelningsfrekvens"] = ft
        if xd:     d.loc[m, "Ex-Date"] = xd
        d.loc[m, "Källa"] = "Yahoo"
        if vals.get("uppdaterad"):
            d.loc[m, "Senaste uppdatering"] = vals["uppdaterad"]

        d = beräkna(d)
        spara_data(d)
        st.success(f"{tkr} uppdaterad och sparad.")
    return d

# --- Massuppdatera Alla ---
def massuppdatera(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("⏩ Massuppdatera ALLA från Yahoo")
    d = säkerställ_kolumner(df).copy()
    if d.empty:
        st.info("Ingen data i portföljen ännu."); return d

    tickers = d["Ticker"].astype(str).tolist()
    st.write(f"Antal tickers: {len(tickers)}")

    if not st.button("Starta massuppdatering"):
        return d

    pb = st.progress(0)
    status = st.empty()
    errors = []

    for i, tkr in enumerate(tickers, start=1):
        status.text(f"Hämtar {tkr} ({i}/{len(tickers)})…")
        try:
            vals = hamta_yahoo_data(tkr)
            if vals:
                m = d["Ticker"] == tkr
                locked = bool(d.loc[m, "Lås utdelning"].iloc[0])
                new_div = float(vals.get("utdelning") or 0.0)

                d.loc[m, "Aktuell kurs"] = vals.get("kurs") or d.loc[m, "Aktuell kurs"]
                if vals.get("valuta"): d.loc[m, "Valuta"] = vals["valuta"]
                if new_div > 0 and not locked:
                    d.loc[m, "Utdelning/år"] = new_div

                f  = int(vals.get("frekvens") or 0)
                ft = vals.get("frekvens_text") or ""
                xd = vals.get("ex_date") or ""
                if f  > 0: d.loc[m, "Frekvens/år"] = f
                if ft:     d.loc[m, "Utdelningsfrekvens"] = ft
                if xd:     d.loc[m, "Ex-Date"] = xd
                d.loc[m, "Källa"] = "Yahoo"
                if vals.get("uppdaterad"):
                    d.loc[m, "Senaste uppdatering"] = vals["uppdaterad"]
            else:
                errors.append(tkr)
        except Exception as e:
            errors.append(f"{tkr} ({e})")

        pb.progress(i / len(tickers))
        time.sleep(1.0)

    d = beräkna(d)
    spara_data(d)
    status.text("Massuppdatering klar ✅")
    if errors:
        st.warning("Klar, men problem med: " + ", ".join(errors))
    else:
        st.success("Alla tickers uppdaterade utan fel.")
    return d

# --- Köp/Sälj (in-memory, ändrar ej Sheets) ---
def block_trading(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🛒 Köp / 📤 Sälj (in-memory)")
    if df.empty:
        st.info("Lägg till minst en ticker först.")
        return df

    d = säkerställ_kolumner(df).copy()
    tickers = d["Ticker"].astype(str).tolist()
    tkr = st.selectbox("Ticker", options=tickers)
    side = st.radio("Typ", ["KÖP","SÄLJ"], horizontal=True)
    qty  = st.number_input("Antal", min_value=1, value=10, step=1)

    cur_default = d.loc[d["Ticker"]==tkr, "Valuta"].iloc[0] if (d["Ticker"]==tkr).any() else "SEK"
    px_local = st.number_input("Pris (lokal valuta)", min_value=0.0, value=float(d.loc[d["Ticker"]==tkr,"Aktuell kurs"].iloc[0] if (d["Ticker"]==tkr).any() else 0.0), step=0.01)
    fx = fx_for(cur_default)
    px_sek = round(px_local * fx, 6)
    gross  = round(px_sek * qty, 2)
    fee_court, fee_fx, fee_tot = calc_fees(gross, foreign=is_foreign(cur_default))
    net = round(gross + fee_tot, 2) if side=="KÖP" else round(gross - fee_tot, 2)

    st.caption(f"Pris (SEK): {px_sek} | Brutto: {gross} | Avgifter: {fee_tot} | {'Totalkostnad' if side=='KÖP' else 'Nettolikvid'}: {net}")

    if st.button("Lägg order (in-memory)"):
        i = d.index[d["Ticker"]==tkr][0]
        if side == "KÖP":
            d.at[i,"Antal aktier"] = float(d.at[i,"Antal aktier"] or 0.0) + qty
        else:
            new_q = float(d.at[i,"Antal aktier"] or 0.0) - qty
            if new_q < 0: st.error("Sälj ger negativt antal."); return d
            d.at[i,"Antal aktier"] = new_q
        st.session_state["working_df"] = beräkna(d)
        st.success(f"{side} registrerad i minnet (ej sparad).")
    return st.session_state.get("working_df", d)

# --- Kalender ---
def _gen_payment_dates(first_ex_date: str, freq_per_year: float, payment_lag_days: float, months_ahead: int = 12):
    ts = pd.to_datetime(first_ex_date, errors="coerce")
    if pd.isna(ts): return []
    exd = ts.date()
    try: freq = int(float(freq_per_year))
    except: freq = 4
    freq = max(freq, 1)
    try: lag = int(float(payment_lag_days))
    except: lag = 30
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

def page_calendar(df: pd.DataFrame):
    st.subheader("📅 Utdelningskalender")
    months = st.selectbox("Prognoshorisont", options=[12,24,36], index=0)
    d = beräkna(df).copy()
    rows = []
    for _, r in d.iterrows():
        per_share_local = float(r.get("Utdelning/år_eff",0.0)) / max(1.0, float(r.get("Frekvens/år",4)))
        qty = float(r.get("Antal aktier",0.0))
        fx  = fx_for(r.get("Valuta","SEK"))
        per_pay_sek = per_share_local * fx * qty
        if per_pay_sek <= 0: continue
        pays = _gen_payment_dates(r.get("Ex-Date",""), r.get("Frekvens/år",4), r.get("Payment-lag (dagar)",30), months_ahead=months)
        for p in pays:
            rows.append({"Datum": p, "Ticker": r["Ticker"], "Belopp (SEK)": round(per_pay_sek,2)})
    if not rows:
        st.info("Ingen prognos – saknar data.")
        return
    cal = pd.DataFrame(rows)
    cal["Månad"] = cal["Datum"].apply(lambda d: f"{d.year}-{str(d.month).zfill(2)}")
    monthly = cal.groupby("Månad", as_index=False)["Belopp (SEK)"].sum().rename(columns={"Belopp (SEK)":"Utdelning (SEK)"}).sort_values("Månad")
    st.dataframe(monthly, use_container_width=True)
    st.bar_chart(monthly.set_index("Månad")["Utdelning (SEK)"])

# --- Spara manuellt ---
def page_save_now(df: pd.DataFrame):
    st.subheader("💾 Spara till Google Sheets")
    preview = beräkna( säkerställ_kolumner(df) )
    st.write("Antal rader som sparas:", len(preview))
    st.dataframe(preview, use_container_width=True)
    if st.button("✅ Bekräfta och spara"):
        spara_data(preview)
        st.success("Data sparad!")
    return preview

# --- Fördelning & bolagstak (Settings-vy) ---
def page_settings_distribution(df: pd.DataFrame):
    st.subheader("⚖️ Mål per kategori & bolagstak")
    targets, ignore_empty, name_default, name_overrides = load_settings()

    st.markdown("#### Mål/tak per kategori (%)")
    edits = {}
    cols = st.columns(3)
    for i, cat in enumerate(CATEGORY_CHOICES):
        with cols[i % 3]:
            edits[cat] = st.number_input(f"{cat}", min_value=0.0, max_value=100.0,
                                         value=float(targets.get(cat, 0.0)), step=1.0)

    st.markdown("---")
    st.markdown("#### Max %-vikt per enskilt bolag")
    c1, c2 = st.columns([1,2])
    with c1:
        new_default = st.number_input("Standardtak per bolag (%)", min_value=1.0, max_value=100.0,
                                      value=float(name_default), step=0.5,
                                      help="Gäller alla bolag som inte har ett eget override.")
    with c2:
        st.caption("Overrides per ticker (valfritt). Lämna tomt för att använda standardtaket.")
        d = säkerställ_kolumner(df).copy()
        tickers = sorted(d["Ticker"].astype(str).unique().tolist())
        ov_inputs = {}
        cols2 = st.columns(3)
        for i, tkr in enumerate(tickers):
            with cols2[i % 3]:
                current = name_overrides.get(tkr.upper(), None)
                val = st.text_input(f"{tkr} max %", value=(str(current) if current is not None else ""))
                ov_inputs[tkr.upper()] = val

    st.markdown("---")
    c3, c4 = st.columns([1,1])
    with c3:
        ignore = st.checkbox("Ignorera kategorier utan ägda aktier", value=ignore_empty)
    if c4.button("💾 Spara till Google Sheets"):
        clean_ov = {}
        for tkr, txt in ov_inputs.items():
            s = str(txt).strip()
            if not s:
                continue
            try:
                clean_ov[tkr] = float(s)
            except:
                pass
        save_settings(edits, ignore, new_default, clean_ov)
        st.success("Mål, bolagstak och overrides sparade.")
        _rerun()

# --- Sidopanel ---
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

    if st.sidebar.button("↩︎ Återställ FX till standard"):
        for k, v in DEF_FX.items(): st.session_state[k] = v
        _rerun()

# --- Main ---
def main():
    st.title("Relative Yield – utdelningsportfölj")

    # Läs in Google Sheet & beräkna direkt
    if "working_df" not in st.session_state:
        loaded = migrate_sheet_columns()
        st.session_state["working_df"] = beräkna(loaded)

    base = beräkna(st.session_state["working_df"])
    st.session_state["working_df"] = base

    sidopanel()
    st.sidebar.caption(f"📄 Rader i databasen: {len(base)}")

    page = st.sidebar.radio(
        "Meny",
        [
            "➕ Lägg till / ✏ Uppdatera bolag",
            "📦 Portföljöversikt",
            "🎯 Köpförslag",
            "🧩 Planerare 500 kr‑block",
            "⚖️ Fördelning & bolagstak",
            "🛒 Köp/Sälj",
            "🔄 Uppdatera EN",
            "⏩ Massuppdatera alla",
            "📅 Utdelningskalender",
            "💾 Spara",
        ],
        index=1
    )

    if page == "➕ Lägg till / ✏ Uppdatera bolag":
        base = lagg_till_eller_uppdatera(base)
    elif page == "📦 Portföljöversikt":
        base = portfolj_oversikt(base)
    elif page == "🎯 Köpförslag":
        page_buy_suggestions(base)
    elif page == "🧩 Planerare 500 kr‑block":
        page_planner(base)
    elif page == "⚖️ Fördelning & bolagstak":
        page_settings_distribution(base)
    elif page == "🛒 Köp/Sälj":
        base = block_trading(base)
    elif page == "🔄 Uppdatera EN":
        base = uppdatera_enskild(base)
    elif page == "⏩ Massuppdatera alla":
        base = massuppdatera(base)
    elif page == "📅 Utdelningskalender":
        page_calendar(base)
    elif page == "💾 Spara":
        page_save_now(base)

    st.session_state["working_df"] = säkerställ_kolumner(base)

if __name__ == "__main__":
    main()
