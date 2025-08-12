import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time, math
from datetime import datetime, timedelta, date
from google.oauth2.service_account import Credentials

# ── Rerun shim
try:
    _rerun = st.rerun
except AttributeError:
    _rerun = st.experimental_rerun

st.set_page_config(page_title="Utdelningsportfölj", layout="wide")

# ── Secrets
SHEET_URL  = st.secrets["SHEET_URL"]
SHEET_NAME = "Bolag"

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
credentials = Credentials.from_service_account_info(
    st.secrets["GOOGLE_CREDENTIALS"], scopes=scope
)
client = gspread.authorize(credentials)

# ── Kolumnschema
COLUMNS = [
    "Ticker","Bolagsnamn","Aktuell kurs","Valuta","Kategori",
    "Direktavkastning (%)","Utdelning/år","Utdelning/år (manuell)","Lås utdelning",
    "Frekvens/år","Utdelningsfrekvens","Frekvenskälla",
    "Payment-lag (dagar)","Ex-Date","Nästa utbetalning (est)",
    "Antal aktier","GAV","Portföljandel (%)","Årlig utdelning (SEK)",
    "Kurs (SEK)","Utdelningstillväxt (%)","Utdelningskälla",
    "Senaste uppdatering","Källa","Marknadsvärde (SEK)"
]

# ── Robust GAV/nummer-parser
def _coerce_decimal(val) -> float:
    if pd.isna(val): return 0.0
    s = str(val).strip()
    if not s: return 0.0
    s = s.replace(" ", "").replace("−", "-")
    if ":" in s: s = s.replace(":", ".")
    s = "".join(ch for ch in s if ch.isdigit() or ch in ",.-")
    if not s: return 0.0
    if "," in s and "." in s:
        last = max(s.rfind(","), s.rfind("."))
        int_part = "".join(c for c in s[:last] if c.isdigit() or c == "-")
        frac_part = "".join(c for c in s[last+1:] if c.isdigit())
        norm = f"{int_part}.{frac_part}" if int_part not in ("","-") else f"0.{frac_part}"
    else:
        norm = s.replace(",", ".")
        if norm.count(".") > 1:
            parts = norm.split("."); norm = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(norm)
    except Exception:
        x = pd.to_numeric(norm, errors="coerce")
        return float(x) if pd.notna(x) else 0.0

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty: d = pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in d.columns: d[c] = ""
    d["Ticker"] = d["Ticker"].astype(str).str.strip().str.upper()
    d["Bolagsnamn"] = d["Bolagsnamn"].astype(str)
    d["Valuta"] = d["Valuta"].astype(str).str.strip().str.upper()
    d["Kategori"] = d["Kategori"].astype(str).replace({"": "QUALITY"})
    for c in ["Aktuell kurs","Utdelning/år","Utdelning/år (manuell)",
              "Frekvens/år","Payment-lag (dagar)","Antal aktier","Marknadsvärde (SEK)"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["GAV"] = d["GAV"].apply(_coerce_decimal).fillna(0.0)
    d["Lås utdelning"] = d.get("Lås utdelning", False).apply(lambda x: bool(x) if pd.notna(x) else False)
    for add in ["Frekvenskälla","Utdelningskälla","Senaste uppdatering","Källa",
                "Utdelningsfrekvens","Ex-Date","Nästa utbetalning (est)"]:
        if add not in d.columns: d[add] = ""
    return d[COLUMNS].copy()

# ── Kategoritak (tak ≠ målvikt)
MAX_CAT = {
    "QUALITY": 40.0, "REIT": 25.0, "mREIT": 10.0, "BDC": 15.0, "MLP": 20.0,
    "Shipping": 25.0, "Telecom": 20.0, "Tobacco": 20.0, "Utility": 20.0,
    "Tech": 25.0, "Bank": 20.0, "Industrial": 20.0, "Energy": 25.0,
    "Finance": 20.0, "Other": 10.0,
}
CATEGORY_CHOICES = list(MAX_CAT.keys())
GLOBAL_MAX_NAME = 7.0
def get_cat_max(cat: str) -> float:
    return float(MAX_CAT.get(str(cat or "").strip() or "QUALITY", 100.0))

# ── FX (standardvärden – kan ändras i sidopanelen)
DEF = {"USDSEK": 9.60, "NOKSEK": 0.94, "CADSEK": 6.95, "EURSEK": 11.10}
for k, v in DEF.items():
    if k not in st.session_state:
        st.session_state[k] = v

def fx_for(cur: str) -> float:
    if pd.isna(cur): return 1.0
    c = str(cur).strip().upper()
    m = {
        "USD": st.session_state.get("USDSEK", DEF["USDSEK"]),
        "EUR": st.session_state.get("EURSEK", DEF["EURSEK"]),
        "CAD": st.session_state.get("CADSEK", DEF["CADSEK"]),
        "NOK": st.session_state.get("NOKSEK", DEF["NOKSEK"]),
        "SEK": 1.0
    }
    return float(m.get(c, 1.0))

# ── Google Sheets helpers
def _open_sheet():
    try:
        return client.open_by_url(SHEET_URL)
    except Exception as e:
        st.error("❌ Kunde inte öppna kalkylbladet."); st.caption(e); return None

def _ensure_worksheet(sh, title):
    if sh is None: return None
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        try:
            ws = sh.add_worksheet(title=title, rows=2000, cols=len(COLUMNS)+5)
            if title == SHEET_NAME: ws.update([COLUMNS])
            return ws
        except Exception as e:
            st.error(f"❌ Kunde inte skapa fliken '{title}'."); st.caption(e); return None

def skapa_koppling():
    return _ensure_worksheet(_open_sheet(), SHEET_NAME)

def hamta_data():
    ws = skapa_koppling()
    if ws is None: return pd.DataFrame(columns=COLUMNS)
    try:
        rows = ws.get_all_values()
        if not rows: return pd.DataFrame(columns=COLUMNS)
        header = rows[0]; data_rows = rows[1:] if len(rows)>1 else []
        df = pd.DataFrame(data_rows, columns=header) if data_rows else pd.DataFrame(columns=header)
        return säkerställ_kolumner(df)
    except Exception as e:
        st.error("❌ Kunde inte läsa från Google Sheets."); st.caption(e)
        return pd.DataFrame(columns=COLUMNS)

def migrate_sheet_columns():
    return säkerställ_kolumner(hamta_data())

def cleanup_old_backups(days: int = 7):
    try:
        sh = _open_sheet()
        cutoff = datetime.now() - timedelta(days=days)
        for ws in sh.worksheets():
            title = (ws.title or "").strip()
            if not title.startswith("Backup_"): continue
            ts_part = title.replace("Backup_", "")
            try: ts = datetime.strptime(ts_part, "%Y%m%d_%H%M%S")
            except: continue
            if ts < cutoff:
                try: sh.del_worksheet(ws)
                except: pass
    except Exception:
        pass

def spara_data(df: pd.DataFrame):
    ws = skapa_koppling()
    if ws is None: return
    d = säkerställ_kolumner(df).copy()
    if d["Ticker"].astype(str).str.strip().eq("").all():
        st.error("Inget att spara: inga tickers."); return
    try:
        sh = _open_sheet()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bkp_title = f"Backup_{ts}"
        try:
            ws_b = _ensure_worksheet(sh, bkp_title)
            ws_b.update([d.columns.tolist()] + d.astype(str).values.tolist(), value_input_option="USER_ENTERED")
        except Exception as e:
            st.warning(f"Backup misslyckades: {e}")
        ws.clear()
        ws.update([d.columns.tolist()] + d.astype(str).values.tolist(), value_input_option="USER_ENTERED")
        cleanup_old_backups(days=7)
        st.success(f"✅ Sparade {len(d)} rader. (Backup: {bkp_title})")
    except Exception as e:
        st.error("❌ Fel vid sparande."); st.caption(e)

# ── RESET (in‑memory)
def reset_in_memory():
    """Återställ appens in-memory-data till det som ligger i Google Sheets."""
    try:
        fresh = migrate_sheet_columns()
        st.session_state["working_df"] = säkerställ_kolumner(fresh)
    except Exception:
        st.session_state["working_df"] = säkerställ_kolumner(pd.DataFrame())
    for k in ["pending_txs", "last_yahoo_vals"]:
        if k in st.session_state:
            st.session_state.pop(k)

# ── SETTINGS (målvikter) i egen flik
SETTINGS_SHEET = "Settings"
DEFAULT_TARGETS = {
    "Shipping": 10.0,
    "mREIT":   10.0,
    "REIT":    15.0,
    "Bank":    10.0,
    "BDC":     15.0,
    "Telecom": 15.0,
    "Finance": 10.0,
    "Tech":    15.0,
}
def ensure_settings_ws():
    sh = _open_sheet()
    try:
        return sh.worksheet(SETTINGS_SHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=SETTINGS_SHEET, rows=100, cols=4)
        rows = [["Nyckel","Kategori","Värde","Kommentar"],
                ["IGNORE_EMPTY_CATEGORIES","", "1","1=ignorera kategorier utan ägda aktier"],]
        for cat in CATEGORY_CHOICES:
            val = DEFAULT_TARGETS.get(cat, 0.0)
            rows.append(["TARGET", cat, str(val), "Mål %"])
        ws.update(rows, value_input_option="USER_ENTERED")
        return ws

def load_settings():
    try:
        ws = ensure_settings_ws()
        rows = ws.get_all_values()
        if not rows or len(rows) < 2: return {}, True
        header = rows[0]; data = rows[1:]
        df = pd.DataFrame(data, columns=header)
        df["Värde"] = pd.to_numeric(df["Värde"], errors="coerce").fillna(0.0)
        ignore_empty = bool(df.loc[df["Nyckel"]=="IGNORE_EMPTY_CATEGORIES","Värde"].max() > 0.0)
        tgt_df = df[df["Nyckel"]=="TARGET"]
        targets = {str(r["Kategori"]): float(r["Värde"]) for _, r in tgt_df.iterrows() if str(r["Kategori"]) in CATEGORY_CHOICES}
        return targets, ignore_empty
    except Exception:
        return {}, True

def save_settings(targets: dict, ignore_empty: bool):
    ws = ensure_settings_ws()
    rows = ws.get_all_values()
    header = rows[0] if rows else ["Nyckel","Kategori","Värde","Kommentar"]
    df = pd.DataFrame(rows[1:], columns=header) if len(rows) > 1 else pd.DataFrame(columns=header)
    for cat in CATEGORY_CHOICES:
        if not ((df["Nyckel"]=="TARGET") & (df["Kategori"]==cat)).any():
            df.loc[len(df)] = ["TARGET", cat, "0", "Mål %"]
    if not (df["Nyckel"]=="IGNORE_EMPTY_CATEGORIES").any():
        df.loc[len(df)] = ["IGNORE_EMPTY_CATEGORIES","", "1", "1=ignorera tomma kategorier"]
    for cat, pct in targets.items():
        df.loc[(df["Nyckel"]=="TARGET") & (df["Kategori"]==cat), "Värde"] = str(float(pct))
    df.loc[df["Nyckel"]=="IGNORE_EMPTY_CATEGORIES","Värde"] = "1" if ignore_empty else "0"
    out = pd.concat([pd.DataFrame([["Nyckel","Kategori","Värde","Kommentar"]]), df], ignore_index=True)
    ws.clear(); ws.update(out.values.tolist(), value_input_option="USER_ENTERED")

def get_target_cat():
    targets, _ = load_settings()
    return {k: float(v) for k, v in targets.items() if float(v) > 0.0}

def get_ignore_empty_flag():
    _, flag = load_settings()
    return bool(flag)

# ── Frekvens från historik
def _infer_frequency_from_divs(divs: pd.Series):
    if divs is None or divs.empty:
        return 0, "Oregelbunden", "Ingen historik"
    divs = divs.sort_index()
    now = pd.Timestamp.utcnow()
    last24 = divs[divs.index >= (now - pd.Timedelta(days=730))]
    last12 = divs[divs.index >= (now - pd.Timedelta(days=365))]

    def label(n): return {12:"Månads",4:"Kvartals",2:"Halvårs",1:"Års"}.get(n,"Oregelbunden")

    def freq_by_intervals(series, src_label):
        idx = series.index.sort_values()
        if len(idx) < 2:
            n = min(max(len(series), 0), 1)
            return n, label(n), f"{src_label} (count)"
        diffs = (idx[1:] - idx[:-1]).days
        med = float(pd.Series(diffs).median())
        if 20 <= med <= 45:   return 12, "Månads",  f"{src_label} (median≈{med:.0f}d)"
        if 60 <= med <= 110:  return 4,  "Kvartals",f"{src_label} (median≈{med:.0f}d)"
        if 130 <= med <= 210: return 2,  "Halvårs", f"{src_label} (median≈{med:.0f}d)"
        if 300 <= med <= 430: return 1,  "Års",     f"{src_label} (median≈{med:.0f}d)"
        n = len(series)
        if n >= 10: return 12, "Månads",  f"{src_label} (>=10 st)"
        if 3 <= n <= 5: return 4, "Kvartals", f"{src_label} (3–5 st)"
        if n == 2: return 2, "Halvårs", f"{src_label} (2 st)"
        if n == 1: return 1, "Års", f"{src_label} (1 st)"
        return 0, "Oregelbunden", "Spridda intervall"

    if len(last24) >= 2:
        f, t, src = freq_by_intervals(last24, "Historik 24m")
        if f in (12,4,2,1): return f, t, src
    if len(last12) >= 1:
        f, t, src = freq_by_intervals(last12, "Historik 12m")
        if f in (12,4,2,1): return f, t, src
    recent = divs.tail(10)
    if not recent.empty: return freq_by_intervals(recent, "Senaste 10")
    return 0, "Oregelbunden", "Ingen historik"

# ── Yahoo
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
            price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price in (None, ""):
            try:
                h = t.history(period="5d")
                if not h.empty: price = float(h["Close"].iloc[-1])
            except Exception: price = None
        price = float(price) if price not in (None, "") else 0.0

        name = info.get("shortName") or info.get("longName") or ticker
        currency = (info.get("currency") or "").upper()
        if not currency:
            try: currency = (t.fast_info.get("currency") or "").upper()
            except Exception: currency = "SEK"

        div_rate, freq, freq_text, freq_src, ex_date_str = 0.0, 0, "Oregelbunden", "Ingen historik", ""
        try:
            divs = t.dividends
            if divs is not None and not divs.empty:
                cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
                last12 = divs[divs.index >= cutoff]
                div_rate = float(last12.sum()) if not last12.empty else 0.0
                f, ft, src = _infer_frequency_from_divs(divs)
                freq, freq_text, freq_src = f, ft, src
                ex_date_str = pd.to_datetime(divs.index.max()).strftime("%Y-%m-%d")
        except Exception:
            pass

        if div_rate == 0.0:
            try:
                fwd = info.get("forwardAnnualDividendRate")
                if fwd not in (None, "", 0): div_rate = float(fwd)
            except Exception: pass
        if div_rate == 0.0:
            try:
                trailing = info.get("trailingAnnualDividendRate")
                if trailing not in (None, "", 0): div_rate = float(trailing)
            except Exception: pass

        if not ex_date_str:
            try:
                ts = info.get("exDividendDate")
                if ts not in (None, "", 0):
                    ex_date_str = pd.to_datetime(int(ts), unit="s", utc=True).strftime("%Y-%m-%d")
            except Exception: ex_date_str = ""

        return {
            "namn": name, "kurs": price, "valuta": currency,
            "utdelning": div_rate, "frekvens": freq,
            "frekvens_text": freq_text, "frekvens_källa": freq_src,
            "ex_date": ex_date_str, "källa": "Yahoo",
            "uppdaterad": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    except Exception as e:
        st.warning(f"Kunde inte hämta Yahoo-data för {ticker}: {e}")
        return None

# ── Beräkningar
def beräkna(df: pd.DataFrame) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()

    use_manual = (d["Lås utdelning"] == True) & (d["Utdelning/år (manuell)"] > 0)
    d["Utdelning/år_eff"] = pd.to_numeric(d["Utdelning/år"], errors="coerce").fillna(0.0)
    d.loc[use_manual, "Utdelning/år_eff"] = pd.to_numeric(d["Utdelning/år (manuell)"], errors="coerce").fillna(0.0)

    d["Utdelningskälla"] = [
        "Manuell 🔒" if (bool(d.at[i,"Lås utdelning"]) and float(d.at[i,"Utdelning/år (manuell)"])>0.0) else "Yahoo"
        for i in d.index
    ]

    d["Aktuell kurs"] = pd.to_numeric(d["Aktuell kurs"], errors="coerce").fillna(0.0)
    rates = d["Valuta"].apply(fx_for).astype(float)
    d["Kurs (SEK)"] = (d["Aktuell kurs"] * rates).round(6)

    d["Antal aktier"] = pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0)
    d["Årlig utdelning (SEK)"] = (d["Antal aktier"] * d["Utdelning/år_eff"] * rates).round(2)

    d["Direktavkastning (%)"] = 0.0
    ok = (d["Aktuell kurs"] > 0) & (d["Utdelning/år_eff"] > 0)
    d.loc[ok, "Direktavkastning (%)"] = (100.0 * d.loc[ok, "Utdelning/år_eff"] / d.loc[ok, "Aktuell kurs"]).round(2)

    mv = (d["Antal aktier"] * d["Kurs (SEK)"]).round(2)
    tot_mv = float(mv.sum()) if mv.sum() else 1.0
    d["Marknadsvärde (SEK)"] = mv
    d["Portföljandel (%)"] = (100.0 * mv / tot_mv).round(2)

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
        next_pay(d.at[i,"Ex-Date"], d.at[i,"Frekvens/år"], d.at[i,"Payment-lag (dagar)"]) for i in d.index
    ]
    return d

# ── Prognos
def _gen_payment_dates(first_ex_date: str, freq_per_year: float, payment_lag_days: float, months_ahead: int = 12):
    ts = pd.to_datetime(first_ex_date, errors="coerce")
    if pd.isna(ts): return []
    exd = ts.date()
    try: freq = int(float(freq_per_year))
    except: freq = 4
    try: lag = int(float(payment_lag_days))
    except: lag = 30
    freq = max(freq, 1); lag = max(lag, 0)
    step_days = max(1, int(round(365.0 / freq)))
    today_d = date.today()
    horizon = today_d + timedelta(days=int(round(months_ahead * 30.44)))
    while exd < today_d:
        exd = exd + timedelta(days=step_days)
    dates, pay = [], exd + timedelta(days=lag)
    while pay <= horizon:
        dates.append(pay); exd = exd + timedelta(days=step_days); pay = exd + timedelta(days=lag)
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
            if per_payment_sek <= 0: continue
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

# ── Mass‑refresh helper
def refresh_all_from_yahoo(df: pd.DataFrame, sleep_s: float = 1.0, show_progress: bool = True) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()
    if d.empty: return d
    tickers = d["Ticker"].astype(str).tolist(); n = len(tickers)
    prog = st.progress(0) if show_progress else None
    msg  = st.empty() if show_progress else None
    for i, tkr in enumerate(tickers, start=1):
        if show_progress:
            prog.progress(int(i * 100 / n)); msg.text(f"Uppdaterar {tkr} ({i}/{n})…")
        vals = hamta_yahoo_data(tkr)
        m = (d["Ticker"] == tkr)
        if vals:
            if vals.get("namn"):  d.loc[m, "Bolagsnamn"] = vals["namn"]
            if vals.get("kurs") is not None: d.loc[m, "Aktuell kurs"] = vals["kurs"]
            if vals.get("valuta"): d.loc[m, "Valuta"] = vals["valuta"]
            locked = bool(d.loc[m, "Lås utdelning"].iloc[0]) if m.any() else False
            if not locked:
                new_div = float(vals.get("utdelning") or 0.0)
                if new_div > 0: d.loc[m, "Utdelning/år"] = new_div
            f  = int(vals.get("frekvens") or 0)
            ft = vals.get("frekvens_text") or ""
            fs = vals.get("frekvens_källa") or ""
            xd = vals.get("ex_date") or ""
            if f > 0: d.loc[m, "Frekvens/år"] = f
            if ft:    d.loc[m, "Utdelningsfrekvens"] = ft
            if fs:    d.loc[m, "Frekvenskälla"]      = fs
            if xd:    d.loc[m, "Ex-Date"]            = xd
            d.loc[m, "Källa"] = "Yahoo"
            if vals.get("uppdaterad"):
                d.loc[m, "Senaste uppdatering"] = vals["uppdaterad"]
        time.sleep(sleep_s)
    return beräkna(d)

# ── Lägg till / Uppdatera
def lagg_till_eller_uppdatera(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("➕ Lägg till / ✏︎ Uppdatera bolag")
    d = säkerställ_kolumner(df).copy()
    tickers = ["Ny"] + sorted(d["Ticker"].astype(str).tolist())
    val = st.selectbox("Välj bolag", options=tickers)

    if val == "Ny":
        ticker   = st.text_input("Ticker", placeholder="t.ex. VICI eller 2020.OL").strip().upper()
        antal    = st.number_input("Antal aktier", min_value=0, value=0, step=1)
        gav      = st.number_input("GAV (SEK)", min_value=0.0, value=0.0, step=0.01)
        kategori = st.selectbox("Kategori", options=CATEGORY_CHOICES, index=CATEGORY_CHOICES.index("QUALITY"))
        man_utd  = st.number_input("Utdelning/år (manuell, lokal valuta)", min_value=0.0, step=0.01)
        lås      = st.checkbox("Lås utdelning (använd manuell)", value=False)
    else:
        r = d[d["Ticker"] == val].iloc[0]
        ticker   = st.text_input("Ticker", value=r["Ticker"]).strip().upper()
        antal    = st.number_input("Antal aktier", min_value=0, value=int(float(r["Antal aktier"] or 0)), step=1)
        gav      = st.number_input("GAV (SEK)", min_value=0.0, value=float(_coerce_decimal(r["GAV"])), step=0.01)
        cur_cat  = r.get("Kategori","QUALITY"); cur_idx = CATEGORY_CHOICES.index(cur_cat) if cur_cat in CATEGORY_CHOICES else 0
        kategori = st.selectbox("Kategori", options=CATEGORY_CHOICES, index=cur_idx)
        man_utd  = st.number_input("Utdelning/år (manuell, lokal valuta)", min_value=0.0, value=float(r.get("Utdelning/år (manuell)",0.0)), step=0.01)
        lås      = st.checkbox("Lås utdelning (använd manuell)", value=bool(r.get("Lås utdelning", False)))

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("🌐 Hämta data från Yahoo"):
            if not ticker:
                st.error("Ange en ticker först."); return d
            vals = hamta_yahoo_data(ticker)
            st.session_state["last_yahoo_vals"] = vals
            if vals: st.success(f"Hämtat från Yahoo för {ticker}.")
            else:    st.warning("Ingen data från Yahoo.")
    with c2:
        if st.button("➕ Lägg till / uppdatera IN-MEMORY"):
            if not ticker:
                st.error("Ticker måste anges."); return d
            if (d["Ticker"] == ticker).any():
                m = (d["Ticker"] == ticker)
            else:
                d = pd.concat([d, pd.DataFrame([{"Ticker": ticker}])], ignore_index=True)
                m = (d["Ticker"] == ticker)
            d.loc[m, "Antal aktier"] = float(antal)
            d.loc[m, "GAV"]         = float(gav)
            d.loc[m, "Kategori"]    = kategori
            d.loc[m, "Utdelning/år (manuell)"] = float(man_utd)
            d.loc[m, "Lås utdelning"] = bool(lås)

            vals = st.session_state.get("last_yahoo_vals")
            if vals:
                if vals.get("namn"): d.loc[m, "Bolagsnamn"] = vals["namn"]
                if vals.get("kurs") is not None: d.loc[m, "Aktuell kurs"] = vals["kurs"]
                if vals.get("valuta"): d.loc[m, "Valuta"] = vals["valuta"]
                if not lås:
                    new_div = float(vals.get("utdelning") or 0.0)
                    if new_div > 0: d.loc[m, "Utdelning/år"] = new_div
                f  = int(vals.get("frekvens") or 0)
                ft = vals.get("frekvens_text") or ""
                fs = vals.get("frekvens_källa") or ""
                xd = vals.get("ex_date") or ""
                if f > 0: d.loc[m, "Frekvens/år"] = f
                if ft:    d.loc[m, "Utdelningsfrekvens"] = ft
                if fs:    d.loc[m, "Frekvenskälla"]      = fs
                if xd:    d.loc[m, "Ex-Date"]            = xd
                d.loc[m, "Källa"] = "Yahoo"
                if vals.get("uppdaterad"):
                    d.loc[m, "Senaste uppdatering"] = vals["uppdaterad"]

            st.session_state["working_df"] = beräkna(d)
            st.success("Bolag uppdaterat i minnet.")
            _rerun()
    with c3:
        if st.button("💾 Spara hela tabellen till Google Sheets NU"):
            out = beräkna(d)
            spara_data(out)
            st.session_state["working_df"] = out
            _rerun()

    st.markdown("### Förhandsgranskning (in‑memory)")
    st.dataframe(beräkna(d)[["Ticker","Bolagsnamn","Valuta","Kategori","Antal aktier","GAV",
                             "Aktuell kurs","Utdelning/år","Utdelning/år (manuell)","Lås utdelning",
                             "Frekvens/år","Utdelningsfrekvens","Ex-Date"]],
                 use_container_width=True)
    return d

def page_update_single(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🔄 Uppdatera EN ticker från Yahoo")
    d = säkerställ_kolumner(df).copy()
    if d.empty:
        st.info("Lägg till minst en ticker först."); return d
    tkr = st.selectbox("Ticker", options=sorted(d["Ticker"].astype(str).tolist()))
    if st.button("Uppdatera från Yahoo"):
        vals = hamta_yahoo_data(tkr)
        m = (d["Ticker"] == tkr)
        if vals:
            if vals.get("namn"):  d.loc[m, "Bolagsnamn"] = vals["namn"]
            if vals.get("kurs") is not None: d.loc[m, "Aktuell kurs"] = vals["kurs"]
            if vals.get("valuta"): d.loc[m, "Valuta"] = vals["valuta"]
            if not bool(d.loc[m,"Lås utdelning"].iloc[0]):
                new_div = float(vals.get("utdelning") or 0.0)
                if new_div > 0: d.loc[m, "Utdelning/år"] = new_div
            f  = int(vals.get("frekvens") or 0)
            ft = vals.get("frekvens_text") or ""
            fs = vals.get("frekvens_källa") or ""
            xd = vals.get("ex_date") or ""
            if f > 0: d.loc[m, "Frekvens/år"] = f
            if ft:    d.loc[m, "Utdelningsfrekvens"] = ft
            if fs:    d.loc[m, "Frekvenskälla"]      = fs
            if xd:    d.loc[m, "Ex-Date"]            = xd
            d.loc[m, "Källa"] = "Yahoo"
            if vals.get("uppdaterad"):
                d.loc[m, "Senaste uppdatering"] = vals["uppdaterad"]
            st.session_state["working_df"] = beräkna(d)
            st.success(f"{tkr} uppdaterad.")
    return d

def page_mass_update(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("⏩ Massuppdatera alla från Yahoo")
    d = säkerställ_kolumner(df).copy()
    if d.empty:
        st.info("Ingen data att uppdatera."); return d
    sleep_s = st.slider("Paus mellan anrop (sek)", 0.0, 2.0, 1.0, 0.1)
    if st.button("Starta massuppdatering"):
        with st.spinner("Hämtar från Yahoo…"):
            d2 = refresh_all_from_yahoo(d, sleep_s=sleep_s, show_progress=True)
        st.session_state["working_df"] = d2
        st.success("Klart! Glöm inte att spara till Google Sheets om du vill skriva tillbaka.")
    return st.session_state.get("working_df", d)

# ── Enkel portföljvy
def portfolj_oversikt(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("📦 Portföljöversikt")
    d = beräkna(df).copy()
    d["Insatt (SEK)"] = (pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0) * pd.to_numeric(d["GAV"], errors="coerce").fillna(0.0)).round(2)
    d["Orealiserad P/L (SEK)"] = (d["Marknadsvärde (SEK)"] - d["Insatt (SEK)"]).round(2)
    tot_mv, tot_ins = float(d["Marknadsvärde (SEK)"].sum()), float(d["Insatt (SEK)"].sum())
    tot_pl, tot_div = float(d["Orealiserad P/L (SEK)"].sum()), float(d["Årlig utdelning (SEK)"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portföljvärde", f"{round(tot_mv,2):,}".replace(",", " "))
    c2.metric("Insatt kapital", f"{round(tot_ins,2):,}".replace(",", " "))
    c3.metric("Orealiserad P/L", f"{round(tot_pl,2):,}".replace(",", " "))
    c4.metric("Årsutdelning", f"{round(tot_div,2):,}".replace(",", " "))
    st.dataframe(d[["Ticker","Bolagsnamn","Valuta","Kategori","Antal aktier","GAV","Kurs (SEK)","Marknadsvärde (SEK)","Direktavkastning (%)","Årlig utdelning (SEK)","Ex-Date","Nästa utbetalning (est)"]], use_container_width=True)
    return d

# ── Settings-sida (mål)
def page_settings_distribution(df: pd.DataFrame):
    st.subheader("⚖️ Mål & fördelning per kategori")
    targets, ignore_empty = load_settings()
    cur = {cat: float(targets.get(cat, 0.0)) for cat in CATEGORY_CHOICES}
    edits = {}
    cols = st.columns(3)
    for i, cat in enumerate(CATEGORY_CHOICES):
        with cols[i % 3]:
            edits[cat] = st.number_input(f"{cat}", min_value=0.0, max_value=100.0,
                                         value=float(cur.get(cat, 0.0)), step=1.0, key=f"tgt_{cat}")
    st.markdown("---")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        ignore = st.checkbox("Ignorera kategorier utan ägda aktier", value=ignore_empty)
    with c2:
        if st.button("💾 Spara mål till Google Sheets"):
            save_settings(edits, ignore); st.success("Mål & flagga sparade.")
    with c3:
        total = sum(edits.values())
        st.metric("Summa angivna mål (%)", f"{total:.1f}")

# ── Avgifter (mini)
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

# ── Köpförslag (mål + ignorera tomma kategorier)
def suggest_buys(df: pd.DataFrame, cash_sek: float, w_val: float=0.5, w_cat: float=0.35, w_time: float=0.15, topk: int=5) -> pd.DataFrame:
    d = beräkna(df).copy()
    if d.empty or cash_sek <= 0:
        return pd.DataFrame(columns=["Ticker","Kategori","Poäng","DA %","Vikt %","Nästa utb","Föreslagna st","Kostnad ca","Motivering"])

    targets = get_target_cat()
    present_cats = set(d["Kategori"].astype(str).unique())
    targets = {k: v for k, v in targets.items() if k in present_cats}
    ignore_empty = get_ignore_empty_flag()

    d_owned = d.copy()
    d_owned["Antal aktier"] = pd.to_numeric(d_owned["Antal aktier"], errors="coerce").fillna(0.0)
    d_owned = d_owned[d_owned["Antal aktier"] > 0]
    T_now = float((d_owned["Antal aktier"] * d_owned["Kurs (SEK)"]).sum())
    if T_now > 0:
        cat_now = d_owned.groupby("Kategori", as_index=False).agg(**{"Värde (SEK)": ("Marknadsvärde (SEK)", "sum")})
        cat_now["Nu %"] = 100.0 * cat_now["Värde (SEK)"] / T_now
        now_map = dict(zip(cat_now["Kategori"], cat_now["Nu %"]))
    else:
        now_map = {}

    diags, keep_idx = [], []
    T = float(d["Marknadsvärde (SEK)"].sum())
    cat_values = d.groupby("Kategori", as_index=False)["Marknadsvärde (SEK)"].sum().set_index("Kategori")["Marknadsvärde (SEK)"].to_dict()
    for i, r in d.iterrows():
        price = float(pd.to_numeric(r["Kurs (SEK)"], errors="coerce") or 0.0)
        if price <= 0: continue
        cat = str(r["Kategori"]) if str(r["Kategori"]).strip() else "QUALITY"
        Vi = float(r["Marknadsvärde (SEK)"]); C = float(cat_values.get(cat, 0.0))

        m = GLOBAL_MAX_NAME/100.0
        if (Vi + price) / (T + price) > m:
            diags.append((r["Ticker"], "1 st skulle överskrida bolagstak 12.0%")); continue
        M = get_cat_max(cat)/100.0
        if (C + price) / (T + price) > M:
            diags.append((r["Ticker"], f"1 st skulle överskrida kategori-tak {get_cat_max(cat):.1f}%")); continue
        keep_idx.append(i)

    if not keep_idx:
        st.info("Inga köpförslag som klarar reglerna just nu.")
        if diags:
            with st.expander("Varför filtrerades vissa bort? (diagnostik)"):
                st.dataframe(pd.DataFrame(diags, columns=["Ticker","Skäl"]))
        return pd.DataFrame(columns=["Ticker","Kategori","Poäng","DA %","Vikt %","Nästa utb","Föreslagna st","Kostnad ca","Motivering"])

    da = pd.to_numeric(d.loc[keep_idx, "Direktavkastning (%)"], errors="coerce").fillna(0.0)
    da_score = (da.clip(lower=0, upper=15) / 15.0) * 100.0

    cat_gap_list = []
    for i in keep_idx:
        cat = str(d.at[i,"Kategori"])
        tgt = float(targets.get(cat, 0.0))
        cur = float(now_map.get(cat, 0.0))
        if ignore_empty and (T_now > 0) and (tgt > 0.0) and (cur == 0.0):
            gap = 0.0
        else:
            gap = max(0.0, tgt - cur)
        cat_gap_list.append(gap)
    cat_gap = pd.Series(cat_gap_list, index=keep_idx, dtype=float)
    max_gap = float(cat_gap.max()) if len(cat_gap)>0 else 0.0
    cat_score = (100.0 * cat_gap / max_gap) if max_gap > 0 else cat_gap*0.0

    def _days_to(date_str: str) -> int:
        try:
            dt = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(dt): return 9999
            return max(0, (dt.date() - date.today()).days)
        except Exception:
            return 9999
    days = d.loc[keep_idx, "Nästa utbetalning (est)"].apply(_days_to)
    time_score = ((90 - days.clip(upper=90)) / 90.0).clip(lower=0) * 100.0

    totw = max(1e-9, (w_val + w_cat + w_time))
    w_val, w_cat, w_time = w_val/totw, w_cat/totw, w_time/totw
    total_score = (w_val*da_score + w_cat*cat_score + w_time*time_score)

    order = total_score.sort_values(ascending=False).index
    rows = []; used = 0.0
    for i in order:
        tkr = d.at[i,"Ticker"]
        price = float(pd.to_numeric(d.at[i,"Kurs (SEK)"], errors="coerce") or 0.0)
        if price <= 0: continue
        cat = str(d.at[i,"Kategori"]) if str(d.at[i,"Kategori"]).strip() else "QUALITY"
        Vi = float(d.at[i,"Marknadsvärde (SEK)"]); C = float(cat_values.get(cat, 0.0))

        rem = max(0.0, cash_sek - used)
        if rem < price: continue
        m = GLOBAL_MAX_NAME/100.0
        n_name = max(0, math.floor(((m*(T+rem)) - Vi) / max(1e-9, (1-m)*price)))
        M = get_cat_max(cat)/100.0
        n_cat  = max(0, math.floor(((M*(T+rem)) - C) / max(1e-9, (1-M)*price)))
        n_cap = int(max(0, min(n_name, n_cat)))
        if n_cap <= 0: continue

        n = n_cap
        gross = price * n
        _, _, fee = calc_fees(gross, is_foreign(d.at[i,"Valuta"]))
        while n > 0 and gross + fee > rem + 1e-9:
            n -= 1; gross = price * n; _, _, fee = calc_fees(gross, is_foreign(d.at[i,"Valuta"]))
        if n <= 0: continue

        cost = round(gross + fee, 2)
        rows.append({"Ticker": tkr, "Kategori": cat, "Poäng": round(float(total_score.at[i]),1),
                     "DA %": round(float(da.at[i]),2), "Vikt %": float(d.at[i,"Portföljandel (%)"]),
                     "Nästa utb": d.at[i,"Nästa utbetalning (est)"], "Föreslagna st": int(n),
                     "Kostnad ca": cost, "Motivering": f"Kategori-mål & tak ok"})
        used += cost; Vi += gross; C += gross; T += gross; cat_values[cat] = C
        if used >= cash_sek - 1e-9 or len(rows) >= topk: break

    cols = ["Ticker","Kategori","Poäng","DA %","Vikt %","Nästa utb","Föreslagna st","Kostnad ca","Motivering"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)

# ── Köpförslag‑UI + “Använd köpplanen”
def page_buy_suggestions(df: pd.DataFrame):
    st.subheader("🎯 Köpförslag (målvikter & tak beaktas)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cash = st.number_input("Tillgänglig kassa (SEK)", min_value=0.0, value=500.0, step=100.0)
    with c2:
        w_val = st.slider("Vikt: Värdering (DA)", 0.0, 1.0, 0.50, 0.05)
    with c3:
        w_cat = st.slider("Vikt: Kategori-gap vs mål", 0.0, 1.0, 0.35, 0.05)
    with c4:
        w_time = st.slider("Vikt: Timing (nära utdelning)", 0.0, 1.0, 0.15, 0.05)

    if st.button("Beräkna köpförslag"):
        sug = suggest_buys(df, cash_sek=cash, w_val=w_val, w_cat=w_cat, w_time=w_time, topk=5)
        if sug.empty:
            st.info("Inga köpförslag som klarar reglerna just nu.")
        else:
            st.dataframe(sug, use_container_width=True)
            st.caption("Poäng = DA + kategori-gap + hur nära nästa utdelning.")

            # ——— Använd köpplanen direkt (in‑memory) ———
            apply_now = st.checkbox("Använd köpplanen direkt (in‑memory)")
            if apply_now and st.button("🚀 Genomför föreslagna köp i minnet"):
                base = säkerställ_kolumner(st.session_state.get("working_df", df)).copy()
                if "pending_txs" not in st.session_state:
                    st.session_state["pending_txs"] = []
                for _, row in sug.iterrows():
                    tkr = str(row["Ticker"])
                    n   = int(row["Föreslagna st"])
                    if n <= 0: 
                        continue
                    m = base["Ticker"].astype(str) == tkr
                    if not m.any():
                        st.warning(f"{tkr} finns inte i tabellen – hoppar över.")
                        continue

                    px_local = float(pd.to_numeric(base.loc[m, "Aktuell kurs"], errors="coerce").fillna(0.0).iloc[0])
                    ccy      = str(base.loc[m, "Valuta"].iloc[0]).upper() or "SEK"
                    fx       = fx_for(ccy)
                    px_sek   = round(px_local * fx, 6)
                    gross    = round(px_sek * n, 2)
                    foreign  = (ccy != "SEK")
                    fee_c, fee_fx, fee_tot = calc_fees(gross, foreign)
                    cost_sek = gross + fee_tot

                    old_qty = float(pd.to_numeric(base.loc[m, "Antal aktier"], errors="coerce").fillna(0.0).iloc[0])
                    old_gav = float(pd.to_numeric(base.loc[m, "GAV"], errors="coerce").fillna(0.0).iloc[0])
                    new_qty = old_qty + n
                    new_gav = 0.0 if new_qty == 0 else round(((old_gav * old_qty) + cost_sek) / new_qty, 6)
                    base.loc[m, "Antal aktier"] = new_qty
                    base.loc[m, "GAV"] = new_gav

                    st.session_state["pending_txs"].append({
                        "Tid": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Typ": "KÖP", "Ticker": tkr, "Antal": n,
                        "Pris (lokal)": float(px_local), "Valuta": ccy, "FX": float(fx),
                        "Pris (SEK)": float(px_sek), "Belopp (SEK)": float(gross),
                        "Courtage (SEK)": float(fee_c), "FX-avgift (SEK)": float(fee_fx),
                        "Tot.avgifter (SEK)": float(fee_tot), "Kommentar": "auto från köpförslag"
                    })

                st.session_state["working_df"] = beräkna(base)
                st.success("Köpplanen är inlagd i minnet. Gå till “💾 Spara” om du vill skriva till Google Sheets.")
                _rerun()

# ── Kalender-sida
def page_calendar(df: pd.DataFrame):
    st.subheader("📅 Utdelningskalender")
    months = st.selectbox("Prognoshorisont", options=[12, 24, 36], index=0)
    monthly, cal = prognos_kalender(df, months_ahead=months)
    if monthly.empty:
        st.info("Ingen prognos – saknar Ex-Date/frekvens/utdelningsdata."); return
    st.write(f"**Månadsvis prognos ({months} mån) i SEK:**")
    st.dataframe(monthly, use_container_width=True)
    st.bar_chart(monthly.set_index("Månad")["Utdelning (SEK)"])
    if not cal.empty:
        with st.expander("Detaljerade kommande betalningar per ticker"):
            st.dataframe(cal.sort_values("Datum"), use_container_width=True)

# ── Spara-sida
def page_save_now(df: pd.DataFrame):
    st.subheader("💾 Spara till Google Sheets")
    preview = beräkna( säkerställ_kolumner(df) )
    st.write("Antal rader som sparas:", len(preview))
    st.dataframe(preview[["Ticker","Bolagsnamn","Valuta","Kategori","Antal aktier","GAV","Aktuell kurs","Utdelning/år","Kurs (SEK)","Årlig utdelning (SEK)"]], use_container_width=True)

    with st.expander("In‑memory kontroll"):
        c1, c2 = st.columns([1,2])
        with c1:
            clear_ok2 = st.checkbox("Bekräfta rensning här också")
        with c2:
            if st.button("🧹 Rensa in‑memory och ladda från Sheets"):
                if clear_ok2:
                    reset_in_memory()
                    st.success("In‑memory rensat. Återläst från Google Sheets.")
                    _rerun()
                else:
                    st.warning("Bocka i bekräftelsen först.")

    if st.button("✅ Bekräfta och spara"):
        if preview["Ticker"].astype(str).str.strip().eq("").all():
            st.error("Inget att spara: inga tickers."); return df
        spara_data(preview)
        st.success("Data sparad!")
    return preview

# ── Sidopanel
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
        for k, v in DEF.items(): st.session_state[k] = v
        _rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("🔌 Ladda data från Google Sheets NU"):
        with st.sidebar:
            with st.spinner("Hämtar data från Google Sheets…"):
                try:
                    loaded = migrate_sheet_columns()
                    st.session_state["working_df"] = säkerställ_kolumner(loaded)
                    st.success("Data inläst!"); _rerun()
                except Exception as e:
                    st.error("Kunde inte läsa från Google Sheets."); st.caption(e)

    st.sidebar.markdown("---")
    one = st.sidebar.text_input("Uppdatera EN ticker (Yahoo)", placeholder="t.ex. VICI").strip().upper()
    if st.sidebar.button("🔄 Uppdatera EN NU"):
        if one:
            base = säkerställ_kolumner(st.session_state.get("working_df", pd.DataFrame()))
            if one not in base["Ticker"].tolist():
                base = pd.concat([base, pd.DataFrame([{"Ticker": one, "Kategori": "QUALITY"}])], ignore_index=True)
            st.session_state["working_df"] = page_update_single(base)
            st.sidebar.success(f"{one} uppdaterad (in‑memory).")

    st.sidebar.markdown("---")
    clear_ok = st.sidebar.checkbox("Bekräfta rensning (återställ till Google Sheets)")
    if st.sidebar.button("🧹 Rensa ändringar (in‑memory)"):
        if clear_ok:
            reset_in_memory()
            st.sidebar.success("In‑memory ändringar rensade. Data återställd från Google Sheets.")
            _rerun()
        else:
            st.sidebar.warning("Bocka i bekräftelsen först.")

# ── Main
def main():
    st.title("Relative Yield – utdelningsportfölj")

    # Starta utan att läsa Sheets (on‑demand i sidopanelen)
    if "working_df" not in st.session_state:
        st.session_state["working_df"] = säkerställ_kolumner(pd.DataFrame())
    base = säkerställ_kolumner(st.session_state["working_df"])

    sidopanel()
    st.sidebar.caption(f"📄 Rader i databasen: {len(base)}")
    if base.empty:
        st.info("Ingen data inläst än. Använd sidopanelen: **🔌 Ladda data från Google Sheets NU**.")

    page = st.sidebar.radio(
        "Meny",
        [
            "➕ Lägg till / ✏︎ Uppdatera bolag",
            "🔄 Uppdatera EN",
            "⏩ Massuppdatera alla",
            "📦 Portföljöversikt",
            "🎯 Köpförslag",
            "⚖️ Mål & fördelning",
            "📅 Utdelningskalender",
            "💾 Spara",
        ],
        index=0
    )

    if page == "➕ Lägg till / ✏︎ Uppdatera bolag":
        base = lagg_till_eller_uppdatera(base)
    elif page == "🔄 Uppdatera EN":
        base = page_update_single(base)
    elif page == "⏩ Massuppdatera alla":
        base = page_mass_update(base)
    elif page == "📦 Portföljöversikt":
        base = portfolj_oversikt(base)
    elif page == "🎯 Köpförslag":
        page_buy_suggestions(base)
    elif page == "⚖️ Mål & fördelning":
        page_settings_distribution(base)
    elif page == "📅 Utdelningskalender":
        page_calendar(base)
    elif page == "💾 Spara":
        page_save_now(base)

    st.session_state["working_df"] = säkerställ_kolumner(base)

if __name__ == "__main__":
    main()
