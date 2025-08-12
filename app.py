import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time
import math
from datetime import datetime, timedelta, date
from google.oauth2.service_account import Credentials

# ── Streamlit rerun shim ────────────────────────────────────────────────────
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

# ── Kolumnschema ───────────────────────────────────────────────────────────
COLUMNS = [
    "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Kategori",
    "Direktavkastning (%)", "Utdelning/år", "Utdelning/år (manuell)", "Lås utdelning",
    "Frekvens/år", "Utdelningsfrekvens", "Frekvenskälla",
    "Payment-lag (dagar)", "Ex-Date", "Nästa utbetalning (est)",
    "Antal aktier", "GAV", "Portföljandel (%)", "Årlig utdelning (SEK)",
    "Kurs (SEK)", "Utdelningstillväxt (%)", "Utdelningskälla",
    "Senaste uppdatering", "Källa", "Marknadsvärde (SEK)"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        d = pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in d.columns:
            d[c] = ""
    # typer/defaults
    d["Ticker"] = d["Ticker"].astype(str).str.strip().str.upper()
    d["Bolagsnamn"] = d["Bolagsnamn"].astype(str)
    d["Valuta"] = d["Valuta"].astype(str).str.strip().str.upper()
    d["Kategori"] = d["Kategori"].astype(str).replace({"": "QUALITY"})
    # numeriska
    num_cols = ["Aktuell kurs","Utdelning/år","Utdelning/år (manuell)","Frekvens/år",
                "Payment-lag (dagar)","Antal aktier","GAV","Marknadsvärde (SEK)"]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    # bool
    if "Lås utdelning" in d.columns:
        d["Lås utdelning"] = d["Lås utdelning"].apply(lambda x: bool(x) if pd.notna(x) else False)
    else:
        d["Lås utdelning"] = False
    for add in ["Frekvenskälla","Utdelningskälla","Senaste uppdatering","Källa",
                "Utdelningsfrekvens","Ex-Date","Nästa utbetalning (est)"]:
        if add not in d.columns:
            d[add] = ""
    if "Utdelningskälla" not in d.columns:
        d["Utdelningskälla"] = "Yahoo"
    return d[COLUMNS].copy()

# ── Kategorier & max-tak ───────────────────────────────────────────────────
MAX_CAT = {
    "QUALITY": 40.0, "REIT": 25.0, "mREIT": 10.0, "BDC": 15.0, "MLP": 20.0,
    "Shipping": 25.0, "Telecom": 20.0, "Tobacco": 20.0, "Utility": 20.0,
    "Tech": 25.0, "Bank": 20.0, "Industrial": 20.0, "Energy": 25.0,
    "Finance": 20.0, "Other": 10.0,
}
CATEGORY_CHOICES = list(MAX_CAT.keys())
GLOBAL_MAX_NAME = 12.0  # max-vikt per enskilt bolag i %

def get_cat_max(cat: str) -> float:
    return float(MAX_CAT.get(str(cat or "").strip() or "QUALITY", 100.0))

# ── Standard FX-kurser (kan ändras i sidopanelen) ──────────────────────────
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

# ── Google Sheets helpers (READ‑ONLY för läsning) ──────────────────────────
def _open_sheet():
    try:
        sh = client.open_by_url(SHEET_URL)
        return sh
    except Exception as e:
        st.error("❌ Kunde inte öppna kalkylbladet. Kontrollera SHEET_URL och behörigheter.")
        st.caption(f"Tekniskt fel: {e}")
        return None

def _ensure_worksheet(sh, title="Bolag"):
    if sh is None:
        return None
    try:
        ws = sh.worksheet(title)
        return ws
    except gspread.WorksheetNotFound:
        try:
            ws = sh.add_worksheet(title=title, rows=2000, cols=len(COLUMNS)+5)
            ws.update([COLUMNS])  # skriv header
            return ws
        except Exception as e:
            st.error("❌ Kunde inte skapa fliken 'Bolag'.")
            st.caption(f"Tekniskt fel: {e}")
            return None

def skapa_koppling():
    sh = _open_sheet()
    ws = _ensure_worksheet(sh, title=SHEET_NAME)
    return ws

def hamta_data():
    """READ‑ONLY: hämtar alla rader utan att skriva något tillbaka."""
    ws = skapa_koppling()
    if ws is None:
        return pd.DataFrame(columns=COLUMNS)
    try:
        rows = ws.get_all_values()
        if not rows:
            return pd.DataFrame(columns=COLUMNS)
        header = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []
        df = pd.DataFrame(data_rows, columns=header) if data_rows else pd.DataFrame(columns=header)
        return säkerställ_kolumner(df)
    except Exception as e:
        st.error("❌ Kunde inte läsa från Google Sheets.")
        st.caption(f"Tekniskt fel: {e}")
        return pd.DataFrame(columns=COLUMNS)

def migrate_sheet_columns():
    """READ‑ONLY: mappa mjukt till COLUMNS i minnet; skriv aldrig till Sheets här."""
    df = hamta_data()
    return säkerställ_kolumner(df)

# ── Backup‑städning (>7 dagar) och spara (enda stället som skriver) ───────
def cleanup_old_backups(days: int = 7):
    """Tar bort flikar Backup_YYYYMMDD_HHMMSS äldre än 'days' dagar."""
    try:
        sh = client.open_by_url(SHEET_URL)
        cutoff = datetime.now() - timedelta(days=days)
        for ws in sh.worksheets():
            title = (ws.title or "").strip()
            if not title.startswith("Backup_"):
                continue
            ts_part = title.replace("Backup_", "")
            try:
                ts = datetime.strptime(ts_part, "%Y%m%d_%H%M%S")
            except Exception:
                continue
            if ts < cutoff:
                try:
                    sh.del_worksheet(ws)
                except Exception:
                    pass
    except Exception:
        pass

def spara_data(df: pd.DataFrame):
    ws = skapa_koppling()
    if ws is None:
        return
    d = säkerställ_kolumner(df).copy()
    if d["Ticker"].astype(str).str.strip().eq("").all():
        st.error("Inget att spara: inga tickers.")
        return
    try:
        sh = client.open_by_url(SHEET_URL)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bkp_title = f"Backup_{ts}"

        # 1) Backup till ny flik
        try:
            ws_b = sh.add_worksheet(title=bkp_title, rows=max(2000, len(d)+5), cols=len(COLUMNS)+5)
            ws_b.update([d.columns.tolist()] + d.astype(str).values.tolist(), value_input_option="USER_ENTERED")
        except Exception as e:
            st.warning(f"Backup misslyckades: {e}")

        # 2) Skriv till huvudfliken
        ws.clear()
        ws.update([d.columns.tolist()] + d.astype(str).values.tolist(), value_input_option="USER_ENTERED")

        # 3) Städning av gamla backupflikar
        cleanup_old_backups(days=7)

        st.success(f"✅ Sparade {len(d)} rader. (Backup: {bkp_title}, äldre än 7 dagar rensade)")
    except Exception as e:
        st.error("❌ Fel vid sparande till Google Sheets.")
        st.caption(f"Tekniskt fel: {e}")

# ── Intervall‑baserad frekvensdetektion ─────────────────────────────────────
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
        return 0, "Oregelbunden", f"{src_label} (spridda)"

    if len(last24) >= 2:
        f, t, src = freq_by_intervals(last24, "Historik 24m")
        if f in (12,4,2,1):
            return f, t, src
    if len(last12) >= 1:
        f, t, src = freq_by_intervals(last12, "Historik 12m")
        if f in (12,4,2,1):
            return f, t, src
    recent = divs.tail(10)
    if not recent.empty:
        f, t, src = freq_by_intervals(recent, "Senaste 10")
        return f, t, src
    return 0, "Oregelbunden", "Ingen historik"

# ── Yahoo Finance: pris, valuta, utdelning, frekvens, ex‑date ───────────────
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
        price = float(price) if price not in (None, "") else 0.0

        name = info.get("shortName") or info.get("longName") or ticker
        currency = (info.get("currency") or "").upper()
        if not currency:
            try:
                currency = (t.fast_info.get("currency") or "").upper()
            except Exception:
                currency = "SEK"

        div_rate = 0.0
        freq = 0
        freq_text = "Oregelbunden"
        freq_src = "Ingen historik"
        ex_date_str = ""
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
                if fwd not in (None, "", 0):
                    div_rate = float(fwd)
            except Exception:
                pass
        if div_rate == 0.0:
            try:
                trailing = info.get("trailingAnnualDividendRate")
                if trailing not in (None, "", 0):
                    div_rate = float(trailing)
            except Exception:
                pass

        if not ex_date_str:
            try:
                ts = info.get("exDividendDate")
                if ts not in (None, "", 0):
                    ex_date_str = pd.to_datetime(int(ts), unit="s", utc=True).strftime("%Y-%m-%d")
            except Exception:
                ex_date_str = ""

        return {
            "namn": name,
            "kurs": price,
            "valuta": currency,
            "utdelning": div_rate,
            "frekvens": freq,
            "frekvens_text": freq_text,
            "frekvens_källa": freq_src,
            "ex_date": ex_date_str,
            "källa": "Yahoo",
            "uppdaterad": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    except Exception as e:
        st.warning(f"Kunde inte hämta Yahoo-data för {ticker}: {e}")
        return None

# ── Full Yahoo‑refresh (Yahoo>0 annars manuell) + PROGRESS ─────────────────
def refresh_all_from_yahoo(df: pd.DataFrame, sleep_s: float = 1.0, show_progress: bool = True) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()
    if d.empty:
        return d

    tickers = d["Ticker"].astype(str).tolist()
    n = len(tickers)

    prog = st.progress(0) if show_progress else None
    msg  = st.empty() if show_progress else None

    for i, tkr in enumerate(tickers, start=1):
        if show_progress:
            prog.progress(int(i * 100 / n))
            msg.text(f"Uppdaterar {tkr} ({i}/{n})…")

        vals = hamta_yahoo_data(tkr)
        if vals:
            m = d["Ticker"] == tkr

            # Namn/kurs/valuta
            d.loc[m, "Bolagsnamn"] = vals.get("namn", tkr)
            if vals.get("kurs") is not None:
                d.loc[m, "Aktuell kurs"] = float(vals.get("kurs") or 0.0)
            if vals.get("valuta"):
                d.loc[m, "Valuta"] = vals["valuta"]

            # Utdelning enligt regeln: Yahoo om >0, annars manuell om >0, annars 0
            manual = float(pd.to_numeric(d.loc[m, "Utdelning/år (manuell)"].iloc[0], errors="coerce") or 0.0)
            new_div = float(vals.get("utdelning") or 0.0)
            if new_div > 0:
                d.loc[m, "Utdelning/år"] = new_div
            elif manual > 0:
                d.loc[m, "Utdelning/år"] = manual
            else:
                d.loc[m, "Utdelning/år"] = 0.0

            # Frekvens / ex‑date
            f  = int(vals.get("frekvens") or 0)
            ft = vals.get("frekvens_text") or ""
            fsrc = vals.get("frekvens_källa") or ""
            xd = vals.get("ex_date") or ""
            if f  > 0: d.loc[m, "Frekvens/år"] = f
            if ft:     d.loc[m, "Utdelningsfrekvens"] = ft
            if fsrc:   d.loc[m, "Frekvenskälla"] = fsrc
            if xd:     d.loc[m, "Ex-Date"] = xd

            d.loc[m, "Källa"] = "Yahoo"
            if vals.get("uppdaterad"):
                d.loc[m, "Senaste uppdatering"] = vals["uppdaterad"]

        if sleep_s and sleep_s > 0:
            time.sleep(sleep_s)

    if show_progress:
        prog.progress(100)
        msg.text("Klart. Räknar om…")

    return beräkna(d)

# ── Full beräkning ─────────────────────────────────────────────────────────
def beräkna(df: pd.DataFrame) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()

    # utdelningskälla-tag
    use_manual = pd.to_numeric(d["Utdelning/år (manuell)"], errors="coerce").fillna(0.0) > 0
    from_yahoo = pd.to_numeric(d["Utdelning/år"], errors="coerce").fillna(0.0) > 0
    d["Utdelningskälla"] = ["Yahoo" if from_yahoo.iloc[i] else ("Manuell" if use_manual.iloc[i] else "") for i in range(len(d))]

    # priser & FX
    d["Aktuell kurs"] = pd.to_numeric(d["Aktuell kurs"], errors="coerce").fillna(0.0)
    rates = d["Valuta"].apply(fx_for).astype(float)
    d["Kurs (SEK)"] = (d["Aktuell kurs"] * rates).round(6)

    d["Antal aktier"] = pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0)
    d["Årlig utdelning (SEK)"] = (d["Antal aktier"] * pd.to_numeric(d["Utdelning/år"], errors="coerce").fillna(0.0) * rates).round(2)

    ok = (d["Aktuell kurs"] > 0) & (pd.to_numeric(d["Utdelning/år"], errors="coerce").fillna(0.0) > 0)
    d["Direktavkastning (%)"] = 0.0
    d.loc[ok, "Direktavkastning (%)"] = (100.0 * pd.to_numeric(d["Utdelning/år"], errors="coerce").fillna(0.0) / d["Aktuell kurs"]).round(2)

    d["Marknadsvärde (SEK)"] = (d["Antal aktier"] * d["Kurs (SEK)"]).round(2)
    tot_mv = float(d["Marknadsvärde (SEK)"].sum()) if not d.empty else 1.0
    d["Portföljandel (%)"] = (100.0 * d["Marknadsvärde (SEK)"] / (tot_mv if tot_mv != 0 else 1.0)).round(2)

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

# ── Köpmotor som respekterar bolags‑ & kategori‑tak, med diagnostik ────────
def suggest_buys(df: pd.DataFrame,
                 w_val: float=0.5, w_under: float=0.35, w_time: float=0.15,
                 topk: int=5, allow_margin: float=0.0, return_debug: bool=False):
    d = beräkna(df).copy()
    cols = ["Ticker","Kategori","Poäng","DA %","Vikt %","Nästa utb",
            "Rek. (st)","Max enl. regler (st)","Kostnad 1 st (SEK)","Motivering"]
    diag = []

    if d.empty:
        return (pd.DataFrame(columns=cols), pd.DataFrame(columns=["Ticker","Skäl"])) if return_debug else pd.DataFrame(columns=cols)

    # totalvärde + aktuella vikter
    T = float(d["Marknadsvärde (SEK)"].sum())
    d["Kategori"] = d["Kategori"].astype(str).replace({"": "QUALITY"})
    cat_values = d.groupby("Kategori", as_index=False)["Marknadsvärde (SEK)"].sum()
    cat_values = cat_values.set_index("Kategori")["Marknadsvärde (SEK)"].to_dict()

    tol = float(allow_margin)
    keep_idx = []
    for i, r in d.iterrows():
        tkr = str(r["Ticker"])
        cat = str(r["Kategori"]) if str(r["Kategori"]).strip() else "QUALITY"
        Vi  = float(r["Marknadsvärde (SEK)"])
        w_ticker = float(r.get("Portföljandel (%)", 0.0))
        if w_ticker >= (GLOBAL_MAX_NAME - 1e-9):
            diag.append({"Ticker": tkr, "Skäl": f"Över bolagstak {GLOBAL_MAX_NAME:.1f}%"})
            continue
        C = float(cat_values.get(cat, 0.0))
        w_cat = (0.0 if T <= 0 else 100.0 * C / T)
        if w_cat >= (MAX_CAT.get(cat, 100.0) - 1e-9):
            diag.append({"Ticker": tkr, "Skäl": f"Kategorin '{cat}' vid/över tak {MAX_CAT.get(cat,100):.1f}%"})
            continue
        keep_idx.append(i)

    if not keep_idx:
        out = pd.DataFrame(columns=cols)
        diag_df = pd.DataFrame(diag) if diag else pd.DataFrame(columns=["Ticker","Skäl"])
        return (out, diag_df) if return_debug else out

    da = pd.to_numeric(d.loc[keep_idx, "Direktavkastning (%)"], errors="coerce").fillna(0.0)
    da_score = (da.clip(lower=0, upper=15) / 15.0) * 100.0
    under = (GLOBAL_MAX_NAME - d.loc[keep_idx, "Portföljandel (%)"]).clip(lower=0)
    under_score = (under / GLOBAL_MAX_NAME) * 100.0

    def _days_to(date_str: str) -> int:
        try:
            dt = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(dt): return 9999
            return max(0, (dt.date() - date.today()).days)
        except Exception:
            return 9999
    days = d.loc[keep_idx, "Nästa utbetalning (est)"].apply(_days_to)
    time_score = ((90 - days.clip(upper=90)) / 90.0).clip(lower=0) * 100.0

    totw = max(1e-9, (w_val + w_under + w_time))
    w_val, w_under, w_time = w_val/totw, w_under/totw, w_time/totw
    total_score = (w_val*da_score + w_under*under_score + w_time*time_score)

    order = total_score.sort_values(ascending=False).index
    rows = []

    def _cap_by_weight(Vi: float, Tot: float, price_sek: float, max_pct: float) -> int:
        if price_sek <= 0: return 0
        m = max_pct / 100.0
        numer = m*Tot - Vi
        denom = (1.0 - m) * price_sek
        if denom <= 0: return 0
        return int(max(0, math.floor(numer / denom)))

    def _cap_by_category(C: float, Tot: float, price_sek: float, cat_max_pct: float) -> int:
        if price_sek <= 0: return 0
        M = cat_max_pct / 100.0
        numer = M*Tot - C
        denom = (1.0 - M) * price_sek
        if denom <= 0: return 0
        return int(max(0, math.floor(numer / denom)))

    # --- bootstrap‑fix: hoppa över 1‑aktie‑spärren när portföljen saknar värde ---
    T_now_global = float(d["Marknadsvärde (SEK)"].sum())
    bootstrap_mode = (T_now_global <= 0)

    for i in order:
        tkr = str(d.at[i,"Ticker"])
        price = float(pd.to_numeric(d.at[i,"Kurs (SEK)"], errors="coerce") or 0.0)
        if price <= 0:
            diag.append({"Ticker": tkr, "Skäl": "Pris saknas/0"})
            continue

        cat = str(d.at[i,"Kategori"]) if str(d.at[i,"Kategori"]).strip() else "QUALITY"
        Vi  = float(d.at[i,"Marknadsvärde (SEK)"])
        C   = float(cat_values.get(cat, 0.0))

        if not bootstrap_mode:
            T_now = float(d["Marknadsvärde (SEK)"].sum())
            Vi2 = Vi + price
            T2  = T_now + price
            w_after = 100.0 * Vi2 / T2 if T2 > 0 else 0.0
            if T2 > 0 and w_after > (GLOBAL_MAX_NAME + tol) + 1e-9:
                diag.append({"Ticker": tkr, "Skäl": f"1 st skulle överskrida bolagstak {GLOBAL_MAX_NAME:.1f}%"})
                continue
            C2 = C + price
            cat_after = 100.0 * C2 / T2 if T2 > 0 else 0.0
            if T2 > 0 and cat_after > (MAX_CAT.get(cat, 100.0) + tol) + 1e-9:
                diag.append({"Ticker": tkr, "Skäl": f"1 st skulle överskrida kategori‑tak {MAX_CAT.get(cat,100):.1f}%"})
                continue

        # beräkna max kapacitet enligt regler
        T = float(d["Marknadsvärde (SEK)"].sum())
        if T <= 0:
            n_name_cap = 10**9
            n_cat_cap  = 10**9
        else:
            n_name_cap = _cap_by_weight(Vi, T, price, GLOBAL_MAX_NAME + tol)
            n_cat_cap  = _cap_by_category(C, T, price, MAX_CAT.get(cat, 100.0) + tol)

        n_max = int(max(1, min(n_name_cap, n_cat_cap)))
        n_reco = 1

        rows.append({
            "Ticker": tkr,
            "Kategori": cat,
            "Poäng": round(float(total_score.at[i]), 1),
            "DA %": round(float(d.at[i,"Direktavkastning (%)"]), 2),
            "Vikt %": float(d.at[i,"Portföljandel (%)"]),
            "Nästa utb": d.at[i,"Nästa utbetalning (est)"],
            "Rek. (st)": int(n_reco),
            "Max enl. regler (st)": int(n_max),
            "Kostnad 1 st (SEK)": round(price,2),
            "Motivering": f"{'Bootstrap-läge – första köp tillåts. ' if bootstrap_mode else ''}Ryms inom {GLOBAL_MAX_NAME:.0f}% & kategori≤{MAX_CAT.get(cat,100):.0f}%"
        })

        if len(rows) >= topk:
            break

    out = pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)
    diag_df = pd.DataFrame(diag) if diag else pd.DataFrame(columns=["Ticker","Skäl"])
    return (out, diag_df) if return_debug else out

# ── Auto‑invest simulering (trancher ~500 kr, summering per ticker) ────────
def simulate_auto_invest(df: pd.DataFrame,
                         cash_sek: float,
                         tranche_sek: float = 500.0,
                         w_val: float = 0.5, w_under: float = 0.35, w_time: float = 0.15,
                         allow_margin: float = 0.0,
                         max_steps: int = 100):
    work = beräkna(df).copy()
    remaining = float(cash_sek)
    tol = float(allow_margin)

    def _cap_by_weight(Vi: float, Tot: float, price_sek: float, max_pct: float) -> int:
        if price_sek <= 0: return 0
        m = max_pct / 100.0
        numer = m*Tot - Vi
        denom = (1.0 - m) * price_sek
        if denom <= 0: return 0
        return int(max(0, math.floor(numer / denom)))

    def _cap_by_category(C: float, Tot: float, price_sek: float, cat_max_pct: float) -> int:
        if price_sek <= 0: return 0
        M = cat_max_pct / 100.0
        numer = M*Tot - C
        denom = (1.0 - M) * price_sek
        if denom <= 0: return 0
        return int(max(0, math.floor(numer / denom)))

    steps = []
    step_no = 0

    while remaining > 0 and step_no < max_steps:
        sug = suggest_buys(work, w_val=w_val, w_under=w_under, w_time=w_time,
                           topk=10, allow_margin=tol, return_debug=False)
        if sug is None or sug.empty:
            break

        bought_this_round = False

        for _, row in sug.iterrows():
            tkr = str(row["Ticker"])
            m = work["Ticker"] == tkr
            if not m.any():
                continue
            price = float(pd.to_numeric(work.loc[m, "Kurs (SEK)"], errors="coerce").fillna(0.0).iloc[0])
            if price <= 0:
                continue

            desired_n = max(1, math.ceil(tranche_sek / price))
            n_cash_cap = int(remaining // price)
            if n_cash_cap <= 0:
                continue

            Vi = float(work.loc[m, "Marknadsvärde (SEK)"].iloc[0])
            T  = float(work["Marknadsvärde (SEK)"].sum())
            cat = str(work.loc[m, "Kategori"].iloc[0]) if str(work.loc[m, "Kategori"].iloc[0]).strip() else "QUALITY"
            C  = float(work.groupby("Kategori")["Marknadsvärde (SEK)"].sum().get(cat, 0.0))

            if T <= 0:
                n_name_cap = 10**9
                n_cat_cap  = 10**9
            else:
                n_name_cap = _cap_by_weight(Vi, T, price, GLOBAL_MAX_NAME + tol)
                n_cat_cap  = _cap_by_category(C, T, price, MAX_CAT.get(cat, 100.0) + tol)

            n = int(max(0, min(desired_n, n_cash_cap, n_name_cap, n_cat_cap)))
            if n <= 0:
                n = 1
                if n > n_cash_cap:
                    continue
                Vi2 = Vi + price
                T2  = T + price if T > 0 else price
                w_after = 100.0 * Vi2 / T2 if T2 > 0 else 0.0
                if T > 0 and w_after > (GLOBAL_MAX_NAME + tol) + 1e-9:
                    continue
                C2 = C + price
                cat_after = 100.0 * C2 / (T + price) if T > 0 else 0.0
                if T > 0 and cat_after > (MAX_CAT.get(cat, 100.0) + tol) + 1e-9:
                    continue

            if n > 0:
                old_q = float(pd.to_numeric(work.loc[m, "Antal aktier"], errors="coerce").fillna(0.0).iloc[0])
                work.loc[m, "Antal aktier"] = old_q + n
                work = beräkna(work)
                cost = round(n * price, 2)
                remaining = round(remaining - cost, 2)
                new_weight = float(work.loc[m, "Portföljandel (%)"].iloc[0])
                steps.append({
                    "Steg": step_no + 1,
                    "Ticker": tkr,
                    "Antal": int(n),
                    "Pris (SEK)": round(price, 2),
                    "Summa (SEK)": cost,
                    "Vikt efter (%)": round(new_weight, 2)
                })
                step_no += 1
                bought_this_round = True
                break  # kör om ranking efter varje köp

        if not bought_this_round:
            break

    steps_df = pd.DataFrame(steps) if steps else pd.DataFrame(columns=["Steg","Ticker","Antal","Pris (SEK)","Summa (SEK)","Vikt efter (%)"])
    if steps_df.empty:
        summary_df = pd.DataFrame(columns=["Ticker","Antal totalt","Summa totalt (SEK)"])
    else:
        summary_df = steps_df.groupby("Ticker", as_index=False).agg(
            **{"Antal totalt": ("Antal","sum"), "Summa totalt (SEK)": ("Summa (SEK)","sum")}
        ).sort_values("Summa totalt (SEK)", ascending=False)
        summary_df["Summa totalt (SEK)"] = summary_df["Summa totalt (SEK)"].round(2)

    leftover = round(max(0.0, remaining), 2)
    return steps_df, summary_df, leftover

# ── Pending‑kö: init & helpers ─────────────────────────────────────────────
def ensure_pending():
    if "pending_rows" not in st.session_state:
        st.session_state["pending_rows"] = []

def _pending_to_df() -> pd.DataFrame:
    ensure_pending()
    if not st.session_state["pending_rows"]:
        return pd.DataFrame(columns=COLUMNS)
    dfp = pd.DataFrame(st.session_state["pending_rows"])
    return säkerställ_kolumner(dfp)

def _merge_pending_into_df(base: pd.DataFrame) -> pd.DataFrame:
    ensure_pending()
    d = säkerställ_kolumner(base).copy()
    if not st.session_state["pending_rows"]:
        return d
    for row in st.session_state["pending_rows"]:
        tkr = str(row.get("Ticker","")).upper().strip()
        if not tkr:
            continue
        if (d["Ticker"] == tkr).any():
            m = d["Ticker"] == tkr
        else:
            d = pd.concat([d, pd.DataFrame([{"Ticker": tkr}])], ignore_index=True)
            m = d["Ticker"] == tkr
        for k, v in row.items():
            if k in d.columns:
                d.loc[m, k] = v
    d = beräkna(d)
    return d

def pending_panel(show_actions=True, key_prefix=""):
    ensure_pending()
    st.markdown("### 🧺 Pending (inte sparat ännu)")
    dfp = _pending_to_df()
    st.dataframe(dfp[["Ticker","Bolagsnamn","Kategori","Valuta","Antal aktier","GAV",
                      "Aktuell kurs","Utdelning/år","Frekvens/år","Ex-Date"]],
                 use_container_width=True)
    st.caption(f"Rader i pending: {len(dfp)}")
    if not show_actions:
        return

    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        if st.button("➖ Ta bort vald", key=f"{key_prefix}pending_remove"):
            if len(dfp) == 0:
                st.warning("Inget att ta bort.")
            else:
                tkr = st.session_state.get(f"{key_prefix}pending_remove_choice", "")
                if not tkr:
                    st.warning("Välj ticker nedan och tryck igen.")
                else:
                    st.session_state["pending_rows"] = [r for r in st.session_state["pending_rows"] if str(r.get("Ticker","")).upper() != tkr]
                    st.success(f"Tog bort {tkr} från pending.")
                    _rerun()
    with colp2:
        if st.button("🧹 Rensa pending", key=f"{key_prefix}pending_clear"):
            st.session_state["pending_rows"] = []
            st.success("Pending tömd.")
            _rerun()
    with colp3:
        if st.button("💾 Spara pending till Google Sheets", key=f"{key_prefix}pending_save"):
            base = säkerställ_kolumner(st.session_state.get("working_df", pd.DataFrame()))
            merged = _merge_pending_into_df(base)
            merged_refreshed = refresh_all_from_yahoo(merged, sleep_s=1.0, show_progress=True)
            spara_data(merged_refreshed)
            st.session_state["pending_rows"] = []
            st.session_state["working_df"] = merged_refreshed
            st.success("Pending sparat. All data uppdaterad från Yahoo och skriven till Google Sheets.")
            _rerun()

    options = _pending_to_df()["Ticker"].astype(str).tolist()
    st.selectbox("Välj ticker att ta bort ur pending", options=[""]+options, key=f"{key_prefix}pending_remove_choice")

# ── Lägg till / Uppdatera bolag (IN‑MEMORY + pending) ──────────────────────
def lagg_till_eller_uppdatera(df: pd.DataFrame) -> pd.DataFrame:
    ensure_pending()
    st.subheader("➕ Lägg till / ✏ Uppdatera bolag")
    tickers = ["Ny"] + sorted(df["Ticker"].unique().tolist())
    val = st.selectbox("Välj bolag", options=tickers)

    if val == "Ny":
        ticker = st.text_input("Ticker").strip().upper()
        antal = st.number_input("Antal aktier", min_value=0, value=0, step=1)
        gav = st.number_input("GAV (SEK)", min_value=0.0, value=0.0, step=0.01)
        kategori = st.selectbox("Kategori", options=CATEGORY_CHOICES, index=CATEGORY_CHOICES.index("QUALITY"))
        man_utd = st.number_input("Utdelning/år (manuell)", min_value=0.0, step=0.01)
        lås = st.checkbox("Lås utdelning (använd manuell)", value=False)
    else:
        rad = df[df["Ticker"] == val].iloc[0]
        ticker = st.text_input("Ticker", value=rad["Ticker"]).strip().upper()
        antal = st.number_input("Antal aktier", min_value=0, value=int(float(rad.get("Antal aktier",0))), step=1)
        gav = st.number_input("GAV (SEK)", min_value=0.0, value=float(rad.get("GAV",0.0)), step=0.01)
        kategori = st.selectbox("Kategori", options=CATEGORY_CHOICES,
                                index=CATEGORY_CHOICES.index(str(rad.get("Kategori","QUALITY")) if str(rad.get("Kategori","QUALITY")) in CATEGORY_CHOICES else "QUALITY"))
        man_utd = st.number_input("Utdelning/år (manuell)", min_value=0.0, value=float(rad.get("Utdelning/år (manuell)",0.0)), step=0.01)
        lås = st.checkbox("Lås utdelning (använd manuell)", value=bool(rad.get("Lås utdelning", False)))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Lägg till i minnet (pending)"):
            if not ticker:
                st.error("Ticker måste anges."); return df
            row = {
                "Ticker": ticker, "Antal aktier": float(antal), "GAV": float(gav),
                "Kategori": kategori, "Utdelning/år (manuell)": float(man_utd),
                "Lås utdelning": bool(lås)
            }
            vals = hamta_yahoo_data(ticker)
            if vals:
                row.update({
                    "Bolagsnamn": vals.get("namn", ""),
                    "Aktuell kurs": float(vals.get("kurs") or 0.0),
                    "Valuta": vals.get("valuta", ""),
                    "Utdelning/år": float(vals.get("utdelning") or 0.0),
                    "Frekvens/år": int(vals.get("frekvens") or 0),
                    "Utdelningsfrekvens": vals.get("frekvens_text",""),
                    "Ex-Date": vals.get("ex_date",""),
                    "Källa": "Yahoo", "Senaste uppdatering": vals.get("uppdaterad","")
                })
            st.session_state["pending_rows"].append(row)
            st.success(f"{ticker} tillagd i pending.")
            _rerun()
    with col2:
        if st.button("💾 Spara till Google Sheets NU (inkl. pending)"):
            base = säkerställ_kolumner(st.session_state.get("working_df", pd.DataFrame()))
            merged = _merge_pending_into_df(base)
            merged_refreshed = refresh_all_from_yahoo(merged, sleep_s=1.0, show_progress=True)
            spara_data(merged_refreshed)
            st.session_state["pending_rows"] = []
            st.session_state["working_df"] = merged_refreshed
            st.success("Sparat & uppdaterat.")
            _rerun()

    pending_panel(show_actions=True, key_prefix="addupd_")
    return säkerställ_kolumner(st.session_state.get("working_df", df))

# ── Uppdatera enskilt bolag ────────────────────────────────────────────────
def uppdatera_bolag(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🔄 Uppdatera enskilt bolag")
    if df.empty:
        st.info("Ingen data att uppdatera."); return df
    val = st.selectbox("Välj bolag", options=sorted(df["Ticker"].unique().tolist()))
    if st.button("Uppdatera från Yahoo"):
        merged = df.copy()
        vals = hamta_yahoo_data(val)
        if vals:
            m = merged["Ticker"] == val
            manual_locked = bool(merged.loc[m, "Lås utdelning"].iloc[0])
            new_div = float(vals.get("utdelning") or 0.0)
            if new_div > 0 or not manual_locked:
                merged.loc[m, "Utdelning/år"] = new_div
            merged.loc[m, "Bolagsnamn"] = vals.get("namn", val)
            merged.loc[m, "Aktuell kurs"] = vals.get("kurs") or merged.loc[m, "Aktuell kurs"]
            if vals.get("valuta"): merged.loc[m, "Valuta"] = vals["valuta"]
            f = int(vals.get("frekvens") or 0)
            ft = vals.get("frekvens_text") or ""
            fsrc = vals.get("frekvens_källa") or ""
            xd = vals.get("ex_date") or ""
            if f  > 0: merged.loc[m, "Frekvens/år"] = f
            if ft:     merged.loc[m, "Utdelningsfrekvens"] = ft
            if fsrc:   merged.loc[m, "Frekvenskälla"] = fsrc
            if xd:     merged.loc[m, "Ex-Date"] = xd
            merged.loc[m, "Källa"] = "Yahoo"
            if vals.get("uppdaterad"): merged.loc[m, "Senaste uppdatering"] = vals["uppdaterad"]
            merged = beräkna(merged)
            st.success(f"{val} uppdaterad (in-memory).")
            return merged
        else:
            st.warning(f"Kunde inte hämta data för {val}")
    return df

# ── Massuppdatera alla ─────────────────────────────────────────────────────
def massuppdatera(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("⏩ Massuppdatera alla bolag från Yahoo")
    if df.empty:
        st.info("Ingen data att uppdatera."); return df
    if st.button("Starta massuppdatering"):
        merged = refresh_all_from_yahoo(df, sleep_s=1.0, show_progress=True)
        st.success("Massuppdatering klar (in-memory). Glöm inte spara om du vill skriva till Google Sheets.")
        return merged
    return df

# ── Utdelningskalender (12/24/36 mån) ──────────────────────────────────────
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
            per_share_local = float(r.get("Utdelning/år", 0.0)) / max(1.0, float(r.get("Frekvens/år", 4.0)))
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

# ── Portföljöversikt ───────────────────────────────────────────────────────
def portfolj_oversikt(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("📦 Portföljöversikt")
    d = beräkna(df).copy()

    d["Insatt (SEK)"] = (pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0) *
                         pd.to_numeric(d["GAV"], errors="coerce").fillna(0.0)).round(2)
    d["Orealiserad P/L (SEK)"] = (d["Marknadsvärde (SEK)"] - d["Insatt (SEK)"]).round(2)
    d["Orealiserad P/L (%)"] = (100.0 * d["Orealiserad P/L (SEK)"] /
                                d["Insatt (SEK)"].replace({0: pd.NA})).fillna(0.0).round(2)

    tot_mv, tot_ins = float(d["Marknadsvärde (SEK)"].sum()), float(d["Insatt (SEK)"].sum())
    tot_pl, tot_div = float(d["Orealiserad P/L (SEK)"].sum()), float(d["Årlig utdelning (SEK)"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portföljvärde", f"{round(tot_mv,2):,}".replace(",", " "))
    c2.metric("Insatt kapital", f"{round(tot_ins,2):,}".replace(",", " "))
    c3.metric("Orealiserad P/L", f"{round(tot_pl,2):,}".replace(",", " "),
              delta=f"{(0 if tot_ins==0 else (100*tot_pl/tot_ins)):.2f}%")
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
    st.dataframe(d[show_cols], use_container_width=True)
    return d

# ── Köpförslag-sida: ranking + diagnostik + auto‑invest simulering ────────
def page_buy_suggestions(df: pd.DataFrame):
    st.subheader("🎯 Köpförslag (respekterar 12% bolagstak + kategori‑tak)")

    # — A) Vanlig ranking med diagnostik —
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        w_val = st.slider("Vikt: Värdering (DA)", 0.0, 1.0, 0.50, 0.05)
    with c2:
        w_under = st.slider("Vikt: Undervikt mot 12%", 0.0, 1.0, 0.35, 0.05)
    with c3:
        w_time = st.slider("Vikt: Timing (nära utdelning)", 0.0, 1.0, 0.15, 0.05)
    with c4:
        allow_margin = st.number_input("Marginal över 12%-tak (p)", min_value=0.0, value=0.0, step=0.1,
                                       help="Tillåten buffert i procentenheter över tak vid prövning av 1 st")

    if st.button("Beräkna köpförslag"):
        sug, diag = suggest_buys(
            df,
            w_val=w_val, w_under=w_under, w_time=w_time,
            topk=5, allow_margin=allow_margin,
            return_debug=True
        )
        if sug.empty:
            st.info("Inga köpförslag som klarar reglerna just nu.")
        else:
            st.dataframe(sug, use_container_width=True)
            st.caption("Poäng = viktad mix av direktavkastning, undervikt mot 12% och hur nära nästa utdelning bolaget är.")
        if not diag.empty:
            with st.expander("Varför filtrerades vissa bort? (diagnostik)"):
                st.dataframe(diag, use_container_width=True)

    st.markdown("---")

    # — B) Auto‑invest simulering —
    st.subheader("🧠 Auto‑invest simulering (kassa → inköpslista)")
    cA, cB, cC = st.columns([1,1,1])
    with cA:
        cash = st.number_input("Kassa (SEK)", min_value=0.0, value=2500.0, step=100.0)
    with cB:
        tranche = st.number_input("Tranche per köp (SEK)", min_value=50.0, value=500.0, step=50.0,
                                  help="Varje köp försöker ligga runt detta belopp (avrundat UPPÅT i antal aktier).")
    with cC:
        max_steps = st.number_input("Max antal köp", min_value=1, value=20, step=1)

    if st.button("Simulera inköp utifrån kassa"):
        steps_df, summary_df, leftover = simulate_auto_invest(
            df=df,
            cash_sek=cash,
            tranche_sek=tranche,
            w_val=w_val, w_under=w_under, w_time=w_time,
            allow_margin=allow_margin,
            max_steps=int(max_steps)
        )
        if steps_df.empty:
            st.info("Ingen simulering kunde genomföras (kolla kassa/tak/priser).")
        else:
            st.write("**Föreslagna köp (i ordning):**")
            st.dataframe(steps_df, use_container_width=True)
            st.write("**Summering per ticker:**")
            st.dataframe(summary_df, use_container_width=True)
            st.success(f"Kvarvarande kassa efter simulering: {leftover:.2f} SEK")

# ── Kalender-sida ──────────────────────────────────────────────────────────
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

# ── Spara-sida ─────────────────────────────────────────────────────────────
def page_save_now(df: pd.DataFrame):
    st.subheader("💾 Spara till Google Sheets")
    preview = beräkna( säkerställ_kolumner(df) )
    st.write("Antal rader som sparas:", len(preview))
    st.dataframe(preview[["Ticker","Bolagsnamn","Valuta","Kategori","Antal aktier","GAV",
                          "Aktuell kurs","Utdelning/år","Kurs (SEK)","Årlig utdelning (SEK)"]],
                 use_container_width=True)
    if st.button("✅ Bekräfta och spara"):
        if preview["Ticker"].astype(str).str.strip().eq("").all():
            st.error("Inget att spara: inga tickers i tabellen."); return df
        spara_data(preview)
        st.success("Data sparade till Google Sheets!")
    return preview

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
    st.session_state["USDSEK"], st.session_state["EURSEK"], st.session_state["CADSEK"], st.session_state["NOKSEK"] = usd, eur, cad, nok

    if st.sidebar.button("↩︎ Återställ FX till standard"):
        for k, v in DEF.items(): st.session_state[k] = v
        _rerun()

    st.sidebar.markdown("---")
    one_ticker = st.sidebar.text_input("Uppdatera EN ticker (Yahoo)", placeholder="t.ex. VICI").strip().upper()
    if st.sidebar.button("🔄 Uppdatera EN"):
        base = säkerställ_kolumner(st.session_state.get("working_df", pd.DataFrame()))
        if one_ticker:
            if one_ticker not in base["Ticker"].tolist():
                base = pd.concat([base, pd.DataFrame([{"Ticker": one_ticker, "Kategori": "QUALITY"}])], ignore_index=True)
            vals = hamta_yahoo_data(one_ticker)
            if vals:
                m = base["Ticker"] == one_ticker
                manual_locked = bool(base.loc[m, "Lås utdelning"].iloc[0])
                new_div = float(vals.get("utdelning") or 0.0)
                if new_div > 0 or not manual_locked:
                    base.loc[m, "Utdelning/år"] = new_div
                base.loc[m, "Bolagsnamn"] = vals.get("namn", one_ticker)
                base.loc[m, "Aktuell kurs"] = vals.get("kurs") or base.loc[m, "Aktuell kurs"]
                if vals.get("valuta"): base.loc[m, "Valuta"] = vals.get("valuta")
                f  = int(vals.get("frekvens") or 0)
                ft = vals.get("frekvens_text") or ""
                fsrc = vals.get("frekvens_källa") or ""
                xd = vals.get("ex_date") or ""
                if f  > 0: base.loc[m, "Frekvens/år"] = f
                if ft:     base.loc[m, "Utdelningsfrekvens"] = ft
                if fsrc:   base.loc[m, "Frekvenskälla"] = fsrc
                if xd:     base.loc[m, "Ex-Date"] = xd
                base.loc[m, "Källa"] = "Yahoo"
                if vals.get("uppdaterad"): base.loc[m, "Senaste uppdatering"] = vals["uppdaterad"]
                st.session_state["working_df"] = beräkna(base)
                st.sidebar.success(f"{one_ticker} uppdaterad.")

# ── Main (router/meny) ─────────────────────────────────────────────────────
def main():
    st.title("Relative Yield – utdelningsportfölj")

    if "working_df" not in st.session_state:
        try:
            st.session_state["working_df"] = migrate_sheet_columns()
        except Exception:
            st.session_state["working_df"] = säkerställ_kolumner(pd.DataFrame())
    base = säkerställ_kolumner(st.session_state["working_df"])

    sidopanel()
    st.sidebar.caption(f"📄 Rader i databasen: {len(base)}")

    page = st.sidebar.radio(
        "Meny",
        [
            "➕ Lägg till / ✏ Uppdatera bolag",
            "🔄 Uppdatera EN",
            "⏩ Massuppdatera alla",
            "📦 Portföljöversikt",
            "🎯 Köpförslag",
            "📅 Utdelningskalender",
            "💾 Spara",
        ],
        index=0
    )

    if page == "➕ Lägg till / ✏ Uppdatera bolag":
        base = lagg_till_eller_uppdatera(base)
    elif page == "🔄 Uppdatera EN":
        base = uppdatera_bolag(base)
    elif page == "⏩ Massuppdatera alla":
        base = massuppdatera(base)
    elif page == "📦 Portföljöversikt":
        base = portfolj_oversikt(base)
    elif page == "🎯 Köpförslag":
        page_buy_suggestions(base)
    elif page == "📅 Utdelningskalender":
        page_calendar(base)
    elif page == "💾 Spara":
        page_save_now(base)

    st.session_state["working_df"] = säkerställ_kolumner(base)

if __name__ == "__main__":
    main()
