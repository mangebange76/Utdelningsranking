import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time, re, math
from datetime import datetime, timedelta, date
from google.oauth2.service_account import Credentials

# ── Rerun shim & page cfg ─────────────────────────────────────────────────
try:
    _rerun = st.rerun
except AttributeError:
    _rerun = st.experimental_rerun

st.set_page_config(page_title="Relative Yield – utdelningsportfölj", layout="wide")

# ── Secrets / Google Sheets ────────────────────────────────────────────────
SHEET_URL  = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"          # databasflik
SET_SHEET  = "Settings"       # regler/mål
TX_SHEET   = "Transaktioner"  # transaktionslogg

scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def _throttle(min_gap=0.5):
    last = st.session_state.get("_last_call_ts")
    now  = time.time()
    if last and now - last < min_gap:
        time.sleep(min_gap - (now - last))
    st.session_state["_last_call_ts"] = time.time()

def _open_sheet():
    _throttle(0.5)
    return client.open_by_url(SHEET_URL)

def skapa_koppling():
    sh = _open_sheet()
    try:
        return sh.worksheet(SHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=SHEET_NAME, rows=1000, cols=60)
        ws.update([["Ticker"]], value_input_option="RAW")
        return ws

# ── Robust talparser (hindrar “tid”, kommatecken mm) ──────────────────────
def _to_float(x):
    if pd.isna(x): return 0.0
    s = str(x).strip()
    if ":" in s:                # "09:18" → "9.18"
        s = s.replace(":", ".")
        m = re.match(r"^(\d+)\.(\d+)", s)
        if m: s = f"{m.group(1)}.{m.group(2)}"
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s) if s not in ("", ".", "-", "-.") else 0.0
    except Exception:
        return 0.0

# ── FX defaults ───────────────────────────────────────────────────────────
DEF_FX = {"USDSEK": 9.60, "NOKSEK": 0.94, "CADSEK": 6.95, "EURSEK": 11.10}
for k, v in DEF_FX.items():
    st.session_state.setdefault(k, v)

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

# ── Kolumnschema (inkl. manuella lås för utd & frekvens) ──────────────────
COLUMNS = [
    "Ticker","Bolagsnamn","Aktuell kurs","Valuta","Kategori",
    "Direktavkastning (%)","Utdelning/år","Utdelning/år (manuell)","Lås utdelning",
    "Frekvens/år","Frekvens/år (manuell)","Lås frekvens",
    "Utdelningsfrekvens","Payment-lag (dagar)","Ex-Date","Nästa utbetalning (est)",
    "Antal aktier","GAV",
    "Kurs (SEK)","Marknadsvärde (SEK)","Portföljandel (%)",
    "Insatt (SEK)","Årlig utdelning (SEK)","Utdelningstillväxt (%)",
    "Utdelning/år_eff","Frekvens/år_eff",           # ← eff-fält som används i alla beräkningar
    "Utdelningskälla","Senaste uppdatering","Källa"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        d = pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in d.columns:
            d[c] = ""

    d["Ticker"]   = d["Ticker"].astype(str).str.strip().str.upper()
    d["Valuta"]   = d["Valuta"].astype(str).str.strip().str.upper()
    d["Kategori"] = d["Kategori"].astype(str).replace({"": "QUALITY"})

    # booleans
    if "Lås utdelning" in d.columns:
        d["Lås utdelning"] = d["Lås utdelning"].apply(lambda x: bool(x) if pd.notna(x) else False)
    else:
        d["Lås utdelning"] = False
    if "Lås frekvens" in d.columns:
        d["Lås frekvens"] = d["Lås frekvens"].apply(lambda x: bool(x) if pd.notna(x) else False)
    else:
        d["Lås frekvens"] = False

    # numeriska normaliseras
    num_cols = [
        "Aktuell kurs","Utdelning/år","Utdelning/år (manuell)","Frekvens/år","Frekvens/år (manuell)",
        "Payment-lag (dagar)","Antal aktier","GAV","Kurs (SEK)","Marknadsvärde (SEK)",
        "Portföljandel (%)","Insatt (SEK)","Årlig utdelning (SEK)","Direktavkastning (%)",
        "Utdelning/år_eff","Frekvens/år_eff"
    ]
    for c in num_cols:
        d[c] = d[c].apply(_to_float)

    if "Utdelningskälla" not in d.columns:
        d["Utdelningskälla"] = "Yahoo"

    return d[COLUMNS].copy()

def hamta_data():
    try:
        ws = skapa_koppling()
        rows = ws.get_all_records()
        df = pd.DataFrame(rows)
        return säkerställ_kolumner(df)
    except Exception as e:
        st.warning(f"Kunde inte läsa Google Sheet: {e}")
        return säkerställ_kolumner(pd.DataFrame())

# ── Settings (GLOBAL_MAX + kategori-mål) ──────────────────────────────────
DEFAULT_GLOBAL_MAX = 12.0
DEFAULT_CAT_TARGETS = {
    "QUALITY": 40.0, "REIT": 25.0, "mREIT": 10.0, "BDC": 15.0,
    "Shipping": 25.0, "Telecom": 20.0, "Tech": 25.0, "Bank": 20.0,
    "Finance": 20.0, "Energy": 25.0, "Industrial": 20.0, "Other": 10.0
}

def _ensure_settings_sheet():
    sh = _open_sheet()
    try:
        return sh.worksheet(SET_SHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=SET_SHEET, rows=200, cols=4)
        rows = [["Key","Value","Type","Note"],
                ["GLOBAL_MAX_NAME", str(DEFAULT_GLOBAL_MAX), "float","max vikt per bolag i %"]]
        for k,v in DEFAULT_CAT_TARGETS.items():
            rows.append([f"CAT_{k}", str(v), "float","kategori-mål %"])
        ws.update(rows, value_input_option="RAW")
        return ws

def load_settings():
    ws = _ensure_settings_sheet()
    recs = ws.get_all_records()
    gmax = DEFAULT_GLOBAL_MAX
    cats = DEFAULT_CAT_TARGETS.copy()
    for r in recs:
        key = str(r.get("Key",""))
        val = _to_float(r.get("Value",""))
        if key == "GLOBAL_MAX_NAME" and val>0:
            gmax = float(val)
        elif key.startswith("CAT_"):
            cats[key[4:]] = float(val)
    return gmax, cats

def save_settings(global_max, cat_targets: dict):
    ws = _ensure_settings_sheet()
    rows = [["Key","Value","Type","Note"],
            ["GLOBAL_MAX_NAME", str(float(global_max)), "float","max vikt per bolag i %"]]
    for k,v in cat_targets.items():
        rows.append([f"CAT_{k}", str(float(v)), "float","kategori-mål %"])
    ws.clear()
    ws.update(rows, value_input_option="RAW")

# ── Autosnap (backup var 5:e minut, behåll 10 st) ─────────────────────────
def _list_backup_titles(sh):
    try:
        return [ws.title for ws in sh.worksheets() if ws.title.startswith("_Backup_")]
    except Exception:
        return []

def autosnap_now():
    try:
        sh = _open_sheet()
        ws = skapa_koppling()
        snap_title = f"_Backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_ws = sh.add_worksheet(title=snap_title, rows=1, cols=ws.col_count or 60)
        cur = ws.get_all_values()
        if cur:
            backup_ws.update(cur, value_input_option="RAW")
        titles = sorted(_list_backup_titles(sh))
        if len(titles) > 10:
            for t in titles[:-10]:
                try: sh.del_worksheet(sh.worksheet(t))
                except Exception: pass
        st.sidebar.success(f"Autosnap: skapade {snap_title}")
    except Exception as e:
        st.sidebar.warning(f"Autosnap misslyckades: {e}")

def autosnap_if_due(interval_sec=300):
    last = st.session_state.get("_autosnap_last_ts")
    now  = time.time()
    if (last is None) or (now - last >= interval_sec):
        autosnap_now()
        st.session_state["_autosnap_last_ts"] = now

# ── JSON-safe/sanering & anti-wipe i sparning ─────────────────────────────
def _is_finite_number(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False

def _sanitize_for_sheets(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "Aktuell kurs","Utdelning/år","Utdelning/år (manuell)","Frekvens/år","Frekvens/år (manuell)",
        "Payment-lag (dagar)","Antal aktier","GAV","Kurs (SEK)","Marknadsvärde (SEK)",
        "Portföljandel (%)","Insatt (SEK)","Årlig utdelning (SEK)","Direktavkastning (%)",
        "Utdelning/år_eff","Frekvens/år_eff"
    ]
    out = df.copy()
    for c in numeric_cols:
        if c in out.columns:
            out[c] = out[c].apply(_to_float).apply(lambda v: float(v) if _is_finite_number(v) else 0.0)
    for c in out.columns:
        if c not in numeric_cols:
            out[c] = out[c].apply(lambda v: "" if (pd.isna(v) or str(v).lower()=="nan") else v)
    out = out.replace([pd.NA, float("inf"), float("-inf")], "").fillna("")
    return out

def spara_data_safe(df: pd.DataFrame, max_retries: int = 3):
    ws = skapa_koppling()
    out = säkerställ_kolumner(df).copy()

    # Skydda mot tom sparning
    if out.empty or out["Ticker"].astype(str).str.strip().eq("").all():
        st.error("Sparning avbruten: tom data eller inga tickers.")
        return

    # Läs nuvarande blad för jämförelse (anti-wipe)
    try:
        current_rows = ws.get_all_records()
    except Exception:
        current_rows = []
    current_df = säkerställ_kolumner(pd.DataFrame(current_rows))

    def _count_tickers(d):
        return int(d["Ticker"].astype(str).str.strip().ne("").sum())
    old_n = _count_tickers(current_df)
    new_n = _count_tickers(out)
    if old_n > 0 and new_n < max(1, int(0.5 * old_n)):
        st.error(f"Sparning stoppad: nya datasetet ({new_n} tickers) är mycket mindre än nuvarande ({old_n}).")
        st.info("Kontrollera datan, eller spara igen efter att du fyllt på.")
        return

    # Backup före skrivning
    try:
        sh = _open_sheet()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_title = f"_Backup_{timestamp}"
        backup_ws = sh.add_worksheet(title=backup_title, rows=1, cols=max(1, len(out.columns)))
        try:
            cur_rows = ws.get_all_values()
        except Exception:
            cur_rows = []
        if cur_rows:
            backup_ws.update(cur_rows, value_input_option="RAW")
        # trimma äldre backups
        titles = sorted([t for t in _list_backup_titles(sh)])
        if len(titles) > 10:
            for t in titles[:-10]:
                try:
                    sh.del_worksheet(sh.worksheet(t))
                except Exception:
                    pass
    except Exception as e:
        st.warning(f"Kunde inte skapa backupflik (fortsätter ändå): {e}")

    # Sanera data
    out = _sanitize_for_sheets(out)

    header = [out.columns.tolist()]
    body   = out.values.tolist()

    # Uppdatera med retry/backoff
    for attempt in range(1, max_retries+1):
        try:
            ws.update(header + body, value_input_option="RAW")
            break
        except Exception as e:
            msg = str(e)
            if ("Quota exceeded" in msg or "429" in msg) and attempt < max_retries:
                time.sleep(2 * attempt)
                continue
            st.error(f"Sparfel (avbryter): {e}")
            return

    st.success("✅ Sparning klar (säkert läge).")

# 🔄 Alias – använd spara_data() i koden
spara_data = spara_data_safe

# ── Yahoo: frekvens-inferens 24m + TTM-skalning ───────────────────────────
def _infer_frequency_from_dates(idx: pd.Index) -> int:
    if idx is None or len(idx) == 0:
        return 0
    dates = pd.to_datetime(idx).sort_values()
    if len(dates) < 2:
        return 1
    diffs = (dates[1:] - dates[:-1]).days
    if len(diffs) == 0:
        return 1
    med = float(pd.Series(diffs).median())
    candidates = {12:30, 4:90, 2:180, 1:365}
    best = sorted(candidates.items(), key=lambda kv: abs(kv[1] - med))[0][0]
    return best

def fetch_yahoo(ticker: str) -> dict:
    try:
        _throttle(1.0)
        t = (ticker or "").strip().upper()
        if not t:
            return {}

        yf_t = yf.Ticker(t)

        info = {}
        try:
            info = yf_t.get_info() or {}
        except Exception:
            try:
                info = yf_t.info or {}
            except Exception:
                info = {}

        # Pris
        price = None
        try:
            price = getattr(yf_t, "fast_info", {}).get("last_price", None)
        except Exception:
            price = None
        if price in (None, ""):
            price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price in (None, ""):
            try:
                h = yf_t.history(period="5d")
                if not h.empty:
                    price = float(h["Close"].iloc[-1])
            except Exception:
                price = None
        price = _to_float(price)

        currency = (info.get("currency") or "").upper() or "SEK"
        name = info.get("shortName") or info.get("longName") or t

        # Utdelning – 24m för frekvens, TTM för belopp (skalning vid ofullständigt fönster)
        div_year = 0.0
        freq_inferred = 0
        ex_date = ""

        try:
            divs = yf_t.dividends
            if divs is not None and not divs.empty:
                ex_date = pd.to_datetime(divs.index.max()).strftime("%Y-%m-%d")

                cutoff24 = pd.Timestamp.utcnow() - pd.Timedelta(days=730)
                last24 = divs[divs.index >= cutoff24]
                if not last24.empty:
                    freq_inferred = _infer_frequency_from_dates(last24.index)

                cutoff12 = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
                last12 = divs[divs.index >= cutoff12]
                ttm_sum = float(last12.sum()) if not last12.empty else 0.0
                cnt12 = int(last12.shape[0]) if not last12.empty else 0

                if ttm_sum > 0 and freq_inferred > 0:
                    if cnt12 > 0 and cnt12 < freq_inferred:
                        scale = freq_inferred / max(1, cnt12)
                        scale = max(0.75, min(1.50, scale))
                        div_year = ttm_sum * scale
                    else:
                        div_year = ttm_sum
                else:
                    if freq_inferred > 0 and len(divs) > 0:
                        last_amt = float(divs.iloc[-1])
                        if last_amt > 0:
                            div_year = last_amt * freq_inferred

        except Exception:
            pass

        # Fallback forward/trailing om allt annat 0
        if _to_float(div_year) <= 0.0:
            fwd = _to_float(info.get("forwardAnnualDividendRate", 0))
            trl = _to_float(info.get("trailingAnnualDividendRate", 0))
            if fwd > 0: div_year = fwd
            elif trl > 0: div_year = trl

        return {
            "Aktuell kurs": price,
            "Valuta": currency,
            "Bolagsnamn": name,
            "Utdelning/år": _to_float(div_year),
            "Frekvens/år": int(freq_inferred),
            "Ex-Date": ex_date,
            "Källa": "Yahoo"
        }
    except Exception as e:
        st.warning(f"Yahoo-fel {ticker}: {e}")
        return {}

# ── Nästa utdelning (estimerad) – använder Frekvens/år_eff ────────────────
def nästa_utd_datum(row):
    try:
        freq = int(_to_float(row.get("Frekvens/år_eff", row.get("Frekvens/år", 0))))
        if freq <= 0:
            return ""
        exdate_str = str(row.get("Ex-Date", "")).strip()
        if not exdate_str or exdate_str.lower() == "nan":
            return ""
        exdate = datetime.strptime(exdate_str, "%Y-%m-%d").date()
        pay_lag = int(_to_float(row.get("Payment-lag (dagar)", 30)))

        step_days = max(1, int(round(365.0 / max(freq, 1))))
        today_d = date.today()
        while exdate < today_d:
            exdate = exdate + timedelta(days=step_days)
        pay_date = exdate + timedelta(days=max(0, pay_lag))
        return pay_date.strftime("%Y-%m-%d")
    except Exception:
        return ""

def uppdatera_nästa_utd(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Nästa utbetalning (est)"] = d.apply(nästa_utd_datum, axis=1)
    return d

# ── Beräkningar (använder lås & eff-fält) ─────────────────────────────────
def beräkna_allt(df: pd.DataFrame) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()
    d["Kategori"] = d["Kategori"].astype(str).replace({"": "QUALITY"})

    # utdelning – lås om manuell + lås utd = True
    lock_div = d["Lås utdelning"].apply(lambda x: bool(x))
    div_manual = pd.to_numeric(d["Utdelning/år (manuell)"].apply(_to_float), errors="coerce").fillna(0.0).astype(float)
    div_yahoo  = pd.to_numeric(d["Utdelning/år"].apply(_to_float),        errors="coerce").fillna(0.0).astype(float)
    d["Utdelning/år_eff"] = div_yahoo.copy()
    d.loc[(lock_div) & (div_manual > 0), "Utdelning/år_eff"] = div_manual
    d["Utdelningskälla"] = ["Manuell 🔒" if (l and m>0) else "Yahoo" for l, m in zip(lock_div, div_manual)]

    # frekvens – lås om manuell + lås frekvens = True
    lock_freq = d["Lås frekvens"].apply(lambda x: bool(x))
    freq_manual = pd.to_numeric(d["Frekvens/år (manuell)"].apply(_to_float), errors="coerce").fillna(0.0).astype(float)
    freq_yahoo  = pd.to_numeric(d["Frekvens/år"].apply(_to_float), errors="coerce").fillna(0.0).astype(float)
    d["Frekvens/år_eff"] = freq_yahoo.copy()
    d.loc[(lock_freq) & (freq_manual > 0), "Frekvens/år_eff"] = freq_manual

    # priser & FX
    prices = pd.to_numeric(d["Aktuell kurs"].apply(_to_float), errors="coerce").fillna(0.0).astype(float)
    rates  = pd.to_numeric(d["Valuta"].apply(fx_for),          errors="coerce").fillna(1.0).astype(float)
    d["Kurs (SEK)"] = (prices * rates).astype(float).round(6)

    qty = pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0).astype(float)
    gav = pd.to_numeric(d["GAV"],          errors="coerce").fillna(0.0).astype(float)  # GAV i lokal valuta
    div_eff = pd.to_numeric(d["Utdelning/år_eff"], errors="coerce").fillna(0.0).astype(float)

    d["Marknadsvärde (SEK)"] = (qty * d["Kurs (SEK)"]).astype(float).round(2)
    tot_mv = float(d["Marknadsvärde (SEK)"].sum()) or 1.0
    d["Portföljandel (%)"] = (100.0 * d["Marknadsvärde (SEK)"] / tot_mv).astype(float).round(2)

    d["Insatt (SEK)"] = (qty * gav * rates).astype(float).round(2)
    d["Årlig utdelning (SEK)"] = (qty * div_eff * rates).astype(float).round(2)

    safe_price = prices.replace(0, pd.NA)
    d["Direktavkastning (%)"] = (100.0 * div_eff / safe_price).fillna(0.0).astype(float).round(2)

    return d

# ── Avgifter (Avanza/Nordnet mini + FX) ───────────────────────────────────
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

# ── Sidebar: Växelkurser + EN-uppdatering + backup ───────────────────────
def sidebar_tools():
    st.sidebar.header("⚙️ Inställningar")
    st.sidebar.markdown("**Växelkurser (SEK)**")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        st.session_state["USDSEK"] = st.number_input("USD/SEK", 0.0, value=float(st.session_state["USDSEK"]), step=0.01, format="%.4f")
        st.session_state["EURSEK"] = st.number_input("EUR/SEK", 0.0, value=float(st.session_state["EURSEK"]), step=0.01, format="%.4f")
    with c2:
        st.session_state["CADSEK"] = st.number_input("CAD/SEK", 0.0, value=float(st.session_state["CADSEK"]), step=0.01, format="%.4f")
        st.session_state["NOKSEK"] = st.number_input("NOK/SEK", 0.0, value=float(st.session_state["NOKSEK"]), step=0.01, format="%.4f")
    if st.sidebar.button("↩︎ Återställ FX"):
        for k,v in DEF_FX.items(): st.session_state[k] = v
        st.sidebar.success("Standardkurser återställda.")

    st.sidebar.markdown("---")
    if st.sidebar.button("📸 Ta backup nu"):
        autosnap_now()

    st.sidebar.markdown("---")
    one = st.sidebar.text_input("Uppdatera EN ticker (Yahoo)", placeholder="t.ex. VICI").strip().upper()
    if st.sidebar.button("🔄 Uppdatera EN"):
        base = säkerställ_kolumner(st.session_state.get("working_df", pd.DataFrame()))
        if one:
            if one not in base["Ticker"].tolist():
                base = pd.concat([base, pd.DataFrame([{"Ticker":one, "Kategori":"QUALITY"}])], ignore_index=True)
            vals = fetch_yahoo(one)
            m = base["Ticker"]==one
            # skriv inte över låst utdelning / låst frekvens
            if "Aktuell kurs" in vals and vals["Aktuell kurs"] not in (None,""): base.loc[m,"Aktuell kurs"] = vals["Aktuell kurs"]
            if "Valuta" in vals and vals["Valuta"]: base.loc[m,"Valuta"] = vals["Valuta"]
            if "Bolagsnamn" in vals and vals["Bolagsnamn"]: base.loc[m,"Bolagsnamn"] = vals["Bolagsnamn"]
            if "Utdelning/år" in vals and vals["Utdelning/år"] not in (None,"") and (not bool(base.loc[m,"Lås utdelning"].iloc[0])):
                base.loc[m,"Utdelning/år"] = vals["Utdelning/år"]
            if "Frekvens/år" in vals and vals["Frekvens/år"] and (not bool(base.loc[m,"Lås frekvens"].iloc[0])):
                base.loc[m,"Frekvens/år"] = vals["Frekvens/år"]
            if "Ex-Date" in vals: base.loc[m,"Ex-Date"] = vals["Ex-Date"]
            base = beräkna_allt(base)
            base = uppdatera_nästa_utd(base)
            st.session_state["working_df"] = base
            st.sidebar.success(f"{one} uppdaterad i minnet. Spara via menyn när du är klar.")

# ── Settings-sida (max per bolag & kategori-mål) ─────────────────────────
def page_settings(df: pd.DataFrame):
    st.subheader("⚖️ Regler & mål")
    gmax, cats = load_settings()

    present = sorted([c for c in df["Kategori"].dropna().astype(str).unique().tolist()])
    cats_view = {k: float(cats.get(k, 0.0)) for k in present}

    col = st.columns(2)
    with col[0]:
        gmax_new = st.number_input("Max vikt per bolag (%)", min_value=1.0, max_value=100.0, value=float(gmax), step=0.5)
    with col[1]:
        st.caption("Kategorimål (%) – används för att dämpa överviktade kategorier i köpförslag.")
    edit_df = pd.DataFrame([{"Kategori":k, "Mål (%)":v} for k,v in cats_view.items()]).sort_values("Kategori")
    edited = st.data_editor(
        edit_df, hide_index=True, use_container_width=True,
        column_config={
            "Kategori": st.column_config.TextColumn(disabled=True),
            "Mål (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.2f")
        }
    )
    if st.button("💾 Spara regler"):
        new_cats = {row["Kategori"]: float(row["Mål (%)"]) for _, row in edited.iterrows()}
        save_settings(gmax_new, new_cats)
        st.success("Regler sparade till Settings.")

# ── Lägg till / uppdatera bolag (in-memory, spara via ”Spara”) ────────────
CATEGORY_CHOICES = ["QUALITY","REIT","mREIT","BDC","Shipping","Telecom","Tech","Bank","Finance","Energy","Industrial","Other"]

def page_add_or_update(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("➕ Lägg till / ✏ Uppdatera bolag")
    base = säkerställ_kolumner(df).copy()

    tickers = ["Ny"] + sorted(base["Ticker"].unique().tolist())
    val = st.selectbox("Välj bolag", tickers)

    if val == "Ny":
        tkr = st.text_input("Ticker").strip().upper()
        antal = st.number_input("Antal aktier", min_value=0, value=0, step=1)
        gav   = st.number_input("GAV (i **lokal** valuta)", min_value=0.0, value=0.0, step=0.01)
        kategori = st.selectbox("Kategori", options=CATEGORY_CHOICES, index=0)
        c1,c2 = st.columns(2)
        with c1:
            freq_man = st.number_input("Frekvens/år (manuell)", min_value=0, max_value=12, value=0, step=1, help="0 = ej satt")
            lock_freq = st.checkbox("Lås frekvens (använd manuell)", value=False)
            man_div = st.number_input("Utdelning/år (manuell)", min_value=0.0, value=0.0, step=0.01)
            lock_div = st.checkbox("Lås utdelning (använd manuell)", value=False)
        with c2:
            if st.button("🌐 Hämta från Yahoo"):
                if not tkr:
                    st.warning("Ange ticker först.")
                else:
                    vals = fetch_yahoo(tkr)
                    if vals:
                        st.info(f"{vals.get('Bolagsnamn',tkr)} | {vals.get('Valuta','?')} | Kurs {vals.get('Aktuell kurs',0)} | Utd/år {vals.get('Utdelning/år',0)} | Freq {vals.get('Frekvens/år',0)} | ExDate {vals.get('Ex-Date','')}")
        if st.button("➕ Lägg till i minnet"):
            if not tkr:
                st.error("Ticker måste anges.")
            else:
                row = {
                    "Ticker":tkr,"Bolagsnamn":tkr,"Kategori":kategori,
                    "Antal aktier":antal,"GAV":gav,
                    "Valuta":"SEK","Aktuell kurs":0.0,"Utdelning/år":0.0,"Frekvens/år":0,"Ex-Date":"",
                    "Frekvens/år (manuell)":freq_man,"Lås frekvens":lock_freq,
                    "Utdelning/år (manuell)":man_div,"Lås utdelning":lock_div
                }
                vals = fetch_yahoo(tkr)
                for k in ["Aktuell kurs","Valuta","Bolagsnamn","Utdelning/år","Frekvens/år","Ex-Date","Källa"]:
                    if vals.get(k) not in (None,""):
                        # respektera lås:
                        if k == "Utdelning/år" and lock_div: 
                            pass
                        elif k == "Frekvens/år" and lock_freq:
                            pass
                        else:
                            row[k] = vals[k]
                base = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
                base = beräkna_allt(base)
                base = uppdatera_nästa_utd(base)
                st.session_state["working_df"] = base
                st.success(f"{tkr} tillagt i minnet. Gå till 💾 Spara för att skriva till Sheets.")
    else:
        r = base[base["Ticker"]==val].iloc[0]
        tkr = st.text_input("Ticker", value=r["Ticker"]).strip().upper()
        antal = st.number_input("Antal aktier", min_value=0, value=int(_to_float(r["Antal aktier"])), step=1)
        gav   = st.number_input("GAV (i **lokal** valuta)", min_value=0.0, value=float(_to_float(r["GAV"])), step=0.01)
        kategori = st.selectbox("Kategori", options=CATEGORY_CHOICES, index=CATEGORY_CHOICES.index(str(r.get("Kategori","QUALITY"))))

        c1,c2,c3 = st.columns(3)
        with c1:
            freq_man = st.number_input("Frekvens/år (manuell)", min_value=0, max_value=12, value=int(_to_float(r.get("Frekvens/år (manuell)",0))), step=1)
            lock_freq = st.checkbox("Lås frekvens (använd manuell)", value=bool(r.get("Lås frekvens", False)))
        with c2:
            man_div = st.number_input("Utdelning/år (manuell)", min_value=0.0, value=float(_to_float(r.get("Utdelning/år (manuell)",0.0))), step=0.01)
            lock_div = st.checkbox("Lås utdelning (använd manuell)", value=bool(r.get("Lås utdelning", False)))
        with c3:
            if st.button("🌐 Uppdatera från Yahoo"):
                vals = fetch_yahoo(tkr)
                m = base["Ticker"]==val
                if "Aktuell kurs" in vals and vals["Aktuell kurs"] not in (None,""): base.loc[m,"Aktuell kurs"] = vals["Aktuell kurs"]
                if "Valuta" in vals and vals["Valuta"]: base.loc[m,"Valuta"] = vals["Valuta"]
                if "Bolagsnamn" in vals and vals["Bolagsnamn"]: base.loc[m,"Bolagsnamn"] = vals["Bolagsnamn"]
                if "Utdelning/år" in vals and vals["Utdelning/år"] not in (None,"") and (not lock_div):
                    base.loc[m,"Utdelning/år"] = vals["Utdelning/år"]
                if "Frekvens/år" in vals and vals["Frekvens/år"] and (not lock_freq):
                    base.loc[m,"Frekvens/år"] = vals["Frekvens/år"]
                if "Ex-Date" in vals: base.loc[m,"Ex-Date"] = vals["Ex-Date"]

                base.loc[m,"Ticker"] = tkr
                base.loc[m,"Antal aktier"] = antal
                base.loc[m,"GAV"] = gav
                base.loc[m,"Kategori"] = kategori
                base.loc[m,"Frekvens/år (manuell)"] = freq_man
                base.loc[m,"Lås frekvens"] = lock_freq
                base.loc[m,"Utdelning/år (manuell)"] = man_div
                base.loc[m,"Lås utdelning"] = lock_div

                base = beräkna_allt(base)
                base = uppdatera_nästa_utd(base)
                st.session_state["working_df"] = base
                st.success(f"{tkr} uppdaterad i minnet.")

        if st.button("✏ Uppdatera fält (minne)"):
            m = base["Ticker"]==val
            base.loc[m,"Ticker"] = tkr
            base.loc[m,"Antal aktier"] = antal
            base.loc[m,"GAV"] = gav
            base.loc[m,"Kategori"] = kategori
            base.loc[m,"Frekvens/år (manuell)"] = freq_man
            base.loc[m,"Lås frekvens"] = lock_freq
            base.loc[m,"Utdelning/år (manuell)"] = man_div
            base.loc[m,"Lås utdelning"] = lock_div
            base = beräkna_allt(base)
            base = uppdatera_nästa_utd(base)
            st.session_state["working_df"] = base
            st.success(f"{tkr} uppdaterad i minnet.")

        if st.button("🗑 Ta bort (minne)"):
            base = base[base["Ticker"]!=val].reset_index(drop=True)
            base = beräkna_allt(base)
            st.session_state["working_df"] = base
            st.success(f"{val} borttagen i minnet.")

    st.markdown("---")
    if st.button("💾 Spara alla ändringar till Google Sheets"):
        spara_data(beräkna_allt(st.session_state["working_df"]))
        st.success("Sparat till Sheets.")
    return st.session_state.get("working_df", base)

# ── Portfölj ──────────────────────────────────────────────────────────────
def page_portfolio(df: pd.DataFrame):
    st.subheader("📦 Portföljöversikt")
    d = uppdatera_nästa_utd(beräkna_allt(df).copy())
    if d.empty:
        st.info("Lägg till minst ett bolag.")
        return
    tot_mv  = float(d["Marknadsvärde (SEK)"].sum())
    tot_ins = float(d["Insatt (SEK)"].sum())
    tot_div = float(d["Årlig utdelning (SEK)"].sum())

    c1,c2,c3 = st.columns(3)
    c1.metric("Portföljvärde (SEK)", f"{tot_mv:,.0f}".replace(","," "))
    c2.metric("Insatt (SEK)", f"{tot_ins:,.0f}".replace(","," "))
    c3.metric("Årlig utdelning (SEK)", f"{tot_div:,.0f}".replace(","," "))

    show_cols = [
        "Ticker","Bolagsnamn","Valuta","Kategori","Antal aktier","GAV",
        "Aktuell kurs","Kurs (SEK)","Marknadsvärde (SEK)","Portföljandel (%)",
        "Utdelning/år","Utdelning/år_eff","Frekvens/år","Frekvens/år_eff",
        "Årlig utdelning (SEK)","Ex-Date","Nästa utbetalning (est)"
    ]
    st.dataframe(d[show_cols], use_container_width=True)

# ── Kalender ──────────────────────────────────────────────────────────────
def page_calendar(df: pd.DataFrame):
    st.subheader("📅 Utdelningskalender")
    months = st.selectbox("Prognoshorisont (mån)", options=[12,24,36], index=0)

    def _gen(first_date, freq, lag, months_ahead):
        ts = pd.to_datetime(first_date, errors="coerce")
        if pd.isna(ts): return []
        exd = ts.date()
        try: f = max(1, int(_to_float(freq)))
        except: f = 4
        try: L = max(0, int(_to_float(lag)))
        except: L = 30
        step = max(1, int(round(365.0 / f)))
        today_d = date.today()
        horizon = today_d + timedelta(days=int(round(months_ahead*30.44)))
        while exd < today_d:
            exd = exd + timedelta(days=step)
        pays = []
        pay = exd + timedelta(days=L)
        while pay <= horizon:
            pays.append(pay)
            exd = exd + timedelta(days=step)
            pay = exd + timedelta(days=L)
        return pays

    d = beräkna_allt(df).copy()
    rows = []
    for _, r in d.iterrows():
        per_share_local = _to_float(r["Utdelning/år_eff"]) / max(1.0, _to_float(r.get("Frekvens/år_eff", r.get("Frekvens/år",4))))
        qty = _to_float(r.get("Antal aktier",0.0))
        fx  = fx_for(r.get("Valuta","SEK"))
        per_payment_sek = per_share_local * fx * qty
        if per_payment_sek <= 0: continue
        for p in _gen(r.get("Ex-Date",""), r.get("Frekvens/år_eff", r.get("Frekvens/år",4)), r.get("Payment-lag (dagar)",30), months):
            rows.append({"Datum": p, "Ticker": r["Ticker"], "Belopp (SEK)": round(per_payment_sek,2)})
    if not rows:
        st.info("Ingen prognos – saknar data.")
        return
    cal = pd.DataFrame(rows)
    cal["Månad"] = cal["Datum"].apply(lambda d: f"{d.year}-{str(d.month).zfill(2)}")
    monthly = cal.groupby("Månad", as_index=False)["Belopp (SEK)"].sum().rename(columns={"Belopp (SEK)":"Utdelning (SEK)"}).sort_values("Månad")
    st.dataframe(monthly, use_container_width=True)
    st.bar_chart(monthly.set_index("Månad")["Utdelning (SEK)"])
    with st.expander("Detaljer per betalning"):
        st.dataframe(cal.sort_values("Datum"), use_container_width=True)

# ── Massuppdatering ───────────────────────────────────────────────────────
def page_mass_update(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("⏩ Massuppdatera alla bolag (Yahoo)")
    base = säkerställ_kolumner(df).copy()
    if base.empty:
        st.info("Inga bolag i databasen ännu.")
        return base

    if st.button("Starta massuppdatering"):
        progress = st.progress(0)
        status   = st.empty()
        N = len(base)
        for i, tkr in enumerate(base["Ticker"].tolist(), start=1):
            status.write(f"Uppdaterar {tkr} ({i}/{N}) …")
            vals = fetch_yahoo(tkr)
            m = base["Ticker"]==tkr

            # Respektera lås:
            if "Aktuell kurs" in vals and vals["Aktuell kurs"] not in (None,""): base.loc[m,"Aktuell kurs"] = vals["Aktuell kurs"]
            if "Valuta" in vals and vals["Valuta"]: base.loc[m,"Valuta"] = vals["Valuta"]
            if "Bolagsnamn" in vals and vals["Bolagsnamn"]: base.loc[m,"Bolagsnamn"] = vals["Bolagsnamn"]
            if "Utdelning/år" in vals and vals["Utdelning/år"] not in (None,"") and (not bool(base.loc[m,"Lås utdelning"].iloc[0])):
                base.loc[m,"Utdelning/år"] = vals["Utdelning/år"]
            if "Frekvens/år" in vals and vals["Frekvens/år"] and (not bool(base.loc[m,"Lås frekvens"].iloc[0])):
                base.loc[m,"Frekvens/år"] = vals["Frekvens/år"]
            if "Ex-Date" in vals: base.loc[m,"Ex-Date"] = vals["Ex-Date"]

            base = beräkna_allt(base)
            base = uppdatera_nästa_utd(base)
            progress.progress(int(i*100/N))
            time.sleep(1.0)  # respekt mot Yahoo
        st.session_state["working_df"] = base
        st.success("Massuppdatering klar (i minnet). Gå till 💾 Spara för att skriva till Sheets.")
    return st.session_state.get("working_df", base)

# ── Spara ─────────────────────────────────────────────────────────────────
def page_save(df: pd.DataFrame):
    st.subheader("💾 Spara nu till Google Sheets")
    preview = uppdatera_nästa_utd(beräkna_allt(säkerställ_kolumner(df)))
    st.write("Rader som sparas:", len(preview))
    st.dataframe(
        preview[["Ticker","Bolagsnamn","Valuta","Kategori","Antal aktier","GAV","Aktuell kurs","Utdelning/år","Utdelning/år_eff","Frekvens/år","Frekvens/år_eff","Kurs (SEK)","Årlig utdelning (SEK)"]],
        use_container_width=True
    )
    danger = st.checkbox("⚠️ Tillåt riskabel överskrivning (anti-wipe finns kvar men use with care)", value=False)
    if st.button("✅ Bekräfta och spara"):
        spara_data(preview)

# ── Köpsimulator (≈500 kr-lotter) ─────────────────────────────────────────
def _n_affordable(price_sek, cash, foreign):
    if price_sek <= 0 or cash <= 0: return 0
    approx = int(max(1, cash // price_sek))
    for n in range(approx, 0, -1):
        gross = price_sek * n
        _, _, fee = calc_fees(gross, foreign)
        if gross + fee <= cash + 1e-9:
            return n
    return 0

def _cap_shares_limit(current_value, total_value, px, limit_pct):
    if px <= 0: return 0
    m = limit_pct/100.0
    numer = m*total_value - current_value
    denom = (1.0 - m) * px
    if denom <= 0: return 0
    return int(max(0, math.floor(numer/denom)))

def page_buy_planner(df: pd.DataFrame):
    st.subheader("🎯 Köpförslag & plan (≈500 kr per köp)")
    base = uppdatera_nästa_utd(beräkna_allt(df).copy())

    gmax, cat_targets = load_settings()
    present_cats = set(base["Kategori"].astype(str).unique().tolist())
    cat_limits = {k: v for k, v in cat_targets.items() if k in present_cats}

    c1,c2,c3 = st.columns(3)
    with c1:
        cash = st.number_input("Tillgänglig kassa (SEK)", min_value=0.0, value=2000.0, step=100.0)
    with c2:
        lot  = st.number_input("Belopp per köp (≈)", min_value=100.0, value=500.0, step=50.0)
    with c3:
        gmax_ui = st.number_input("Max per bolag (%)", min_value=1.0, max_value=100.0, value=float(gmax), step=0.5)

    def _score(r):
        da = float(_to_float(r["Direktavkastning (%)"]))
        da_score = (min(max(da,0),15)/15.0)*100.0
        under = max(0.0, gmax_ui - float(_to_float(r["Portföljandel (%)"])))
        under_score = (under/max(gmax_ui,1e-9))*100.0
        dt = pd.to_datetime(r.get("Nästa utbetalning (est)",""), errors="coerce")
        days = 9999 if pd.isna(dt) else max(0,(dt.date()-date.today()).days)
        time_score = ((90 - min(days,90))/90.0)*100.0
        return 0.5*da_score + 0.35*under_score + 0.15*time_score

    cand = base.copy()
    cand["Poäng"] = cand.apply(_score, axis=1)
    cand = cand.sort_values("Poäng", ascending=False).reset_index(drop=True)

    T = float(base["Marknadsvärde (SEK)"].sum())
    if T <= 0: T = 1.0
    cat_val = base.groupby("Kategori", as_index=False)["Marknadsvärde (SEK)"].sum().set_index("Kategori")["Marknadsvärde (SEK)"].to_dict()
    qty_map = base.set_index("Ticker")["Antal aktier"].to_dict()

    steps = []
    used  = 0.0
    while cash - used >= min(50.0, lot):
        picked = None
        for _, r in cand.iterrows():
            tkr = r["Ticker"]; cat = r["Kategori"]
            price = float(_to_float(r["Kurs (SEK)"]))
            if price <= 0: continue
            foreign = str(r["Valuta"]).upper() != "SEK"
            Vi = float(_to_float(r["Marknadsvärde (SEK)"]))
            C  = float(cat_val.get(cat, 0.0))
            n_name = _cap_shares_limit(Vi, T, price, gmax_ui)
            n_cat  = _cap_shares_limit(C,  T, price, float(cat_limits.get(cat, 100.0)))
            if min(n_name, n_cat) <= 0: continue
            n_cash = _n_affordable(price, lot, foreign)
            n = max(1, min(n_name, n_cat, n_cash))
            gross = price * n
            c_fee, fx_fee, tot_fee = calc_fees(gross, foreign)
            total_cost = gross + tot_fee
            if used + total_cost > cash + 1e-9:
                continue
            picked = {
                "Ticker": tkr, "Kategori": cat, "Antal": int(n),
                "Pris (SEK)": round(price,2), "Kostnad (SEK)": round(total_cost,2),
                "Courtage": c_fee, "FX-avg": fx_fee, "Poäng": round(float(r["Poäng"]),1),
                "Kommentar": f"Lot ~{int(lot)} kr, under {gmax_ui:.0f}% & kat≤{cat_limits.get(cat,100):.0f}%"
            }
            used += total_cost
            qty_map[tkr] = qty_map.get(tkr, 0.0) + n
            add_value = price * n
            Vi += add_value; C += add_value; T += add_value
            cat_val[cat] = C
            steps.append(picked)
            break
        if picked is None:
            break

    if not steps:
        st.info("Ingen plan kunde skapas givet reglerna/kassan.")
        return

    plan = pd.DataFrame(steps)
    per_ticker = (plan.groupby(["Ticker","Kategori"], as_index=False)
                        .agg({"Antal":"sum","Kostnad (SEK)":"sum","Pris (SEK)":"last","Poäng":"max"}))
    st.write("**Plan – steg för steg:**")
    st.dataframe(plan, use_container_width=True)
    st.write("**Summering per ticker:**")
    st.dataframe(per_ticker, use_container_width=True)

# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    st.title("Relative Yield – utdelningsportfölj")

    # Initiera in-memory tabellen från Google Sheets en gång
    if "working_df" not in st.session_state:
        try:
            st.session_state["working_df"] = hamta_data()
        except Exception:
            st.session_state["working_df"] = säkerställ_kolumner(pd.DataFrame())

    # Autosnap var 5 min
    autosnap_if_due(300)

    # Sidebar
    sidebar_tools()

    page = st.sidebar.radio(
        "Meny",
        [
            "📦 Portföljöversikt",
            "⚖️ Regler & mål",
            "➕ Lägg till / ✏ Uppdatera bolag",
            "⏩ Massuppdatera alla",
            "🎯 Köpförslag & plan",
            "📅 Utdelningskalender",
            "💾 Spara"
        ],
        index=0
    )

    base = säkerställ_kolumner(st.session_state["working_df"]).copy()

    if page == "📦 Portföljöversikt":
        page_portfolio(base)
    elif page == "⚖️ Regler & mål":
        _ = load_settings()  # säkerställ att sheetet finns
        page_settings(base)
    elif page == "➕ Lägg till / ✏ Uppdatera bolag":
        base = page_add_or_update(base)
    elif page == "⏩ Massuppdatera alla":
        base = page_mass_update(base)
    elif page == "🎯 Köpförslag & plan":
        page_buy_planner(base)
    elif page == "📅 Utdelningskalender":
        page_calendar(base)
    elif page == "💾 Spara":
        page_save(base)

    st.session_state["working_df"] = säkerställ_kolumner(base)

if __name__ == "__main__":
    main()
