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

st.set_page_config(page_title="Utdelningsranking & Portfölj", layout="wide")

# ── Secrets / Google Sheets ────────────────────────────────────────────────
SHEET_URL  = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"          # databasflik
SET_SHEET  = "Settings"       # regler/mål + parametrar
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
        ws = sh.add_worksheet(title=SHEET_NAME, rows=1000, cols=50)
        ws.update([["Ticker"]], value_input_option="RAW")
        return ws

# ── Robust talparser (hindrar tid/datum, kommatecken etc) ─────────────────
def _to_float(x):
    if pd.isna(x): return 0.0
    s = str(x).strip()

    # Tid 09:15 → "9.15"
    if re.fullmatch(r"\d{1,2}:\d{2}(\.\d+)?", s):
        s = s.replace(":", ".")

    # Vanliga tusentals-/decimaltecken
    # ex: "1 234,56" | "1.234,56" | "1,234.56"
    s = s.replace(" ", "")
    if s.count(",") == 1 and s.count(".") > 1:
        # "1.234.567,89" → byt bort tusenpunkter
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")

    s = re.sub(r"[^0-9\.\-eE]", "", s)
    try:
        v = float(s)
        if not math.isfinite(v): return 0.0
        return v
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

# ── Kolumnschema ───────────────────────────────────────────────────────────
COLUMNS = [
    "Ticker","Bolagsnamn","Aktuell kurs","Valuta","Kategori",
    "Direktavkastning (%)","Utdelning/år","Utdelning/år (manuell)","Lås utdelning",
    "Frekvens/år","Utdelningsfrekvens","Payment-lag (dagar)","Ex-Date","Nästa utbetalning (est)",
    "Antal aktier","GAV",
    "Kurs (SEK)","Marknadsvärde (SEK)","Portföljandel (%)",
    "Insatt (SEK)","Årlig utdelning (SEK)","Utdelningstillväxt (%)",
    "Utdelningskälla","Senaste uppdatering","Källa"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        d = pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in d.columns:
            d[c] = "" if c not in ("Lås utdelning",) else False

    # normalisera str
    d["Ticker"] = d["Ticker"].astype(str).str.strip().str.upper()
    d["Valuta"] = d["Valuta"].astype(str).str.strip().str.upper()
    d["Kategori"] = d["Kategori"].astype(str).str.strip()
    d.loc[d["Kategori"].isin(["", "nan", "NaN", "None"]), "Kategori"] = "QUALITY"

    # numeriska kolumner → floats
    num_cols = ["Aktuell kurs","Utdelning/år","Utdelning/år (manuell)","Frekvens/år","Payment-lag (dagar)",
                "Antal aktier","GAV","Kurs (SEK)","Marknadsvärde (SEK)","Portföljandel (%)",
                "Insatt (SEK)","Årlig utdelning (SEK)","Direktavkastning (%)","Utdelningstillväxt (%)"]
    for c in num_cols:
        d[c] = d[c].apply(_to_float)

    # boolean
    d["Lås utdelning"] = d["Lås utdelning"].apply(lambda x: bool(x) if pd.notna(x) else False)

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
        st.warning(f"Kunde inte läsa data från Google Sheets: {e}")
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
        backup_ws = sh.add_worksheet(title=snap_title, rows=1, cols=ws.col_count or 50)
        cur = ws.get_all_values()
        if cur:
            backup_ws.update(cur, value_input_option="RAW")
        # trimma äldre backups (behåll 10 senaste)
        titles = sorted(_list_backup_titles(sh))
        if len(titles) > 10:
            for t in titles[:-10]:
                try:
                    sh.del_worksheet(sh.worksheet(t))
                except Exception:
                    pass
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
    numeric_cols = ["Aktuell kurs","Utdelning/år","Utdelning/år (manuell)","Frekvens/år","Payment-lag (dagar)",
                    "Antal aktier","GAV","Kurs (SEK)","Marknadsvärde (SEK)","Portföljandel (%)",
                    "Insatt (SEK)","Årlig utdelning (SEK)","Direktavkastning (%)","Utdelningstillväxt (%)"]
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

# ── Yahoo Finance-hämtning ────────────────────────────────────────────────
def fetch_yahoo(ticker: str) -> dict:
    """Hämtar pris, valuta, namn, utdelning 12m och frekvens ~12m från yfinance."""
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

        # Pris (lokal)
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

        # Utdelningar
        div_year = 0.0
        freq = 0
        ex_date = ""
        try:
            divs = yf_t.dividends
            if divs is not None and not divs.empty:
                cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
                last12 = divs[divs.index >= cutoff]
                div_year = float(last12.tail(12).sum()) if not last12.empty else 0.0
                # frekvens = antal utdelningstillfällen senaste 12 månaderna (clamp 0..12)
                freq = int(max(0, min(12, last12.shape[0]))) if not last12.empty else 0
                ex_date = pd.to_datetime(divs.index.max()).strftime("%Y-%m-%d")
        except Exception:
            pass

        return {
            "Aktuell kurs": price,
            "Valuta": currency,
            "Bolagsnamn": name,
            "Utdelning/år": div_year,
            "Frekvens/år": freq,
            "Ex-Date": ex_date,
            "Källa": "Yahoo"
        }
    except Exception as e:
        st.warning(f"Yahoo-fel {ticker}: {e}")
        return {}

# ── Nästa utdelning (estimerad) ───────────────────────────────────────────
def nästa_utd_datum(row):
    try:
        freq = int(_to_float(row.get("Frekvens/år", 0)))
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

# ── Beräkningar (robusta casts till float) ────────────────────────────────
def beräkna_allt(df: pd.DataFrame) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()
    d["Kategori"] = d["Kategori"].astype(str).replace({"": "QUALITY"})

    # låsning av utd/år
    lock = d["Lås utdelning"].apply(lambda x: bool(x))
    div_manual = pd.to_numeric(d["Utdelning/år (manuell)"].apply(_to_float), errors="coerce").fillna(0.0).astype(float)
    div_yahoo  = pd.to_numeric(d["Utdelning/år"].apply(_to_float),        errors="coerce").fillna(0.0).astype(float)
    d["Utdelning/år_eff"] = div_yahoo.copy()
    d.loc[(lock) & (div_manual > 0), "Utdelning/år_eff"] = div_manual
    d["Utdelningskälla"] = ["Manuell 🔒" if (l and m>0) else "Yahoo" for l, m in zip(lock, div_manual)]

    prices = pd.to_numeric(d["Aktuell kurs"].apply(_to_float), errors="coerce").fillna(0.0).astype(float)
    rates  = pd.to_numeric(d["Valuta"].apply(fx_for),          errors="coerce").fillna(1.0).astype(float)
    d["Kurs (SEK)"] = (prices * rates).astype(float).round(6)

    qty = pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0).astype(float)
    gav = pd.to_numeric(d["GAV"],          errors="coerce").fillna(0.0).astype(float)
    div_eff = pd.to_numeric(d["Utdelning/år_eff"], errors="coerce").fillna(0.0).astype(float)

    d["Marknadsvärde (SEK)"] = (qty * d["Kurs (SEK)"]).astype(float).round(2)
    tot_mv = float(d["Marknadsvärde (SEK)"].sum()) or 1.0
    d["Portföljandel (%)"] = (100.0 * d["Marknadsvärde (SEK)"] / tot_mv).astype(float).round(2)

    d["Insatt (SEK)"] = (qty * gav * rates).astype(float).round(2)
    d["Årlig utdelning (SEK)"] = (qty * div_eff * rates).astype(float).round(2)

    safe_price = prices.replace(0, pd.NA)
    d["Direktavkastning (%)"] = (100.0 * div_eff / safe_price).fillna(0.0).astype(float).round(2)

    # sanity guard – om något spårar ur (t.ex. 1e12) nollställ hellre än att visa galet
    for col in ["Kurs (SEK)","Marknadsvärde (SEK)","Insatt (SEK)","Årlig utdelning (SEK)","Direktavkastning (%)"]:
        d[col] = d[col].apply(lambda v: 0.0 if (pd.isna(v) or not math.isfinite(float(v)) or abs(float(v))>1e9) else float(v))

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

# ── Sidebar: Växelkurser + snabb EN-uppdatering + manuell backup ─────────
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
            for k in ["Aktuell kurs","Valuta","Bolagsnamn","Utdelning/år","Frekvens/år","Ex-Date","Källa"]:
                if k in vals and vals[k] not in (None,""):
                    base.loc[m, k] = vals[k]
            base = beräkna_allt(base)
            base = uppdatera_nästa_utd(base)
            st.session_state["working_df"] = base
            st.sidebar.success(f"{one} uppdaterad i minnet. Spara via menyn när du är klar.")

# ── Regler & mål (med sammanställningar) ─────────────────────────────────
def page_settings(df: pd.DataFrame):
    st.subheader("⚖️ Regler & mål")
    gmax, cats = load_settings()

    base = beräkna_allt(df)
    # sammanställningar
    if base.empty:
        st.info("Ingen data ännu.")
    else:
        st.markdown("**Kategori-fördelning (nuvarande)**")
        cat_sum = (base.groupby("Kategori", as_index=False)
                        .agg({"Marknadsvärde (SEK)":"sum"}))
        T = float(cat_sum["Marknadsvärde (SEK)"].sum()) or 1.0
        cat_sum["Andel (%)"] = (100.0 * cat_sum["Marknadsvärde (SEK)"]/T).round(2)
        st.dataframe(cat_sum.sort_values("Andel (%)", ascending=False), use_container_width=True)

        st.markdown("**Bolag – andel av portföljen (topp→botten)**")
        comp = base[["Ticker","Bolagsnamn","Kategori","Marknadsvärde (SEK)","Portföljandel (%)"]].sort_values("Portföljandel (%)", ascending=False)
        st.dataframe(comp, use_container_width=True)

    # visa bara kategorier som finns i databasen
    present = sorted([c for c in base["Kategori"].dropna().astype(str).unique().tolist()]) if not base.empty else sorted(cats.keys())
    cats_view = {k: float(cats.get(k, 0.0)) for k in present}

    col = st.columns(2)
    with col[0]:
        gmax_new = st.number_input("Max vikt per bolag (%)", min_value=1.0, max_value=100.0, value=float(gmax), step=0.5)
    with col[1]:
        st.caption("Kategorimål (%) – används för att dämpa överviktade kategorier i köpförslag samt i trim/sälj.")

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

# ── Lägg till / uppdatera bolag (in-memory; spara via ”Spara”-sida) ──────
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
            if st.button("🌐 Hämta från Yahoo"):
                if not tkr:
                    st.warning("Ange ticker först.")
                else:
                    vals = fetch_yahoo(tkr)
                    if vals:
                        st.info(f"{vals.get('Bolagsnamn',tkr)} | {vals.get('Valuta','?')} | Kurs {vals.get('Aktuell kurs',0)} | Utd/år {vals.get('Utdelning/år',0)} | Freq {vals.get('Frekvens/år',0)} | ExDate {vals.get('Ex-Date','')}")
        with c2:
            if st.button("➕ Lägg till i minnet"):
                if not tkr:
                    st.error("Ticker måste anges.")
                else:
                    row = {"Ticker":tkr,"Bolagsnamn":tkr,"Kategori":kategori,"Antal aktier":antal,"GAV":gav,
                           "Valuta":"SEK","Aktuell kurs":0.0,"Utdelning/år":0.0,"Frekvens/år":0,"Ex-Date":""}
                    vals = fetch_yahoo(tkr)
                    for k in ["Aktuell kurs","Valuta","Bolagsnamn","Utdelning/år","Frekvens/år","Ex-Date","Källa"]:
                        if vals.get(k) not in (None,""):
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
            if st.button("🌐 Uppdatera från Yahoo"):
                vals = fetch_yahoo(tkr)
                m = base["Ticker"]==val
                for k in ["Aktuell kurs","Valuta","Bolagsnamn","Utdelning/år","Frekvens/år","Ex-Date","Källa"]:
                    if k in vals and vals[k] not in (None,""):
                        base.loc[m,k] = vals[k]
                base.loc[m,"Ticker"] = tkr
                base.loc[m,"Antal aktier"] = antal
                base.loc[m,"GAV"] = gav
                base.loc[m,"Kategori"] = kategori
                base = beräkna_allt(base)
                base = uppdatera_nästa_utd(base)
                st.session_state["working_df"] = base
                st.success(f"{tkr} uppdaterad i minnet.")
        with c2:
            if st.button("✏ Uppdatera fält (minne)"):
                m = base["Ticker"]==val
                base.loc[m,"Ticker"] = tkr
                base.loc[m,"Antal aktier"] = antal
                base.loc[m,"GAV"] = gav
                base.loc[m,"Kategori"] = kategori
                base = beräkna_allt(base)
                base = uppdatera_nästa_utd(base)
                st.session_state["working_df"] = base
                st.success(f"{tkr} uppdaterad i minnet.")
        with c3:
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
    """Max extra antal aktier som kan köpas utan att bolagsvikt > limit_pct efter köpet."""
    if px <= 0: return 0
    m = limit_pct/100.0
    numer = m*total_value - current_value
    denom = (1.0 - m) * px
    if denom <= 0: return 0
    return int(max(0, math.floor(numer/denom)))

def _cap_shares_by_cat(cat_value, total_value, px, cat_limit_pct):
    """Max extra antal aktier utan att kategorivikt > cat_limit_pct."""
    if px <= 0: return 0
    M = cat_limit_pct/100.0
    numer = M*total_value - cat_value
    denom = (1.0 - M) * px
    if denom <= 0: return 0
    return int(max(0, math.floor(numer/denom)))

def _score_row(r, gmax_ui):
    da = float(_to_float(r["Direktavkastning (%)"]))
    da_score = (min(max(da,0),15)/15.0)*100.0
    under = max(0.0, gmax_ui - float(_to_float(r["Portföljandel (%)"])))
    under_score = (under/max(gmax_ui,1e-9))*100.0
    dt = pd.to_datetime(r.get("Nästa utbetalning (est)",""), errors="coerce")
    days = 9999 if pd.isna(dt) else max(0,(dt.date()-date.today()).days)
    time_score = ((90 - min(days,90))/90.0)*100.0
    return 0.5*da_score + 0.35*under_score + 0.15*time_score

def page_buy_planner(df: pd.DataFrame):
    st.subheader("🎯 Köpförslag & plan (≈500 kr per köp) + trim/sälj-kandidater")
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

    # Rangordna kandidater
    cand = base.copy()
    cand["Poäng"] = cand.apply(lambda r: _score_row(r, gmax_ui), axis=1)
    cand = cand.sort_values("Poäng", ascending=False).reset_index(drop=True)

    # Simulera köp i ~lot-storlek tills kassan slut
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
            n_cat  = _cap_shares_by_cat(C,  T, price, float(cat_limits.get(cat, 100.0)))
            if min(n_name, n_cat) <= 0: 
                continue

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

    # Visa plan
    if steps:
        plan = pd.DataFrame(steps)
        per_ticker = (plan.groupby(["Ticker","Kategori"], as_index=False)
                            .agg({"Antal":"sum","Kostnad (SEK)":"sum","Pris (SEK)":"last","Poäng":"max"}))
        st.write("**Plan – steg för steg:**")
        st.dataframe(plan, use_container_width=True)
        st.write("**Summering per ticker:**")
        st.dataframe(per_ticker, use_container_width=True)

        st.session_state["last_plan"] = plan.to_dict(orient="records")

        # Spara plan till transaktionslogg (utan att påverka antal aktier)
        if st.button("📝 Spara plan som transaktionslogg"):
            rows = []
            base_ccy = base.set_index("Ticker")["Valuta"].to_dict()
            for r in steps:
                rows.append(["KÖP", r["Ticker"], r["Antal"], r["Pris (SEK)"], base_ccy.get(r["Ticker"], "SEK"), r["Kostnad (SEK)"], "Plan"])
            logga_transaktioner(rows)
            st.success("Plan sparad i fliken Transaktioner.")

        if st.button("🗑 Rensa plan (minne)"):
            st.session_state["last_plan"] = []
            st.info("Plan rensad (minne).")
    else:
        st.info("Ingen plan kunde skapas givet reglerna/kassan.")

    st.markdown("---")
    # Trim/sälj-kandidater
    st.subheader("✂️ Trim/sälj-kandidater")
    # över  gmax
    over = base[ base["Portföljandel (%)"] > gmax_ui ].copy()
    if over.empty and base.empty==False:
        st.success("Inget enskilt bolag över max-vikten just nu.")
    else:
        if not over.empty:
            st.dataframe(over[["Ticker","Bolagsnamn","Kategori","Portföljandel (%)","Kurs (SEK)","Antal aktier"]].sort_values("Portföljandel (%)", ascending=False), use_container_width=True)

    # Kategorier över mål
    cat_sum = (base.groupby("Kategori", as_index=False)
                    .agg({"Marknadsvärde (SEK)":"sum"}))
    T = float(cat_sum["Marknadsvärde (SEK)"].sum()) or 1.0
    cat_sum["Andel (%)"] = (100.0 * cat_sum["Marknadsvärde (SEK)"]/T)
    over_cats = []
    for _, r in cat_sum.iterrows():
        lim = float(cat_limits.get(r["Kategori"], 100.0))
        if float(r["Andel (%)"]) > lim:
            over_cats.append((r["Kategori"], float(r["Andel (%)"]), lim))
    if over_cats:
        st.warning("Kategorier över mål:")
        st.write(pd.DataFrame([{"Kategori":k,"Nu (%)":nu,"Mål (%)":lim} for k,nu,lim in over_cats]))
        # föreslå trim inom dessa kategorier: sälj från tyngsta namn
        trims = []
        for cat, nu, lim in over_cats:
            sub = base[base["Kategori"]==cat].copy()
            sub = sub.sort_values("Portföljandel (%)", ascending=False)
            # sälj från det tyngsta namnet först (en enkel heuristik)
            for _, r in sub.iterrows():
                price = float(_to_float(r["Kurs (SEK)"]))
                if price <= 0: continue
                Vi = float(_to_float(r["Marknadsvärde (SEK)"]))
                qty = float(_to_float(r["Antal aktier"]))
                # hur många aktier måste bort för att kat-vikten nå lim? (approx: sälj i ett namn)
                cat_value = float(cat_val.get(cat, 0.0))
                excess = max(0.0, (nu - lim)/100.0 * T)
                if excess <= 0: break
                n = int(min(qty, math.ceil(excess / price)))
                if n <= 0: continue
                trims.append({"Ticker":r["Ticker"],"Kategori":cat,"Föreslagen sälj (st)":n,
                              "Pris (SEK)":round(price,2),"Ca likvid (SEK)":round(n*price,2)})
                break
        if trims:
            st.dataframe(pd.DataFrame(trims), use_container_width=True)
        else:
            st.info("Ingen tydlig trim att föreslå baserat på enkel heuristik.")

# ── Transaktionslogg (för ”Spara plan som logg”) ──────────────────────────
def _ensure_tx_sheet():
    sh = _open_sheet()
    try:
        return sh.worksheet(TX_SHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=TX_SHEET, rows=1000, cols=8)
        ws.update([["Datum","Typ","Ticker","Antal","Pris (SEK)","Valuta","Total SEK","Kommentar"]],
                  value_input_option="RAW")
        return ws

def logga_transaktioner(rows: list):
    """
    rows: lista av [Typ ('KÖP'/'SÄLJ'), Ticker, Antal, Pris (SEK), Valuta, Total SEK, Kommentar]
    """
    ws = _ensure_tx_sheet()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = []
    for r in rows:
        out.append([now] + r)
    if out:
        ws.append_rows(out, value_input_option="RAW")

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
            for k in ["Aktuell kurs","Valuta","Bolagsnamn","Utdelning/år","Frekvens/år","Ex-Date","Källa"]:
                if k in vals and vals[k] not in (None,""):
                    base.loc[m,k] = vals[k]
            base = beräkna_allt(base)
            base = uppdatera_nästa_utd(base)
            progress.progress(int(i*100/N))
            time.sleep(1.0)  # respekt mot Yahoo
        st.session_state["working_df"] = base
        st.success("Massuppdatering klar (i minnet). Gå till 💾 Spara för att skriva till Sheets.")
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
        "Utdelning/år","Årlig utdelning (SEK)","Frekvens/år","Ex-Date","Nästa utbetalning (est)"
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
        per_share_local = _to_float(r["Utdelning/år"]) / max(1.0, _to_float(r.get("Frekvens/år",4)))
        qty = _to_float(r.get("Antal aktier",0.0))
        fx  = fx_for(r.get("Valuta","SEK"))
        per_payment_sek = per_share_local * fx * qty
        if per_payment_sek <= 0: continue
        for p in _gen(r.get("Ex-Date",""), r.get("Frekvens/år",4), r.get("Payment-lag (dagar)",30), months):
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

# ── Spara ─────────────────────────────────────────────────────────────────
def page_save(df: pd.DataFrame):
    st.subheader("💾 Spara nu till Google Sheets")
    preview = uppdatera_nästa_utd(beräkna_allt(säkerställ_kolumner(df)))
    st.write("Rader som sparas:", len(preview))
    st.dataframe(
        preview[["Ticker","Bolagsnamn","Valuta","Kategori","Antal aktier","GAV","Aktuell kurs","Utdelning/år","Kurs (SEK)","Årlig utdelning (SEK)"]],
        use_container_width=True
    )
    danger = st.checkbox("⚠️ Tillåt riskabel överskrivning (anti-wipe skyddar i normalfallet)", value=False)
    if st.button("✅ Bekräfta och spara"):
        if not danger:
            spara_data(preview)   # har anti-wipe inuti
        else:
            spara_data(preview)

# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    st.title("Relative Yield – utdelningsportfölj")

    # Initiera in-memory tabellen från Google Sheets en gång
    if "working_df" not in st.session_state:
        try:
            st.session_state["working_df"] = hamta_data()
        except Exception:
            st.session_state["working_df"] = säkerställ_kolumner(pd.DataFrame())

    # Kör autosnap var 5 min (skapar _Backup_-flikar och trimmar till 10 st)
    autosnap_if_due(300)

    # Sidebar (FX + snabb “uppdatera EN” + backup)
    sidebar_tools()

    # Meny
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

    # Bas-data (sanera innan varje vy)
    base = säkerställ_kolumner(st.session_state["working_df"]).copy()

    if page == "📦 Portföljöversikt":
        page_portfolio(base)

    elif page == "⚖️ Regler & mål":
        _ = load_settings()  # se till att Settings-fliken finns
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

    # Lägg tillbaka i sessionen efter ev. ändringar
    st.session_state["working_df"] = säkerställ_kolumner(base)


if __name__ == "__main__":
    main()
