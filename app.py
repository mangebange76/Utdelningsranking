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
        ws = sh.add_worksheet(title=SHEET_NAME, rows=1000, cols=50)
        ws.update([["Ticker"]], value_input_option="RAW")
        return ws

# ── Robust talparser (hindrar “tid” osv) ───────────────────────────────────
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

# ── FX defaults ────────────────────────────────────────────────────────────
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
    "Antal aktier","GAV",                       # GAV i lokal valuta
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
            d[c] = ""
    d["Ticker"]   = d["Ticker"].astype(str).str.strip().str.upper()
    d["Valuta"]   = d["Valuta"].astype(str).str.strip().str.upper()
    d["Kategori"] = d["Kategori"].astype(str).replace({"": "QUALITY"})
    # normalisera numeriska fält till floats
    num_cols = ["Aktuell kurs","Utdelning/år","Utdelning/år (manuell)","Frekvens/år","Payment-lag (dagar)",
                "Antal aktier","GAV","Kurs (SEK)","Marknadsvärde (SEK)","Portföljandel (%)",
                "Insatt (SEK)","Årlig utdelning (SEK)","Direktavkastning (%)"]
    for c in num_cols:
        d[c] = d[c].apply(_to_float)
    if "Lås utdelning" in d.columns:
        d["Lås utdelning"] = d["Lås utdelning"].apply(lambda x: bool(x) if pd.notna(x) else False)
    else:
        d["Lås utdelning"] = False
    if "Utdelningskälla" not in d.columns:
        d["Utdelningskälla"] = "Yahoo"
    return d[COLUMNS].copy()

# ── Spara/Läsa till Sheets (SÄKERT läge) ──────────────────────────────────
def spara_data_safe(df: pd.DataFrame, max_retries: int = 3):
    ws = skapa_koppling()
    out = säkerställ_kolumner(df).copy()

    # 1) Skydda mot tom sparning
    if out.empty or out["Ticker"].astype(str).str.strip().eq("").all():
        st.error("Sparning avbruten: tom data eller inga tickers.")
        return

    # 2) Backup av nuvarande blad innan skrivning
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
    except Exception as e:
        st.warning(f"Kunde inte skapa backupflik (fortsätter ändå): {e}")

    # 3) Säkerställ numeriska kolumner
    numeric_cols = ["Aktuell kurs","Utdelning/år","Utdelning/år (manuell)","Frekvens/år","Payment-lag (dagar)",
                    "Antal aktier","GAV","Kurs (SEK)","Marknadsvärde (SEK)","Portföljandel (%)",
                    "Insatt (SEK)","Årlig utdelning (SEK)","Direktavkastning (%)"]
    for c in numeric_cols:
        out[c] = out[c].apply(_to_float)

    header = [out.columns.tolist()]
    body   = out.values.tolist()

    # 4) Skriv FÖRST (utan clear) med retry/backoff vid 429
    for attempt in range(1, max_retries+1):
        try:
            ws.update(header + body, value_input_option="RAW")
            break
        except Exception as e:
            msg = str(e)
            if ("Quota exceeded" in msg or "429" in msg) and attempt < max_retries:
                time.sleep(2 * attempt)  # enkel backoff
                continue
            st.error(f"Sparfel (avbryter, ingen clear gjord): {e}")
            return

    # 5) Rensa bara överflödiga rader (om ny data är kortare)
    try:
        old = ws.get_all_values()
        new_rows = len(header + body)
        if len(old) > new_rows and old:
            start_row = new_rows + 1
            end_row   = len(old)
            start_col = 1
            end_col   = len(old[0])
            ws.batch_clear([
                f"{gspread.utils.rowcol_to_a1(start_row, start_col)}:"
                f"{gspread.utils.rowcol_to_a1(end_row, end_col)}"
            ])
    except Exception as e:
        st.warning(f"Kunde inte rensa överflödiga rader (inte kritiskt): {e}")

    # 6) Sätt tal- och datumformat efter lyckad skrivning
    try:
        nrows = max(1, len(out)+1)
        def col_idx(name):
            try: return list(out.columns).index(name)
            except ValueError: return None
        requests = []
        for name in numeric_cols:
            idx = col_idx(name)
            if idx is None: continue
            requests.append({
                "repeatCell": {
                    "range": {"sheetId": ws.id, "startRowIndex": 1, "endRowIndex": nrows,
                              "startColumnIndex": idx, "endColumnIndex": idx+1},
                    "cell": {"userEnteredFormat": {"numberFormat": {"type":"NUMBER","pattern":"0.00########"}}},
                    "fields": "userEnteredFormat.numberFormat"
                }
            })
        for date_col in ["Ex-Date","Nästa utbetalning (est)","Senaste uppdatering"]:
            idx = col_idx(date_col)
            if idx is None: continue
            requests.append({
                "repeatCell": {
                    "range": {"sheetId": ws.id, "startRowIndex": 1, "endRowIndex": nrows,
                              "startColumnIndex": idx, "endColumnIndex": idx+1},
                    "cell": {"userEnteredFormat": {"numberFormat": {"type":"DATE","pattern":"yyyy-mm-dd"}}},
                    "fields": "userEnteredFormat.numberFormat"
                }
            })
        if requests:
            ws.spreadsheet.batch_update({"requests": requests})
    except Exception as e:
        st.warning(f"Kunde inte sätta format: {e}")

    st.success("✅ Sparning klar (säkert läge).")

# 🔄 Alias – alla befintliga spara_data-anrop funkar oförändrat
spara_data = spara_data_safe

def hamta_data():
    try:
        ws = skapa_koppling()
        rows = ws.get_all_records()
        df = pd.DataFrame(rows)
        return säkerställ_kolumner(df)
    except Exception as e:
        st.warning(f"Kunde inte läsa Google Sheet: {e}")
        return säkerställ_kolumner(pd.DataFrame())

# ── Settings (GLOBAL_MAX + kategori‑mål) ──────────────────────────────────
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

# ── Yahoo Finance-hämtning ────────────────────────────────────────────────
def fetch_yahoo(ticker: str) -> dict:
    try:
        _throttle(1.0)
        t = ticker.strip().upper()
        yf_ticker = yf.Ticker(t)
        info = yf_ticker.info if hasattr(yf_ticker, "info") else {}
        hist_divs = getattr(yf_ticker, "dividends", pd.Series(dtype=float))

        ex_date = None
        if hist_divs is not None and not hist_divs.empty:
            ex_date = hist_divs.index[-1].strftime("%Y-%m-%d")

        # Summera senaste 12 mån utdelningar
        div_year = 0.0
        freq = 0
        if hist_divs is not None and not hist_divs.empty:
            last12 = hist_divs[hist_divs.index >= (pd.Timestamp.utcnow() - pd.Timedelta(days=365))]
            div_year = float((last12.tail(12)).sum())
            freq = int(last12.shape[0])

        return {
            "Aktuell kurs": _to_float(info.get("currentPrice", 0)),
            "Valuta": info.get("currency", "USD"),
            "Bolagsnamn": info.get("shortName", t),
            "Utdelning/år": div_year,
            "Frekvens/år": freq,
            "Ex-Date": ex_date,
            "Källa": "Yahoo"
        }
    except Exception as e:
        st.warning(f"Yahoo-fel {ticker}: {e}")
        return {}

# ── Beräkningar ───────────────────────────────────────────────────────────
def beräkna_allt(df: pd.DataFrame) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()

    # Val av utdelningskälla
    d["Utdelning/år_eff"] = d.apply(
        lambda r: _to_float(r["Utdelning/år (manuell)"]) if bool(r.get("Lås utdelning")) and _to_float(r["Utdelning/år (manuell)"])>0
        else _to_float(r["Utdelning/år"]), axis=1
    )
    d["Utdelningskälla"] = d.apply(
        lambda r: "Manuell 🔒" if bool(r.get("Lås utdelning")) and _to_float(r["Utdelning/år (manuell)"])>0 else "Yahoo", axis=1
    )

    # Kurs i SEK
    d["Kurs (SEK)"] = (d["Aktuell kurs"].apply(_to_float) * d["Valuta"].map(fx_for)).round(6)

    # Marknadsvärde och andel
    d["Marknadsvärde (SEK)"] = (d["Antal aktier"].apply(_to_float) * d["Kurs (SEK)"]).round(2)
    tot_mv = float(d["Marknadsvärde (SEK)"].sum()) or 1.0
    d["Portföljandel (%)"] = (100.0 * d["Marknadsvärde (SEK)"] / tot_mv).round(2)

    # Insatt i SEK (GAV i lokal * qty * FX)
    d["Insatt (SEK)"] = (d["Antal aktier"].apply(_to_float) * d["GAV"].apply(_to_float) * d["Valuta"].map(fx_for)).round(2)

    # Utdelning i SEK & DA
    d["Årlig utdelning (SEK)"] = (d["Antal aktier"].apply(_to_float) * d["Utdelning/år_eff"].apply(_to_float) * d["Valuta"].map(fx_for)).round(2)
    d["Direktavkastning (%)"] = d.apply(lambda r: (100.0 * _to_float(r["Utdelning/år_eff"]) / _to_float(r["Aktuell kurs"])) if _to_float(r["Aktuell kurs"])>0 else 0.0, axis=1).round(2)

    return d

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
        # approximera nästa ex-date framåt
        step_days = max(1, int(round(365.0 / max(freq,1))))
        today_d = date.today()
        while exdate < today_d:
            exdate = exdate + timedelta(days=step_days)
        pay_date = exdate + timedelta(days=pay_lag)
        return pay_date.strftime("%Y-%m-%d")
    except Exception:
        return ""

def uppdatera_nästa_utd(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Nästa utbetalning (est)"] = d.apply(nästa_utd_datum, axis=1)
    return d

# ── Transaktionslogg ──────────────────────────────────────────────────────
def _ensure_tx_sheet():
    sh = _open_sheet()
    try:
        return sh.worksheet(TX_SHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=TX_SHEET, rows=1000, cols=8)
        ws.update([["Datum","Ticker","Bolagsnamn","Antal","Kurs","Valuta","Total SEK","Kommentar"]],
                  value_input_option="RAW")
        return ws

def logga_transaktioner(rows: list):
    ws = _ensure_tx_sheet()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for r in rows:
        ws.append_row([now] + r, value_input_option="RAW")

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

# ── Sidebar: Växelkurser + snabb EN‑uppdatering ───────────────────────────
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
            spara_data(base)
            st.session_state["working_df"] = base
            st.sidebar.success(f"{one} uppdaterad.")

# ── Settings‑sida (max per bolag & kategori‑mål) ─────────────────────────
def page_settings(df: pd.DataFrame):
    st.subheader("⚖️ Regler & mål")
    gmax, cats = load_settings()

    # visa bara kategorier som finns i databasen
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

# ── Lägg till / uppdatera bolag ───────────────────────────────────────────
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
        c1,c2,c3 = st.columns(3)
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
                    spara_data(base)
                    st.success(f"{tkr} tillagt och sparat.")
        with c3:
            if st.button("💾 Spara hela tabellen nu"):
                base = beräkna_allt(base)
                base = uppdatera_nästa_utd(base)
                spara_data(base)
                st.success("Sparat.")
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
                spara_data(base)
                st.success(f"{tkr} uppdaterad och sparad.")
        with c2:
            if st.button("✏ Uppdatera fält och spara"):
                m = base["Ticker"]==val
                base.loc[m,"Ticker"] = tkr
                base.loc[m,"Antal aktier"] = antal
                base.loc[m,"GAV"] = gav
                base.loc[m,"Kategori"] = kategori
                base = beräkna_allt(base)
                base = uppdatera_nästa_utd(base)
                spara_data(base)
                st.success(f"{tkr} uppdaterad.")
        with c3:
            if st.button("🗑 Ta bort bolag"):
                base = base[base["Ticker"]!=val].reset_index(drop=True)
                base = beräkna_allt(base)
                spara_data(base)
                st.success(f"{val} borttagen.")

    return base

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
        spara_data(base)
        st.success("Massuppdatering klar!")
    return base

# ── Köpsimulator (≈500 kr‑lotter) ─────────────────────────────────────────
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

    # poäng: DA + undervikt vs gmax + nära utdelning
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

    # init totals
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
            # begränsningar
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
            # accept
            picked = {
                "Ticker": tkr, "Kategori": cat, "Antal": int(n),
                "Pris (SEK)": round(price,2), "Kostnad (SEK)": round(total_cost,2),
                "Courtage": c_fee, "FX-avg": fx_fee, "Poäng": round(float(r["Poäng"]),1),
                "Kommentar": f"Lot ~{int(lot)} kr, under {gmax_ui:.0f}% & kat≤{cat_limits.get(cat,100):.0f}%"
            }
            # uppdatera simulering
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

    st.session_state["buy_plan_rows"] = plan.to_dict("records")
    if st.button("📝 Spara planen som transaktionslogg (utan att ändra innehav)"):
        rows = []
        for r in st.session_state["buy_plan_rows"]:
            rows.append([r["Ticker"], "", r["Antal"], r["Pris (SEK)"], "SEK", r["Kostnad (SEK)"], "plan"])
        logga_transaktioner(rows)
        st.success("Planen sparades i fliken Transaktioner.")

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
    if st.button("✅ Bekräfta och spara"):
        spara_data(preview)
        st.success("Sparat.")

# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    st.title("Relative Yield – utdelningsportfölj")

    # Initiera in‑memory tabellen från Google Sheets en gång
    if "working_df" not in st.session_state:
        try:
            st.session_state["working_df"] = hamta_data()
        except Exception:
            st.session_state["working_df"] = säkerställ_kolumner(pd.DataFrame())

    # Sidebar (FX + snabb “uppdatera EN”)
    sidebar_tools()

    # Välj sida
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

    # Arbeta på en lokal kopia, spara tillbaka i slutet av anropet
    base = säkerställ_kolumner(st.session_state["working_df"]).copy()

    if page == "📦 Portföljöversikt":
        page_portfolio(base)

    elif page == "⚖️ Regler & mål":
        # Läs nuvarande settings så UI visar rätt värden
        _ = load_settings()
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

    # Skriv tillbaka in‑memory efter varje sidkörning
    st.session_state["working_df"] = säkerställ_kolumner(base)

if __name__ == "__main__":
    main()
