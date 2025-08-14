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

def skapa_koppling_google_sheet():
    return _open_sheet()

# ── Rätta Google Sheet (formatering → TEXT) ───────────────────────────────
def ratta_sheet_format(sheet_name=SHEET_NAME):
    try:
        sh = _open_sheet()
        ws = sh.worksheet(sheet_name)
        sheet_id = ws._properties["sheetId"]
        body = {
            "requests": [
                {
                    "repeatCell": {
                        "range": {"sheetId": sheet_id},
                        "cell": {"userEnteredFormat": {"numberFormat": {"type": "TEXT"}}},
                        "fields": "userEnteredFormat.numberFormat"
                    }
                }
            ]
        }
        sh.batch_update(body)
        return True, "Formatering satt till TEXT för hela bladet."
    except Exception as e:
        return False, f"Kunde inte rätta format: {e}"

# ── Robust talparser (hindrar tid/%/datum) ────────────────────────────────
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
    "Antal aktier","GAV",  # GAV i lokal valuta
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

def _sanitize_for_sheets_as_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Skriv ALLT som TEXT till Sheets (för att undvika datum/%-korruption).
    Appen kastar till float vid läsning.
    """
    out = df.copy()
    out = out.fillna("")
    # konvertera allt till str (med punkt som decimal)
    def _fmt(v):
        if pd.isna(v): return ""
        try:
            f = float(_to_float(v))
            # behåll rimlig precision
            return f"{f:.10g}"
        except Exception:
            return str(v)
    for c in out.columns:
        out[c] = out[c].apply(_fmt)
    return out

def spara_data_safe(df: pd.DataFrame, max_retries: int = 3):
    ws = skapa_koppling()
    out = säkerställ_kolumner(df).copy()

    if out.empty or out["Ticker"].astype(str).str.strip().eq("").all():
        st.error("Sparning avbruten: tom data eller inga tickers.")
        return

    # Anti-wipe: jämför antal tickers
    try:
        current_rows = ws.get_all_records()
    except Exception:
        current_rows = []
    current_df = säkerställ_kolumner(pd.DataFrame(current_rows))
    old_n = int(current_df["Ticker"].astype(str).str.strip().ne("").sum()) if not current_df.empty else 0
    new_n = int(out["Ticker"].astype(str).str.strip().ne("").sum())

    if old_n > 0 and new_n < max(1, int(0.5*old_n)):
        st.error(f"Sparning stoppad: nya datasetet ({new_n} tickers) << nuvarande ({old_n}).")
        return

    # Försök rätta formatering innan skrivning
    try:
        ok, msg = ratta_sheet_format(SHEET_NAME)
        if ok: st.caption("Sheet-format: TEXT (säkrat).")
        else:  st.warning(msg)
    except Exception as e:
        st.warning(f"Format-försök misslyckades: {e}")

    # Backup före skrivning
    try:
        sh = _open_sheet()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_title = f"_Backup_{timestamp}"
        backup_ws = sh.add_worksheet(title=backup_title, rows=1, cols=max(1, len(out.columns)))
        cur_rows = ws.get_all_values()
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

    # Sanera: skriv allt som TEXT
    out_txt = _sanitize_for_sheets_as_text(out)
    header = [out_txt.columns.tolist()]
    body   = out_txt.values.tolist()

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

    st.success("✅ Sparning klar (säker och textformat).")

# Alias
spara_data = spara_data_safe

# ── Yahoo Finance-hämtning ────────────────────────────────────────────────
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

        div_year = 0.0
        freq = 0
        ex_date = ""
        try:
            divs = yf_t.dividends
            if divs is not None and not divs.empty:
                cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
                last12 = divs[divs.index >= cutoff]
                div_year = float(last12.tail(12).sum()) if not last12.empty else 0.0
                freq = int(last12.shape[0]) if not last12.empty else 0
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

# ── Sidebar: FX + backup + Rätta Sheet + uppdatera EN ────────────────────
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

    if st.sidebar.button("🧹 Rätta Google Sheet (TEXT-format)"):
        ok, msg = ratta_sheet_format(SHEET_NAME)
        (st.sidebar.success if ok else st.sidebar.error)(msg)

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

# ── Settings-sida (mål & regler) + översikter + EXPORT TrimCandidates ────
def page_settings(df: pd.DataFrame):
    st.subheader("⚖️ Regler & mål")

    gmax, cats = load_settings()
    present = sorted([c for c in df["Kategori"].dropna().astype(str).unique().tolist()])
    cats_view = {k: float(cats.get(k, 0.0)) for k in (present if present else cats.keys())}

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
        gmax, cats = gmax_new, new_cats

    st.markdown("---")

    d = uppdatera_nästa_utd(beräkna_allt(df).copy())
    if d.empty:
        st.info("Ingen data i portföljen ännu.")
        return

    st.markdown("### 🧭 Kategoriöversikt")
    T = float(d["Marknadsvärde (SEK)"].sum()) or 1.0
    cat_now = (d.groupby("Kategori", as_index=False)["Marknadsvärde (SEK)"].sum()
                 .rename(columns={"Marknadsvärde (SEK)":"MV"}))
    cat_now["Nu (%)"] = (100.0 * cat_now["MV"] / T).round(2)
    cat_now["Mål (%)"] = cat_now["Kategori"].apply(lambda c: float(cats.get(c, 0.0)))
    cat_now["Över/Under mål (pp)"] = (cat_now["Nu (%)"] - cat_now["Mål (%)"]).round(2)
    cat_now = cat_now.sort_values("Nu (%)", ascending=False)
    st.dataframe(cat_now[["Kategori","Nu (%)","Mål (%)","Över/Under mål (pp)"]], use_container_width=True)

    st.markdown("### 🧩 Bolagsöversikt (andelar)")
    d["Vikt (%)"] = d["Portföljandel (%)"].round(2)
    comp_view = d[["Ticker","Bolagsnamn","Kategori","Vikt (%)","Marknadsvärde (SEK)"]]\
                   .sort_values("Vikt (%)", ascending=False)\
                   .reset_index(drop=True)
    st.dataframe(comp_view, use_container_width=True)

    st.markdown("### ✂️ Trim-kandidater")
    over_name = d[d["Vikt (%)"] > float(gmax)][["Ticker","Bolagsnamn","Kategori","Vikt (%)"]].copy()
    over_name["Skäl"] = [f"Över {gmax:.1f}% bolagstak"] * len(over_name)
    over_cats = cat_now[cat_now["Över/Under mål (pp)"] > 0.0]["Kategori"].tolist()

    rows = []
    if over_cats:
        subset = d[d["Kategori"].isin(over_cats)].copy()
        subset = subset.sort_values(["Kategori","Vikt (%)"], ascending=[True, False])
        for _, r in subset.iterrows():
            rows.append({
                "Ticker": r["Ticker"],
                "Bolagsnamn": r["Bolagsnamn"],
                "Kategori": r["Kategori"],
                "Vikt (%)": round(float(r["Vikt (%)"]), 2),
                "Skäl": "Kategori övervikt"
            })
    over_cat_df = pd.DataFrame(rows)
    trim_df = pd.concat([over_name, over_cat_df], ignore_index=True).drop_duplicates(subset=["Ticker","Skäl"])

    if trim_df.empty:
        st.success("Inga tydliga trim-kandidater just nu.")
    else:
        st.dataframe(trim_df.sort_values(["Skäl","Vikt (%)"], ascending=[True, False]), use_container_width=True)
        if st.button("📤 Exportera trim-kandidater till Google Sheets"):
            try:
                sh = skapa_koppling_google_sheet()
                try:
                    ws = sh.worksheet("TrimCandidates")
                except gspread.WorksheetNotFound:
                    ws = sh.add_worksheet(title="TrimCandidates", rows="500", cols="10")
                ws.clear()
                ws.update([trim_df.columns.tolist()] + trim_df.astype(str).values.tolist(), value_input_option="RAW")
                st.success("Trim-kandidater exporterade till fliken 'TrimCandidates'.")
            except Exception as e:
                st.error(f"Kunde inte exportera: {e}")

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

# ── Köpförslag & plan (≈500 kr lotter) ────────────────────────────────────
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

    # Visa ev. exporterade trim-kandidater överst
    try:
        sh = skapa_koppling_google_sheet()
        ws = sh.worksheet("TrimCandidates")
        trim_data = pd.DataFrame(ws.get_all_records())
        if not trim_data.empty:
            st.markdown("### ✂️ Trimma/sälj-kandidater (från Regler & mål)")
            st.dataframe(trim_data, use_container_width=True)
    except Exception:
        pass

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

# ── Spara ─────────────────────────────────────────────────────────────────
def page_save(df: pd.DataFrame):
    st.subheader("💾 Spara nu till Google Sheets")
    preview = uppdatera_nästa_utd(beräkna_allt(säkerställ_kolumner(df)))
    st.write("Rader som sparas:", len(preview))
    st.dataframe(
        preview[["Ticker","Bolagsnamn","Valuta","Kategori","Antal aktier","GAV","Aktuell kurs","Utdelning/år","Kurs (SEK)","Årlig utdelning (SEK)"]],
        use_container_width=True
    )
    danger = st.checkbox("⚠️ Tillåt riskabel överskrivning (anti-wipe kvarstår ändå)", value=False)
    if st.button("✅ Bekräfta och spara"):
        spara_data(preview)

# ── Transaktionslogg (”Spara plan till Transaktioner”) ────────────────────
def _ensure_tx_sheet():
    sh = _open_sheet()
    try:
        return sh.worksheet(TX_SHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=TX_SHEET, rows=1000, cols=10)
        ws.update([["Tid","Typ","Ticker","Antal","Pris (SEK)","Valuta","Belopp (SEK)","Courtage","FX-avg","Kommentar"]],
                  value_input_option="RAW")
        return ws

def save_sell_plan_to_log(rows: list):
    ws = _ensure_tx_sheet()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = []
    for r in rows:
        out.append([now,"SÄLJ-PLAN", r["Ticker"], r["Antal"], r["Pris (SEK)"], r["Valuta"],
                    r["Belopp (SEK)"], r["Courtage"], r["FX-avg"], r.get("Kommentar","")])
    if out:
        ws.append_rows(out, value_input_option="RAW")

# ── Trim/Sälj-simulator ───────────────────────────────────────────────────
def _sell_needed_name_cap(Vi, T, p, m_pct):
    """n >= (Vi - m*T) / (p*(1-m))"""
    if p <= 0: return 0
    m = m_pct/100.0
    numer = Vi - m*T
    denom = p*(1.0 - m)
    if numer <= 0 or denom <= 0: return 0
    return int(math.ceil(numer/denom))

def _sell_needed_cat_cap(C, T, p, M_pct):
    """n >= (C - M*T) / (p*(1-M))"""
    if p <= 0: return 0
    M = M_pct/100.0
    numer = C - M*T
    denom = p*(1.0 - M)
    if numer <= 0 or denom <= 0: return 0
    return int(math.ceil(numer/denom))

def page_trim_simulator(df: pd.DataFrame):
    st.subheader("✂️ Sälj/Trim-simulator (utan att röra innehaven)")

    d = uppdatera_nästa_utd(beräkna_allt(df).copy())
    if d.empty:
        st.info("Ingen data i portföljen ännu.")
        return

    gmax, cats = load_settings()
    T = float(d["Marknadsvärde (SEK)"].sum()) or 1.0
    cat_now = d.groupby("Kategori", as_index=False)["Marknadsvärde (SEK)"].sum().rename(columns={"Marknadsvärde (SEK)":"C"})
    cat_map = dict(zip(cat_now["Kategori"], cat_now["C"]))

    # Kandidater: sådana som bryter mot bolagstak eller ligger i överviktade kategorier
    over_cat_list = []
    for k, C in cat_map.items():
        M = float(cats.get(k, 100.0))
        if (100.0*C/T) > M + 1e-9:
            over_cat_list.append(k)

    rows = []
    for _, r in d.iterrows():
        tkr = r["Ticker"]; cat = r["Kategori"]; val = str(r["Valuta"]).upper()
        p = float(_to_float(r["Kurs (SEK)"]))
        qty = int(_to_float(r["Antal aktier"]))
        Vi = float(_to_float(r["Marknadsvärde (SEK)"]))
        C  = float(cat_map.get(cat, 0.0))
        foreign = (val != "SEK")

        n_name = _sell_needed_name_cap(Vi, T, p, gmax)
        n_cat  = _sell_needed_cat_cap(C,  T, p, float(cats.get(cat, 100.0)))
        n_req  = max(n_name, n_cat) if (cat in over_cat_list or n_name>0) else 0
        if n_req <= 0: 
            continue
        n = min(n_req, qty)
        if n <= 0:
            continue

        gross = round(p * n, 2)
        c_fee, fx_fee, tot_fee = calc_fees(gross, foreign)
        net = round(gross - tot_fee, 2)
        rows.append({
            "Ticker": tkr, "Kategori": cat, "Valuta": val, "Pris (SEK)": round(p,2),
            "Antal": int(n), "Belopp (SEK)": gross, "Courtage": c_fee, "FX-avg": fx_fee,
            "Nettolikvid (SEK)": net,
            "Skäl": ("Bolag>tak" if n_name>0 else "") + (" & " if (n_name>0 and n_cat>0) else "") + ("Kat>mål" if n_cat>0 else "")
        })

    if not rows:
        st.success("Inga sälj/trim behövs enligt reglerna just nu.")
        return

    sug = pd.DataFrame(rows).sort_values(["Skäl","Belopp (SEK)"], ascending=[True, False])
    st.dataframe(sug, use_container_width=True)

    # Plan i minnet
    if "sell_plan" not in st.session_state:
        st.session_state["sell_plan"] = []

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("➕ Lägg ALLA i säljplan (minne)"):
            for _, r in sug.iterrows():
                st.session_state["sell_plan"].append({
                    "Ticker": r["Ticker"], "Kategori": r["Kategori"], "Valuta": r["Valuta"],
                    "Antal": int(r["Antal"]), "Pris (SEK)": float(r["Pris (SEK)"]),
                    "Belopp (SEK)": float(r["Belopp (SEK)"]), "Courtage": float(r["Courtage"]),
                    "FX-avg": float(r["FX-avg"]), "Kommentar": r["Skäl"]
                })
            st.success("Säljplan uppdaterad i minnet.")
    with colB:
        if st.button("🧹 Töm säljplan (minne)"):
            st.session_state["sell_plan"] = []
            st.success("Säljplan tömd.")
    with colC:
        if st.button("📝 Spara säljplan till Transaktioner"):
            if not st.session_state["sell_plan"]:
                st.warning("Planen är tom.")
            else:
                save_sell_plan_to_log(st.session_state["sell_plan"])
                st.success("Säljplan sparad till fliken 'Transaktioner' (utan att ändra antal).")

    if st.session_state["sell_plan"]:
        st.markdown("### Säljplan i minnet")
        st.dataframe(pd.DataFrame(st.session_state["sell_plan"]), use_container_width=True)

# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    st.title("Relative Yield – utdelningsportfölj")

    # Initiera in-memory tabellen från Google Sheets en gång
    if "working_df" not in st.session_state:
        try:
            st.session_state["working_df"] = hamta_data()
        except Exception:
            st.session_state["working_df"] = säkerställ_kolumner(pd.DataFrame())

    # Kör autosnap var 5 min (säkerhetskopia av huvudbladet)
    autosnap_if_due(300)

    # Sidebar (FX + snabb “uppdatera EN” + backup + rätta sheet)
    sidebar_tools()

    page = st.sidebar.radio(
        "Meny",
        [
            "📦 Portföljöversikt",
            "⚖️ Regler & mål",
            "➕ Lägg till / ✏ Uppdatera bolag",
            "⏩ Massuppdatera alla",
            "🎯 Köpförslag & plan",
            "✂️ Sälj/Trim-simulator",
            "📅 Utdelningskalender",
            "💾 Spara"
        ],
        index=0
    )

    base = säkerställ_kolumner(st.session_state["working_df"]).copy()

    if page == "📦 Portföljöversikt":
        page_portfolio(base)
    elif page == "⚖️ Regler & mål":
        _ = load_settings()  # se till att Settings-bladet finns
        page_settings(base)
    elif page == "➕ Lägg till / ✏ Uppdatera bolag":
        base = page_add_or_update(base)
    elif page == "⏩ Massuppdatera alla":
        base = page_mass_update(base)
    elif page == "🎯 Köpförslag & plan":
        page_buy_planner(base)
    elif page == "✂️ Sälj/Trim-simulator":
        page_trim_simulator(base)
    elif page == "📅 Utdelningskalender":
        page_calendar(base)
    elif page == "💾 Spara":
        page_save(base)

    st.session_state["working_df"] = säkerställ_kolumner(base)

if __name__ == "__main__":
    main()
