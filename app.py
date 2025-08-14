import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time, math
from datetime import datetime, timedelta, date
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Relative Yield – utdelningsportfölj", layout="wide")

# ── Google Sheets konfig ───────────────────────────────────────────────────
SHEET_URL  = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"          # databasblad
SET_SHEET  = "Settings"       # regler/mål
TX_SHEET   = "Transaktioner"  # (reserverat, används ej här)

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def _open_sheet():
    return client.open_by_url(SHEET_URL)

def skapa_koppling():
    sh = _open_sheet()
    try:
        return sh.worksheet(SHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=SHEET_NAME, rows=1000, cols=50)
        ws.update([["Ticker"]], value_input_option="RAW")
        return ws

# ── Hjälpare: säkra tal & kolumner ────────────────────────────────────────
def _to_float(val):
    try:
        return float(str(val).replace(",", ".").replace(" ", "").strip())
    except Exception:
        return 0.0

COLUMNS = [
    "Ticker","Bolagsnamn","Valuta","Kategori",
    "Antal aktier","GAV","Aktuell kurs","Kurs (SEK)",
    "Marknadsvärde (SEK)","Insatt (SEK)","Årlig utdelning (SEK)",
    "Direktavkastning (%)","Portföljandel (%)",
    "Utdelning/år","Frekvens/år","Payment-lag (dagar)","Ex-Date","Nästa utbetalning (est)",
    "Källa"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        d = pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in d.columns:
            d[c] = ""
    # Normalisera nyckelfält
    d["Ticker"]   = d["Ticker"].astype(str).str.strip().str.upper()
    d["Valuta"]   = d["Valuta"].astype(str).str.strip().str.upper().replace({"": "SEK"})
    d["Kategori"] = d["Kategori"].astype(str).str.strip().replace({"": "QUALITY"})
    # Säkerställ numeriska fält är flyttal i minnet
    for c in ["Aktuell kurs","Antal aktier","GAV","Kurs (SEK)","Marknadsvärde (SEK)",
              "Insatt (SEK)","Årlig utdelning (SEK)","Direktavkastning (%)","Portföljandel (%)",
              "Utdelning/år","Frekvens/år","Payment-lag (dagar)"]:
        d[c] = d[c].apply(_to_float)
    return d[COLUMNS].copy()

# ── Läs/spara (med säker sanering) ────────────────────────────────────────
def hamta_data() -> pd.DataFrame:
    try:
        ws = skapa_koppling()
        rows = ws.get_all_records()
        return säkerställ_kolumner(pd.DataFrame(rows))
    except Exception as e:
        st.warning(f"Kunde inte läsa Google Sheet: {e}")
        return säkerställ_kolumner(pd.DataFrame())

def _sanitize_for_sheets_as_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Skriv data som text så Google inte gör om 125,5 till datum etc.
    Numeriska kolumner formateras som talsträngar med punkt.
    """
    numeric_cols = {
        "Aktuell kurs","Antal aktier","GAV","Kurs (SEK)","Marknadsvärde (SEK)",
        "Insatt (SEK)","Årlig utdelning (SEK)","Direktavkastning (%)","Portföljandel (%)",
        "Utdelning/år","Frekvens/år","Payment-lag (dagar)"
    }
    out = df.copy().fillna("")
    def _fmt_num(v):
        try:
            f = float(_to_float(v))
            return f"{f:.10g}"   # stabil sträng
        except Exception:
            return "0"
    for c in out.columns:
        if c in numeric_cols:
            out[c] = out[c].apply(_fmt_num)
        else:
            out[c] = out[c].apply(lambda x: "" if pd.isna(x) else str(x).strip())
    return out

def spara_data(df: pd.DataFrame):
    """Säkert skrivläge: snapshot-backup, sedan skrivning som RAW."""
    ws = skapa_koppling()
    sh = _open_sheet()
    # autosnap (dup av nuvarande blad innan skrivning)
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sh.duplicate_sheet(ws.id, new_sheet_name=f"_Backup_{timestamp}")
    except Exception:
        pass
    # skriv
    ws.clear()
    safe = _sanitize_for_sheets_as_text(säkerställ_kolumner(df))
    ws.update([safe.columns.tolist()] + safe.values.tolist(), value_input_option="RAW")

# ── Autosnap även tidsstyrt ───────────────────────────────────────────────
def _list_backup_titles(sh):
    try:
        return sorted([w.title for w in sh.worksheets() if w.title.startswith("_Backup_")])
    except Exception:
        return []

def autosnap_now():
    """Skapa backupflik av Blad1."""
    try:
        sh = _open_sheet()
        ws = skapa_koppling()
        title = f"_Backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sh.duplicate_sheet(ws.id, new_sheet_name=title)
        # rensa gamla (behåll 10)
        titles = _list_backup_titles(sh)
        if len(titles) > 10:
            for t in titles[:-10]:
                try:
                    sh.del_worksheet(sh.worksheet(t))
                except Exception:
                    pass
        st.sidebar.success(f"Autosnap skapad: {title}")
    except Exception as e:
        st.sidebar.warning(f"Autosnap misslyckades: {e}")

def autosnap_if_due(interval_sec=300):
    last = st.session_state.get("_autosnap_last_ts")
    now  = time.time()
    if (last is None) or (now - last >= interval_sec):
        autosnap_now()
        st.session_state["_autosnap_last_ts"] = now

# ── Settings-blad (GLOBAL_MAX & kategori-mål) ─────────────────────────────
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
        for k, v in DEFAULT_CAT_TARGETS.items():
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
        if key == "GLOBAL_MAX_NAME" and val > 0:
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

# ── Reparera in-memory (motverka knäppa värden) ───────────────────────────
NUM_COLS = [
    "Aktuell kurs","Antal aktier","GAV","Kurs (SEK)","Marknadsvärde (SEK)",
    "Insatt (SEK)","Årlig utdelning (SEK)","Direktavkastning (%)","Portföljandel (%)",
    "Utdelning/år","Frekvens/år","Payment-lag (dagar)"
]

def repair_in_memory(df: pd.DataFrame) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()
    d["Valuta"]   = d["Valuta"].astype(str).str.upper().str.strip().replace({"": "SEK"})
    d["Kategori"] = d["Kategori"].astype(str).str.strip().replace({"", "nan": "QUALITY"})
    for c in NUM_COLS:
        if c in d.columns:
            d[c] = d[c].apply(_to_float)
    d["Ticker"]     = d["Ticker"].astype(str).str.strip().str.upper()
    d["Bolagsnamn"] = d["Bolagsnamn"].astype(str).str.strip()
    return d

# ── FX-kurser (SEK) ───────────────────────────────────────────────────────
DEF_FX = {"USD": 9.60, "EUR": 11.10, "CAD": 6.95, "NOK": 0.94}
for k, v in DEF_FX.items():
    st.session_state.setdefault(f"{k}SEK", v)

def fx_for(cur: str) -> float:
    c = (cur or "SEK").upper()
    if c == "SEK":
        return 1.0
    return float(st.session_state.get(f"{c}SEK", DEF_FX.get(c, 1.0)))

# ── Yahoo-hämtare (robust) ────────────────────────────────────────────────
def fetch_yahoo(ticker: str) -> dict:
    t = (ticker or "").strip().upper()
    if not t:
        return {}
    try:
        y = yf.Ticker(t)

        # Pris
        price = None
        try:
            fi = getattr(y, "fast_info", {}) or {}
            price = fi.get("last_price")
        except Exception:
            price = None
        if price is None:
            try:
                info = y.get_info() or {}
            except Exception:
                info = {}
            price = info.get("regularMarketPrice") or info.get("currentPrice")
        if price is None:
            hist = y.history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])

        # Valuta & namn
        try:
            info2 = y.get_info() or {}
        except Exception:
            info2 = {}
        currency = (info2.get("currency") or "SEK").upper()
        name = info2.get("shortName") or info2.get("longName") or t

        # Utdelningar – summera 12 mån, räkna antal
        div_year = 0.0
        freq = 0
        ex_date = ""
        try:
            divs = y.dividends
            if divs is not None and not divs.empty:
                cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
                last12 = divs[divs.index >= cutoff]
                div_year = float(last12.sum()) if not last12.empty else 0.0
                freq = int(last12.shape[0]) if not last12.empty else 0
                ex_date = pd.to_datetime(divs.index.max()).strftime("%Y-%m-%d")
        except Exception:
            pass

        return {
            "Aktuell kurs": _to_float(price),
            "Valuta": currency,
            "Bolagsnamn": name,
            "Utdelning/år": _to_float(div_year),
            "Frekvens/år": int(freq),
            "Ex-Date": ex_date,
            "Källa": "Yahoo"
        }
    except Exception as e:
        st.warning(f"Yahoo-fel {ticker}: {e}")
        return {}

# ── Beräkningar (i SEK) ───────────────────────────────────────────────────
def beräkna_allt(df: pd.DataFrame) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()
    price = d["Aktuell kurs"].apply(_to_float)
    qty   = d["Antal aktier"].apply(_to_float)
    gav   = d["GAV"].apply(_to_float)
    fx    = d["Valuta"].apply(fx_for)

    d["Kurs (SEK)"]          = (price * fx).round(6)
    d["Marknadsvärde (SEK)"] = (qty * d["Kurs (SEK)"]).round(2)
    d["Insatt (SEK)"]        = (qty * gav * fx).round(2)

    if "Utdelning/år" in d.columns:
        d["Årlig utdelning (SEK)"] = (qty * d["Utdelning/år"].apply(_to_float) * fx).round(2)
    else:
        d["Årlig utdelning (SEK)"] = 0.0

    d["Direktavkastning (%)"] = (
        (d["Utdelning/år"].apply(_to_float) / price.replace(0, pd.NA)) * 100.0
    ).fillna(0.0).round(2)

    tot = float(d["Marknadsvärde (SEK)"].sum()) or 1.0
    d["Portföljandel (%)"] = (100.0 * d["Marknadsvärde (SEK)"] / tot).round(2)
    return d

# ── Sidomeny: FX + backup + reparera ──────────────────────────────────────
def restore_latest_backup():
    try:
        sh = _open_sheet()
        backups = _list_backup_titles(sh)
        if not backups:
            st.sidebar.warning("Hittar ingen backupflik.")
            return False
        latest = backups[-1]
        ws_src = sh.worksheet(latest)
        ws_dst = sh.worksheet(SHEET_NAME)
        values = ws_src.get_all_values()
        if not values:
            st.sidebar.warning("Backupen är tom.")
            return False
        ws_dst.clear()
        ws_dst.update(values, value_input_option="RAW")
        st.sidebar.success(f"Återställde från {latest}")
        return True
    except Exception as e:
        st.sidebar.error(f"Återställning misslyckades: {e}")
        return False

def sidebar_tools():
    st.sidebar.header("⚙️ Verktyg")

    with st.sidebar.expander("Växelkurser (SEK)"):
        for ccy in ["USD","EUR","CAD","NOK"]:
            key = f"{ccy}SEK"
            st.session_state[key] = st.number_input(
                f"{ccy}/SEK", value=float(st.session_state[key]), step=0.01, format="%.4f"
            )

    st.sidebar.markdown("---")
    if st.sidebar.button("📸 Ta backup nu"):
        autosnap_now()

    if st.sidebar.button("♻️ Återställ från senaste backup"):
        if restore_latest_backup():
            st.session_state["working_df"] = repair_in_memory(hamta_data())
            st.sidebar.success("Läste in återställd data.")

    if st.sidebar.button("🧹 Reparera värden i minnet (utan att spara)"):
        base = st.session_state.get("working_df", pd.DataFrame())
        st.session_state["working_df"] = repair_in_memory(base)
        st.sidebar.success("Datan i minnet har reparerats.")

# ── Nästa utdelningsdatum (est) ───────────────────────────────────────────
def _safe_int(x, d=0):
    try:
        return int(_to_float(x))
    except Exception:
        return d

def nästa_utd_datum(row):
    try:
        f = _safe_int(row.get("Frekvens/år", 0))
        if f <= 0:
            return ""
        ex_str = str(row.get("Ex-Date", "")).strip()
        if not ex_str or ex_str.lower() == "nan":
            return ""
        ex = datetime.strptime(ex_str, "%Y-%m-%d").date()
        lag = _safe_int(row.get("Payment-lag (dagar)", 30), 30)
        step = max(1, int(round(365.0 / max(1, f))))
        today = date.today()
        while ex < today:
            ex = ex + timedelta(days=step)
        pay = ex + timedelta(days=lag)
        return pay.strftime("%Y-%m-%d")
    except Exception:
        return ""

def uppdatera_nästa_utd(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Nästa utbetalning (est)"] = d.apply(nästa_utd_datum, axis=1)
    return d

# ── Sida: Portföljöversikt ────────────────────────────────────────────────
def page_portfolio(df: pd.DataFrame):
    st.subheader("📦 Portföljöversikt")
    d = uppdatera_nästa_utd(beräkna_allt(df))
    if d.empty:
        st.info("Lägg till minst ett bolag.")
        return

    tot_mv  = float(d["Marknadsvärde (SEK)"].sum())
    tot_ins = float(d["Insatt (SEK)"].sum())
    tot_div = float(d["Årlig utdelning (SEK)"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Portföljvärde (SEK)", f"{tot_mv:,.0f}".replace(",", " "))
    c2.metric("Insatt (SEK)", f"{tot_ins:,.0f}".replace(",", " "))
    c3.metric("Årlig utdelning (SEK)", f"{tot_div:,.0f}".replace(",", " "))

    show_cols = [
        "Ticker","Bolagsnamn","Valuta","Kategori","Antal aktier","GAV",
        "Aktuell kurs","Kurs (SEK)","Marknadsvärde (SEK)","Portföljandel (%)",
        "Utdelning/år","Årlig utdelning (SEK)","Frekvens/år","Ex-Date","Nästa utbetalning (est)"
    ]
    st.dataframe(d[show_cols], use_container_width=True)

# ── Sida: Regler & mål + sammanställningar ────────────────────────────────
CATEGORY_CHOICES = [
    "QUALITY","REIT","mREIT","BDC","Shipping","Telecom","Tech","Bank",
    "Finance","Energy","Industrial","Other"
]

def page_settings(df: pd.DataFrame):
    st.subheader("⚖️ Regler & mål")
    gmax, cats = load_settings()

    base = beräkna_allt(df)
    cat_sum = (base.groupby("Kategori", as_index=False)["Marknadsvärde (SEK)"]
                    .sum().rename(columns={"Marknadsvärde (SEK)":"Värde (SEK)"}))
    tot = float(cat_sum["Värde (SEK)"].sum() or 1.0)
    cat_sum["Andel (%)"] = (100.0 * cat_sum["Värde (SEK)"] / tot).round(2)

    stock_sum = base[["Bolagsnamn","Ticker","Marknadsvärde (SEK)"]].copy()
    stock_sum["Andel (%)"] = (100.0 * stock_sum["Marknadsvärde (SEK)"] / (float(base["Marknadsvärde (SEK)"].sum()) or 1.0)).round(2)
    stock_sum = stock_sum.sort_values("Andel (%)", ascending=False)

    st.write("**Nuvarande fördelning per kategori:**")
    st.dataframe(cat_sum.sort_values("Andel (%)", ascending=False), use_container_width=True)
    st.bar_chart(cat_sum.set_index("Kategori")["Andel (%)"])

    st.write("**Största innehav (andel av portfölj):**")
    st.dataframe(stock_sum.sort_values("Andel (%)", ascending=False), use_container_width=True)

    present = sorted(set(base["Kategori"].astype(str).tolist()) | set(cats.keys()))
    edit_rows = [{"Kategori":k, "Mål (%)":float(cats.get(k, 0.0))} for k in present]
    cols = st.columns(2)
    with cols[0]:
        gmax_new = st.number_input("Max vikt per bolag (%)", min_value=1.0, max_value=100.0, value=float(gmax), step=0.5)
    with cols[1]:
        st.caption("Justera kategori-mål nedan.")
    edit_df = pd.DataFrame(edit_rows).sort_values("Kategori")
    edited = st.data_editor(
        edit_df, hide_index=True, use_container_width=True,
        column_config={
            "Kategori": st.column_config.TextColumn(disabled=False),
            "Mål (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.2f")
        }
    )
    if st.button("💾 Spara regler"):
        new_cats = {str(r["Kategori"]): float(_to_float(r["Mål (%)"])) for _, r in edited.iterrows()}
        save_settings(gmax_new, new_cats)
        st.success("Regler sparade.")

# ── Sida: Lägg till / uppdatera bolag (in-memory, spara separat) ─────────
def page_add_or_update(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("➕ Lägg till / ✏ Uppdatera bolag")
    base = säkerställ_kolumner(df).copy()

    tickers = ["Ny"] + sorted(base["Ticker"].unique().tolist())
    val = st.selectbox("Välj bolag", tickers)

    if val == "Ny":
        c1, c2, c3 = st.columns(3)
        with c1:
            tkr = st.text_input("Ticker").strip().upper()
        with c2:
            antal = st.number_input("Antal aktier", min_value=0, value=0, step=1)
        with c3:
            gav   = st.number_input("GAV (i **lokal** valuta)", min_value=0.0, value=0.0, step=0.01)

        kategori = st.selectbox("Kategori", options=CATEGORY_CHOICES, index=0)
        colA, colB = st.columns(2)
        with colA:
            if st.button("🌐 Hämta från Yahoo"):
                if not tkr:
                    st.warning("Ange ticker först.")
                else:
                    vals = fetch_yahoo(tkr)
                    if vals:
                        st.info(f"{vals.get('Bolagsnamn',tkr)} | {vals.get('Valuta','?')} | Kurs {vals.get('Aktuell kurs',0)} | Utd/år {vals.get('Utdelning/år',0)} | Freq {vals.get('Frekvens/år',0)} | ExDate {vals.get('Ex-Date','')}")
        with colB:
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
                    base = uppdatera_nästa_utd(beräkna_allt(base))
                    st.session_state["working_df"] = base
                    st.success(f"{tkr} tillagt i minnet. Spara via 💾 Spara.")
    else:
        r = base[base["Ticker"]==val].iloc[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            tkr = st.text_input("Ticker", value=r["Ticker"]).strip().upper()
        with c2:
            antal = st.number_input("Antal aktier", min_value=0, value=int(_to_float(r["Antal aktier"])), step=1)
        with c3:
            gav   = st.number_input("GAV (i **lokal** valuta)", min_value=0.0, value=float(_to_float(r["GAV"])), step=0.01)
        kategori = st.selectbox("Kategori", options=CATEGORY_CHOICES, index=CATEGORY_CHOICES.index(str(r.get("Kategori","QUALITY"))))

        b1, b2, b3 = st.columns(3)
        with b1:
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
                base = uppdatera_nästa_utd(beräkna_allt(base))
                st.session_state["working_df"] = base
                st.success(f"{tkr} uppdaterad i minnet.")
        with b2:
            if st.button("✏ Uppdatera fält (minne)"):
                m = base["Ticker"]==val
                base.loc[m,"Ticker"] = tkr
                base.loc[m,"Antal aktier"] = antal
                base.loc[m,"GAV"] = gav
                base.loc[m,"Kategori"] = kategori
                base = uppdatera_nästa_utd(beräkna_allt(base))
                st.session_state["working_df"] = base
                st.success(f"{tkr} uppdaterad i minnet.")
        with b3:
            if st.button("🗑 Ta bort (minne)"):
                base = base[base["Ticker"]!=val].reset_index(drop=True)
                base = uppdatera_nästa_utd(beräkna_allt(base))
                st.session_state["working_df"] = base
                st.success(f"{val} borttagen i minnet.")

    st.markdown("---")
    if st.button("💾 Spara alla ändringar till Google Sheets"):
        spara_data(beräkna_allt(st.session_state["working_df"]))
        st.success("Sparat till Sheets.")
    return st.session_state.get("working_df", base)

# ── Sida: Massuppdatera alla (Yahoo) ──────────────────────────────────────
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
            base = uppdatera_nästa_utd(beräkna_allt(base))
            progress.progress(int(i*100/N))
            time.sleep(0.8)  # throttling
        st.session_state["working_df"] = base
        st.success("Massuppdatering klar (i minnet). Spara via 💾 Spara.")
    return st.session_state.get("working_df", base)

# ── Avgifter (enkelt courtage + FX) ───────────────────────────────────────
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

# ── Sida: Köpförslag & plan (≈500 kr) + säljkandidater ───────────────────
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

    T = float(base["Marknadsvärde (SEK)"].sum()) or 1.0
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
                "Courtage": c_fee, "FX-avg": fx_fee, "Poäng": round(float(r["Poäng"]),1)
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

    if steps:
        plan = pd.DataFrame(steps)
        per_ticker = (plan.groupby(["Ticker","Kategori"], as_index=False)
                            .agg({"Antal":"sum","Kostnad (SEK)":"sum","Pris (SEK)":"last","Poäng":"max"}))
        st.write("**Plan – steg för steg:**")
        st.dataframe(plan, use_container_width=True)
        st.write("**Summering per ticker:**")
        st.dataframe(per_ticker, use_container_width=True)
    else:
        st.info("Ingen köpposition passade reglerna.")

    # Säljkandidater
    st.markdown("---")
    st.write("🔻 **Säljkandidater (trimma)**")
    d = base.copy()
    gmax_now, cat_targets_now = load_settings()
    d["Över bolagsmax"] = d["Portföljandel (%)"] > float(gmax_now)
    cat_share = (d.groupby("Kategori", as_index=False)["Marknadsvärde (SEK)"].sum()
                   .rename(columns={"Marknadsvärde (SEK)":"V"}))
    total_v = float(cat_share["V"].sum() or 1.0)
    cat_share["Andel"] = 100.0 * cat_share["V"] / total_v
    cat_share["Mål"] = cat_share["Kategori"].map(lambda k: float(cat_targets_now.get(k, 100.0)))
    over_cats = cat_share[cat_share["Andel"] > cat_share["Mål"]]["Kategori"].tolist()
    d["Kategori övervikt"] = d["Kategori"].isin(over_cats)

    sellers = d[(d["Över bolagsmax"]) | (d["Kategori övervikt"])]
    if sellers.empty:
        st.info("Inga tydliga säljkandidater enligt nuvarande regler.")
    else:
        sellers = sellers.sort_values(["Över bolagsmax","Portföljandel (%)"], ascending=[False, False])
        show = ["Bolagsnamn","Ticker","Kategori","Portföljandel (%)","Marknadsvärde (SEK)","Över bolagsmax","Kategori övervikt"]
        st.dataframe(sellers[show], use_container_width=True)

# ── Sida: Utdelningskalender ──────────────────────────────────────────────
def page_calendar(df: pd.DataFrame):
    st.subheader("📅 Utdelningskalender")
    months = st.selectbox("Prognoshorisont (mån)", options=[12,24,36], index=0)

    def _gen(first_date, freq, lag, months_ahead):
        ts = pd.to_datetime(first_date, errors="coerce")
        if pd.isna(ts): return []
        exd = ts.date()
        f = _safe_int(freq, 4)
        L = _safe_int(lag, 30)
        step = max(1, int(round(365.0 / max(1, f))))
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
        per_share_local = _to_float(r.get("Utdelning/år", 0.0)) / max(1.0, _to_float(r.get("Frekvens/år",4)))
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

# ── Sida: Spara ───────────────────────────────────────────────────────────
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
        st.success("Sparat till Sheets.")

# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    st.title("Relative Yield – utdelningsportfölj")

    # Initiera in-memory tabellen från Google Sheets en gång per session
    if "working_df" not in st.session_state:
        try:
            st.session_state["working_df"] = hamta_data()
        except Exception:
            st.session_state["working_df"] = säkerställ_kolumner(pd.DataFrame())

    # Kör autosnap var 5:e minut
    autosnap_if_due(300)

    # Sidomeny med FX/backup/reparera
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
            "💾 Spara",
        ],
        index=0
    )

    base = säkerställ_kolumner(st.session_state["working_df"]).copy()

    if page == "📦 Portföljöversikt":
        page_portfolio(base)
    elif page == "⚖️ Regler & mål":
        _ = load_settings()  # se till att Settings finns
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
