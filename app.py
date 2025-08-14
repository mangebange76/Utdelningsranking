# app.py — Relative Yield (Utdelningsportfölj)
# Del 1/6  ─────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time, re, math
from datetime import datetime, timedelta, date
from google.oauth2.service_account import Credentials

# ── Rerun shim & page config ───────────────────────────────────────────────
try:
    _rerun = st.rerun
except AttributeError:
    _rerun = st.experimental_rerun

st.set_page_config(page_title="Relative Yield – utdelningsportfölj", layout="wide")

# ── Secrets / Google Sheets ────────────────────────────────────────────────
SHEET_URL  = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"            # databasflik
SET_SHEET  = "Settings"         # regler/mål
TX_SHEET   = "Transaktioner"    # transaktionslogg

scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# ── Taktning mot API (för att undvika rate limits) ────────────────────────
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

# ── A1 / formateringshjälpare (för rätt visning i Sheets) ─────────────────
def _col_to_a1(col_idx: int) -> str:
    # 1 -> A, 2 -> B, ...
    s = ""
    while col_idx > 0:
        col_idx, r = divmod(col_idx - 1, 26)
        s = chr(65 + r) + s
    return s

def _header_index_map(ws) -> dict:
    headers = ws.row_values(1)
    out = {}
    for i, name in enumerate(headers, start=1):
        if name:
            out[str(name).strip()] = i
    return out

def apply_number_formats(ws, sheet_title: str):
    """
    Lås kolumnformat i Google Sheets till NUMBER så datum/tid/komma inte sabbar siffror.
    Körs direkt efter sparning.
    """
    try:
        number_cols = [
            "Aktuell kurs","Utdelning/år","Utdelning/år (manuell)","Frekvens/år","Payment-lag (dagar)",
            "Antal aktier","GAV","Kurs (SEK)","Marknadsvärde (SEK)","Portföljandel (%)",
            "Insatt (SEK)","Årlig utdelning (SEK)","Direktavkastning (%)"
        ]
        idx_map = _header_index_map(ws)
        requests = []
        for col_name in number_cols:
            if col_name not in idx_map:
                continue
            # hela kolumnen (från rad 2 och nedåt)
            requests.append({
                "repeatCell": {
                    "range": {"sheetId": ws.id},
                    "cell": {
                        "userEnteredFormat": {
                            "numberFormat": {"type": "NUMBER", "pattern": "0.########"}
                        }
                    },
                    "fields": "userEnteredFormat.numberFormat"
                }
            })

        if not requests:
            return
        sh = _open_sheet()
        sh.batch_update({"requests": requests})
    except Exception as e:
        st.warning(f"Kunde inte låsa nummerformat i Sheets: {e}")

# ── Robust talparser (hindrar “tid”, kommatecken m.m.) ─────────────────────
def _to_float(x):
    if pd.isna(x): 
        return 0.0
    s = str(x).strip()
    # “09:18” → “9.18”
    if ":" in s:
        s = s.replace(":", ".")
        m = re.match(r"^(\d+)\.(\d+)", s)
        if m:
            s = f"{m.group(1)}.{m.group(2)}"
    # byt komma till punkt, rensa skräp
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s) if s not in ("", ".", "-", "-.") else 0.0
    except Exception:
        return 0.0

def _as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def make_view_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    v = df.copy()
    for c in cols:
        if c in v.columns:
            v[c] = _as_num(v[c])
    return v

# ── FX defaults ────────────────────────────────────────────────────────────
DEF_FX = {"USDSEK": 9.60, "NOKSEK": 0.94, "CADSEK": 6.95, "EURSEK": 11.10}
for k, v in DEF_FX.items():
    st.session_state.setdefault(k, v)

def fx_for(cur: str) -> float:
    if pd.isna(cur): 
        return 1.0
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
    # se till att alla kolumner finns
    for c in COLUMNS:
        if c not in d.columns:
            d[c] = ""

    # normalisera textfält
    d["Ticker"]   = d["Ticker"].astype(str).str.strip().str.upper()
    d["Valuta"]   = d["Valuta"].astype(str).str.strip().str.upper()
    d["Kategori"] = d["Kategori"].astype(str).str.strip()
    d["Kategori"] = d["Kategori"].replace({"": "QUALITY", "nan": "QUALITY", "NaN": "QUALITY"})

    # normalisera numeriska (som str -> float)
    num_cols = [
        "Aktuell kurs","Utdelning/år","Utdelning/år (manuell)","Frekvens/år","Payment-lag (dagar)",
        "Antal aktier","GAV","Kurs (SEK)","Marknadsvärde (SEK)","Portföljandel (%)",
        "Insatt (SEK)","Årlig utdelning (SEK)","Direktavkastning (%)"
    ]
    for c in num_cols:
        d[c] = d[c].apply(_to_float)

    # booleans
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

# ── JSON-safe/sanering & anti-wipe i sparning ─────────────────────────────
def _is_finite_number(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False

def _sanitize_for_sheets(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "Aktuell kurs","Utdelning/år","Utdelning/år (manuell)","Frekvens/år","Payment-lag (dagar)",
        "Antal aktier","GAV","Kurs (SEK)","Marknadsvärde (SEK)","Portföljandel (%)",
        "Insatt (SEK)","Årlig utdelning (SEK)","Direktavkastning (%)"
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

    # Sanera data innan skrivning
    out = _sanitize_for_sheets(out)

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
        titles = [w.title for w in sh.worksheets() if w.title.startswith("_Backup_")]
        titles.sort()
        if len(titles) > 10:
            for t in titles[:-10]:
                try:
                    sh.del_worksheet(sh.worksheet(t))
                except Exception:
                    pass
    except Exception as e:
        st.warning(f"Kunde inte skapa backupflik (fortsätter ändå): {e}")

    header = [out.columns.tolist()]
    body   = out.values.tolist()

    # Uppdatera med retry/backoff
    for attempt in range(1, max_retries+1):
        try:
            ws.update(header + body, value_input_option="RAW")
            # Lås format i sheets efter lyckad uppdatering
            apply_number_formats(ws, SHEET_NAME)
            break
        except Exception as e:
            msg = str(e)
            if ("Quota exceeded" in msg or "429" in msg) and attempt < max_retries:
                time.sleep(2 * attempt)
                continue
            st.error(f"Sparfel (avbryter): {e}")
            return

    st.success("✅ Sparning klar (säkert läge).")

# Alias som resten av koden använder
spara_data = spara_data_safe

# app.py — Relative Yield (Utdelningsportfölj)
# Del 2/6  ─────────────────────────────────────────────────────────────────

# ── Yahoo Finance-hämtning ────────────────────────────────────────────────
def fetch_yahoo(ticker: str) -> dict:
    """
    Hämtar bolagsnamn, kurs, valuta, utdelning och utdelningsfrekvens från Yahoo Finance.
    Returnerar dict med dessa värden (None om okänt).
    """
    try:
        _throttle(0.5)
        t = str(ticker).strip().upper()
        if not t:
            return {}
        data = {}
        y = yf.Ticker(t)

        info = y.info
        if not info:
            return {}

        data["Bolagsnamn"] = info.get("longName") or info.get("shortName")
        data["Aktuell kurs"] = info.get("currentPrice")
        data["Valuta"] = info.get("currency")

        # Utdelning/år i lokal valuta
        div_yield = info.get("dividendYield")
        if div_yield is not None:
            div_per_year = (div_yield or 0) * (data["Aktuell kurs"] or 0)
            data["Utdelning/år"] = div_per_year
        else:
            # Direkt från trailingAnnualDividendRate om finns
            data["Utdelning/år"] = info.get("trailingAnnualDividendRate")

        # Utdelningsfrekvens
        cal = y.dividends
        if not cal.empty:
            diffs = cal.index.to_series().diff().dt.days.dropna()
            if not diffs.empty:
                avg_gap = diffs.mean()
                if avg_gap > 300:
                    data["Frekvens/år"] = 1
                elif avg_gap > 100:
                    data["Frekvens/år"] = 4
                elif avg_gap > 40:
                    data["Frekvens/år"] = 12
                else:
                    data["Frekvens/år"] = round(365 / avg_gap)
        return data
    except Exception as e:
        st.warning(f"Kunde inte hämta från Yahoo för {ticker}: {e}")
        return {}

# ── Beräkningar ───────────────────────────────────────────────────────────
def beräkna_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uppdaterar beräknade kolumner i dataframe baserat på grunddata.
    """
    d = df.copy()
    if d.empty:
        return d

    d = säkerställ_kolumner(d)

    # Kurs (SEK)
    d["Kurs (SEK)"] = d.apply(lambda r: _to_float(r["Aktuell kurs"]) * fx_for(r["Valuta"]), axis=1)

    # Marknadsvärde
    d["Marknadsvärde (SEK)"] = d["Kurs (SEK)"] * _as_num(d["Antal aktier"])

    # Insatt kapital (SEK)
    d["Insatt (SEK)"] = _as_num(d["GAV"]) * _as_num(d["Antal aktier"]) * d.apply(lambda r: fx_for(r["Valuta"]), axis=1)

    # Portföljandel (%)
    tot_mkv = d["Marknadsvärde (SEK)"].sum()
    if tot_mkv > 0:
        d["Portföljandel (%)"] = 100 * d["Marknadsvärde (SEK)"] / tot_mkv
    else:
        d["Portföljandel (%)"] = 0.0

    # Utdelning/år (om lås = False, annars lämna som är)
    for idx, row in d.iterrows():
        if not row.get("Lås utdelning", False):
            # om manuell finns och >0, använd den
            if _to_float(row.get("Utdelning/år (manuell)", 0)) > 0:
                d.at[idx, "Utdelning/år"] = _to_float(row.get("Utdelning/år (manuell)", 0))

    # Årlig utdelning (SEK)
    d["Årlig utdelning (SEK)"] = _as_num(d["Utdelning/år"]) * _as_num(d["Antal aktier"]) * d.apply(lambda r: fx_for(r["Valuta"]), axis=1)

    # Direktavkastning (%)
    d["Direktavkastning (%)"] = d.apply(lambda r: (100 * _to_float(r["Utdelning/år"]) / _to_float(r["Aktuell kurs"])) if _to_float(r["Aktuell kurs"]) > 0 else 0, axis=1)

    # Utdelningskälla
    if "Utdelningskälla" not in d.columns:
        d["Utdelningskälla"] = "Yahoo"
    else:
        d["Utdelningskälla"] = d["Utdelningskälla"].fillna("Yahoo")

    # Senaste uppdatering (om tomt)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    d["Senaste uppdatering"] = d["Senaste uppdatering"].replace("", now_str)
    return d

# ── Formulär för Lägg till/Uppdatera ───────────────────────────────────────
def formulär(df: pd.DataFrame):
    """
    UI-formulär för att lägga till eller uppdatera bolag i portföljen.
    """
    st.subheader("➕ Lägg till / Uppdatera bolag")
    df = säkerställ_kolumner(df)

    tickers = ["(Ny)"] + sorted(df["Ticker"].unique())
    val = st.selectbox("Välj bolag att uppdatera eller '(Ny)' för nytt:", tickers)

    if val != "(Ny)":
        # befintlig rad
        row = df[df["Ticker"] == val].iloc[0].to_dict()
    else:
        row = {c: "" for c in COLUMNS}
        row["Lås utdelning"] = False

    with st.form("form_bolag"):
        ticker = st.text_input("Ticker", value=row.get("Ticker", ""), max_chars=10).upper().strip()
        bolagsnamn = st.text_input("Bolagsnamn", value=row.get("Bolagsnamn", ""))
        aktuell_kurs = st.number_input("Aktuell kurs", value=_to_float(row.get("Aktuell kurs", 0)), format="%.4f")
        valuta = st.selectbox("Valuta", ["SEK","USD","EUR","CAD","NOK"], index=max(0, ["SEK","USD","EUR","CAD","NOK"].index(row.get("Valuta", "SEK")) if row.get("Valuta", "SEK") in ["SEK","USD","EUR","CAD","NOK"] else 0))
        kategori = st.text_input("Kategori", value=row.get("Kategori", "QUALITY"))
        da = st.number_input("Direktavkastning (%)", value=_to_float(row.get("Direktavkastning (%)", 0)), format="%.2f")
        utd_per_år = st.number_input("Utdelning/år", value=_to_float(row.get("Utdelning/år", 0)), format="%.4f")
        utd_man = st.number_input("Utdelning/år (manuell)", value=_to_float(row.get("Utdelning/år (manuell)", 0)), format="%.4f")
        lås = st.checkbox("Lås utdelning", value=bool(row.get("Lås utdelning", False)))
        freq = st.number_input("Frekvens/år", value=_to_float(row.get("Frekvens/år", 0)), format="%.0f")
        utd_freq = st.text_input("Utdelningsfrekvens", value=row.get("Utdelningsfrekvens", ""))
        pay_lag = st.number_input("Payment-lag (dagar)", value=_to_float(row.get("Payment-lag (dagar)", 0)), format="%.0f")
        ex_date = st.text_input("Ex-Date", value=row.get("Ex-Date", ""))
        nästa_utb = st.text_input("Nästa utbetalning (est)", value=row.get("Nästa utbetalning (est)", ""))
        antal = st.number_input("Antal aktier", value=_to_float(row.get("Antal aktier", 0)), format="%.4f")
        gav = st.number_input("GAV", value=_to_float(row.get("GAV", 0)), format="%.4f")

        submitted = st.form_submit_button("💾 Spara bolag")

    if submitted:
        # uppdatera rad
        new_row = {
            "Ticker": ticker,
            "Bolagsnamn": bolagsnamn,
            "Aktuell kurs": aktuell_kurs,
            "Valuta": valuta,
            "Kategori": kategori,
            "Direktavkastning (%)": da,
            "Utdelning/år": utd_per_år,
            "Utdelning/år (manuell)": utd_man,
            "Lås utdelning": lås,
            "Frekvens/år": freq,
            "Utdelningsfrekvens": utd_freq,
            "Payment-lag (dagar)": pay_lag,
            "Ex-Date": ex_date,
            "Nästa utbetalning (est)": nästa_utb,
            "Antal aktier": antal,
            "GAV": gav,
        }

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, list(new_row.keys())] = list(new_row.values())
        else:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df = beräkna_kolumner(df)
        spara_data(df)
        st.success(f"{ticker} sparad!")
        _rerun()

# app.py — Relative Yield (Utdelningsportfölj)
# Del 3/6  ─────────────────────────────────────────────────────────────────

# ── Portföljvy ────────────────────────────────────────────────────────────
def portföljvy(df: pd.DataFrame):
    """
    Visar nuvarande portfölj med summeringar och nyckeltal.
    """
    st.subheader("📊 Portföljöversikt")
    df = säkerställ_kolumner(df)
    df = beräkna_kolumner(df)

    if df.empty:
        st.info("Portföljen är tom.")
        return

    # Summeringar
    totalt_värde = df["Marknadsvärde (SEK)"].sum()
    totalt_insatt = df["Insatt (SEK)"].sum()
    totalt_utd_år = df["Årlig utdelning (SEK)"].sum()
    utd_per_mån = totalt_utd_år / 12 if totalt_utd_år else 0
    avkastning = ((totalt_värde - totalt_insatt) / totalt_insatt * 100) if totalt_insatt > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 Totalt värde (SEK)", f"{totalt_värde:,.0f}")
    col2.metric("📥 Insatt kapital (SEK)", f"{totalt_insatt:,.0f}")
    col3.metric("📈 Avkastning (%)", f"{avkastning:,.2f}%")
    col4.metric("💵 Årlig utdelning (SEK)", f"{totalt_utd_år:,.0f}")
    col5.metric("📆 Utdelning/mån (SEK)", f"{utd_per_mån:,.0f}")

    st.markdown("---")

    # Visa portföljtabell
    vis_cols = [
        "Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta",
        "Kurs (SEK)", "Marknadsvärde (SEK)", "Portföljandel (%)",
        "GAV", "Insatt (SEK)", "Årlig utdelning (SEK)", "Direktavkastning (%)",
        "Lås utdelning", "Utdelning/år", "Utdelning/år (manuell)",
        "Frekvens/år", "Utdelningskälla", "Senaste uppdatering"
    ]
    vis_cols = [c for c in vis_cols if c in df.columns]

    # Sortera på portföljandel
    df_sorted = df.sort_values(by="Portföljandel (%)", ascending=False).reset_index(drop=True)

    st.dataframe(df_sorted[vis_cols].style.format({
        "Antal aktier": "{:,.2f}",
        "Aktuell kurs": "{:,.2f}",
        "Kurs (SEK)": "{:,.2f}",
        "Marknadsvärde (SEK)": "{:,.0f}",
        "Portföljandel (%)": "{:,.2f}",
        "GAV": "{:,.2f}",
        "Insatt (SEK)": "{:,.0f}",
        "Årlig utdelning (SEK)": "{:,.0f}",
        "Direktavkastning (%)": "{:,.2f}",
        "Utdelning/år": "{:,.2f}",
        "Utdelning/år (manuell)": "{:,.2f}",
        "Frekvens/år": "{:,.0f}"
    }), use_container_width=True)

# app.py — Relative Yield (Utdelningsportfölj)
# Del 4/6  ─────────────────────────────────────────────────────────────────

# ── Utdelningskalender ────────────────────────────────────────────────────
def utdelningskalender(df: pd.DataFrame):
    """
    Visar förväntade utdelningsdatum och belopp framåt i tiden.
    """
    st.subheader("📅 Utdelningskalender")

    månader_framåt = st.selectbox(
        "Visa utdelningar för antal månader framåt:",
        options=[6, 12, 24, 36],
        index=1
    )

    if df.empty:
        st.info("Ingen data i portföljen.")
        return

    df = säkerställ_kolumner(df)
    df = beräkna_kolumner(df)

    kalender = []
    idag = pd.Timestamp.today()

    for _, rad in df.iterrows():
        try:
            frekvens = int(rad.get("Frekvens/år", 0))
        except:
            frekvens = 0

        if frekvens <= 0 or pd.isna(rad.get("Ex-date")):
            continue

        ex_date = pd.to_datetime(rad["Ex-date"], errors="coerce")
        if pd.isna(ex_date):
            continue

        lag_dagar = int(rad.get("Utbetalningsfördröjning (dagar)", 30))
        belopp_per_aktie = float(rad.get("Utdelning/år", 0)) / frekvens
        antal_aktier = float(rad.get("Antal aktier", 0))
        kurs_fx = fx_for(rad.get("Valuta", "SEK"))

        utdelning_sek = belopp_per_aktie * antal_aktier * kurs_fx

        # Skapa kommande utdelningar
        steg_dagar = round(365 / frekvens)
        kommande_ex_dates = []

        # Se till att ex-datum hamnar i framtiden
        while ex_date < idag:
            ex_date += pd.Timedelta(days=steg_dagar)

        slutdatum = idag + pd.DateOffset(months=månader_framåt)
        while ex_date <= slutdatum:
            betalningsdatum = ex_date + pd.Timedelta(days=lag_dagar)
            kalender.append({
                "Bolag": rad["Bolagsnamn"],
                "Ex-date": ex_date.date(),
                "Betalningsdatum": betalningsdatum.date(),
                "Utdelning (SEK)": round(utdelning_sek, 2)
            })
            ex_date += pd.Timedelta(days=steg_dagar)

    if not kalender:
        st.info("Inga kommande utdelningar hittades.")
        return

    kalender_df = pd.DataFrame(kalender).sort_values(by="Betalningsdatum")
    st.dataframe(kalender_df, use_container_width=True)

# app.py — Relative Yield (Utdelningsportfölj)
# Del 5/6  ─────────────────────────────────────────────────────────────────

# ── Portföljvy ────────────────────────────────────────────────────────────
def visa_portfölj(df: pd.DataFrame):
    """
    Visar portföljöversikt med summeringar, andelar och potentiella trimningskandidater.
    """
    st.subheader("📊 Portföljöversikt")

    if df.empty:
        st.warning("Ingen data finns i databasen.")
        return

    df = säkerställ_kolumner(df)
    df = beräkna_kolumner(df)

    df_ägda = df[df["Antal aktier"] > 0].copy()
    if df_ägda.empty:
        st.info("Du äger inga bolag just nu.")
        return

    # Summeringar
    total_mv = float(df_ägda["Marknadsvärde (SEK)"].sum())
    total_insatt = float(df_ägda["Insatt (SEK)"].sum())
    total_utdelning = float(df_ägda["Årlig utdelning (SEK)"].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Portföljvärde (SEK)", f"{total_mv:,.0f}".replace(",", " "))
    col2.metric("📥 Insatt kapital (SEK)", f"{total_insatt:,.0f}".replace(",", " "))
    col3.metric("📈 Årlig utdelning (SEK)", f"{total_utdelning:,.0f}".replace(",", " "))

    # Andel per kategori
    if "Kategori" in df_ägda.columns:
        kategori_df = df_ägda.groupby("Kategori")["Marknadsvärde (SEK)"].sum().reset_index()
        kategori_df["Andel av portfölj (%)"] = kategori_df["Marknadsvärde (SEK)"] / total_mv * 100
        st.markdown("#### 📂 Fördelning per kategori")
        st.dataframe(kategori_df.sort_values(by="Andel av portfölj (%)", ascending=False))

    # Andel per bolag
    bolag_df = df_ägda[["Bolagsnamn", "Marknadsvärde (SEK)"]].copy()
    bolag_df["Andel av portfölj (%)"] = bolag_df["Marknadsvärde (SEK)"] / total_mv * 100
    st.markdown("#### 🏢 Fördelning per bolag")
    st.dataframe(bolag_df.sort_values(by="Andel av portfölj (%)", ascending=False))

    # Kandidater att sälja/trimma
    st.markdown("#### ✂️ Kandidater att sälja/trimma")
    max_andel = st.slider("Max andel (%)", min_value=5, max_value=50, value=20)
    kandidater = bolag_df[bolag_df["Andel av portfölj (%)"] > max_andel]
    if not kandidater.empty:
        st.dataframe(kandidater.sort_values(by="Andel av portfölj (%)", ascending=False))
    else:
        st.info("Inga bolag över vald maxandel.")

    # Detaljvy för felsökning
    with st.expander("🔍 Felsök ett bolag"):
        tickers = df_ägda["Ticker"].dropna().unique().tolist()
        if tickers:
            val = st.selectbox("Välj ticker", tickers)
            st.write(df_ägda[df_ägda["Ticker"] == val].T)

# app.py — Relative Yield (Utdelningsportfölj)
# Del 6/6  ─────────────────────────────────────────────────────────────────

# ── Huvudprogram ──────────────────────────────────────────────────────────
def main():
    st.title("💹 Aktieanalys & Utdelningsportfölj")

    df = hamta_data()
    df = säkerställ_kolumner(df)

    menyval = st.sidebar.radio(
        "Välj vy",
        ["📊 Analys", "📦 Portfölj", "➕ Lägg till / uppdatera", "🔄 Massuppdatera"]
    )

    if menyval == "📊 Analys":
        visa_analys(df)

    elif menyval == "📦 Portfölj":
        visa_portfölj(df)

    elif menyval == "➕ Lägg till / uppdatera":
        visa_formulär(df)

    elif menyval == "🔄 Massuppdatera":
        massuppdatera(df)


# ── Start ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
