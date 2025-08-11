import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time
import math
from datetime import datetime, timedelta, date
from google.oauth2.service_account import Credentials

# ── Sidkonfiguration ────────────────────────────────────────────────────────
st.set_page_config(page_title="Utdelningsportfölj", layout="wide")

# ── Google Sheets-konfig ────────────────────────────────────────────────────
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = st.secrets.get("SHEET_NAME", "Bolag")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
CREDS  = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=SCOPES)
client = gspread.authorize(CREDS)

def skapa_koppling():
    sh = client.open_by_url(SHEET_URL)
    try:
        return sh.worksheet(SHEET_NAME)
    except gspread.WorksheetNotFound:
        try:
            return sh.worksheet("Blad1")
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=SHEET_NAME, rows=1, cols=50)
            return ws

def hamta_df():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def ensure_or_create_ws(title: str, headers: list[str] | None = None):
    sh = client.open_by_url(SHEET_URL)
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=1000, cols=50)
        if headers:
            ws.update([headers])
    return ws

def save_df_to_sheet(df: pd.DataFrame, title: str):
    ws = ensure_or_create_ws(title, headers=df.columns.tolist())
    ws.clear()
    ws.update([df.columns.tolist()] + df.astype(str).values.tolist(), value_input_option="USER_ENTERED")

# ── Globala regler ─────────────────────────────────────────────────────────
GLOBAL_MAX_NAME = 12.0  # max % per innehav (hård gräns)

# Max per kategori (% av portföljen)
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
def get_cat_max(cat: str) -> float:
    return float(MAX_CAT.get(str(cat or "").strip() or "QUALITY", 100.0))

# Väljbara kategorier (UI)
CATEGORY_CHOICES = [
    "QUALITY", "REIT", "mREIT", "BDC", "MLP",
    "Shipping", "Telecom", "Tobacco", "Utility",
    "Tech", "Bank", "Industrial", "Energy", "Finance", "Other"
]

# ── Kolumnschema ────────────────────────────────────────────────────────────
COLUMNS = [
    "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Kategori",
    "Direktavkastning (%)", "Utdelning/år", "Utdelning/år (manuell)", "Lås utdelning",
    "Frekvens/år", "Utdelningsfrekvens", "Payment-lag (dagar)", "Ex-Date", "Nästa utbetalning (est)",
    "Antal aktier", "GAV", "Portföljandel (%)", "Årlig utdelning (SEK)",
    "Kurs (SEK)", "Utdelningstillväxt (%)", "Senaste uppdatering", "Källa"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        d = pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in d.columns:
            d[c] = ""
    d["Ticker"] = d["Ticker"].astype(str).str.strip().str.upper()
    d["Valuta"] = d["Valuta"].astype(str).str.strip().str.upper()
    # Defaults
    d["Kategori"] = d["Kategori"].astype(str).replace({"": "QUALITY"})
    # numeriska
    for c in ["Aktuell kurs","Utdelning/år","Utdelning/år (manuell)","Frekvens/år","Payment-lag (dagar)","Antal aktier","GAV"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    # booleans
    d["Lås utdelning"] = d["Lås utdelning"].apply(lambda x: bool(x) if pd.notna(x) else False)
    return d[COLUMNS].copy()

def migrate_sheet_columns():
    try:
        df = hamta_df()
    except Exception:
        df = pd.DataFrame()
    df2 = säkerställ_kolumner(df)
    if list(df.columns) != list(df2.columns) or df.shape[1] != df2.shape[1]:
        spara_df(df2)
    return df2

# ── Sparning till huvudark ─────────────────────────────────────────────────
def spara_df(df: pd.DataFrame):
    ws = skapa_koppling()
    out = säkerställ_kolumner(df).copy()
    out = out[out["Ticker"].astype(str).str.strip() != ""]
    if out.empty:
        st.warning("Inget att spara: inga tickers i tabellen.")
        return
    ws.clear()
    ws.update([out.columns.tolist()] + out.astype(str).values.tolist(), value_input_option="USER_ENTERED")

# ── FX ─────────────────────────────────────────────────────────────────────
def fx_for(cur):
    if pd.isna(cur):
        return 1.0
    c = str(cur).strip().upper()
    rate_map = {
        "USD": st.session_state.get("USDSEK", 9.60),
        "EUR": st.session_state.get("EURSEK", 11.10),
        "CAD": st.session_state.get("CADSEK", 6.95),
        "NOK": st.session_state.get("NOKSEK", 0.94),
        "SEK": 1.0,
    }
    try:
        return float(rate_map.get(c, 1.0))
    except:
        return 1.0

# ── Yahoo Finance-hämtning (pris, valuta, utdelning, frekvens, ex-date) ────
def hamta_yahoo(ticker: str) -> dict:
    t = yf.Ticker(ticker)

    # info
    info = {}
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}

    # pris
    price = None
    try:
        price = t.fast_info.get("last_price")
    except Exception:
        pass
    if price in (None, ""):
        price = info.get("currentPrice")
    if price in (None, ""):
        try:
            h = t.history(period="5d")
            if not h.empty:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None
    price = float(price) if price not in (None, "") else 0.0

    # valuta
    currency = (info.get("currency") or "").upper()
    if not currency:
        try:
            currency = (t.fast_info.get("currency") or "").upper()
        except Exception:
            currency = "SEK"

    # utdelning/år + frekvens från historik (12m)
    div_rate_local = 0.0
    last_ex_date = ""
    freq_per_year = 0
    freq_label = "Oregelbunden"
    try:
        divs = t.dividends
        if divs is not None and not divs.empty:
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
            div_rate_local = float(divs[divs.index >= cutoff].sum())
            cnt_12m = int((divs[divs.index >= cutoff].shape[0]))
            if cnt_12m >= 10:  # månads
                freq_per_year, freq_label = 12, "Månads"
            elif cnt_12m >= 3: # kvartal
                freq_per_year, freq_label = 4, "Kvartals"
            elif cnt_12m >= 2: # halvår
                freq_per_year, freq_label = 2, "Halvårs"
            elif cnt_12m >= 1: # års
                freq_per_year, freq_label = 1, "Års"
            else:
                freq_per_year, freq_label = 0, "Oregelbunden"
    except Exception:
        pass

    if div_rate_local == 0.0:
        try:
            fwd = info.get("forwardAnnualDividendRate")
            if fwd not in (None, ""):
                div_rate_local = float(fwd)
        except Exception:
            pass
    if div_rate_local == 0.0:
        try:
            trailing = info.get("trailingAnnualDividendRate")
            if trailing not in (None, ""):
                div_rate_local = float(trailing)
        except Exception:
            pass

    # ex-date
    try:
        ts = info.get("exDividendDate")
        if ts not in (None, "", 0):
            last_ex_date = pd.to_datetime(int(ts), unit="s", utc=True).strftime("%Y-%m-%d")
    except Exception:
        pass

    dy_pct = 0.0
    if price > 0 and div_rate_local > 0:
        dy_pct = round(100.0 * div_rate_local / price, 2)

    return {
        "Bolagsnamn": info.get("longName") or info.get("shortName") or "",
        "Aktuell kurs": price,
        "Valuta": currency,
        "Utdelning/år": round(float(div_rate_local), 4),
        "Direktavkastning (%)": dy_pct,
        "Frekvens/år": freq_per_year if freq_per_year else "",
        "Utdelningsfrekvens": freq_label,
        "Ex-Date": last_ex_date,
        "Senaste uppdatering": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Källa": "Yahoo Finance",
    }

# ── Beräkningar (SEK, DA, portföljandel, nästa utbetalning, eff.utdelning) ─
def beräkna(df: pd.DataFrame) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()

    d["Aktuell kurs"]   = pd.to_numeric(d["Aktuell kurs"], errors="coerce").fillna(0.0).astype(float)
    d["Utdelning/år"]   = pd.to_numeric(d["Utdelning/år"], errors="coerce").fillna(0.0).astype(float)
    d["Utdelning/år (manuell)"] = pd.to_numeric(d["Utdelning/år (manuell)"], errors="coerce").fillna(0.0).astype(float)
    d["Antal aktier"]   = pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0).astype(float)
    d["GAV"]            = pd.to_numeric(d["GAV"], errors="coerce").fillna(0.0).astype(float)
    d["Frekvens/år"]    = pd.to_numeric(d["Frekvens/år"], errors="coerce").fillna(0.0).astype(float).replace(0, 4)
    d["Payment-lag (dagar)"] = pd.to_numeric(d["Payment-lag (dagar)"], errors="coerce").fillna(0.0).astype(float).replace(0, 30)

    rates = d["Valuta"].apply(fx_for).astype(float)
    d["Kurs (SEK)"] = (d["Aktuell kurs"] * rates).astype(float).round(6)

    # Effektiv utd/år: respektera låsningen
    use_manual = (d["Lås utdelning"] == True) & (d["Utdelning/år (manuell)"] > 0)
    d["Utdelning/år_eff"] = d["Utdelning/år"]
    d.loc[use_manual, "Utdelning/år_eff"] = d.loc[use_manual, "Utdelning/år (manuell)"]

    d["Årlig utdelning (SEK)"] = (d["Antal aktier"] * d["Utdelning/år_eff"] * rates).round(2)

    d["Direktavkastning (%)"] = 0.0
    ok = (d["Aktuell kurs"] > 0) & (d["Utdelning/år_eff"] > 0)
    d.loc[ok, "Direktavkastning (%)"] = (100.0 * d.loc[ok, "Utdelning/år_eff"] / d.loc[ok, "Aktuell kurs"]).round(2)

    mv = (d["Antal aktier"] * d["Kurs (SEK)"]).round(2)
    tot_mv = float(mv.sum()) if mv.sum() else 1.0
    d["Portföljandel (%)"] = (100.0 * mv / tot_mv).round(2)

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

# ── Prognos (12/24/36 mån) ─────────────────────────────────────────────────
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

# ── Uppdatera/lägg till tickers (respektera manuell utd.) ──────────────────
def add_or_update_ticker_row(ticker: str) -> pd.DataFrame:
    base = säkerställ_kolumner(st.session_state["working_df"])
    ticker = (ticker or "").strip().upper()
    if not ticker:
        st.warning("Ange en ticker.")
        return base
    if not (base["Ticker"] == ticker).any():
        base = pd.concat([base, pd.DataFrame([{"Ticker": ticker, "Antal aktier": 0.0, "GAV": 0.0, "Kategori": "QUALITY"}])], ignore_index=True)
    try:
        vals = hamta_yahoo(ticker)
        # skriv inte över Utdelning/år med 0
        old = float(pd.to_numeric(base.loc[base["Ticker"] == ticker, "Utdelning/år"], errors="coerce").fillna(0.0))
        new = float(vals.get("Utdelning/år", 0.0) or 0.0)
        if new > 0:
            base.loc[base["Ticker"] == ticker, "Utdelning/år"] = new
        for k, v in vals.items():
            if k in ["Utdelning/år"]: 
                continue
            base.loc[base["Ticker"] == ticker, k] = v
        base = beräkna(base)
        st.success(f"Ticker {ticker} uppdaterad (in-memory).")
    except Exception as e:
        st.error(f"Kunde inte hämta {ticker}: {e}")
    return base

def update_some_tickers(tickers: list) -> pd.DataFrame:
    base = säkerställ_kolumner(st.session_state["working_df"])
    if not tickers:
        st.warning("Välj minst en ticker.")
        return base
    bar = st.progress(0.0)
    for i, tkr in enumerate(tickers, 1):
        try:
            vals = hamta_yahoo(tkr)
            old = float(pd.to_numeric(base.loc[base["Ticker"] == tkr, "Utdelning/år"], errors="coerce").fillna(0.0))
            new = float(vals.get("Utdelning/år", 0.0) or 0.0)
            if new > 0:
                base.loc[base["Ticker"] == tkr, "Utdelning/år"] = new
            for k, v in vals.items():
                if k in ["Utdelning/år"]:
                    continue
                base.loc[base["Ticker"] == tkr, k] = v
        except Exception as e:
            st.warning(f"{tkr}: misslyckades ({e})")
        time.sleep(1.0)
        bar.progress(i/len(tickers))
    base = beräkna(base)
    st.success(f"Uppdaterade {len(tickers)} ticker(s) (in-memory).")
    return base

# ── FX-sidopanel (med reset) ───────────────────────────────────────────────
def sidopanel(df: pd.DataFrame):
    st.sidebar.header("⚙️ Inställningar")
    st.sidebar.markdown("**Växelkurser (SEK)**")
    DEF = {"USDSEK": 9.60, "EURSEK": 11.10, "CADSEK": 6.95, "NOKSEK": 0.94}
    for k, v in DEF.items():
        st.session_state.setdefault(k, v)
    colA, colB = st.sidebar.columns(2)
    with colA:
        USD = st.number_input("USD/SEK", min_value=0.0, value=float(st.session_state["USDSEK"]), step=0.01, format="%.4f")
        EUR = st.number_input("EUR/SEK", min_value=0.0, value=float(st.session_state["EURSEK"]), step=0.01, format="%.4f")
    with colB:
        CAD = st.number_input("CAD/SEK", min_value=0.0, value=float(st.session_state["CADSEK"]), step=0.01, format="%.4f")
        NOK = st.number_input("NOK/SEK", min_value=0.0, value=float(st.session_state["NOKSEK"]), step=0.01, format="%.4f")
    st.session_state["USDSEK"], st.session_state["EURSEK"], st.session_state["CADSEK"], st.session_state["NOKSEK"] = USD, EUR, CAD, NOK
    if st.sidebar.button("↩︎ Återställ FX till standard"):
        for k, v in DEF.items():
            st.session_state[k] = v
        st.experimental_rerun()

    st.sidebar.markdown("---")
    one_ticker = st.sidebar.text_input("Uppdatera EN ticker (Yahoo)", placeholder="t.ex. EPD")
    if st.sidebar.button("🔄 Uppdatera EN"):
        st.session_state["working_df"] = add_or_update_ticker_row(one_ticker)

# ── Lägg till bolag ────────────────────────────────────────────────────────
def page_add_company(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("➕ Lägg till bolag")

    col1, col2, col3, col4 = st.columns([1.3, 1, 1, 1])
    with col1:
        tkr = st.text_input("Ticker *", placeholder="t.ex. VICI eller 2020.OL").strip().upper()
    with col2:
        qty = st.number_input("Antal aktier", min_value=0, value=0, step=1)
    with col3:
        gav = st.number_input("GAV (SEK)", min_value=0.0, value=0.0, step=0.01)
    with col4:
        kat = st.selectbox("Kategori", options=CATEGORY_CHOICES, index=CATEGORY_CHOICES.index("QUALITY"))

    st.caption("Enda obligatoriska fältet är Ticker. Antal och GAV får vara 0. Övrig data hämtas från Yahoo vid sparning.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🌐 Hämta från Yahoo (förhandsgranskning)"):
            if not tkr:
                st.error("Ange Ticker först.")
            else:
                _tmp = säkerställ_kolumner(df)
                if not (_tmp["Ticker"] == tkr).any():
                    _tmp = pd.concat([_tmp, pd.DataFrame([{"Ticker": tkr, "Kategori": kat}])], ignore_index=True)
                try:
                    vals = hamta_yahoo(tkr)
                    for k, v in vals.items():
                        if k == "Utdelning/år":
                            old = float(pd.to_numeric(_tmp.loc[_tmp["Ticker"]==tkr, "Utdelning/år"], errors="coerce").fillna(0.0))
                            new = float(vals.get("Utdelning/år", 0.0) or 0.0)
                            if new > 0:
                                _tmp.loc[_tmp["Ticker"]==tkr, "Utdelning/år"] = new
                        elif k != "Kategori":
                            _tmp.loc[_tmp["Ticker"] == tkr, k] = v
                    df = _tmp
                    st.success(f"Hämtade Yahoo-data för {tkr}. Detta sparas först när du trycker 'Spara NU'.")
                except Exception as e:
                    st.warning(f"Kunde inte hämta data: {e}")

    with c2:
        if st.button("💾 Spara NU till Google Sheets"):
            if not tkr:
                st.error("Ticker är obligatoriskt.")
                return df
            base = säkerställ_kolumner(df)

            # upsert & kategori
            if (base["Ticker"] == tkr).any():
                i = base.index[base["Ticker"] == tkr][0]
                base.at[i, "Antal aktier"] = float(qty)
                base.at[i, "GAV"] = float(gav)
                base.at[i, "Kategori"] = kat
            else:
                base = pd.concat([base, pd.DataFrame([{
                    "Ticker": tkr, "Antal aktier": float(qty), "GAV": float(gav), "Kategori": kat
                }])], ignore_index=True)

            # hämta/uppdatera Yahoo
            try:
                vals = hamta_yahoo(tkr)
                old = float(pd.to_numeric(base.loc[base["Ticker"]==tkr, "Utdelning/år"], errors="coerce").fillna(0.0))
                new = float(vals.get("Utdelning/år", 0.0) or 0.0)
                if new > 0:
                    base.loc[base["Ticker"]==tkr, "Utdelning/år"] = new
                for k, v in vals.items():
                    if k in ["Utdelning/år", "Kategori"]:
                        continue
                    base.loc[base["Ticker"] == tkr, k] = v
            except Exception as e:
                st.warning(f"Kunde inte hämta Yahoo-data just nu ({e}). Sparar ändå grunderna.")

            base = beräkna(base)
            spara_df(base)
            st.session_state["working_df"] = base
            st.success(f"{tkr} sparad till Google Sheets.")
            return base

    st.divider()
    st.caption("Förhandsgranskning (in-memory)")
    st.dataframe(
        beräkna(säkerställ_kolumner(df))[["Ticker","Bolagsnamn","Valuta","Kategori","Antal aktier","GAV","Aktuell kurs","Utdelning/år","Utdelningsfrekvens","Kurs (SEK)","Årlig utdelning (SEK)"]],
        use_container_width=True
    )
    return df

# ── Uppdatera innehav ───────────────────────────────────────────────────────
def page_update_holdings(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🔄 Uppdatera befintligt innehav")
    col1, col2 = st.columns([1.6, 1])
    with col1:
        t_single = st.text_input("Uppdatera EN ticker", placeholder="t.ex. XOM, VICI, FTS").strip().upper()
    with col2:
        if st.button("🚀 Hämta EN från Yahoo"):
            df = add_or_update_ticker_row(t_single)

    st.caption("Uppdatera flera samtidigt (1 sekund paus per ticker)")
    valbara = df["Ticker"].astype(str).tolist() if not df.empty else []
    selection = st.multiselect("Välj tickers", options=valbara)

    cA, cB = st.columns(2)
    with cA:
        if st.button("🔁 Uppdatera valda"):
            df = update_some_tickers(selection)
    with cB:
        if st.button("🌀 Uppdatera ALLA"):
            df = update_some_tickers(valbara)

    st.divider()
    st.caption("Senaste data")
    st.dataframe(beräkna(df)[["Ticker","Bolagsnamn","Valuta","Kategori","Aktuell kurs","Utdelningsfrekvens","Utdelning/år","Direktavkastning (%)","Ex-Date","Nästa utbetalning (est)"]], use_container_width=True)
    return df

# ── Portföljöversikt m. regler & trimförslag ───────────────────────────────
def rule_check(df: pd.DataFrame):
    d = beräkna(df).copy()
    if d.empty:
        return d, pd.DataFrame(), []

    d["Kategori"] = d.get("Kategori", "QUALITY").astype(str).replace({"": "QUALITY"})
    d["Marknadsvärde (SEK)"] = (
        pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0)
        * pd.to_numeric(d["Kurs (SEK)"], errors="coerce").fillna(0.0)
    ).astype(float)
    T = float(d["Marknadsvärde (SEK)"].sum()) if d["Marknadsvärde (SEK)"].sum() else 1.0
    d["Vikt (%)"] = (100.0 * d["Marknadsvärde (SEK)"] / T).round(2)

    cat_w = d.groupby("Kategori", as_index=False)["Marknadsvärde (SEK)"].sum()
    cat_w["Kategori-vikt (%)"] = (100.0 * cat_w["Marknadsvärde (SEK)"] / T).round(2)

    breaches = []
    for _, r in d.iterrows():
        if float(r["Vikt (%)"]) > GLOBAL_MAX_NAME + 1e-9:
            breaches.append(f"{r['Ticker']}: vikt {r['Vikt (%)']:.2f}% > max {GLOBAL_MAX_NAME:.2f}%")
    for _, r in cat_w.iterrows():
        cat = r["Kategori"]; w = float(r["Kategori-vikt (%)"]); limit = get_cat_max(cat)
        if w > limit + 1e-9:
            breaches.append(f"Kategori {cat}: {w:.2f}% > max {limit:.2f}%")
    return d, cat_w[["Kategori","Kategori-vikt (%)"]], breaches

def trim_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    d = beräkna(df).copy()
    if d.empty: return pd.DataFrame(columns=["Ticker","Vikt (%)","Kurs (SEK)","Föreslagen sälj (st)","Nettolikvid ca (SEK)"])
    d["Marknadsvärde (SEK)"] = (pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0)
                                * pd.to_numeric(d["Kurs (SEK)"], errors="coerce").fillna(0.0)).astype(float)
    T = float(d["Marknadsvärde (SEK)"].sum()) if d["Marknadsvärde (SEK)"].sum() else 0.0
    if T <= 0: return pd.DataFrame(columns=["Ticker","Vikt (%)","Kurs (SEK)","Föreslagen sälj (st)","Nettolikvid ca (SEK)"])
    rows = []
    for _, r in d.iterrows():
        V = float(r["Marknadsvärde (SEK)"]); w = 100.0 * V / T if T else 0.0
        if w <= GLOBAL_MAX_NAME + 1e-9: continue
        price = float(pd.to_numeric(r["Kurs (SEK)"], errors="coerce") or 0.0)
        qty   = float(pd.to_numeric(r["Antal aktier"], errors="coerce") or 0.0)
        if price <= 0 or qty <= 0: continue
        # (V - nP) / (T - nP) <= m ⇒ n >= (V - mT) / ((1-m)P)
        n_min = (V - (GLOBAL_MAX_NAME/100.0)*T) / ((1 - GLOBAL_MAX_NAME/100.0) * price)
        n = max(0, math.ceil(n_min)); n = int(min(n, qty))
        if n > 0:
            gross = round(price * n, 2)
            foreign = str(r.get("Valuta","SEK")).upper() != "SEK"
            fee_court, fee_fx, fee_tot = calc_fees(gross, foreign)
            net = round(gross - fee_tot, 2)
            rows.append({"Ticker": r["Ticker"], "Vikt (%)": round(w,2), "Kurs (SEK)": round(price,2),
                         "Föreslagen sälj (st)": n, "Nettolikvid ca (SEK)": net, "Kommentar": f"Ner till {GLOBAL_MAX_NAME:.0f}%"})
    return pd.DataFrame(rows)

def block_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("📦 Portföljöversikt")
    d = beräkna(df).copy()
    d["Marknadsvärde (SEK)"] = (pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0) * pd.to_numeric(d["Kurs (SEK)"], errors="coerce").fillna(0.0)).round(2)
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

    edit_cols = ["Antal aktier", "GAV", "Frekvens/år", "Payment-lag (dagar)", "Kategori", "Utdelning/år (manuell)", "Lås utdelning"]
    show_cols = [
        "Ticker","Bolagsnamn","Valuta","Kategori","Aktuell kurs","Kurs (SEK)",
        "Antal aktier","GAV","Insatt (SEK)","Marknadsvärde (SEK)",
        "Orealiserad P/L (SEK)","Orealiserad P/L (%)",
        "Direktavkastning (%)","Utdelning/år","Utdelning/år (manuell)","Lås utdelning",
        "Utdelningsfrekvens","Frekvens/år","Årlig utdelning (SEK)",
        "Ex-Date","Nästa utbetalning (est)","Portföljandel (%)","Senaste uppdatering"
    ]
    view = d[show_cols].copy()
    editor = st.data_editor(
        view, hide_index=True, num_rows="dynamic", use_container_width=True,
        column_config={
            "Kategori": st.column_config.SelectboxColumn("Kategori", options=CATEGORY_CHOICES, default="QUALITY", required=True)
        }
    )
    if st.button("💾 Spara ändringar (in-memory)"):
        base = säkerställ_kolumner(st.session_state["working_df"])
        for _, r in editor.iterrows():
            t = str(r["Ticker"]).upper()
            mask = base["Ticker"].astype(str).str.upper() == t
            if not mask.any(): continue
            for c in edit_cols:
                base.loc[mask, c] = r[c]
        st.session_state["working_df"] = beräkna(base)
        st.success("Ändringar sparade (i appens minne).")
        return st.session_state["working_df"]

    # Regler & vikter, kategori-tak & trim
    d_rules, cat_w, breaches = rule_check(d)
    with st.expander("📏 Regler & vikter"):
        st.write("**Kategori-vikter (%) och max**")
        if not cat_w.empty:
            max_df = pd.DataFrame([{"Kategori": k, "Max (%)": v} for k, v in MAX_CAT.items()])
            merge = pd.merge(max_df, cat_w, on="Kategori", how="left").fillna({"Kategori-vikt (%)": 0.0})
            merge = merge.rename(columns={"Kategori-vikt (%)":"Nu (%)"})
            st.dataframe(merge.sort_values("Kategori"), use_container_width=True)
        # Trimförslag
        trim_df = trim_suggestions(d)
        if not trim_df.empty:
            st.warning("Följande innehav ligger över 12% – förslag att skala ned:")
            st.dataframe(trim_df, use_container_width=True)
        # Breaches
        if breaches:
            for b in breaches:
                st.error("⚠️ " + b)
        else:
            st.success("Alla innehav/kategorier inom satta gränser.")
    return d

# ── Ranking ────────────────────────────────────────────────────────────────
def block_top_card(df: pd.DataFrame):
    d = beräkna(df)
    if d.empty:
        st.info("Ingen data ännu. Lägg till tickers och uppdatera från Yahoo.")
        return
    d["Direktavkastning (%)"] = pd.to_numeric(d["Direktavkastning (%)"], errors="coerce").fillna(0.0)
    top = d.sort_values("Direktavkastning (%)", ascending=False).iloc[0]
    c1, c2, c3 = st.columns([1.6, 1, 1])
    with c1:
        st.subheader(f"🏆 Högst DA: **{top['Ticker']}** — {top.get('Bolagsnamn','')}")
        st.write(
            f"- Direktavkastning: **{top['Direktavkastning (%)']:.2f}%**  \n"
            f"- Utd/år (lokal, eff.): **{round(float(top['Utdelning/år_eff']),2)}**  \n"
            f"- Ex-Date: **{top.get('Ex-Date','')}**, nästa est: **{top.get('Nästa utbetalning (est)','')}**"
        )
    with c2:
        st.metric("Kurs (SEK)", f"{top.get('Kurs (SEK)','')}")
        st.metric("Årsutd (SEK)", f"{top.get('Årlig utdelning (SEK)','')}")
    with c3:
        st.metric("Valuta", top.get("Valuta",""))
        st.metric("Uppdaterad", top.get("Senaste uppdatering",""))

def block_ranking(df: pd.DataFrame):
    st.subheader("📊 Ranking (sorterat på direktavkastning)")
    d = beräkna(df).copy()
    d["Direktavkastning (%)"] = pd.to_numeric(d["Direktavkastning (%)"], errors="coerce").fillna(0.0)
    d = d.sort_values(["Direktavkastning (%)","Årlig utdelning (SEK)"], ascending=[False, False])
    cols = ["Ticker","Bolagsnamn","Valuta","Kategori","Kurs (SEK)","Direktavkastning (%)","Utdelning/år_eff","Utdelningsfrekvens","Ex-Date","Nästa utbetalning (est)","Portföljandel (%)","Senaste uppdatering"]
    st.dataframe(d[cols], use_container_width=True)

# ── Kalender-sida ──────────────────────────────────────────────────────────
def page_calendar(df: pd.DataFrame):
    st.subheader("📅 Utdelningskalender")
    months = st.selectbox("Prognoshorisont", options=[12, 24, 36], index=0)
    monthly, cal = prognos_kalender(df, months_ahead=months)
    if monthly.empty:
        st.info("Ingen prognos ännu – saknar Ex-Date/frekvens/utdelningsdata.")
        return
    st.write(f"**Månadsvis prognos ({months} mån) i SEK:**")
    st.dataframe(monthly, use_container_width=True)
    st.bar_chart(monthly.set_index("Månad")["Utdelning (SEK)"])
    if not cal.empty:
        with st.expander("Detaljerade kommande betalningar per ticker"):
            st.dataframe(cal.sort_values("Datum"), use_container_width=True)
    st.divider()
    if st.button("💾 Spara prognos till Google Sheets"):
        try:
            save_df_to_sheet(monthly, "Prognos_Månad")
            if not cal.empty:
                cal_sorted = cal.sort_values("Datum").copy()
                cal_sorted["Datum"] = cal_sorted["Datum"].apply(lambda d: d.strftime("%Y-%m-%d"))
                save_df_to_sheet(cal_sorted, "Prognos_Detalj")
            st.success("Prognosen sparad till arken 'Prognos_Månad' och 'Prognos_Detalj'.")
        except Exception as e:
            st.error(f"Kunde inte spara prognosen: {e}")

# ── Spara-sida ─────────────────────────────────────────────────────────────
def page_save_now():
    st.subheader("💾 Spara till Google Sheets")
    preview = beräkna( säkerställ_kolumner(st.session_state["working_df"]) )
    st.write("Antal rader som sparas:", len(preview))
    st.dataframe(preview[["Ticker","Bolagsnamn","Valuta","Kategori","Antal aktier","GAV","Aktuell kurs","Utdelning/år_eff","Kurs (SEK)","Årlig utdelning (SEK)"]], use_container_width=True)
    if st.button("✅ Bekräfta och spara"):
        if preview["Ticker"].astype(str).str.strip().eq("").all():
            st.error("Inget att spara: inga tickers i tabellen.")
            return
        spara_df(preview)
        save_pending_transactions()
        st.success("Data och transaktioner sparade till Google Sheets!")

# ── Trading (Köp/Sälj) med avgifter ────────────────────────────────────────
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
    values = [[r["Tid"], r["Typ"], r["Ticker"], r["Antal"], r["Pris (lokal)"], r["Valuta"], r["FX"],
               r["Pris (SEK)"], r["Belopp (SEK)"], r["Courtage (SEK)"], r["FX-avgift (SEK)"], r["Tot.avgifter (SEK)"], r["Kommentar"]] for r in rows]
    ws_tx.append_rows(values, value_input_option="USER_ENTERED")
    st.session_state["pending_txs"] = []

def block_trading(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🛒 Köp / 📤 Sälj (avgifter, in-memory)")
    if df.empty:
        st.info("Lägg till minst en ticker först."); 
        return df

    tickers = df["Ticker"].astype(str).tolist()
    tkr = st.selectbox("Ticker", options=tickers)
    side = st.radio("Typ", ["KÖP", "SÄLJ"], horizontal=True)
    qty  = st.number_input("Antal", min_value=1, value=10, step=1)
    px_local = st.number_input("Pris (lokal valuta)", min_value=0.0, value=10.0

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

    # --- Regelkontroll-knapp (hård gräns 12% per innehav) ------------------
    if st.button("Kontrollera mot 12%-regeln", key="check_rules_btn2"):
        base = säkerställ_kolumner(st.session_state["working_df"]).copy()
        if not (base["Ticker"] == tkr).any():
            st.error("Ticker finns inte i portföljen ännu. Lägg till under '➕ Lägg till bolag'.")
            return df

        i = base.index[base["Ticker"] == tkr][0]
        sim = base.copy()
        # simulera antal
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

        max_cap = globals().get("GLOBAL_MAX_NAME", 12.0)
        if side == "KÖP" and w_after > max_cap + 1e-9:
            st.error(f"KÖP skulle ge vikt {w_after:.2f}% > max {max_cap:.2f}% – blockerat.")
        elif side == "SÄLJ" and 0 < w_after < (globals().get("GLOBAL_MIN_NAME", 3.0)):
            st.warning(f"SÄLJ skulle lämna {tkr} under {globals().get('GLOBAL_MIN_NAME', 3.0):.0f}% (efter affär {w_after:.2f}%). "
                       "Överväg att sälja allt eller toppa upp senare.")
        else:
            st.success("OK enligt reglerna.")

    # --- Lägg order i minnet (respekterar 12%) ------------------------------
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
                st.error("Sälj ger negativt antal."); 
                return df
            sim.at[i, "Antal aktier"] = new_q

        d_chk = beräkna(sim)
        mv_chk = (
            pd.to_numeric(d_chk["Antal aktier"], errors="coerce").fillna(0.0)
            * pd.to_numeric(d_chk["Kurs (SEK)"], errors="coerce").fillna(0.0)
        )
        tot_mv_chk = float(mv_chk.sum()) if mv_chk.sum() else 1.0
        w_after = float(100.0 * float(mv_chk.loc[d_chk["Ticker"]==tkr].sum()) / tot_mv_chk)

        max_cap = globals().get("GLOBAL_MAX_NAME", 12.0)
        if side == "KÖP" and w_after > max_cap + 1e-9:
            st.error(f"Order stoppad: {tkr} skulle väga {w_after:.2f}% > {max_cap:.2f}%.")
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

# ── Main (router/meny) ─────────────────────────────────────────────────────
def main():
    st.title("Relative Yield – utdelningsportfölj")

    # Initiera arbetskopia från Google Sheets första gången
    if "working_df" not in st.session_state:
        try:
            st.session_state["working_df"] = migrate_sheet_columns()
        except Exception:
            # Fallback om arket är tomt eller saknas
            st.session_state["working_df"] = säkerställ_kolumner(pd.DataFrame())

    base = st.session_state["working_df"]

    # Sidopanel (FX mm.)
    sidopanel(base)

    # Meny
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Meny",
        [
            "➕ Lägg till bolag",
            "📦 Portföljöversikt",
            "🔄 Uppdatera innehav",
            "🛒 Köp/Sälj",
            "📊 Ranking & köpförslag",
            "📅 Utdelningskalender",
            "🎯 Köpförslag",
            "💾 Spara",
        ],
        index=0
    )

    # Router
    if page == "➕ Lägg till bolag":
        base = page_add_company(base)
    elif page == "📦 Portföljöversikt":
        base = block_portfolio(base)
    elif page == "🔄 Uppdatera innehav":
        base = page_update_holdings(base)
    elif page == "🛒 Köp/Sälj":
        base = block_trading(base)
    elif page == "📊 Ranking & köpförslag":
        block_top_card(base)
        st.divider()
        block_ranking(base)
    elif page == "📅 Utdelningskalender":
        page_calendar(base)
    elif page == "🎯 Köpförslag":
        page_buy_suggestions(base)
    elif page == "💾 Spara":
        page_save_now()

    # Uppdatera state
    st.session_state["working_df"] = base

# Streamlit entrypoint
if __name__ == "__main__":
    main()
