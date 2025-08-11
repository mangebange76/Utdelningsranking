import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time
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

def spara_df(df: pd.DataFrame):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.tolist()] + df.astype(str).values.tolist())

# ── Kolumnschema ────────────────────────────────────────────────────────────
COLUMNS = [
    "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Direktavkastning (%)", "Utdelning/år",
    "Frekvens/år", "Payment-lag (dagar)", "Ex-Date", "Nästa utbetalning (est)",
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
    # numerik defaults
    for c in ["Aktuell kurs","Utdelning/år","Frekvens/år","Payment-lag (dagar)","Antal aktier","GAV"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
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

# ── FX-hjälpare (robust) ───────────────────────────────────────────────────
def fx_for(cur):
    if pd.isna(cur):
        return 1.0
    c = str(cur).strip().upper()
    rate_map = {
        "USD": st.session_state.get("USDSEK", 10.50),
        "EUR": st.session_state.get("EURSEK", 11.50),
        "CAD": st.session_state.get("CADSEK", 7.80),
        "NOK": st.session_state.get("NOKSEK", 1.00),
        "SEK": 1.0,
    }
    try:
        return float(rate_map.get(c, 1.0))
    except:
        return 1.0

# ── Yahoo Finance-hämtning (med TTM-utdelning) ─────────────────────────────
def hamta_yahoo(ticker: str) -> dict:
    t = yf.Ticker(ticker)

    # Info
    info = {}
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}

    # Pris (lokal) med fallback
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

    # Valuta
    currency = (info.get("currency") or "").upper()
    if not currency:
        try:
            currency = (t.fast_info.get("currency") or "").upper()
        except Exception:
            currency = "SEK"

    # Utdelning/år (lokal) – TTM över dividends; fallback: forward/trailing
    div_rate_local = 0.0
    last_ex_date = ""

    try:
        divs = t.dividends  # Series med lokala belopp
        if divs is not None and not divs.empty:
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
            div_rate_local = float(divs[divs.index >= cutoff].sum())
            last_ex_date = pd.to_datetime(divs.index.max()).strftime("%Y-%m-%d")
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

    if not last_ex_date:
        try:
            ts = info.get("exDividendDate")
            if ts not in (None, "", 0):
                last_ex_date = pd.to_datetime(int(ts), unit="s", utc=True).strftime("%Y-%m-%d")
        except Exception:
            last_ex_date = ""

    # DA = utd/år / pris
    dy_pct = 0.0
    if price > 0 and div_rate_local > 0:
        dy_pct = round(100.0 * div_rate_local / price, 2)

    return {
        "Bolagsnamn": info.get("longName") or info.get("shortName") or "",
        "Aktuell kurs": price,
        "Valuta": currency,
        "Utdelning/år": div_rate_local,
        "Direktavkastning (%)": dy_pct,
        "Ex-Date": last_ex_date,
        "Senaste uppdatering": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Källa": "Yahoo Finance",
    }

# ── Beräkningar (räkna DA om, SEK, portföljandel, nästa utbetalning) ──────
def beräkna(df: pd.DataFrame) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()

    d["Aktuell kurs"]   = pd.to_numeric(d["Aktuell kurs"], errors="coerce").fillna(0.0).astype(float)
    d["Utdelning/år"]   = pd.to_numeric(d["Utdelning/år"], errors="coerce").fillna(0.0).astype(float)
    d["Antal aktier"]   = pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0).astype(float)
    d["GAV"]            = pd.to_numeric(d["GAV"], errors="coerce").fillna(0.0).astype(float)
    d["Frekvens/år"]    = pd.to_numeric(d["Frekvens/år"], errors="coerce").fillna(0.0).astype(float).replace(0, 4)
    d["Payment-lag (dagar)"] = pd.to_numeric(d["Payment-lag (dagar)"], errors="coerce").fillna(0.0).astype(float).replace(0, 30)

    # SEK-kurs
    rates = d["Valuta"].apply(fx_for).astype(float)
    d["Kurs (SEK)"] = (d["Aktuell kurs"] * rates).astype(float).round(6)

    # Årsutdelning SEK
    d["Årlig utdelning (SEK)"] = (d["Antal aktier"] * d["Utdelning/år"] * rates).round(2)

    # Direktavkastning (%) = omräknad (inte Yahoo-fält)
    d["Direktavkastning (%)"] = 0.0
    ok = (d["Aktuell kurs"] > 0) & (d["Utdelning/år"] > 0)
    d.loc[ok, "Direktavkastning (%)"] = (100.0 * d.loc[ok, "Utdelning/år"] / d.loc[ok, "Aktuell kurs"]).round(2)

    # Portföljandel
    mv = (d["Antal aktier"] * d["Kurs (SEK)"]).round(2)
    tot_mv = float(mv.sum()) if mv.sum() else 1.0
    d["Portföljandel (%)"] = (100.0 * mv / tot_mv).round(2)

    # Nästa utbetalning (est)
    def next_pay(ex_date_str, freq_per_year, payment_lag_days):
        ts = pd.to_datetime(ex_date_str, errors="coerce")
        if pd.isna(ts):
            return ""
        exd = ts.date()
        try:
            freq = int(float(freq_per_year))
        except Exception:
            freq = 4
        try:
            lag = int(float(payment_lag_days))
        except Exception:
            lag = 30
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

# ── Lägg till/uppdatera EN (in-memory) ─────────────────────────────────────
def add_or_update_ticker_row(ticker: str) -> pd.DataFrame:
    base = säkerställ_kolumner(st.session_state["working_df"])
    ticker = (ticker or "").strip().upper()
    if not ticker:
        st.warning("Ange en ticker.")
        return base
    if not (base["Ticker"] == ticker).any():
        base = pd.concat([base, pd.DataFrame([{"Ticker": ticker, "Antal aktier": 0.0, "GAV": 0.0}])], ignore_index=True)
    try:
        vals = hamta_yahoo(ticker)
        for k, v in vals.items():
            base.loc[base["Ticker"] == ticker, k] = v
        base = beräkna(base)
        st.success(f"Ticker {ticker} uppdaterad (in-memory).")
    except Exception as e:
        st.error(f"Kunde inte hämta {ticker}: {e}")
    return base

# ── Uppdatera FLERA/ALLA (1 s paus, in-memory) ─────────────────────────────
def update_some_tickers(tickers: list) -> pd.DataFrame:
    base = säkerställ_kolumner(st.session_state["working_df"])
    if not tickers:
        st.warning("Välj minst en ticker.")
        return base
    bar = st.progress(0.0)
    for i, tkr in enumerate(tickers, 1):
        try:
            vals = hamta_yahoo(tkr)
            for k, v in vals.items():
                base.loc[base["Ticker"] == tkr, k] = v
        except Exception as e:
            st.warning(f"{tkr}: misslyckades ({e})")
        time.sleep(1.0)
        bar.progress(i/len(tickers))
    base = beräkna(base)
    st.success(f"Uppdaterade {len(tickers)} ticker(s) (in-memory).")
    return base

# ── Sidopanel (FX + uppdatera EN) ──────────────────────────────────────────
def sidopanel(df: pd.DataFrame):
    st.sidebar.header("⚙️ Inställningar")
    st.sidebar.markdown("**Växelkurser (SEK)**")
    st.session_state["USDSEK"] = float(st.sidebar.text_input("USD/SEK", value=str(st.session_state.get("USDSEK", 10.50))))
    st.session_state["EURSEK"] = float(st.sidebar.text_input("EUR/SEK", value=str(st.session_state.get("EURSEK", 11.50))))
    st.session_state["CADSEK"] = float(st.sidebar.text_input("CAD/SEK", value=str(st.session_state.get("CADSEK", 7.80))))
    st.session_state["NOKSEK"] = float(st.sidebar.text_input("NOK/SEK", value=str(st.session_state.get("NOKSEK", 1.00))))
    st.sidebar.markdown("---")
    one_ticker = st.sidebar.text_input("Uppdatera EN ticker (Yahoo)", placeholder="t.ex. EPD")
    if st.sidebar.button("🔄 Uppdatera EN"):
        st.session_state["working_df"] = add_or_update_ticker_row(one_ticker)

# ── Snabbverktyg (EN / FLERA / ALLA) ───────────────────────────────────────
def block_quick_update(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("⚡ Snabbverktyg – lägg till/uppdatera från Yahoo (in-memory)")
    c1, c2 = st.columns([1.6, 1])
    with c1:
        t_single = st.text_input("Lägg till/uppdatera EN ticker", placeholder="t.ex. XOM, VICI, FTS")
    with c2:
        if st.button("🚀 Hämta & spara EN"):
            df = add_or_update_ticker_row(t_single)

    st.caption("Välj några av dina befintliga tickers att uppdatera (eller klicka 'ALLA').")
    valbara = df["Ticker"].astype(str).tolist() if not df.empty else []
    selection = st.multiselect("Tickers att uppdatera", options=valbara)

    cA, cB = st.columns(2)
    with cA:
        if st.button("🔁 Uppdatera valda"):
            df = update_some_tickers(selection)
    with cB:
        if st.button("🌀 Uppdatera ALLA"):
            df = update_some_tickers(valbara)
    return df

# ── Portföljöversikt (redigerbar) ──────────────────────────────────────────
def block_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("📦 Portföljöversikt")
    d = beräkna(df).copy()

    d["Marknadsvärde (SEK)"] = (pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0) *
                                pd.to_numeric(d["Kurs (SEK)"], errors="coerce").fillna(0.0)).round(2)
    d["Insatt (SEK)"] = (pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0) *
                         pd.to_numeric(d["GAV"], errors="coerce").fillna(0.0)).round(2)
    d["Orealiserad P/L (SEK)"] = (d["Marknadsvärde (SEK)"] - d["Insatt (SEK)"]).round(2)
    d["Orealiserad P/L (%)"] = (100.0 * d["Orealiserad P/L (SEK)"] / d["Insatt (SEK)"].replace({0: pd.NA})).fillna(0.0).round(2)

    tot_mv  = float(d["Marknadsvärde (SEK)"].sum())
    tot_ins = float(d["Insatt (SEK)"].sum())
    tot_pl  = float(d["Orealiserad P/L (SEK)"].sum())
    tot_div = float(d["Årlig utdelning (SEK)"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portföljvärde", f"{round(tot_mv,2):,}".replace(",", " "))
    c2.metric("Insatt kapital", f"{round(tot_ins,2):,}".replace(",", " "))
    c3.metric("Orealiserad P/L", f"{round(tot_pl,2):,}".replace(",", " "), delta=f"{(0 if tot_ins==0 else (100*tot_pl/tot_ins)):.2f}%")
    c4.metric("Årsutdelning", f"{round(tot_div,2):,}".replace(",", " "))

    edit_cols = ["Antal aktier", "GAV", "Frekvens/år", "Payment-lag (dagar)"]
    show_cols = [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Kurs (SEK)",
        "Antal aktier","GAV","Insatt (SEK)","Marknadsvärde (SEK)",
        "Orealiserad P/L (SEK)","Orealiserad P/L (%)",
        "Direktavkastning (%)","Utdelning/år","Årlig utdelning (SEK)",
        "Ex-Date","Nästa utbetalning (est)","Portföljandel (%)","Senaste uppdatering"
    ]
    view = d[show_cols].copy()
    edited = st.data_editor(view, hide_index=True, num_rows="dynamic", use_container_width=True)

    with st.expander("➕ Lägg till/uppdatera rad manuellt (in-memory)"):
        col1, col2, col3 = st.columns([1.2, 1, 1])
        with col1: t_new = st.text_input("Ticker", placeholder="t.ex. VZ")
        with col2: qty_new = st.number_input("Antal", min_value=0, value=0, step=1)
        with col3: gav_new = st.number_input("GAV (SEK)", min_value=0.0, value=0.0, step=0.01)

        cA, cB = st.columns(2)
        with cA:
            if st.button("💾 Spara rad (utan Yahoo)"):
                base = säkerställ_kolumner(st.session_state["working_df"])
                t_norm = (t_new or "").strip().upper()
                if t_norm:
                    if (base["Ticker"] == t_norm).any():
                        i = base.index[base["Ticker"] == t_norm][0]
                        base.at[i, "Antal aktier"] = qty_new
                        base.at[i, "GAV"] = gav_new
                    else:
                        base = pd.concat([base, pd.DataFrame([{"Ticker": t_norm, "Antal aktier": qty_new, "GAV": gav_new}])], ignore_index=True)
                    base = beräkna(base)
                    st.session_state["working_df"] = base
                    st.success(f"Sparade {t_norm} (in-memory).")
                    d = base
        with cB:
            if st.button("🌐 Hämta Yahoo & spara rad"):
                base = add_or_update_ticker_row(t_new)
                if (t_new or "").strip():
                    mask = base["Ticker"] == (t_new.strip().upper())
                    if mask.any():
                        base.loc[mask, "Antal aktier"] = qty_new
                        base.loc[mask, "GAV"] = gav_new
                        base = beräkna(base)
                st.session_state["working_df"] = base
                d = base

    if st.button("💾 Spara ändringar (in-memory)"):
        base = säkerställ_kolumner(st.session_state["working_df"])
        for _, r in edited.iterrows():
            t = str(r["Ticker"]).upper()
            mask = base["Ticker"].astype(str).str.upper() == t
            if not mask.any():
                continue
            for c in edit_cols:
                base.loc[mask, c] = r[c]
        base = beräkna(base)
        st.session_state["working_df"] = base
        st.success("Ändringar sparade (i appens minne).")
        return base

    return d

# ── Toppkort & ranking ─────────────────────────────────────────────────────
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
            f"- Utd/år (lokal): **{top['Utdelning/år']}**  \n"
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
    cols = [
        "Ticker","Bolagsnamn","Valuta","Kurs (SEK)","Direktavkastning (%)",
        "Utdelning/år","Årlig utdelning (SEK)",
        "Ex-Date","Nästa utbetalning (est)","Portföljandel (%)","Senaste uppdatering"
    ]
    st.dataframe(d[cols], use_container_width=True)

# ── Trading (Köp/Sälj in-memory med avgifter) ──────────────────────────────
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

def block_trading(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🛒 Köp / 📤 Sälj (avgifter, in-memory)")
    if df.empty:
        st.info("Lägg till minst en ticker först."); 
        return df

    tickers = df["Ticker"].astype(str).tolist()
    tkr = st.selectbox("Ticker", options=tickers)
    side = st.radio("Typ", ["KÖP", "SÄLJ"], horizontal=True)
    qty  = st.number_input("Antal", min_value=1, value=10, step=1)
    px_local = st.number_input("Pris (lokal valuta)", min_value=0.0, value=10.0, step=0.01)
    ccy_default = df.loc[df["Ticker"]==tkr, "Valuta"].iloc[0] if (df["Ticker"]==tkr).any() else "SEK"
    ccy = st.selectbox("Valuta", options=["SEK","USD","EUR","CAD","NOK"],
                       index=["SEK","USD","EUR","CAD","NOK"].index(ccy_default) if ccy_default in ["SEK","USD","EUR","CAD","NOK"] else 0)

    fx_rate = fx_for(ccy)
    px_sek = round(px_local * fx_rate, 6)
    gross  = round(px_sek * qty, 2)
    fee_court, fee_fx, fee_tot = calc_fees(gross, is_foreign(ccy))
    net = round(gross + fee_tot, 2) if side == "KÖP" else round(gross - fee_tot, 2)

    st.caption(f"Pris (SEK): **{px_sek}** | Brutto: **{gross} SEK** | Courtage: **{fee_court}** | FX-avgift: **{fee_fx}** | {'Totalt' if side=='KÖP' else 'Nettolikvid'}: **{net} SEK**")

    if st.button("Lägg order i minnet"):
        base = säkerställ_kolumner(st.session_state["working_df"])
        i = base.index[base["Ticker"] == tkr][0]

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

        # köa transaktionen till sparknapp
        if "pending_txs" not in st.session_state:
            st.session_state["pending_txs"] = []
        st.session_state["pending_txs"].append({
            "Tid": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Typ": side, "Ticker": tkr, "Antal": qty,
            "Pris (lokal)": px_local, "Valuta": ccy, "FX": fx_rate,
            "Pris (SEK)": px_sek, "Belopp (SEK)": gross,
            "Courtage (SEK)": fee_court, "FX-avgift (SEK)": fee_fx,
            "Tot.avgifter (SEK)": fee_tot, "Kommentar": "in-memory"
        })

        base = beräkna(base)
        st.session_state["working_df"] = base
        st.success(f"{side} registrerad i minnet. Tryck 'Spara till Google Sheets' för att skriva till arket.")
        return base

    if "pending_txs" in st.session_state and st.session_state["pending_txs"]:
        st.info(f"Ej sparade transaktioner: {len(st.session_state['pending_txs'])} st")
        st.dataframe(pd.DataFrame(st.session_state["pending_txs"]), use_container_width=True)

    return df

# ── Transaktionsark + sparning ─────────────────────────────────────────────
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

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    st.title("Relative Yield – utdelningsportfölj")

    # Initiera in-memory arbetskopia
    if "working_df" not in st.session_state:
        st.session_state["working_df"] = migrate_sheet_columns()

    base = st.session_state["working_df"]

    # Sidopanel
    sidopanel(base)

    # Snabbverktyg
    st.divider()
    base = block_quick_update(base)

    # Toppkort
    st.divider()
    block_top_card(base)

    # Portfölj
    st.divider()
    base = block_portfolio(base)

    # Trading
    st.divider()
    base = block_trading(base)

    # Ranking
    st.divider()
    block_ranking(base)

    # Uppdatera in-memory
    st.session_state["working_df"] = base

    # MANUELL SPAR-KNAPP
    st.divider()
    if st.button("💾 Spara till Google Sheets"):
        spara_df( beräkna(st.session_state["working_df"]) )
        save_pending_transactions()
        st.success("Data och transaktioner sparade till Google Sheets!")

if __name__ == "__main__":
    main()
