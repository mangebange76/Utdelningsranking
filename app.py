import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials

# --- SIDKONFIGURATION ---
st.set_page_config(page_title="Utdelningsranking", layout="wide")

# --- GOOGLE SHEETS KONFIG ---
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Bolag"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_df():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_df(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# --- SÄKERSTÄLLA KOLUMNER ---
def säkerställ_kolumner(df):
    kolumner = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Direktavkastning (%)", "Utdelning/år",
        "Frekvens/år", "Payment-lag (dagar)", "Ex-Date", "Nästa utbetalning (est)",
        "Antal aktier", "GAV", "Portföljandel (%)", "Årlig utdelning (SEK)",
        "Kurs (SEK)", "Utdelningstillväxt (%)", "Senaste uppdatering", "Källa"
    ]
    for kol in kolumner:
        if kol not in df.columns:
            df[kol] = ""
    return df[kolumner]

# --- VALUTAKARTAN ---
def fx_for(cur):
    c = (cur or "").upper()
    if c == "SEK":
        return 1.0
    fx_map = {
        "USD": st.session_state.get("USDSEK", 10.5),
        "EUR": st.session_state.get("EURSEK", 11.5),
        "CAD": st.session_state.get("CADSEK", 7.8),
        "NOK": st.session_state.get("NOKSEK", 1.0)
    }
    return fx_map.get(c, 1.0)

# --- HÄMTA FRÅN YAHOO FINANCE ---
def hamta_yahoo(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}

    def pct(x):
        if x in (None, "", 0): return 0.0
        try: return round(float(x) * 100.0, 2)
        except: return 0.0

    def ts_to_date(ts):
        if ts in (None, "", 0): return ""
        try: return pd.to_datetime(int(ts), unit="s", utc=True).strftime("%Y-%m-%d")
        except: return ""

    # Pris (robust)
    price = None
    try:
        price = t.fast_info.get("last_price")
    except Exception:
        pass
    if price in (None, ""):
        price = info.get("currentPrice")
    if price in (None, ""):
        try:
            h = t.history(period="1d")
            if not h.empty:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None

    # Utdelning per aktie (årlig, lokal valuta) – "forwardAnnualDividendRate"
    forward_div_rate = float(info.get("forwardAnnualDividendRate") or 0.0)

    data = {
        "Bolagsnamn": info.get("longName") or info.get("shortName") or "",
        "Aktuell kurs": float(price) if price not in (None, "") else 0.0,
        "Valuta": (info.get("currency") or "").upper(),
        "Direktavkastning (%)": pct(info.get("forwardAnnualDividendYield", info.get("dividendYield"))),
        "Utdelning/år": forward_div_rate,  # per aktie, i lokal valuta
        "Ex-Date": ts_to_date(info.get("exDividendDate")),
        "Senaste uppdatering": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Källa": "Yahoo Finance",
    }
    return data

# --- BERÄKNINGAR (SEK, portfölj, utdelningar, nästa utbetalning) ---
def beräkna(df: pd.DataFrame) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()

    # Typer / defaults
    d["Ticker"] = d["Ticker"].astype(str).str.strip().str.upper()
    d["Valuta"] = d["Valuta"].astype(str).str.strip().str.upper()

    # Numerik
    for col in ["Aktuell kurs","Utdelning/år","Antal aktier","GAV","Frekvens/år","Payment-lag (dagar)"]:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    # SEK-konvertering
    d["Kurs (SEK)"] = (d["Aktuell kurs"] * d["Valuta"].apply(fx_for)).round(6)

    # Årlig utdelning i SEK = antal * utd/år (lokal) * FX
    d["Årlig utdelning (SEK)"] = (d["Antal aktier"] * d["Utdelning/år"] * d["Valuta"].apply(fx_for)).round(2)

    # Portföljandel = MV / total MV
    mv = (d["Antal aktier"] * d["Kurs (SEK)"]).round(2)
    d["Portföljandel (%)"] = (100.0 * mv / max(mv.sum(), 1.0)).round(2)

    # Nästa utbetalning (est) = Ex-date framskjuten per frekvens + payment-lag
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

        today_d = datetime.today().date()
        while exd < today_d:
            exd = exd + timedelta(days=step_days)
        pay_date = exd + timedelta(days=lag)
        return pay_date.strftime("%Y-%m-%d")

    # Se till att Frekvens/år & Payment-lag finns (default 4 & 30)
    d["Frekvens/år"] = d["Frekvens/år"].replace(0, 4)
    d["Payment-lag (dagar)"] = d["Payment-lag (dagar)"].replace(0, 30)

    d["Nästa utbetalning (est)"] = [
        next_pay(d.at[i, "Ex-Date"], d.at[i, "Frekvens/år"], d.at[i, "Payment-lag (dagar)"]) for i in d.index
    ]

    return d

# --- MIGRATION / SÄKER START ---
def migrate_sheet_columns():
    try:
        df = hamta_df()
    except Exception:
        df = pd.DataFrame()
    df2 = säkerställ_kolumner(df)
    if list(df.columns) != list(df2.columns) or df.shape[1] != df2.shape[1]:
        spara_df(df2)  # skriv rubriker i rätt ordning om något saknas/fel
    return df2

# --- LÄGG TILL/UPPDATERA EN TICKER (med Yahoo) ---
def add_or_update_ticker_row(ticker: str) -> pd.DataFrame:
    ticker = (ticker or "").strip().upper()
    if not ticker:
        st.warning("Ange en ticker."); 
        return säkerställ_kolumner(hamta_df())
    base = migrate_sheet_columns()
    if not (base["Ticker"] == ticker).any():
        base = pd.concat([base, pd.DataFrame([{"Ticker": ticker, "Antal aktier": 0, "GAV": 0.0}])], ignore_index=True)
    # hämta Yahoo och uppdatera fält
    try:
        vals = hamta_yahoo(ticker)
        for k, v in vals.items():
            base.loc[base["Ticker"] == ticker, k] = v
        base = beräkna(base)
        spara_df(base)
        st.success(f"Ticker {ticker} tillagd/uppdaterad.")
    except Exception as e:
        st.error(f"Kunde inte hämta {ticker}: {e}")
    return base

# --- UPPDATERA FLERA / ALLA TICKERS (1s paus) ---
def update_some_tickers(tickers: list) -> pd.DataFrame:
    if not tickers:
        st.warning("Välj minst en ticker.")
        return säkerställ_kolumner(hamta_df())
    base = migrate_sheet_columns()
    bar = st.progress(0.0)
    for i, tkr in enumerate(tickers, 1):
        try:
            vals = hamta_yahoo(tkr)
            for k, v in vals.items():
                base.loc[base["Ticker"] == tkr, k] = v
        except Exception as e:
            st.warning(f"{tkr}: misslyckades ({e})")
        time.sleep(1.0)  # ← 1 sekund mellan varje anrop
        bar.progress(i/len(tickers))
    base = beräkna(base)
    spara_df(base)
    st.success(f"Uppdaterade {len(tickers)} ticker(s).")
    return base

# === SIDOPANEL (FX + snabbuppdatering av en ticker) ===
def sidopanel(df: pd.DataFrame):
    st.sidebar.header("⚙️ Inställningar")

    # Växelkurser (lagras i session_state)
    st.sidebar.markdown("**Växelkurser (SEK)**")
    st.session_state["USDSEK"] = float(st.sidebar.text_input("USD/SEK", value=str(st.session_state.get("USDSEK", 10.50))))
    st.session_state["EURSEK"] = float(st.sidebar.text_input("EUR/SEK", value=str(st.session_state.get("EURSEK", 11.50))))
    st.session_state["CADSEK"] = float(st.sidebar.text_input("CAD/SEK", value=str(st.session_state.get("CADSEK", 7.80))))
    st.session_state["NOKSEK"] = float(st.sidebar.text_input("NOK/SEK", value=str(st.session_state.get("NOKSEK", 1.00))))

    st.sidebar.markdown("---")
    one_ticker = st.sidebar.text_input("Uppdatera EN ticker (Yahoo)", placeholder="t.ex. EPD")
    if st.sidebar.button("🔄 Uppdatera EN"):
        _ = add_or_update_ticker_row(one_ticker)


# === SNABBVERKTYG: lägg till/uppdatera EN, uppdatera FLERA/ALLA (1 s paus) ===
def block_quick_update(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("⚡ Snabbverktyg – lägg till/uppdatera från Yahoo")
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


# === PORTFÖLJVY (redigerbar) ===
def block_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("📦 Portföljöversikt")
    d = beräkna(df).copy()

    # Beräkna totalsiffror
    d["Marknadsvärde (SEK)"] = (pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0) *
                                pd.to_numeric(d["Kurs (SEK)"], errors="coerce").fillna(0.0)).round(2)
    d["Insatt (SEK)"] = (pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0) *
                         pd.to_numeric(d["GAV"], errors="coerce").fillna(0.0)).round(2)
    d["Orealiserad P/L (SEK)"] = (d["Marknadsvärde (SEK)"] - d["Insatt (SEK)"]).round(2)
    d["Orealiserad P/L (%)"] = (
        100.0 * d["Orealiserad P/L (SEK)"] / d["Insatt (SEK)"].replace({0: pd.NA})
    ).fillna(0.0).round(2)

    tot_mv  = float(d["Marknadsvärde (SEK)"].sum())
    tot_ins = float(d["Insatt (SEK)"].sum())
    tot_pl  = float(d["Orealiserad P/L (SEK)"].sum())
    tot_div = float(d["Årlig utdelning (SEK)"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portföljvärde", f"{round(tot_mv,2):,}".replace(",", " "))
    c2.metric("Insatt kapital", f"{round(tot_ins,2):,}".replace(",", " "))
    c3.metric("Orealiserad P/L", f"{round(tot_pl,2):,}".replace(",", " "),
              delta=f"{(0 if tot_ins==0 else (100*tot_pl/tot_ins)):.2f}%")
    c4.metric("Årsutdelning", f"{round(tot_div,2):,}".replace(",", " "))

    # Redigerbara kolumner
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

    # Snabb add/uppdatera rad manuellt (t.ex. innan Yahoo-data finns)
    with st.expander("➕ Lägg till/uppdatera rad manuellt"):
        col1, col2, col3 = st.columns([1.2, 1, 1])
        with col1:
            t_new = st.text_input("Ticker", placeholder="t.ex. VZ")
        with col2:
            qty_new = st.number_input("Antal", min_value=0, value=0, step=1)
        with col3:
            gav_new = st.number_input("GAV (SEK)", min_value=0.0, value=0.0, step=0.01)

        cA, cB = st.columns(2)
        with cA:
            if st.button("💾 Spara rad (utan Yahoo)"):
                base = säkerställ_kolumner(hamta_df())
                t_norm = (t_new or "").strip().upper()
                if t_norm:
                    if (base["Ticker"] == t_norm).any():
                        i = base.index[base["Ticker"] == t_norm][0]
                        base.at[i, "Antal aktier"] = qty_new
                        base.at[i, "GAV"] = gav_new
                    else:
                        base = pd.concat([base, pd.DataFrame([{"Ticker": t_norm, "Antal aktier": qty_new, "GAV": gav_new}])], ignore_index=True)
                    base = beräkna(base)
                    spara_df(base)
                    st.success(f"Sparade {t_norm}.")
                    d = base
        with cB:
            if st.button("🌐 Hämta Yahoo & spara rad"):
                d = add_or_update_ticker_row(t_new)
                if (t_new or "").strip():
                    base = säkerställ_kolumner(hamta_df())
                    mask = base["Ticker"] == (t_new.strip().upper())
                    if mask.any():
                        base.loc[mask, "Antal aktier"] = qty_new
                        base.loc[mask, "GAV"] = gav_new
                        base = beräkna(base); spara_df(base)
                        d = base

    if st.button("💾 Spara ändringar (antal, GAV, frekvens, lag)"):
        base = säkerställ_kolumner(hamta_df())
        # Synka tillbaka endast redigerbara fält
        for _, r in edited.iterrows():
            t = r["Ticker"]
            mask = base["Ticker"].astype(str).str.upper() == str(t).upper()
            if not mask.any(): 
                continue
            for c in edit_cols:
                base.loc[mask, c] = r[c]
        base = beräkna(base)
        spara_df(base)
        st.success("Sparat.")
        return base

    return d


# === TOPPKORT & RANKING ===
def block_top_card(df: pd.DataFrame):
    d = beräkna(df)
    if d.empty:
        st.info("Ingen data ännu. Lägg till tickers och uppdatera från Yahoo.")
        return
    # Top: högst direktavkastning (enkel signal)
    d["Direktavkastning (%)"] = pd.to_numeric(d["Direktavkastning (%)"], errors="coerce").fillna(0.0)
    top = d.sort_values("Direktavkastning (%)", ascending=False).iloc[0]
    c1, c2, c3 = st.columns([1.6, 1, 1])
    with c1:
        st.subheader(f"🏆 Högst DA: **{top['Ticker']}** — {top.get('Bolagsnamn','')}")
        st.write(
            f"- Direktavkastning: **{top['Direktavkastning (%)']:.2f}%**  \n"
            f"- Utd/år (lokal): **{top['Utdelning/år']}**  \n"
            f"- Ex-Date: **{top.get('Ex-Date','')}**, nästa utbetalning est: **{top.get('Nästa utbetalning (est)','')}**"
        )
    with c2:
        st.metric("Kurs (SEK)", f"{top.get('Kurs (SEK)','')}")
        st.metric("Årsutd (SEK)", f"{top.get('Årlig utdelning (SEK)','')}")
    with c3:
        st.metric("Valuta", top.get("Valuta",""))
        st.metric("Senast uppdaterad", top.get("Senaste uppdatering",""))

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

def main():
    st.title("Relative Yield – utdelningsportfölj")

    # 1) Säkerställ kolumner i arket och ladda data
    base = migrate_sheet_columns()

    # 2) Sidopanel (FX + snabb "uppdatera EN" från Yahoo)
    sidopanel(base)

    # 3) Snabbverktyg: lägg till/uppdatera EN / uppdatera FLERA/ALLA (1s paus mellan anrop)
    st.divider()
    base = block_quick_update(base)

    # 4) Toppkort (visar högst DA just nu)
    st.divider()
    block_top_card(base)

    # 5) Portföljöversikt (redigerbar: antal, GAV, frekvens, payment-lag + manuell add)
    st.divider()
    base = block_portfolio(base)

    # 6) Ranking
    st.divider()
    block_ranking(base)

if __name__ == "__main__":
    main()
