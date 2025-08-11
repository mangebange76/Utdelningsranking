import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials
from datetime import timedelta, date

st.set_page_config(page_title="Utdelningsranking", layout="wide")

# === Konstanter ===
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Bolag"

# Courtage och växlingsavgifter (standard mini Avanza/Nordnet)
DEFAULT_FX = {
    "USD/SEK": 10.50,
    "CAD/SEK": 7.80,
    "NOK/SEK": 1.00,
    "EUR/SEK": 11.50
}
DEFAULT_COURTAGE = {
    "Avanza": 0.0025,  # 0.25%
    "Nordnet": 0.0025  # 0.25%
}
DEFAULT_VXL = 0.0025  # 0.25% växlingsavgift

# Google Sheets-autentisering
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# === Koppling ===
def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

# === Hämta data från Google Sheets ===
def hamta_df():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# === Spara data till Google Sheets ===
def spara_df_säkert(df: pd.DataFrame):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.tolist()] + df.astype(str).values.tolist())

# === Säkerställ kolumner ===
def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs", "Forward Yield (%)",
        "5Y Avg Yield (%)", "Forward Div Rate", "Trailing Div Rate", "Payout Ratio (%)",
        "Ex-Date", "Frekvens/år", "Payment-lag (dagar)", "Antal",
        "Kurs (SEK)", "Marknadsvärde (SEK)", "Utd/aktie (SEK)",
        "Årsutdelning (SEK)", "Månadsutdelning (SEK)", "Relative Yield (x)",
        "Dagar till X-dag", "Poäng", "Rank", "Nästa utbetalning (est)",
        "Nuvarande vikt (%)", "Senast uppdaterad"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

# === Yahoo Finance: hämta fält per ticker ===
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

    return {
        "Bolagsnamn": info.get("longName") or info.get("shortName") or "",
        "Valuta": info.get("currency") or "",
        "Aktuell kurs": float(price) if price not in (None, "") else 0.0,
        "Forward Yield (%)": pct(info.get("forwardAnnualDividendYield", info.get("dividendYield"))),
        "5Y Avg Yield (%)": pct(info.get("fiveYearAvgDividendYield")),
        "Forward Div Rate": float(info.get("forwardAnnualDividendRate") or 0.0),
        "Trailing Div Rate": float(info.get("trailingAnnualDividendRate") or 0.0),
        "Payout Ratio (%)": pct(info.get("payoutRatio")),
        "Ex-Date": ts_to_date(info.get("exDividendDate")),
        "Senast uppdaterad": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }


# === Beräkningar (SEK, utdelning, Relative Yield, X-dag, poäng) ===
def beräkna(df: pd.DataFrame, fx_map: dict) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()

    # FX-funktion
    def fx_for(cur: str) -> float:
        c = (cur or "").upper()
        if c == "USD": return float(fx_map.get("USD/SEK", DEFAULT_FX["USD/SEK"]))
        if c == "CAD": return float(fx_map.get("CAD/SEK", DEFAULT_FX["CAD/SEK"]))
        if c == "NOK": return float(fx_map.get("NOK/SEK", DEFAULT_FX["NOK/SEK"]))
        if c == "EUR": return float(fx_map.get("EUR/SEK", DEFAULT_FX["EUR/SEK"]))
        return 1.0  # SEK

    # SEK-konvertering & marknadsvärde
    d["Kurs (SEK)"] = (pd.to_numeric(d["Aktuell kurs"], errors="coerce").fillna(0.0) * d["Valuta"].apply(fx_for)).round(6)
    d["Marknadsvärde (SEK)"] = (pd.to_numeric(d["Antal"], errors="coerce").fillna(0.0) * d["Kurs (SEK)"]).round(2)

    # Utdelning i SEK
    d["Utd/aktie (SEK)"] = (pd.to_numeric(d["Forward Div Rate"], errors="coerce").fillna(0.0) * d["Valuta"].apply(fx_for)).round(6)
    d["Årsutdelning (SEK)"] = (pd.to_numeric(d["Antal"], errors="coerce").fillna(0.0) * d["Utd/aktie (SEK)"]).round(2)
    d["Månadsutdelning (SEK)"] = (d["Årsutdelning (SEK)"] / 12.0).round(2)

    # Relative Yield
    fy = pd.to_numeric(d["Forward Yield (%)"], errors="coerce").fillna(0.0)
    ay = pd.to_numeric(d["5Y Avg Yield (%)"], errors="coerce").fillna(0.0)
    with pd.option_context("mode.use_inf_as_na", True):
        d["Relative Yield (x)"] = (fy / ay.replace(0, pd.NA)).fillna(0.0).round(2)

    # Dagar till X-dag
    def days_to_ex(s):
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts): return 99999
        ex = ts.date()
        today = pd.Timestamp.utcnow().date()
        if ex < today: return 99999
        return (ex - today).days
    d["Dagar till X-dag"] = d["Ex-Date"].apply(days_to_ex)

    # Poäng (Relative Yield + liten bonus om ex-dag inom 14 dagar)
    x_bonus = d["Dagar till X-dag"].apply(lambda dd: 0.05 if 0 <= dd <= 14 else 0.0)
    d["Poäng"] = (pd.to_numeric(d["Relative Yield (x)"], errors="coerce").fillna(0.0) + x_bonus).round(3)

    # Ranking
    d = d.sort_values(by=["Poäng", "Dagar till X-dag"], ascending=[False, True]).reset_index(drop=True)
    d["Rank"] = d.index + 1

    # Robust "Nästa utbetalning (est)" = Ex-date framskjuten per frekvens + payment-lag
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

    # Nuvarande vikt
    tot_mv = max(d["Marknadsvärde (SEK)"].sum(), 1.0)
    d["Nuvarande vikt (%)"] = (100.0 * d["Marknadsvärde (SEK)"] / tot_mv).round(2)

    return d


# === Sidopanel: FX-inställningar, tickers & Uppdatera alla ===
def sidopanel(df: pd.DataFrame):
    st.sidebar.header("⚙️ Inställningar")

    # Tickers
    tickers_default = ",".join(df["Ticker"].astype(str).tolist()) if not df.empty else "EPD,VICI,FTS,XOM,CVX,VZ,MO,USB,MGA,AMCR"
    tickers_str = st.sidebar.text_area("Tickers (komma-separerade)", value=tickers_default)

    # Växelkurser
    st.sidebar.markdown("**Växelkurser**")
    fx_map = {}
    for k, v in DEFAULT_FX.items():
        fx_map[k] = float(st.sidebar.text_input(k, value=str(v)))

    do_update = st.sidebar.button("🔄 Uppdatera alla från Yahoo")
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    return tickers, fx_map, do_update


# === Säkerställ att rader finns för alla tickers ===
def säkerställ_tickers(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    d = säkerställ_kolumner(df).copy()
    for t in tickers:
        if not (d["Ticker"] == t).any():
            d = pd.concat([d, pd.DataFrame([{"Ticker": t, "Antal": 0}])], ignore_index=True)
    d = d[d["Ticker"].astype(bool)].reset_index(drop=True)
    return d


# === Massuppdatering (Yahoo) ===
def block_uppdatera_alla(df: pd.DataFrame, tickers: list, fx_map: dict) -> pd.DataFrame:
    if not tickers:
        st.warning("Lägg till minst en ticker först.")
        return df
    st.info("Uppdaterar – undvik för många samtidigt (Yahoo kan rate-limita).")
    bar = st.progress(0.0)
    d = df.copy()
    for i, tkr in enumerate(tickers, 1):
        try:
            vals = hamta_yahoo(tkr)
            for k, v in vals.items():
                d.loc[d["Ticker"] == tkr, k] = v
        except Exception as e:
            st.warning(f"{tkr}: misslyckades ({e})")
        time.sleep(0.7)
        bar.progress(i / len(tickers))
    d = beräkna(d, fx_map)
    spara_df_säkert(d)
    st.success("Uppdaterat och sparat.")
    return d

# === UI: toppkort (mest attraktiv just nu) ===
def block_top_card(df: pd.DataFrame):
    if df.empty:
        st.info("Ingen data ännu. Lägg in tickers i sidopanelen och klicka 'Uppdatera alla'.")
        return
    top = df.iloc[0]
    c1, c2, c3 = st.columns([1.6, 1, 1])
    with c1:
        st.subheader(f"🏆 Mest attraktiv: **{top['Ticker']}** — {top.get('Bolagsnamn','')}")
        st.write(
            f"- Relative Yield: **{top['Relative Yield (x)']}x**  \n"
            f"- Forward: **{top['Forward Yield (%)']}%**, 5Y Avg: **{top['5Y Avg Yield (%)']}%**  \n"
            f"- Ex-Date: **{top.get('Ex-Date','')}** (om {int(top.get('Dagar till X-dag', 99999))} dagar)"
        )
    with c2:
        st.metric("Kurs (SEK)", f"{top.get('Kurs (SEK)','')}")
        st.metric("Årsutd (SEK)", f"{top.get('Årsutdelning (SEK)','')}")
    with c3:
        st.metric("Poäng", f"{top.get('Poäng','')}")
        st.metric("Rank", f"{top.get('Rank','')}")

# === UI: portfölj (redigera Antal & GAV) ===
def block_portfolio(df: pd.DataFrame, fx_map: dict) -> pd.DataFrame:
    st.subheader("📦 Portfölj")
    # Beräkna uppdaterade värden innan visning
    df = beräkna(df, fx_map)

    # Summering
    total_mv = float(df["Marknadsvärde (SEK)"].sum())
    total_y  = float(df["Årsutdelning (SEK)"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Portföljvärde", f"{round(total_mv,2):,}".replace(",", " "))
    c2.metric("Årsutdelning", f"{round(total_y,2):,}".replace(",", " "))
    c3.metric("Utd/månad", f"{round(total_y/12.0,2):,}".replace(",", " "))

    # Redigerbar vy
    edit_cols = ["Antal", "GAV (SEK)", "Minvikt (%)", "Målvikt (%)", "Maxvikt (%)", "Frekvens/år", "Payment-lag (dagar)"]
    show_cols = [
        "Rank","Ticker","Bolagsnamn","Valuta","Aktuell kurs","Kurs (SEK)",
        "Antal","GAV (SEK)","Marknadsvärde (SEK)",
        "Utd/aktie (SEK)","Årsutdelning (SEK)","Månadsutdelning (SEK)",
        "Forward Yield (%)","5Y Avg Yield (%)","Relative Yield (x)",
        "Ex-Date","Dagar till X-dag","Nästa utbetalning (est)",
        "Minvikt (%)","Målvikt (%)","Maxvikt (%)","Nuvarande vikt (%)"
    ]
    view = df[show_cols].copy()
    edited = st.data_editor(view, hide_index=True, num_rows="dynamic", use_container_width=True)

    if st.button("💾 Spara portfölj (antal, GAV, vikter & schema)"):
        base = säkerställ_kolumner(hamta_df())
        for _, r in edited.iterrows():
            t = r["Ticker"]
            mask = base["Ticker"] == t
            if not mask.any():
                continue
            for c in edit_cols:
                base.loc[mask, c] = r[c]
        base = beräkna(base, fx_map)
        spara_df_säkert(base)
        st.success("Sparat.")
        return base

    return df

# === UI: rankingtabell ===
def block_ranking(df: pd.DataFrame):
    st.subheader("📊 Ranking")
    cols = [
        "Rank","Ticker","Bolagsnamn","Poäng",
        "Relative Yield (x)","Forward Yield (%)","5Y Avg Yield (%)",
        "Ex-Date","Dagar till X-dag","Nästa utbetalning (est)",
        "Kurs (SEK)","Årsutdelning (SEK)","Månadsutdelning (SEK)",
        "Nuvarande vikt (%)","Målvikt (%)","Minvikt (%)","Maxvikt (%)"
    ]
    st.dataframe(df[cols], use_container_width=True)

# === UI: enkel köp-simulering (utan avgifter – vi lägger på avgifter i Del 4) ===
def block_buy_sim(df: pd.DataFrame):
    st.subheader("🛒 Köp-simulering (enkel)")
    amount = st.number_input("Belopp (SEK)", min_value=0, value=900, step=100)
    mode = st.radio("Modell", ["1 bolag", "Fördela på topp 3"], horizontal=True)

    if st.button("🎯 Föreslå köp"):
        if amount <= 0 or df.empty:
            st.warning("Ange belopp och uppdatera data först.")
            return

        cand = df[df["Poäng"] > 0].sort_values(["Poäng", "Dagar till X-dag"], ascending=[False, True])
        if cand.empty:
            st.info("Inga kandidater med data.")
            return

        if mode == "1 bolag":
            r = cand.iloc[0]
            price = float(r["Kurs (SEK)"]) if r["Kurs (SEK)"] not in ("", None, "") else 0.0
            qty = int(amount // price) if price > 0 else 0
            cost = round(qty * price, 2)
            st.success(f"Köp **{qty} st {r['Ticker']}** (≈ {cost} SEK)")
            st.caption(f"Poäng {r['Poäng']}, RelYield {r['Relative Yield (x)']}x, Ex-Date {r['Ex-Date']}")
        else:
            picks = cand.head(3).copy()
            per = amount / len(picks)
            rows = []
            for _, x in picks.iterrows():
                price = float(x["Kurs (SEK)"]) if x["Kurs (SEK)"] not in ("", None, "") else 0.0
                qty = int(per // price) if price > 0 else 0
                cost = round(qty * price, 2)
                rows.append({
                    "Ticker": x["Ticker"], "Poäng": x["Poäng"], "Kurs (SEK)": price,
                    "Qty": qty, "Kostnad (SEK)": cost, "Ex-Date": x["Ex-Date"], "Dagar till X-dag": x["Dagar till X-dag"]
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# === Avgifter (Mini) ===
MIN_COURTAGE_RATE = 0.0025   # 0,25 %
MIN_COURTAGE_SEK  = 1.0      # minst 1 kr
FX_FEE_RATE       = 0.0025   # 0,25 % växling när ej SEK

def is_foreign(ccy: str) -> bool:
    return (str(ccy or "").upper() != "SEK")

def calc_fees(order_value_sek: float, foreign: bool):
    courtage = max(MIN_COURTAGE_RATE * order_value_sek, MIN_COURTAGE_SEK)
    fx_fee   = (FX_FEE_RATE * order_value_sek) if foreign else 0.0
    total    = round(courtage + fx_fee, 2)
    return round(courtage,2), round(fx_fee,2), total

# === Transaktionslogg (flik "Transaktioner") ===
TX_SHEET = "Transaktioner"

def ensure_tx_sheet():
    sh = skapa_koppling().spreadsheet
    import gspread
    try:
        ws_tx = sh.worksheet(TX_SHEET)
    except gspread.WorksheetNotFound:
        ws_tx = sh.add_worksheet(title=TX_SHEET, rows=1, cols=14)
        ws_tx.update([[
            "Tid","Typ","Ticker","Antal","Pris (lokal)","Valuta","FX (manuell)",
            "Pris (SEK)","Belopp (SEK)","Courtage (SEK)","FX-avgift (SEK)","Tot.avgifter (SEK)","Kommentar"
        ]])
    return ws_tx

def log_tx(tx_type, ticker, qty, px_local, ccy, fx_used, px_sek, gross_sek, fee_court, fee_fx, note=""):
    ws_tx = ensure_tx_sheet()
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    ws_tx.append_row([
        ts, tx_type, ticker, qty, px_local, ccy, fx_used,
        px_sek, gross_sek, fee_court, fee_fx, round(fee_court+fee_fx,2), note
    ])

# === Position-uppdatering ===
def update_position_after_buy(df: pd.DataFrame, idx: int, qty: int, total_cost_sek: float,
                              fee_court: float, fee_fx: float):
    old_qty = float(df.at[idx, "Antal"] or 0)
    old_gav = float(df.at[idx, "GAV (SEK)"] or 0)
    new_qty = old_qty + qty
    new_gav = 0.0 if new_qty == 0 else round(((old_gav * old_qty) + total_cost_sek) / new_qty, 6)

    df.at[idx, "Antal"] = new_qty
    df.at[idx, "GAV (SEK)"] = new_gav
    df.at[idx, "Sum courtage (SEK)"]  = round(float(df.at[idx, "Sum courtage (SEK)"] or 0) + fee_court, 2)
    df.at[idx, "Sum FX-avgift (SEK)"] = round(float(df.at[idx, "Sum FX-avgift (SEK)"] or 0) + fee_fx, 2)
    return df

def update_position_after_sell(df: pd.DataFrame, idx: int, qty: int,
                               fee_court: float, fee_fx: float):
    old_qty = float(df.at[idx, "Antal"] or 0)
    new_qty = old_qty - qty
    if new_qty < 0:
        raise ValueError("För stort sälj; fler aktier än du äger.")
    df.at[idx, "Antal"] = new_qty
    if new_qty == 0:
        df.at[idx, "GAV (SEK)"] = 0.0
    df.at[idx, "Sum courtage (SEK)"]  = round(float(df.at[idx, "Sum courtage (SEK)"] or 0) + fee_court, 2)
    df.at[idx, "Sum FX-avgift (SEK)"] = round(float(df.at[idx, "Sum FX-avgift (SEK)"] or 0) + fee_fx, 2)
    return df

# === UI: Köp (med avgifter & logg) ===
def block_buy(df: pd.DataFrame, fx_map: dict) -> pd.DataFrame:
    st.subheader("🛒 Köp (med Mini-courtage & växlingsavgift)")
    if df.empty:
        st.info("Lägg till och uppdatera åtminstone en ticker först.")
        return df

    tickers = df["Ticker"].tolist()
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
    with c1:
        tkr = st.selectbox("Ticker", options=tickers, key="buy_tkr")
    with c2:
        qty = st.number_input("Antal", min_value=1, value=10, step=1, key="buy_qty")
    with c3:
        px_local = st.number_input("Pris (lokal valuta)", min_value=0.0, value=10.0, step=0.1, key="buy_px")
    with c4:
        cur = df.loc[df["Ticker"] == tkr, "Valuta"].iloc[0] if (df["Ticker"] == tkr).any() else "USD"
        ccy = st.selectbox("Valuta", options=["SEK", "USD", "CAD", "NOK", "EUR"],
                           index=["SEK","USD","CAD","NOK","EUR"].index(cur) if cur in ["SEK","USD","CAD","NOK","EUR"] else 1,
                           key="buy_ccy")

    # FX till SEK (från sidopanelens värden)
    fx_used = 1.0
    if ccy == "USD": fx_used = float(fx_map.get("USD/SEK", DEFAULT_FX["USD/SEK"]))
    elif ccy == "CAD": fx_used = float(fx_map.get("CAD/SEK", DEFAULT_FX["CAD/SEK"]))
    elif ccy == "NOK": fx_used = float(fx_map.get("NOK/SEK", DEFAULT_FX["NOK/SEK"]))
    elif ccy == "EUR": fx_used = float(fx_map.get("EUR/SEK", DEFAULT_FX["EUR/SEK"]))

    px_sek = round(px_local * fx_used, 6)
    gross  = round(px_sek * qty, 2)
    fee_court, fee_fx, fee_total = calc_fees(gross, is_foreign(ccy))
    net    = round(gross + fee_total, 2)

    st.caption(f"Pris (SEK): **{px_sek}**  |  Brutto: **{gross} SEK**  |  "
               f"Courtage: **{fee_court}**  |  FX-avgift: **{fee_fx}**  |  Totalt: **{net} SEK**")

    if st.button("➕ Genomför köp"):
        i = df.index[df["Ticker"] == tkr][0]
        df = update_position_after_buy(df, i, qty, net, fee_court, fee_fx)
        log_tx("KÖP", tkr, qty, px_local, ccy, fx_used, px_sek, gross, fee_court, fee_fx, note="app")
        df = beräkna(df, fx_map)
        spara_df_säkert(df)
        st.success(f"Köp klart: {qty} {tkr}. Avgifter {fee_total} SEK. GAV uppdaterad.")
    return df

# === UI: Sälj (med avgifter & logg) ===
def block_sell(df: pd.DataFrame, fx_map: dict) -> pd.DataFrame:
    st.subheader("📤 Sälj (med Mini-courtage & växlingsavgift)")
    if df.empty:
        st.info("Lägg till och uppdatera minst en ticker först.")
        return df

    tickers = df["Ticker"].tolist()
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
    with c1:
        tkr = st.selectbox("Ticker", options=tickers, key="sell_tkr")
    with c2:
        qty = st.number_input("Antal", min_value=1, value=10, step=1, key="sell_qty")
    with c3:
        px_local = st.number_input("Pris (lokal valuta)", min_value=0.0, value=10.0, step=0.1, key="sell_px")
    with c4:
        cur = df.loc[df["Ticker"] == tkr, "Valuta"].iloc[0] if (df["Ticker"] == tkr).any() else "USD"
        ccy = st.selectbox("Valuta", options=["SEK", "USD", "CAD", "NOK", "EUR"],
                           index=["SEK","USD","CAD","NOK","EUR"].index(cur) if cur in ["SEK","USD","CAD","NOK","EUR"] else 1,
                           key="sell_ccy")

    fx_used = 1.0
    if ccy == "USD": fx_used = float(fx_map.get("USD/SEK", DEFAULT_FX["USD/SEK"]))
    elif ccy == "CAD": fx_used = float(fx_map.get("CAD/SEK", DEFAULT_FX["CAD/SEK"]))
    elif ccy == "NOK": fx_used = float(fx_map.get("NOK/SEK", DEFAULT_FX["NOK/SEK"]))
    elif ccy == "EUR": fx_used = float(fx_map.get("EUR/SEK", DEFAULT_FX["EUR/SEK"]))

    px_sek = round(px_local * fx_used, 6)
    gross  = round(px_sek * qty, 2)
    fee_court, fee_fx, fee_total = calc_fees(gross, is_foreign(ccy))
    net    = round(gross - fee_total, 2)

    st.caption(f"Pris (SEK): **{px_sek}**  |  Brutto: **{gross} SEK**  |  "
               f"Courtage: **{fee_court}**  |  FX-avgift: **{fee_fx}**  |  Nettolikvid: **{net} SEK**")

    if st.button("➖ Genomför sälj"):
        i = df.index[df["Ticker"] == tkr][0]
        if qty > float(df.at[i, "Antal"] or 0):
            st.error("Du kan inte sälja fler aktier än du äger.")
            return df
        df = update_position_after_sell(df, i, qty, fee_court, fee_fx)
        log_tx("SÄLJ", tkr, -qty, px_local, ccy, fx_used, px_sek, gross, fee_court, fee_fx, note="app")
        df = beräkna(df, fx_map)
        spara_df_säkert(df)
        st.success(f"Sälj klart: {qty} {tkr}. Avgifter {fee_total} SEK. Kvar: {int(df.at[i,'Antal'])} st.")
    return df

# === Köp-simulering med avgifter (valfritt) ===
def block_buy_sim_with_fees(df: pd.DataFrame, fx_map: dict):
    st.subheader("🎯 Köp-simulering (med avgifter)")
    amount = st.number_input("Belopp (SEK)", min_value=0, value=900, step=100, key="sim_amount")
    if st.button("Beräkna förslag"):
        if amount <= 0 or df.empty:
            st.warning("Ange belopp och uppdatera data först.")
            return
        cand = df[df["Poäng"] > 0].sort_values(["Poäng","Dagar till X-dag"], ascending=[False, True])
        if cand.empty:
            st.info("Inga kandidater.")
            return
        r = cand.iloc[0]
        price = float(r["Kurs (SEK)"]) or 0.0
        if price <= 0:
            st.info("Saknar prisdata för toppkandidat.")
            return
        ccy   = r["Valuta"]
        qty   = 0
        while True:
            test_qty = qty + 1
            gross = round(test_qty * price, 2)
            fee_court, fee_fx, fee_tot = calc_fees(gross, is_foreign(ccy))
            total = gross + fee_tot
            if total <= amount:
                qty = test_qty
            else:
                break
        gross = round(qty * price, 2)
        fee_court, fee_fx, fee_tot = calc_fees(gross, is_foreign(ccy))
        total = gross + fee_tot
        st.success(f"Föreslår **{qty} st {r['Ticker']}**  |  Brutto {gross}  |  Avgifter {fee_tot}  |  Totalt {total} SEK")
        st.caption(f"Poäng {r['Poäng']}, RelYield {r['Relative Yield (x)']}x, Ex-Date {r['Ex-Date']}  |  "
                   f"Courtage {fee_court}, FX-avgift {fee_fx}")

def main():
    st.title("Relative Yield – utdelningsåterinvestering (avgifter & transaktioner)")

    # Läs in & sidopanel
    raw = hamta_df()
    raw = säkerställ_kolumner(raw)
    tickers, fx_map, do_update = sidopanel(raw)
    df = säkerställ_tickers(raw, tickers)

    # Uppdatera alla om man klickar
    if do_update and tickers:
        df = block_uppdatera_alla(df, tickers, fx_map)

    # Beräkna & visa
    df = beräkna(df, fx_map)
    st.divider(); block_top_card(df)
    st.divider(); df = block_portfolio(df, fx_map)
    st.divider(); block_ranking(df)

    # Köp / Sälj (med avgifter & logg)
    st.divider(); df = block_buy(df, fx_map)
    st.divider(); df = block_sell(df, fx_map)

    # Köp-simulering med avgifter
    st.divider(); block_buy_sim_with_fees(beräkna(hamta_df(), fx_map), fx_map)

if __name__ == "__main__":
    main()
