import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time
from datetime import datetime
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Utdelningsportfölj", layout="wide")

# --- Ladda Google Sheets credentials ---
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = st.secrets["SHEET_NAME"]

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# --- Säkerställ kolumner ---
def säkerställ_kolumner(df):
    kolumner = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Direktavkastning (%)",
        "Årlig utdelning", "Nästa X-dag", "Nästa utdelning", "Antal aktier",
        "GAV", "Portföljvärde", "Årlig utdelning totalt", "Utdelning/månad",
        "Uppside (%)", "Rekommendation"
    ]
    for col in kolumner:
        if col not in df.columns:
            df[col] = ""
    return df[kolumner]

# --- Hämta data från Yahoo Finance ---
def hamta_yahoo_data(ticker):
    try:
        aktie = yf.Ticker(ticker)
        info = aktie.info

        namn = info.get("longName") or info.get("shortName") or ticker
        kurs = info.get("currentPrice")
        valuta = info.get("currency")
        utdelning = info.get("dividendRate")
        direktavkastning = info.get("dividendYield") * 100 if info.get("dividendYield") else None
        nästa_xdag = info.get("exDividendDate")
        if nästa_xdag:
            nästa_xdag = datetime.fromtimestamp(nästa_xdag).strftime("%Y-%m-%d")
        nästa_utd = info.get("lastDividendValue")

        return {
            "Bolagsnamn": namn,
            "Aktuell kurs": kurs,
            "Valuta": valuta,
            "Årlig utdelning": utdelning,
            "Direktavkastning (%)": direktavkastning,
            "Nästa X-dag": nästa_xdag,
            "Nästa utdelning": nästa_utd
        }
    except Exception as e:
        st.error(f"Kunde inte hämta data för {ticker}: {e}")
        return {}

# --- Uppdatera rader med beräkningar ---
def uppdatera_beräkningar(df):
    for i, rad in df.iterrows():
        try:
            kurs = float(rad.get("Aktuell kurs") or 0)
            antal = float(rad.get("Antal aktier") or 0)
            utdelning_per_aktie = float(rad.get("Årlig utdelning") or 0)

            # Portföljvärde per rad
            portföljvärde = kurs * antal
            df.at[i, "Portföljvärde"] = round(portföljvärde, 2)

            # Årlig utdelning totalt
            total_utd = utdelning_per_aktie * antal
            df.at[i, "Årlig utdelning totalt"] = round(total_utd, 2)

            # Utdelning per månad
            df.at[i, "Utdelning/månad"] = round(total_utd / 12, 2) if total_utd else 0

            # Uppsida – här kan du lägga logik för riktkurs
            riktkurs = kurs * 1.1  # Placeholder: +10% som exempel
            if kurs > 0:
                uppsida = ((riktkurs - kurs) / kurs) * 100
                df.at[i, "Uppside (%)"] = round(uppsida, 2)

            # Rekommendation
            if uppsida > 15:
                df.at[i, "Rekommendation"] = "Köp"
            elif uppsida > 5:
                df.at[i, "Rekommendation"] = "Behåll"
            else:
                df.at[i, "Rekommendation"] = "Sälj"
        except:
            pass
    return df

# --- Formulär för att lägga till eller uppdatera bolag ---
def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till eller uppdatera bolag")
    ticker = st.text_input("Ticker (t.ex. AAPL, 2020.OL)").strip().upper()

    if st.button("Hämta från Yahoo Finance"):
        if ticker:
            data = hamta_yahoo_data(ticker)
            if data:
                if ticker in df["Ticker"].values:
                    idx = df[df["Ticker"] == ticker].index[0]
                    for key, value in data.items():
                        df.at[idx, key] = value
                else:
                    ny_rad = {"Ticker": ticker, **data, "Antal aktier": 0, "GAV": 0}
                    df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
                df = uppdatera_beräkningar(df)
                spara_data(df)
                st.success(f"Data uppdaterad för {ticker}")
        else:
            st.warning("Ange en ticker först.")
    return df

# --- Massuppdatering av alla bolag ---
def massuppdatera(df):
    st.subheader("🔄 Massuppdatera alla bolag")
    if st.button("Uppdatera nu"):
        for i, rad in df.iterrows():
            ticker = rad["Ticker"]
            if ticker:
                data = hamta_yahoo_data(ticker)
                for key, value in data.items():
                    df.at[i, key] = value
                df = uppdatera_beräkningar(df)
                spara_data(df)
                st.write(f"✅ {ticker} uppdaterad")
                time.sleep(1)
        st.success("Alla bolag uppdaterade.")
    return df

# --- Visa portfölj ---
def visa_portfölj(df):
    st.subheader("📊 Portföljöversikt")
    total_värde = df["Portföljvärde"].sum()
    total_utd = df["Årlig utdelning totalt"].sum()
    st.metric("Portföljvärde", f"{total_värde:,.2f} kr")
    st.metric("Årlig utdelning", f"{total_utd:,.2f} kr")
    st.metric("Utdelning/månad", f"{total_utd/12:,.2f} kr")

    st.dataframe(df)

# --- Visa kommande utdelningar ---
def kommande_utdelningar(df):
    st.subheader("📅 Kommande utdelningar")
    df_utd = df[df["Nästa X-dag"] != ""].copy()
    df_utd["Nästa X-dag"] = pd.to_datetime(df_utd["Nästa X-dag"], errors="coerce")
    df_utd = df_utd.sort_values("Nästa X-dag")
    st.dataframe(df_utd[["Ticker", "Bolagsnamn", "Nästa X-dag", "Nästa utdelning"]])

# --- Meny & main ---------------------------------------------------------------
def main():
    st.title("📈 Utdelningsportfölj")

    # Läs data från Google Sheets
    try:
        df = hamta_data()
    except Exception as e:
        st.error(f"Kunde inte läsa Google Sheet: {e}")
        return

    # Säkerställ kolumner även om arket är tomt
    df = säkerställ_kolumner(df)

    # Meny
    meny = st.sidebar.radio(
        "Meny",
        ["Portfölj", "Lägg till/uppdatera bolag", "Massuppdatera", "Kommande utdelningar"]
    )

    if meny == "Portfölj":
        st.subheader("📦 Portfölj")
        # Redigerbar vy för Antal aktier & GAV
        edit_cols = ["Antal aktier", "GAV"]
        vis_cols = [
            "Ticker","Bolagsnamn","Aktuell kurs","Valuta",
            "Antal aktier","GAV","Portföljvärde",
            "Årlig utdelning","Årlig utdelning totalt","Utdelning/månad",
            "Direktavkastning (%)","Nästa X-dag"
        ]
        # Re-beräkna innan visning
        df = uppdatera_beräkningar(df)
        view = df[vis_cols].copy()
        edited = st.data_editor(view, hide_index=True, use_container_width=True, num_rows="dynamic")

        colA, colB = st.columns(2)
        with colA:
            if st.button("💾 Spara portföljändringar"):
                # skriv tillbaka ändringar för Antal aktier & GAV
                for _, r in edited.iterrows():
                    t = r["Ticker"]
                    if t in df["Ticker"].values:
                        idx = df[df["Ticker"] == t].index[0]
                        for c in edit_cols:
                            df.at[idx, c] = r[c]
                df = uppdatera_beräkningar(df)
                spara_data(df)
                st.success("Portföljen sparad.")
        with colB:
            if st.button("🔁 Uppdatera värden (recalc)"):
                df = uppdatera_beräkningar(df)
                spara_data(df)
                st.success("Värden uppdaterade.")

        # Snabb-summering
        total_värde = float(df["Portföljvärde"].fillna(0).sum())
        total_utd   = float(df["Årlig utdelning totalt"].fillna(0).sum())
        c1,c2,c3 = st.columns(3)
        c1.metric("Portföljvärde", f"{total_värde:,.2f} kr")
        c2.metric("Årsutdelning", f"{total_utd:,.2f} kr")
        c3.metric("Utd/månad", f"{(total_utd/12):,.2f} kr")

    elif meny == "Lägg till/uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)

    elif meny == "Massuppdatera":
        df = massuppdatera(df)

    elif meny == "Kommande utdelningar":
        kommande_utdelningar(df)


if __name__ == "__main__":
    main()
