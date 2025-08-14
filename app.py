import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time
import numpy as np
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Utdelningsranking", layout="wide")

# ---- Google Sheets Config ----
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Bolag"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# ---- Koppling till Google Sheets ----
def _open_sheet():
    return client.open_by_url(SHEET_URL)

def skapa_koppling():
    return _open_sheet().worksheet(SHEET_NAME)

# ---- Hämta data ----
def hamta_data():
    ws = skapa_koppling()
    data = ws.get_all_records()
    df = pd.DataFrame(data)

    # Säkerställ att alla kolumner finns och rätt typer
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    return df

# ---- Spara data ----
def spara_data(df):
    ws = skapa_koppling()
    # Konvertera till strängar för att undvika Google Sheets nummerformat-problem
    ws.clear()
    ws.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ---- Säkerställ att kolumner finns ----
def säkerställ_kolumner(df):
    nödvändiga_kolumner = [
        "Ticker", "Bolagsnamn", "Kategori", "Antal aktier", "GAV",
        "Aktuell kurs", "Kurs SEK", "Marknadsvärde", "Årlig utdelning",
        "Utdelningsfrekvens", "Direktavkastning (%)", "Utdelning/år (SEK)",
        "Valuta", "Datakälla"
    ]

    for kol in nödvändiga_kolumner:
        if kol not in df.columns:
            df[kol] = ""

    # Trimma extra kolumner som inte används
    df = df[nödvändiga_kolumner]

    return df

# ---- Konvertera typer ----
def konvertera_typer(df):
    num_cols = [
        "Antal aktier", "GAV", "Aktuell kurs", "Kurs SEK", "Marknadsvärde",
        "Årlig utdelning", "Direktavkastning (%)", "Utdelning/år (SEK)"
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fyll NaN med 0 för numeriska kolumner
    df[num_cols] = df[num_cols].fillna(0)

    # Rensa och fyll kategori
    df["Kategori"] = df["Kategori"].astype(str).str.strip()
    df["Kategori"] = df["Kategori"].replace({"": "QUALITY", "nan": "QUALITY"})

    return df

# ---- Robust talparser (hindrar tid/datum-problem) ----
def _to_float(x):
    if pd.isna(x):
        return 0.0
    s = str(x).strip()

    # Vanliga datum/tidsartefakter
    # "12:30" -> "12.30" (tolkas som 12.30 istället för tid)
    if ":" in s and all(part.isdigit() for part in s.split(":")):
        s = s.replace(":", ".")

    # Ersätt decimalkomma med punkt
    s = s.replace(",", ".")

    # Ta bort allt utom siffror, punkt och minus
    keep = "0123456789.-"
    s = "".join(ch for ch in s if ch in keep)

    # Edge cases
    if s in ("", ".", "-", "-.", ".-"):
        return 0.0

    try:
        return float(s)
    except Exception:
        return 0.0


# ---- Växelkurser (default) ----
DEF_FX = {"USDSEK": 9.60, "NOKSEK": 0.94, "CADSEK": 6.95, "EURSEK": 11.10}
for k, v in DEF_FX.items():
    if k not in st.session_state:
        st.session_state[k] = v

def fx_for(ccy: str) -> float:
    c = (ccy or "").strip().upper()
    if c == "SEK": return 1.0
    if c == "USD": return float(st.session_state.get("USDSEK", DEF_FX["USDSEK"]))
    if c == "EUR": return float(st.session_state.get("EURSEK", DEF_FX["EURSEK"]))
    if c == "CAD": return float(st.session_state.get("CADSEK", DEF_FX["CADSEK"]))
    if c == "NOK": return float(st.session_state.get("NOKSEK", DEF_FX["NOKSEK"]))
    return 1.0


# ---- Frekvensinferens från utdelningsserie ----
def _infer_frequency(div_series: pd.Series) -> str:
    """Returnerar text 'Månads', 'Kvartals', 'Halvårs', 'Års' eller 'Oregelbunden'."""
    try:
        if div_series is None or div_series.empty:
            return "Oregelbunden"
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
        last12 = div_series[div_series.index >= cutoff]
        cnt = int(last12.shape[0]) if not last12.empty else 0
        if cnt >= 10: return "Månads"
        if cnt >= 3:  return "Kvartals"
        if cnt == 2:  return "Halvårs"
        if cnt == 1:  return "Års"
    except Exception:
        pass
    return "Oregelbunden"


# ---- Yahoo-hämtning (pris, valuta, namn, utd/år lokal, frekvens) ----
def hamta_yahoo_data(ticker: str) -> dict:
    t = (ticker or "").strip().upper()
    if not t:
        return {}

    yf_t = yf.Ticker(t)

    # Info
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

    # Valuta & namn
    currency = (info.get("currency") or "").upper() or "SEK"
    name = info.get("shortName") or info.get("longName") or t

    # Utdelningar (lokal valuta per aktie under 12 mån)
    div_year_local = 0.0
    freq_text = "Oregelbunden"
    try:
        divs = yf_t.dividends
        if divs is not None and not divs.empty:
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
            last12 = divs[divs.index >= cutoff]
            div_year_local = float(last12.tail(12).sum()) if not last12.empty else 0.0
            freq_text = _infer_frequency(divs)
    except Exception:
        pass

    return {
        "Bolagsnamn": name,
        "Aktuell kurs": price,
        "Valuta": currency,
        "Utdelning/år (lokal)": div_year_local,  # mellanlagras bara i minnet
        "Utdelningsfrekvens": freq_text,
        "Datakälla": "Yahoo",
    }


# ---- Beräkningar: kurs SEK, marknadsvärde, yield, utdelning SEK ----
def beräkna_portfölj(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Säkerställ kolumner och typer
    d = säkerställ_kolumner(d)
    d = konvertera_typer(d)

    # Kategori: tom eller "nan" -> QUALITY
    d["Kategori"] = (
        d["Kategori"]
        .astype(str)
        .str.strip()
        .replace({"": "QUALITY", "nan": "QUALITY"})
    )

    # Kurs i SEK
    fx = d["Valuta"].apply(fx_for).astype(float)
    d["Kurs SEK"] = (pd.to_numeric(d["Aktuell kurs"], errors="coerce").fillna(0.0) * fx).round(6)

    # Marknadsvärde & yield
    qty = pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0)
    d["Marknadsvärde"] = (qty * d["Kurs SEK"]).round(2)

    # Utdelning per år (SEK per aktie) – antingen från "Utdelning/år (SEK)" om redan fyllt,
    # annars från en mellanlagrad in-memory-kolumn "Utdelning/år (lokal)" * FX om vi satte den.
    # (Om användaren själv fyller "Utdelning/år (SEK)" så respekterar vi den.)
    if "Utdelning/år (lokal)" in d.columns:
        utd_sek_calc = pd.to_numeric(d["Utdelning/år (lokal)"], errors="coerce").fillna(0.0) * fx
        d["Utdelning/år (SEK)"] = pd.to_numeric(d["Utdelning/år (SEK)"], errors="coerce").fillna(0.0)
        # Om befintligt värde är 0, använd kalkylen
        need = d["Utdelning/år (SEK)"] <= 0
        d.loc[need, "Utdelning/år (SEK)"] = utd_sek_calc[need].round(6)
    else:
        d["Utdelning/år (SEK)"] = pd.to_numeric(d["Utdelning/år (SEK)"], errors="coerce").fillna(0.0)

    # Total årlig utdelning (SEK) = per-aktie SEK * antal
    d["Årlig utdelning"] = (d["Utdelning/år (SEK)"] * qty).round(2)

    # Direktavkastning (%) = (Utdelning/år (SEK) / Kurs SEK) * 100
    price_sek = d["Kurs SEK"].replace(0, np.nan)
    d["Direktavkastning (%)"] = ((d["Utdelning/år (SEK)"] / price_sek) * 100.0).fillna(0.0).round(2)

    # Portföljandel
    tot_mv = float(pd.to_numeric(d["Marknadsvärde"], errors="coerce").fillna(0.0).sum()) or 1.0
    d["Portföljandel (%)"] = (100.0 * d["Marknadsvärde"] / tot_mv).round(2)

    # Datakälla — om tom, fyll "Manuell"
    d["Datakälla"] = d["Datakälla"].replace({"": "Manuell", "nan": "Manuell"})

    return d

# ---- Sidebar: FX, backup, uppdatera EN ----
def sidebar_tools():
    st.sidebar.header("⚙️ Inställningar")
    st.sidebar.markdown("**Växelkurser (SEK)**")

    c1, c2 = st.sidebar.columns(2)
    with c1:
        st.session_state["USDSEK"] = st.number_input(
            "USD/SEK", min_value=0.0, value=float(st.session_state["USDSEK"]), step=0.01, format="%.4f"
        )
        st.session_state["EURSEK"] = st.number_input(
            "EUR/SEK", min_value=0.0, value=float(st.session_state["EURSEK"]), step=0.01, format="%.4f"
        )
    with c2:
        st.session_state["CADSEK"] = st.number_input(
            "CAD/SEK", min_value=0.0, value=float(st.session_state["CADSEK"]), step=0.01, format="%.4f"
        )
        st.session_state["NOKSEK"] = st.number_input(
            "NOK/SEK", min_value=0.0, value=float(st.session_state["NOKSEK"]), step=0.01, format="%.4f"
        )

    if st.sidebar.button("↩︎ Återställ FX till standard"):
        for k, v in DEF_FX.items():
            st.session_state[k] = v
        st.sidebar.success("Standardkurser återställda.")

    st.sidebar.markdown("---")
    one_ticker = st.sidebar.text_input("Uppdatera EN ticker (Yahoo)", placeholder="t.ex. VICI").strip().upper()
    if st.sidebar.button("🔄 Uppdatera EN"):
        if not one_ticker:
            st.sidebar.warning("Ange ticker först.")
        else:
            base = st.session_state.get("working_df", pd.DataFrame())
            base = säkerställ_kolumner(base)
            # Om ticker saknas – skapa rad
            if one_ticker not in base["Ticker"].astype(str).tolist():
                base = pd.concat([base, pd.DataFrame([{
                    "Ticker": one_ticker, "Bolagsnamn": one_ticker, "Kategori": "QUALITY",
                    "Antal aktier": 0, "GAV": 0.0, "Valuta": "SEK"
                }])], ignore_index=True)

            vals = hamta_yahoo_data(one_ticker)
            if vals:
                m = base["Ticker"] == one_ticker
                for k_src, k_dst in [
                    ("Bolagsnamn", "Bolagsnamn"),
                    ("Aktuell kurs", "Aktuell kurs"),
                    ("Valuta", "Valuta"),
                    ("Utdelningsfrekvens", "Utdelningsfrekvens"),
                ]:
                    if k_src in vals and vals[k_src] not in (None, ""):
                        base.loc[m, k_dst] = vals[k_src]

                # Lägg mellanlagrad "Utdelning/år (lokal)" i minne (ej sparkrav, men driver beräkning)
                if "Utdelning/år (lokal)" not in base.columns:
                    base["Utdelning/år (lokal)"] = 0.0
                if "Utdelning/år (lokal)" in vals:
                    base.loc[m, "Utdelning/år (lokal)"] = _to_float(vals["Utdelning/år (lokal)"])
                base.loc[m, "Datakälla"] = "Yahoo"

                base = beräkna_portfölj(base)
                st.session_state["working_df"] = base
                st.sidebar.success(f"{one_ticker} uppdaterad i minnet (ej sparad).")
            else:
                st.sidebar.warning("Kunde inte hämta data från Yahoo.")

# ---- Lägg till / Uppdatera bolag (in-memory; spara via Spara-sidan) ----
CATEGORY_CHOICES = ["QUALITY","REIT","mREIT","BDC","Shipping","Telecom","Tech","Bank","Finance","Energy","Industrial","Other"]

def page_add_or_update(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("➕ Lägg till / ✏ Uppdatera bolag")
    base = säkerställ_kolumner(df).copy()

    tickers = ["Ny"] + sorted(base["Ticker"].astype(str).tolist())
    val = st.selectbox("Välj bolag", tickers)

    if val == "Ny":
        tkr = st.text_input("Ticker").strip().upper()
        antal = st.number_input("Antal aktier", min_value=0, value=0, step=1)
        gav   = st.number_input("GAV (i **lokal** valuta)", min_value=0.0, value=0.0, step=0.01)
        kategori = st.selectbox("Kategori", options=CATEGORY_CHOICES, index=0)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🌐 Hämta från Yahoo"):
                if not tkr:
                    st.warning("Ange ticker först.")
                else:
                    vals = hamta_yahoo_data(tkr)
                    if not vals:
                        st.info("Ingen data mottagen (kan vara tillfälligt).")
                    else:
                        st.info(
                            f"**{vals.get('Bolagsnamn', tkr)}** | "
                            f"Valuta: {vals.get('Valuta','?')}, Kurs: {vals.get('Aktuell kurs',0)}, "
                            f"Utd/år lokal: {vals.get('Utdelning/år (lokal)',0)}, "
                            f"Frekvens: {vals.get('Utdelningsfrekvens','-')}"
                        )
        with c2:
            if st.button("➕ Lägg till i minnet"):
                if not tkr:
                    st.error("Ticker måste anges.")
                else:
                    row = {"Ticker": tkr, "Bolagsnamn": tkr, "Kategori": kategori,
                           "Antal aktier": antal, "GAV": gav, "Valuta": "SEK",
                           "Aktuell kurs": 0.0, "Utdelningsfrekvens": "", "Utdelning/år (SEK)": 0.0,
                           "Kurs SEK": 0.0, "Marknadsvärde": 0.0, "Årlig utdelning": 0.0, "Direktavkastning (%)": 0.0,
                           "Datakälla": "Manuell"}
                    vals = hamta_yahoo_data(tkr)
                    if vals:
                        for k_src, k_dst in [
                            ("Bolagsnamn","Bolagsnamn"),
                            ("Aktuell kurs","Aktuell kurs"),
                            ("Valuta","Valuta"),
                            ("Utdelningsfrekvens","Utdelningsfrekvens"),
                        ]:
                            if vals.get(k_src) not in (None,""):
                                row[k_dst] = vals[k_src]
                        # in-memory mellanlagring för beräkning
                        row["Utdelning/år (lokal)"] = _to_float(vals.get("Utdelning/år (lokal)", 0.0))
                        row["Datakälla"] = "Yahoo"
                    base = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
                    base = beräkna_portfölj(base)
                    st.session_state["working_df"] = base
                    st.success(f"{tkr} tillagt i minnet. Spara via '💾 Spara'-sidan.")
    else:
        r = base[base["Ticker"] == val].iloc[0]
        tkr = st.text_input("Ticker", value=r["Ticker"]).strip().upper()
        antal = st.number_input("Antal aktier", min_value=0, value=int(_to_float(r["Antal aktier"])), step=1)
        gav   = st.number_input("GAV (i **lokal** valuta)", min_value=0.0, value=float(_to_float(r["GAV"])), step=0.01)
        kategori = st.selectbox("Kategori", options=CATEGORY_CHOICES,
                                index=CATEGORY_CHOICES.index(str(r.get("Kategori","QUALITY"))))

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🌐 Uppdatera från Yahoo"):
                vals = hamta_yahoo_data(tkr)
                m = base["Ticker"] == val  # uppdatera nuvarande rad
                if vals:
                    for k_src, k_dst in [
                        ("Bolagsnamn","Bolagsnamn"),
                        ("Aktuell kurs","Aktuell kurs"),
                        ("Valuta","Valuta"),
                        ("Utdelningsfrekvens","Utdelningsfrekvens"),
                    ]:
                        if vals.get(k_src) not in (None,""):
                            base.loc[m, k_dst] = vals[k_src]
                    # in-memory "Utdelning/år (lokal)"
                    if "Utdelning/år (lokal)" not in base.columns:
                        base["Utdelning/år (lokal)"] = 0.0
                    base.loc[m, "Utdelning/år (lokal)"] = _to_float(vals.get("Utdelning/år (lokal)", 0.0))
                    base.loc[m, "Datakälla"] = "Yahoo"

                # uppdatera manuella fält
                base.loc[m, "Ticker"] = tkr
                base.loc[m, "Antal aktier"] = antal
                base.loc[m, "GAV"] = gav
                base.loc[m, "Kategori"] = kategori

                base = beräkna_portfölj(base)
                st.session_state["working_df"] = base
                st.success(f"{tkr} uppdaterad i minnet (ej sparad).")

        with c2:
            if st.button("✏ Uppdatera fält (minne)"):
                m = base["Ticker"] == val
                base.loc[m, "Ticker"] = tkr
                base.loc[m, "Antal aktier"] = antal
                base.loc[m, "GAV"] = gav
                base.loc[m, "Kategori"] = kategori
                base = beräkna_portfölj(base)
                st.session_state["working_df"] = base
                st.success(f"{tkr} uppdaterad i minnet.")

        with c3:
            if st.button("🗑 Ta bort (minne)"):
                base = base[base["Ticker"] != val].reset_index(drop=True)
                base = beräkna_portfölj(base)
                st.session_state["working_df"] = base
                st.success(f"{val} borttagen i minnet.")

    st.markdown("---")
    if st.button("💾 Spara alla ändringar till Google Sheets"):
        preview = beräkna_portfölj(st.session_state["working_df"])
        spara_data(preview)
        st.success("Sparat till Sheets.")
    return st.session_state.get("working_df", base)


# ---- Massuppdatering från Yahoo ----
def page_mass_update(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("⏩ Massuppdatera alla bolag (Yahoo)")
    base = säkerställ_kolumner(df).copy()
    if base.empty:
        st.info("Inga bolag i databasen ännu.")
        return base

    if st.button("Starta massuppdatering"):
        N = len(base)
        prog = st.progress(0)
        info = st.empty()
        for i, tkr in enumerate(base["Ticker"].astype(str).tolist(), start=1):
            info.write(f"Uppdaterar {tkr} ({i}/{N}) …")
            vals = hamta_yahoo_data(tkr)
            m = base["Ticker"] == tkr
            if vals:
                for k_src, k_dst in [
                    ("Bolagsnamn","Bolagsnamn"),
                    ("Aktuell kurs","Aktuell kurs"),
                    ("Valuta","Valuta"),
                    ("Utdelningsfrekvens","Utdelningsfrekvens"),
                ]:
                    if vals.get(k_src) not in (None,""):
                        base.loc[m, k_dst] = vals[k_src]
                if "Utdelning/år (lokal)" not in base.columns:
                    base["Utdelning/år (lokal)"] = 0.0
                base.loc[m, "Utdelning/år (lokal)"] = _to_float(vals.get("Utdelning/år (lokal)", 0.0))
                base.loc[m, "Datakälla"] = "Yahoo"

            base = beräkna_portfölj(base)
            prog.progress(int(i * 100 / N))
            time.sleep(1.0)  # respekt mot Yahoo

        st.session_state["working_df"] = base
        st.success("Massuppdatering klar (i minnet). Spara via '💾 Spara'.")
    return st.session_state.get("working_df", base)


# ---- Portföljöversikt ----
def page_portfolio(df: pd.DataFrame):
    st.subheader("📦 Portföljöversikt")
    d = beräkna_portfölj(df).copy()
    if d.empty:
        st.info("Lägg till minst ett bolag.")
        return

    tot_mv  = float(pd.to_numeric(d["Marknadsvärde"], errors="coerce").fillna(0.0).sum())
    tot_div = float(pd.to_numeric(d["Årlig utdelning"], errors="coerce").fillna(0.0).sum())

    c1, c2 = st.columns(2)
    c1.metric("Portföljvärde (SEK)", f"{tot_mv:,.0f}".replace(",", " "))
    c2.metric("Årlig utdelning (SEK)", f"{tot_div:,.0f}".replace(",", " "))

    show_cols = [
        "Ticker","Bolagsnamn","Valuta","Kategori","Antal aktier","GAV",
        "Aktuell kurs","Kurs SEK","Marknadsvärde","Portföljandel (%)",
        "Utdelningsfrekvens","Utdelning/år (SEK)","Årlig utdelning","Direktavkastning (%)","Datakälla"
    ]
    st.dataframe(d[show_cols], use_container_width=True)


# ---- Spara-sida ----
def page_save(df: pd.DataFrame):
    st.subheader("💾 Spara till Google Sheets")
    preview = beräkna_portfölj(säkerställ_kolumner(df))
    st.write("Rader som sparas:", len(preview))
    st.dataframe(
        preview[["Ticker","Bolagsnamn","Valuta","Kategori","Antal aktier","GAV","Aktuell kurs","Kurs SEK","Marknadsvärde","Utdelning/år (SEK)","Årlig utdelning"]],
        use_container_width=True
    )
    if st.button("✅ Bekräfta och spara"):
        spara_data(preview)
        st.success("Sparat!")

# ---- Main navigation ----
def main():
    st.title("📈 Utdelningsranking & Portfölj")
    if "working_df" not in st.session_state:
        try:
            df = hamta_data()
            df = säkerställ_kolumner(df)
            st.session_state["working_df"] = df
        except Exception as e:
            st.session_state["working_df"] = säkerställ_kolumner(pd.DataFrame())
            st.error(f"Kunde inte läsa data från Google Sheets: {e}")

    sidebar_tools()

    menu = st.sidebar.radio(
        "Meny",
        ["📦 Portfölj", "➕ Lägg till/Uppdatera", "⏩ Massuppdatera", "💾 Spara"],
        index=0
    )

    if menu == "📦 Portfölj":
        page_portfolio(st.session_state["working_df"])
    elif menu == "➕ Lägg till/Uppdatera":
        df_new = page_add_or_update(st.session_state["working_df"])
        st.session_state["working_df"] = df_new
    elif menu == "⏩ Massuppdatera":
        df_new = page_mass_update(st.session_state["working_df"])
        st.session_state["working_df"] = df_new
    elif menu == "💾 Spara":
        page_save(st.session_state["working_df"])


if __name__ == "__main__":
    main()
