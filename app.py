# app.py  — Relative Yield (komplett)

import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time, math, json
from datetime import datetime, timedelta, date
from google.oauth2.service_account import Credentials

# Rerun-shim (Streamlit v1.x / v2.x)
try:
    _rerun = st.rerun
except AttributeError:
    _rerun = st.experimental_rerun

st.set_page_config(page_title="Relative Yield – utdelningsportfölj", layout="wide")

# ── Sheets konfiguration ────────────────────────────────────────────────────
SHEET_URL   = st.secrets["SHEET_URL"]
SHEET_NAME  = "Blad1"        # <- enligt din begäran
RULES_SHEET = "Regler"
TX_SHEET    = "Transaktioner"

scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# ── FX standard ─────────────────────────────────────────────────────────────
DEF_FX = {"USDSEK": 9.60, "NOKSEK": 0.94, "CADSEK": 6.95, "EURSEK": 11.10}
for k, v in DEF_FX.items():
    if k not in st.session_state:
        st.session_state[k] = v

def fx_for(cur: str) -> float:
    c = (cur or "SEK").strip().upper()
    return float({
        "SEK": 1.0,
        "USD": st.session_state.get("USDSEK", DEF_FX["USDSEK"]),
        "EUR": st.session_state.get("EURSEK", DEF_FX["EURSEK"]),
        "CAD": st.session_state.get("CADSEK", DEF_FX["CADSEK"]),
        "NOK": st.session_state.get("NOKSEK", DEF_FX["NOKSEK"]),
    }.get(c, 1.0))

# ── Numerik-städning (hindrar 09:30 → tid) ─────────────────────────────────
def _to_float(x):
    try:
        if x is None: return 0.0
        if isinstance(x, (int, float, np.number)): 
            if math.isfinite(float(x)): return float(x)
            return 0.0
        s = str(x).strip().replace(" ", "").replace(",", ".")
        # ta bort kolon / tidsskräp
        if ":" in s:
            s = s.replace(":", "")
        if s in ("", "-", "nan", "NaN", "None"):
            return 0.0
        v = float(s)
        if not math.isfinite(v): return 0.0
        return v
    except Exception:
        return 0.0

def _to_int(x):
    try:
        v = int(round(_to_float(x)))
        return max(0, v)
    except Exception:
        return 0

# ── Ark-hjälpare ───────────────────────────────────────────────────────────
def _open_sheet():
    # Minimal retry (rate limit)
    for i in range(3):
        try:
            return client.open_by_url(SHEET_URL)
        except Exception:
            time.sleep(1 + i*0.5)
    return client.open_by_url(SHEET_URL)

def _ensure_worksheet(sh, title):
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=1000, cols=50)

def skapa_koppling():
    sh = _open_sheet()
    return _ensure_worksheet(sh, SHEET_NAME)

def spara_data(df: pd.DataFrame):
    """Säker skrivning: backup + anti-wipe + normalisering till strängar."""
    sh = _open_sheet()
    ws = _ensure_worksheet(sh, SHEET_NAME)

    safe = df.copy()

    # Normalisera alla numeriska kolumner till str
    for c in safe.columns:
        if c in ("Ticker","Bolagsnamn","Valuta","Kategori","Utdelningsfrekvens","Utdelningskälla","Källa","Ex-Date","Nästa utbetalning (est)","Senaste uppdatering"):
            safe[c] = safe[c].astype(str)
        else:
            safe[c] = safe[c].apply(_to_float).astype(float)

    # Anti-wipe (kräv minst 1 giltig ticker)
    if safe["Ticker"].astype(str).str.strip().eq("").all():
        st.error("Sparning avbröts: inga giltiga tickers.")
        return

    # Backupflik
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        sh.duplicate_sheet(source_sheet_id=ws.id, new_sheet_name=f"Backup_{ts}")
        st.sidebar.success(f"📸 Backup skapad: Backup_{ts}")
    except Exception:
        st.sidebar.warning("Kunde inte skapa backupflik (fortsätter ändå).")

    values = [safe.columns.tolist()] + safe.astype(str).values.tolist()
    ws.clear()
    ws.update(values, value_input_option="USER_ENTERED")

# ── Läs data ────────────────────────────────────────────────────────────────
COLUMNS = [
    "Ticker","Bolagsnamn","Aktuell kurs","Valuta","Kategori",
    "Direktavkastning (%)","Utdelning/år","Utdelning/år (manuell)","Lås utdelning",
    "Frekvens/år","Utdelningsfrekvens","Payment-lag (dagar)","Ex-Date","Nästa utbetalning (est)",
    "Antal aktier","GAV",  # GAV i lokal valuta (frivillig)
    "Portföljandel (%)","Årlig utdelning (SEK)",
    "Kurs (SEK)","Utdelningstillväxt (%)","Utdelningskälla",
    "Senaste uppdatering","Källa","Marknadsvärde (SEK)","Insatt (SEK)","Orealiserad P/L (SEK)","Orealiserad P/L (%)"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        d = pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in d.columns:
            d[c] = ""

    # Grundstädning
    d["Ticker"]      = d["Ticker"].astype(str).str.strip().str.upper()
    d["Valuta"]      = d["Valuta"].astype(str).str.strip().str.upper().replace({"": "SEK"})
    d["Kategori"]    = d["Kategori"].astype(str).replace({"": "QUALITY"})
    d["Bolagsnamn"]  = d["Bolagsnamn"].astype(str)

    # Numeriska
    for c in ["Aktuell kurs","Utdelning/år","Utdelning/år (manuell)","Frekvens/år","Payment-lag (dagar)","Antal aktier","GAV"]:
        d[c] = d[c].apply(_to_float)

    # Bool
    d["Lås utdelning"] = d["Lås utdelning"].apply(lambda x: str(x).strip().upper() in ("TRUE","1","JA","YES"))

    # Tomma
    d["Utdelningskälla"] = d["Utdelningskälla"].replace({"": "Yahoo"})
    d["Källa"] = d["Källa"].replace({"": "Yahoo"})
    return d[COLUMNS].copy()

def hamta_data() -> pd.DataFrame:
    try:
        ws = skapa_koppling()
        data = ws.get_all_records()
        return säkerställ_kolumner(pd.DataFrame(data))
    except Exception as e:
        st.warning(f"Kunde inte läsa Google Sheet just nu: {e}")
        return säkerställ_kolumner(pd.DataFrame())

# ── Regler (sparas i separat flik “Regler”) ────────────────────────────────
DEFAULT_MAX_NAME = 7.0  # standard enligt din begäran
DEFAULT_CAT_GOALS = {
    "BDC": 15.0, "mREIT": 10.0, "REIT": 15.0, "Shipping": 25.0,
    "Telecom": 5.0, "Finance": 10.0, "Tech": 5.0, "Energy": 5.0, "Industrial": 10.0
}

def _read_rules():
    sh = _open_sheet()
    ws = _ensure_worksheet(sh, RULES_SHEET)
    rows = ws.get_all_records()
    if not rows:
        return {"max_name": DEFAULT_MAX_NAME, "cat_goals": DEFAULT_CAT_GOALS}
    max_name = DEFAULT_MAX_NAME
    cat_goals = {}
    for r in rows:
        k = str(r.get("Key","")).strip()
        v = r.get("Value","")
        if k == "max_name":
            max_name = _to_float(v)
        elif k.startswith("cat:"):
            cat = k.split("cat:",1)[1]
            cat_goals[cat] = _to_float(v)
    if not cat_goals:
        cat_goals = DEFAULT_CAT_GOALS
    return {"max_name": max_name, "cat_goals": cat_goals}

def _write_rules(max_name: float, cat_goals: dict):
    sh = _open_sheet()
    ws = _ensure_worksheet(sh, RULES_SHEET)
    out = [["Key","Value"]]
    out.append(["max_name", str(float(max_name))])
    for k, v in cat_goals.items():
        out.append([f"cat:{k}", str(float(v))])
    ws.clear()
    ws.update(out, value_input_option="USER_ENTERED")

# ── Yahoo Finance hämtning ─────────────────────────────────────────────────
def hamta_yahoo_data(ticker: str):
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.get_info() or {}
        except Exception:
            try: info = t.info or {}
            except Exception: info = {}

        # pris
        price = None
        try: price = t.fast_info.get("last_price")
        except Exception: pass
        if price in (None, ""):
            price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price in (None, ""):
            h = t.history(period="5d")
            if not h.empty:
                price = float(h["Close"].iloc[-1])
        price = _to_float(price)

        # valuta
        currency = (info.get("currency") or t.fast_info.get("currency") or "SEK").upper()

        # utdelningar senaste 12m
        div_rate = 0.0
        freq = 0
        freq_text = "Oregelbunden"
        ex_date_str = ""
        try:
            divs = t.dividends
            if divs is not None and not divs.empty:
                cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
                last12 = divs[divs.index >= cutoff]
                div_rate = _to_float(last12.sum()) if not last12.empty else 0.0
                cnt = int(last12.shape[0]) if not last12.empty else 0
                if cnt >= 10: freq, freq_text = 12,"Månads"
                elif cnt >= 3: freq, freq_text = 4,"Kvartals"
                elif cnt == 2: freq, freq_text = 2,"Halvårs"
                elif cnt == 1: freq, freq_text = 1,"Års"
                else: freq, freq_text = 0,"Oregelbunden"
                ex_date_str = pd.to_datetime(divs.index.max()).strftime("%Y-%m-%d")
        except Exception:
            pass

        if div_rate == 0.0:
            for k in ("forwardAnnualDividendRate","trailingAnnualDividendRate"):
                v = info.get(k)
                if _to_float(v) > 0:
                    div_rate = _to_float(v)
                    break

        if not ex_date_str:
            ts = info.get("exDividendDate")
            if _to_float(ts) > 0:
                ex_date_str = pd.to_datetime(int(ts), unit="s", utc=True).strftime("%Y-%m-%d")

        name = info.get("shortName") or info.get("longName") or ticker
        return {
            "name": name,
            "kurs": price,
            "valuta": currency,
            "utdelning": div_rate,
            "frekvens": int(freq),
            "frekvens_text": freq_text,
            "ex_date": ex_date_str,
            "uppdaterad": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    except Exception as e:
        st.warning(f"Kunde inte hämta Yahoo-data för {ticker}: {e}")
        return None

# ── Beräkningar ────────────────────────────────────────────────────────────
def beräkna_allt(df: pd.DataFrame):
    d = säkerställ_kolumner(df).copy()

    # Effektiv utd/år (manuell lås prioriteras)
    use_manual = (d["Lås utdelning"] == True) & (d["Utdelning/år (manuell)"].apply(_to_float) > 0)
    d["Utdelning/år_eff"] = d["Utdelning/år"].apply(_to_float)
    d.loc[use_manual, "Utdelning/år_eff"] = d.loc[use_manual, "Utdelning/år (manuell)"].apply(_to_float)

    d["Utdelningskälla"] = np.where(use_manual, "Manuell 🔒", "Yahoo")

    # SEK-pris
    d["Kurs (SEK)"] = (d["Aktuell kurs"].apply(_to_float) * d["Valuta"].map(fx_for)).round(6)

    # Årlig utdelning i SEK
    d["Antal aktier"] = d["Antal aktier"].apply(_to_float)
    d["Årlig utdelning (SEK)"] = (d["Antal aktier"] * d["Utdelning/år_eff"].apply(_to_float) * d["Valuta"].map(fx_for)).round(2)

    # Direktavkastning
    mask = d["Aktuell kurs"].apply(_to_float) > 0
    d["Direktavkastning (%)"] = 0.0
    d.loc[mask, "Direktavkastning (%)"] = (100.0 * d.loc[mask, "Utdelning/år_eff"].apply(_to_float) / d.loc[mask, "Aktuell kurs"].apply(_to_float)).round(2)

    # Marknadsvärde / vikt
    d["Marknadsvärde (SEK)"] = (d["Antal aktier"] * d["Kurs (SEK)"]).round(2)
    tot_mv = float(d["Marknadsvärde (SEK)"].sum()) or 1.0
    d["Portföljandel (%)"] = (100.0 * d["Marknadsvärde (SEK)"] / tot_mv).round(2)

    # Insatt & P/L (GAV i lokal valuta)
    d["GAV"] = d["GAV"].apply(_to_float)
    d["Insatt (SEK)"] = (d["Antal aktier"] * d["GAV"] * d["Valuta"].map(fx_for)).round(2)
    d["Orealiserad P/L (SEK)"] = (d["Marknadsvärde (SEK)"] - d["Insatt (SEK)"]).round(2)
    d["Orealiserad P/L (%)"]  = np.where(d["Insatt (SEK)"]>0, (100.0*d["Orealiserad P/L (SEK)"]/d["Insatt (SEK)"]).round(2), 0.0)

    # Defaults
    d["Frekvens/år"] = d["Frekvens/år"].apply(_to_float).replace(0, 4)
    d["Payment-lag (dagar)"] = d["Payment-lag (dagar)"].apply(_to_float).replace(0, 30)
    return d

def _next_pay(ex_date_str, freq_per_year, payment_lag_days):
    ts = pd.to_datetime(ex_date_str, errors="coerce")
    if pd.isna(ts): return ""
    exd = ts.date()
    f = int(round(_to_float(freq_per_year))); f = max(f,1)
    lag = int(round(_to_float(payment_lag_days))); lag = max(lag,0)
    step = max(1, int(round(365.0/f)))
    today = date.today()
    while exd < today:
        exd = exd + timedelta(days=step)
    pay = exd + timedelta(days=lag)
    return pay.strftime("%Y-%m-%d")

def uppdatera_nästa_utd(d: pd.DataFrame):
    d = d.copy()
    d["Nästa utbetalning (est)"] = [
        _next_pay(d.at[i,"Ex-Date"], d.at[i,"Frekvens/år"], d.at[i,"Payment-lag (dagar)"]) for i in d.index
    ]
    return d

# ── Kalender ───────────────────────────────────────────────────────────────
def _gen_payment_dates(first_ex_date, fpy, lag_days, months=12):
    ts = pd.to_datetime(first_ex_date, errors="coerce")
    if pd.isna(ts): return []
    exd = ts.date()
    f = max(1, int(round(_to_float(fpy))))
    lag = max(0, int(round(_to_float(lag_days))))
    step = max(1, int(round(365.0/f)))
    today = date.today()
    horizon = today + timedelta(days=int(round(months*30.44)))
    while exd < today:
        exd = exd + timedelta(days=step)
    out = []
    pay = exd + timedelta(days=lag)
    while pay <= horizon:
        out.append(pay)
        exd = exd + timedelta(days=step)
        pay = exd + timedelta(days=lag)
    return out

def prognos_kalender(df: pd.DataFrame, months=12):
    d = uppdatera_nästa_utd(beräkna_allt(df)).copy()
    rows = []
    for _, r in d.iterrows():
        per_share_local = _to_float(r.get("Utdelning/år_eff", 0.0)) / max(1.0, _to_float(r.get("Frekvens/år",4)))
        qty = _to_float(r.get("Antal aktier",0))
        fx = fx_for(r.get("Valuta","SEK"))
        per_payment_sek = round(per_share_local * fx * qty, 2)
        if per_payment_sek <= 0: 
            continue
        pays = _gen_payment_dates(r.get("Ex-Date",""), r.get("Frekvens/år",4), r.get("Payment-lag (dagar)",30), months)
        for p in pays:
            rows.append({"Datum": p, "Ticker": r["Ticker"], "Belopp (SEK)": per_payment_sek})
    if not rows:
        return pd.DataFrame(columns=["Månad","Utdelning (SEK)"]), pd.DataFrame()
    cal = pd.DataFrame(rows)
    cal["Månad"] = cal["Datum"].apply(lambda d: f"{d.year}-{str(d.month).zfill(2)}")
    monthly = cal.groupby("Månad", as_index=False)["Belopp (SEK)"].sum().rename(columns={"Belopp (SEK)":"Utdelning (SEK)"}).sort_values("Månad")
    return monthly, cal

# ── Köpförslag (≈500 kr/köp, respekterar regler) ───────────────────────────
MIN_COURTAGE_RATE = 0.0025
MIN_COURTAGE_SEK  = 1.0
FX_FEE_RATE       = 0.0025

def _fees(order_value_sek: float, foreign: bool):
    courtage = max(MIN_COURTAGE_RATE*order_value_sek, MIN_COURTAGE_SEK)
    fx_fee   = (FX_FEE_RATE*order_value_sek) if foreign else 0.0
    return round(courtage,2), round(fx_fee,2), round(courtage+fx_fee,2)

def _cap_by_name(Vi, T, price_sek, pct):
    m = pct/100.0; num = m*T - Vi; den = (1.0-m)*price_sek
    if den <= 0: return 0
    return int(max(0, math.floor(num/den)))

def _cap_by_cat(C, T, price_sek, pct):
    m = pct/100.0; num = m*T - C; den = (1.0-m)*price_sek
    if den <= 0: return 0
    return int(max(0, math.floor(num/den)))

def suggest_plan(df: pd.DataFrame, cash: float, per_trade: float, max_name_pct: float, cat_goals: dict):
    d = uppdatera_nästa_utd(beräkna_allt(df)).copy()
    if d.empty or cash <= 0: 
        return pd.DataFrame(), pd.DataFrame()

    T = float(d["Marknadsvärde (SEK)"].sum())
    cat_mv = d.groupby("Kategori", as_index=False)["Marknadsvärde (SEK)"].sum().set_index("Kategori")["Marknadsvärde (SEK)"].to_dict()

    # rankingpoäng
    da = d["Direktavkastning (%)"].apply(_to_float).clip(lower=0, upper=20)
    under = (max_name_pct - d["Portföljandel (%)"].apply(_to_float)).clip(lower=0)
    def _days_to(s):
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt): return 9999
        return max(0, (dt.date()-date.today()).days)
    days = d["Nästa utbetalning (est)"].apply(_days_to)
    score = (0.5*(da/20.0) + 0.35*(under/max_name_pct) + 0.15*((90-days.clip(upper=90))/90.0)).fillna(0)*100.0

    d["score"] = score
    d = d.sort_values("score", ascending=False)

    step_rows = []
    used = 0.0
    while used + 1e-9 < cash:
        picked = None
        for _, r in d.iterrows():
            tkr = r["Ticker"]; price = _to_float(r["Kurs (SEK)"])
            if price <= 0: continue
            cat = str(r["Kategori"]) or "Other"
            Vi  = _to_float(r["Marknadsvärde (SEK)"])
            Ci  = float(cat_mv.get(cat, 0.0))
            n_name = _cap_by_name(Vi, T, price, max_name_pct)
            n_cat  = _cap_by_cat(Ci, T, price, float(cat_goals.get(cat, 100.0)))
            # räkna ut ca antal för per_trade
            foreign = str(r["Valuta"]).upper() != "SEK"
            # prova n från 1.. uppåt tills kostnad ~ per_trade (avrundas uppåt)
            n_try = max(1, int(math.ceil(per_trade / price)))
            # ta min mot tak
            n = min(n_try, n_name, n_cat)
            if n <= 0: 
                continue
            gross = n*price
            c,f,tot = _fees(gross, foreign)
            cost = gross + tot
            if used + cost <= cash + 1e-9:
                picked = (tkr, cat, n, price, cost, c, f)
                break
        if not picked:
            break
        tkr, cat, n, price, cost, c, f = picked
        step_rows.append({"Ticker": tkr, "Kategori": cat, "Antal": int(n), "Pris (SEK)": round(price,2),
                          "Kostnad (SEK)": round(cost,2)})
        # uppdatera simulerad portfölj
        T += n*price
        cat_mv[cat] = cat_mv.get(cat,0.0) + n*price
        used += cost

    step = pd.DataFrame(step_rows)
    if step.empty:
        return step, step
    sumrows = step.groupby(["Ticker","Kategori"], as_index=False).agg({"Antal":"sum","Kostnad (SEK)":"sum","Pris (SEK)":"last"})
    return step, sumrows

# ── UI: sidopanel ──────────────────────────────────────────────────────────
def sidopanel():
    st.sidebar.header("⚙️ Inställningar")
    st.sidebar.caption("Växelkurser (SEK)")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        usd = st.number_input("USD/SEK", value=float(st.session_state["USDSEK"]), step=0.01, format="%.4f")
        cad = st.number_input("CAD/SEK", value=float(st.session_state["CADSEK"]), step=0.01, format="%.4f")
    with c2:
        eur = st.number_input("EUR/SEK", value=float(st.session_state["EURSEK"]), step=0.01, format="%.4f")
        nok = st.number_input("NOK/SEK", value=float(st.session_state["NOKSEK"]), step=0.01, format="%.4f")
    st.session_state["USDSEK"], st.session_state["CADSEK"], st.session_state["EURSEK"], st.session_state["NOKSEK"] = usd, cad, eur, nok

    if st.sidebar.button("↩︎ Återställ FX"):
        for k,v in DEF_FX.items(): st.session_state[k] = v
        _rerun()

    if st.sidebar.button("📸 Ta backup nu"):
        sh = _open_sheet(); ws = _ensure_worksheet(sh, SHEET_NAME)
        ts = datetime.now().strftime("Backup_%Y%m%d_%H%M%S")
        try:
            sh.duplicate_sheet(source_sheet_id=ws.id, new_sheet_name=ts)
            st.sidebar.success(f"Backup skapad: {ts}")
        except Exception as e:
            st.sidebar.warning(f"Kunde inte skapa backup: {e}")

    # Autosnap (var 30:e minut)
    last_snap = st.session_state.get("last_snap")  # (ts, sheetname)
    now = time.time()
    if (not last_snap) or (now - last_snap[0] > 60*30):
        try:
            sh = _open_sheet(); ws = _ensure_worksheet(sh, SHEET_NAME)
            name = datetime.now().strftime("_Backup_%Y%m%d_%H%M%S")
            sh.duplicate_sheet(source_sheet_id=ws.id, new_sheet_name=name)
            st.sidebar.success(f"Autosnap: skapade {name}")
            st.session_state["last_snap"] = (now, name)
        except Exception:
            pass

    st.sidebar.markdown("---")
    one = st.sidebar.text_input("Uppdatera EN ticker (Yahoo)", placeholder="t.ex. ARR").strip().upper()
    if st.sidebar.button("🔄 Uppdatera EN"):
        if not one: st.sidebar.warning("Ange ticker."); return
        base = säkerställ_kolumner(st.session_state.get("working_df", pd.DataFrame()))
        if one not in base["Ticker"].tolist():
            base = pd.concat([base, pd.DataFrame([{"Ticker": one, "Kategori":"QUALITY"}])], ignore_index=True)
        vals = hamta_yahoo_data(one)
        if vals:
            m = base["Ticker"]==one
            base.loc[m,"Bolagsnamn"] = vals["name"]
            base.loc[m,"Aktuell kurs"] = vals["kurs"]
            base.loc[m,"Valuta"] = vals["valuta"]
            if not bool(base.loc[m,"Lås utdelning"].iloc[0]) and _to_float(vals["utdelning"])>0:
                base.loc[m,"Utdelning/år"] = vals["utdelning"]
            f, ft, xd = vals["frekvens"], vals["frekvens_text"], vals["ex_date"]
            if f>0: base.loc[m,"Frekvens/år"] = f
            if ft:  base.loc[m,"Utdelningsfrekvens"] = ft
            if xd:  base.loc[m,"Ex-Date"] = xd
            base.loc[m,"Källa"] = "Yahoo"
            base.loc[m,"Senaste uppdatering"] = vals["uppdaterad"]
            st.session_state["working_df"] = beräkna_allt(base)
            st.sidebar.success(f"{one} uppdaterad.")

# ── Portföljöversikt ───────────────────────────────────────────────────────
def page_portfolio(df: pd.DataFrame):
    d = uppdatera_nästa_utd(beräkna_allt(df)).copy()

    tot_mv  = float(d["Marknadsvärde (SEK)"].sum())
    tot_ins = float(d["Insatt (SEK)"].sum())
    tot_pl  = float(d["Orealiserad P/L (SEK)"].sum())
    tot_div = float(d["Årlig utdelning (SEK)"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portföljvärde (SEK)", f"{round(tot_mv,2):,}".replace(",", " "))
    c2.metric("Insatt (SEK)", f"{round(tot_ins,2):,}".replace(",", " "))
    c3.metric("Orealiserad P/L (SEK)", f"{round(tot_pl,2):,}".replace(",", " "),
              delta=(f"{(0 if tot_ins==0 else 100.0*tot_pl/tot_ins):.2f}%"))
    c4.metric("Årlig utdelning (SEK)", f"{round(tot_div,2):,}".replace(",", " "))

    show = [
        "Bolagsnamn","Ticker","Valuta","Kategori",
        "Antal aktier","GAV","Aktuell kurs","Kurs (SEK)","Marknadsvärde (SEK)",
        "Direktavkastning (%)","Utdelning/år","Utdelning/år (manuell)","Lås utdelning","Utdelningskälla",
        "Frekvens/år","Utdelningsfrekvens","Ex-Date","Nästa utbetalning (est)",
        "Årlig utdelning (SEK)","Portföljandel (%)","Senaste uppdatering"
    ]
    st.dataframe(d[show], use_container_width=True)

# ── Regler & mål (sparas i flik ”Regler”) ──────────────────────────────────
def page_rules():
    st.subheader("⚖️ Regler & mål")
    rules = _read_rules()
    max_name = st.number_input("Max vikt per bolag (%)", min_value=1.0, max_value=50.0, value=float(rules["max_name"]), step=0.5)
    st.caption("Kategorimål (%) – påverkar köpförslag (dämpar överviktade kategorier).")
    cats = sorted(set(list(DEFAULT_CAT_GOALS.keys()) + list(rules["cat_goals"].keys())))
    grid = []
    for c in cats:
        grid.append({"Kategori": c, "Mål (%)": float(rules["cat_goals"].get(c, DEFAULT_CAT_GOALS.get(c, 0.0)))})
    edit = st.data_editor(pd.DataFrame(grid), use_container_width=True, num_rows="dynamic")
    if st.button("💾 Spara regler"):
        goals = {str(r["Kategori"]): _to_float(r["Mål (%)"]) for _, r in edit.iterrows()}
        _write_rules(max_name, goals)
        st.success("Regler sparade.")
        _rerun()

# ── Lägg till / uppdatera ──────────────────────────────────────────────────
def page_add_update(df: pd.DataFrame):
    st.subheader("➕ Lägg till / ✏ Uppdatera bolag")
    d = säkerställ_kolumner(df).copy()
    val = st.selectbox("Välj bolag", options=["Ny"] + d["Ticker"].tolist())
    if val == "Ny":
        ticker = st.text_input("Ticker").strip().upper()
        qty = st.number_input("Antal aktier", min_value=0, value=0, step=1)
        gav = st.number_input("GAV (i lokal valuta)", min_value=0.0, value=0.0, step=0.01)
        kategori = st.selectbox("Kategori", options=["QUALITY","REIT","mREIT","BDC","MLP","Shipping","Telecom","Tobacco","Utility","Tech","Bank","Industrial","Energy","Finance","Other"])
    else:
        r = d[d["Ticker"]==val].iloc[0]
        ticker = st.text_input("Ticker", value=r["Ticker"]).strip().upper()
        qty = st.number_input("Antal aktier", min_value=0, value=int(_to_int(r["Antal aktier"])), step=1)
        gav = st.number_input("GAV (i lokal valuta)", min_value=0.0, value=float(_to_float(r["GAV"])), step=0.01)
        kategori = st.selectbox("Kategori", options=["QUALITY","REIT","mREIT","BDC","MLP","Shipping","Telecom","Tobacco","Utility","Tech","Bank","Industrial","Energy","Finance","Other"], index=0)
    c1, c2 = st.columns(2)
    if c1.button("🌐 Hämta från Yahoo"):
        if not ticker:
            st.warning("Ange ticker.")
        else:
            vals = hamta_yahoo_data(ticker)
            if vals:
                m = d["Ticker"]==ticker
                if not m.any():
                    d = pd.concat([d, pd.DataFrame([{"Ticker": ticker, "Kategori": kategori}])], ignore_index=True)
                    m = d["Ticker"]==ticker
                d.loc[m,"Bolagsnamn"] = vals["name"]
                d.loc[m,"Valuta"] = vals["valuta"]
                d.loc[m,"Aktuell kurs"] = vals["kurs"]
                if _to_float(vals["utdelning"])>0 and not bool(d.loc[m,"Lås utdelning"].iloc[0]):
                    d.loc[m,"Utdelning/år"] = vals["utdelning"]
                if vals["frekvens"]>0: d.loc[m,"Frekvens/år"]= vals["frekvens"]
                if vals["frekvens_text"]: d.loc[m,"Utdelningsfrekvens"]= vals["frekvens_text"]
                if vals["ex_date"]: d.loc[m,"Ex-Date"]= vals["ex_date"]
                d.loc[m,"Källa"] = "Yahoo"
                d.loc[m,"Senaste uppdatering"] = vals["uppdaterad"]
                st.session_state["working_df"] = beräkna_allt(d)
                st.success("Hämtat & uppdaterat.")
    if c2.button("➕ Lägg till i minnet"):
        if not ticker:
            st.warning("Ange ticker.")
        else:
            m = d["Ticker"]==ticker
            if not m.any():
                d = pd.concat([d, pd.DataFrame([{"Ticker": ticker}])], ignore_index=True)
                m = d["Ticker"]==ticker
            d.loc[m,"Antal aktier"] = qty
            d.loc[m,"GAV"] = gav
            d.loc[m,"Kategori"] = kategori
            st.session_state["working_df"] = beräkna_allt(d)
            st.success("Sparat i minnet (ej Google Sheets).")

    if st.button("💾 Spara alla ändringar till Google Sheets"):
        safe = beräkna_allt(st.session_state.get("working_df", d))
        spara_data(safe)
        st.success("Sparat till Google Sheets.")

# ── Massuppdatera ──────────────────────────────────────────────────────────
def page_massupdate(df: pd.DataFrame):
    st.subheader("⏩ Massuppdatera alla bolag (Yahoo)")
    d = säkerställ_kolumner(df).copy()
    if d.empty:
        st.info("Lägg till minst en ticker först."); return
    if st.button("Starta massuppdatering"):
        ph = st.progress(0)
        for i, tkr in enumerate(d["Ticker"].tolist(), start=1):
            vals = hamta_yahoo_data(tkr)
            m = d["Ticker"]==tkr
            if vals:
                d.loc[m,"Bolagsnamn"] = vals["name"]
                d.loc[m,"Valuta"] = vals["valuta"]
                d.loc[m,"Aktuell kurs"] = vals["kurs"]
                if _to_float(vals["utdelning"])>0 and not bool(d.loc[m,"Lås utdelning"].iloc[0]):
                    d.loc[m,"Utdelning/år"] = vals["utdelning"]
                if vals["frekvens"]>0: d.loc[m,"Frekvens/år"]= vals["frekvens"]
                if vals["frekvens_text"]: d.loc[m,"Utdelningsfrekvens"]= vals["frekvens_text"]
                if vals["ex_date"]: d.loc[m,"Ex-Date"]= vals["ex_date"]
                d.loc[m,"Källa"] = "Yahoo"
                d.loc[m,"Senaste uppdatering"] = vals["uppdaterad"]
            ph.progress(i/len(d))
            time.sleep(0.6)  # snäll mot Yahoo
        st.session_state["working_df"] = beräkna_allt(d)
        st.success("Massuppdatering klar.")

# ── Kalender ───────────────────────────────────────────────────────────────
def page_calendar(df: pd.DataFrame):
    st.subheader("📅 Utdelningskalender")
    months = st.selectbox("Prognoshorisont (mån)", options=[12,24,36], index=0)
    monthly, cal = prognos_kalender(df, months)
    if monthly.empty:
        st.info("Ingen prognos (saknar ex-date/frekvens/utdelningsdata).")
        return
    st.dataframe(monthly, use_container_width=True)
    st.bar_chart(monthly.set_index("Månad")["Utdelning (SEK)"])
    with st.expander("Detaljer per betalning"):
        st.dataframe(cal.sort_values("Datum"), use_container_width=True)

# ── Köpförslag & plan ──────────────────────────────────────────────────────
def page_buy_plan(df: pd.DataFrame):
    st.subheader("🎯 Köpförslag & plan (≈500 kr per köp)")
    rules = _read_rules()
    cash = st.number_input("Tillgänglig kassa (SEK)", min_value=0.0, value=2000.0, step=100.0)
    per_trade = st.number_input("Belopp per köp (≈)", min_value=200.0, value=500.0, step=50.0)
    max_name = st.number_input("Max per bolag (%)", min_value=1.0, max_value=50.0, value=float(rules["max_name"]), step=0.5)
    if st.button("Beräkna plan"):
        step, summary = suggest_plan(df, cash, per_trade, max_name, rules["cat_goals"])
        if step.empty:
            st.info("Ingen köpkandidat klarade reglerna/given kassa.")
        else:
            st.write("Plan – steg för steg:")
            st.dataframe(step, use_container_width=True)
            st.write("Summering per ticker:")
            st.dataframe(summary, use_container_width=True)

# ── Spara (preview + antiwipe) ─────────────────────────────────────────────
def page_save(df: pd.DataFrame):
    st.subheader("💾 Spara nu till Google Sheets")
    preview = uppdatera_nästa_utd(beräkna_allt(df))
    st.write("Rader som sparas:", len(preview))
    st.dataframe(preview[["Ticker","Bolagsnamn","Valuta","Kategori","Antal aktier","GAV","Aktuell kurs","Utdelning/år","Kurs (SEK)","Årlig utdelning (SEK)"]], use_container_width=True)
    allow = st.checkbox("⚠️ Tillåt riskabel överskrivning (använd bara om du vet vad du gör)", value=True)
    if st.button("✅ Bekräfta och spara"):
        if not allow:
            st.warning("Bocka i rutan för att spara."); return
        spara_data(preview)
        st.success("Sparat.")

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    st.title("Relative Yield – utdelningsportfölj")

    # Läs in arbetskopia
    if "working_df" not in st.session_state:
        st.session_state["working_df"] = hamta_data()

    base = säkerställ_kolumner(st.session_state["working_df"])
    sidopanel()

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

    if page == "📦 Portföljöversikt":
        page_portfolio(base)
    elif page == "⚖️ Regler & mål":
        page_rules()
    elif page == "➕ Lägg till / ✏ Uppdatera bolag":
        page_add_update(base)
    elif page == "⏩ Massuppdatera alla":
        page_massupdate(base)
    elif page == "🎯 Köpförslag & plan":
        page_buy_plan(base)
    elif page == "📅 Utdelningskalender":
        page_calendar(base)
    elif page == "💾 Spara":
        page_save(base)

    # Uppdatera session
    st.session_state["working_df"] = säkerställ_kolumner(st.session_state.get("working_df", base))

if __name__ == "__main__":
    main()
