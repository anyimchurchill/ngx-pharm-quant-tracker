import io
import numpy as np
import pandas as pd

REFERENCE_FUNDAMENTALS_CSV = """ticker,company,share_price_ngn,shares_out_m,market_cap_bn,revenue_ngn_m,cogs_ngn_m,gross_profit_ngn_m,sga_ngn_m,da_ngn_m,operating_profit_ngn_m,interest_expense_ngn_m,net_income_ngn_m,capex_ngn_m,cash_ngn_m,short_term_debt_ngn_m,long_term_debt_ngn_m,total_equity_ngn_m,inventory_ngn_m,receivables_ngn_m,payables_ngn_m,beta_est,reporting_basis
MECURE,MeCure Industries Plc,56.20,4000.0,224.80,60000.0,36000.0,24000.0,16100.0,1600.0,6300.0,1500.0,3800.0,2400.0,3500.0,7000.0,5000.0,18000.0,9500.0,8000.0,7000.0,1.05,Reference / 9M25-Mkt
FIDSON,Fidson Healthcare Plc,91.80,2295.0,210.68,93082.0,63677.0,48935.0,25191.0,2500.0,21244.0,5200.0,11315.0,4200.0,8500.0,14800.0,11000.0,34000.0,18500.0,14000.0,23716.0,0.92,Reference / 9M26-Mkt
MAYBAKER,May & Baker Nigeria Plc,40.00,1725.2,69.01,38300.0,25200.0,13100.0,3601.0,1050.0,8449.0,650.0,6831.0,1500.0,2800.0,2500.0,2000.0,14500.0,6200.0,5100.0,4200.0,0.85,Reference / 9M26-Mkt
NEIMETH,Neimeth International Phar.,7.10,4273.0,30.34,5000.0,3100.0,1900.0,1340.0,220.0,340.0,180.0,940.0,350.0,600.0,2000.0,1200.0,2500.0,1400.0,1100.0,950.0,1.15,Reference / 9M25-Mkt
MORISON,Morison Industries Plc,10.40,1255.8,13.06,1200.0,820.0,380.0,280.0,55.0,45.0,25.0,13.0,80.0,200.0,200.0,100.0,900.0,250.0,180.0,160.0,0.75,Reference / Mkt
"""


def load_reference_fundamentals() -> pd.DataFrame:
    return pd.read_csv(io.StringIO(REFERENCE_FUNDAMENTALS_CSV))


def build_reference_price_panel(periods: int = 252, seed: int = 2026) -> pd.DataFrame:
    """Deterministic 252-session trading panel anchored to current spot prices."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-11-18", periods=periods)
    bench_rets = rng.normal(loc=0.0008, scale=0.0085, size=periods)
    bench_level = 100000.0 * np.cumprod(1.0 + bench_rets)

    specs = {
        "MECURE": (56.20, 1.05, 0.0009, 0.016),
        "FIDSON": (91.80, 0.92, 0.0012, 0.014),
        "MAYBAKER": (40.00, 0.85, 0.0010, 0.013),
        "NEIMETH": (7.10, 1.15, 0.0005, 0.022),
        "MORISON": (10.40, 0.75, 0.0004, 0.019),
    }

    out = {"Date": dates, "NGX_ASI": bench_level}
    for ticker, (spot, beta, alpha, sigma) in specs.items():
        r = alpha + beta * bench_rets + rng.normal(0.0, sigma, size=periods)
        path = np.cumprod(1.0 + r)
        out[ticker] = np.round(path * (spot / path[-1]), 2)
    return pd.DataFrame(out)


def fmt_multiple(x: float) -> str:
    if pd.isna(x) or not np.isfinite(x) or x <= 0:
        return "N/M"
    return f"{x:.1f}x"


def compute_wacc(
    rf: float,
    erp: float,
    beta: float,
    country_risk_prem: float,
    pre_tax_kd: float,
    tax_rate: float,
    debt_val: float,
    equity_val: float,
) -> dict:
    ke = rf + beta * erp + country_risk_prem
    after_tax_kd = pre_tax_kd * (1.0 - tax_rate)
    total_cap = max(max(debt_val, 0.0) + max(equity_val, 0.0), 1e-6)
    we = max(equity_val, 0.0) / total_cap
    wd = max(debt_val, 0.0) / total_cap
    wacc = (we * ke) + (wd * after_tax_kd)
    return {
        "ke": ke,
        "after_tax_kd": after_tax_kd,
        "we": we,
        "wd": wd,
        "wacc": wacc,
    }


def compute_price_analytics(price_df: pd.DataFrame, rf_annual: float) -> pd.DataFrame:
    df = price_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    bench_col = "NGX_ASI" if "NGX_ASI" in df.columns else df.columns[1]
    tickers = [c for c in df.columns if c not in ("Date", bench_col)]

    rets = df[tickers + [bench_col]].pct_change().dropna()
    bench_r = rets[bench_col]
    bench_var = float(bench_r.var())
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0

    rows = []
    n_obs = max(len(rets), 1)
    for t in tickers:
        px = df[t].astype(float)
        r = rets[t].astype(float)

        tot_ret = (px.iloc[-1] / px.iloc[0]) - 1.0
        cagr = ((1.0 + tot_ret) ** (252.0 / n_obs)) - 1.0
        ann_vol = float(r.std() * np.sqrt(252.0))

        downside = np.minimum(r - rf_daily, 0.0)
        downside_ann = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(252.0))

        sharpe = (cagr - rf_annual) / ann_vol if ann_vol > 0 else np.nan
        sortino = (cagr - rf_annual) / downside_ann if downside_ann > 0 else np.nan
        beta_emp = float(r.cov(bench_r) / bench_var) if bench_var > 0 else np.nan

        peak = px.cummax()
        mdd = float(((px - peak) / peak).min())

        rows.append(
            {
                "Ticker": t,
                "Spot (₦)": round(float(px.iloc[-1]), 2),
                "Period Ret (%)": round(tot_ret * 100.0, 2),
                "Ann. Vol (%)": round(ann_vol * 100.0, 2),
                "Max DD (%)": round(mdd * 100.0, 2),
                "Emp. Beta": round(beta_emp, 2),
                "Sharpe": round(sharpe, 2),
                "Sortino": round(sortino, 2),
            }
        )
    return pd.DataFrame(rows)


def compute_financial_statements_metrics(df: pd.DataFrame, tax_rate: float) -> pd.DataFrame:
    out = df.copy()
    out["total_debt_ngn_m"] = out["short_term_debt_ngn_m"] + out["long_term_debt_ngn_m"]
    out["net_debt_ngn_m"] = out["total_debt_ngn_m"] - out["cash_ngn_m"]
    out["ebitda_ngn_m"] = out["operating_profit_ngn_m"] + out["da_ngn_m"]

    out["gross_margin_pct"] = np.where(out["revenue_ngn_m"] > 0, (out["gross_profit_ngn_m"] / out["revenue_ngn_m"]) * 100.0, np.nan)
    out["ebit_margin_pct"] = np.where(out["revenue_ngn_m"] > 0, (out["operating_profit_ngn_m"] / out["revenue_ngn_m"]) * 100.0, np.nan)
    out["ebitda_margin_pct"] = np.where(out["revenue_ngn_m"] > 0, (out["ebitda_ngn_m"] / out["revenue_ngn_m"]) * 100.0, np.nan)
    out["net_margin_pct"] = np.where(out["revenue_ngn_m"] > 0, (out["net_income_ngn_m"] / out["revenue_ngn_m"]) * 100.0, np.nan)

    invested_cap = (out["total_debt_ngn_m"] + out["total_equity_ngn_m"] - out["cash_ngn_m"]).clip(lower=1.0)
    nopat = out["operating_profit_ngn_m"] * (1.0 - tax_rate)
    out["roic_pct"] = (nopat / invested_cap) * 100.0
    out["roe_pct"] = np.where(out["total_equity_ngn_m"] > 0, (out["net_income_ngn_m"] / out["total_equity_ngn_m"]) * 100.0, np.nan)

    out["debt_to_equity"] = np.where(out["total_equity_ngn_m"] > 0, out["total_debt_ngn_m"] / out["total_equity_ngn_m"], np.nan)
    out["net_debt_to_ebitda"] = np.where(out["ebitda_ngn_m"] > 0, out["net_debt_ngn_m"] / out["ebitda_ngn_m"], np.nan)

    out["dio"] = np.where(out["cogs_ngn_m"] > 0, (out["inventory_ngn_m"] / out["cogs_ngn_m"]) * 365.0, np.nan)
    out["dso"] = np.where(out["revenue_ngn_m"] > 0, (out["receivables_ngn_m"] / out["revenue_ngn_m"]) * 365.0, np.nan)
    out["dpo"] = np.where(out["cogs_ngn_m"] > 0, (out["payables_ngn_m"] / out["cogs_ngn_m"]) * 365.0, np.nan)
    out["ccc"] = out["dio"] + out["dso"] - out["dpo"]

    out["ev_ngn_m"] = (out["market_cap_bn"] * 1000.0) + out["net_debt_ngn_m"]
    eps_implied = np.where(out["shares_out_m"] > 0, (out["net_income_ngn_m"] / out["shares_out_m"]), np.nan)
    out["pe"] = np.where(eps_implied > 0, out["share_price_ngn"] / eps_implied, np.nan)
    out["ev_ebitda"] = np.where(out["ebitda_ngn_m"] > 0, out["ev_ngn_m"] / out["ebitda_ngn_m"], np.nan)
    out["ev_sales"] = np.where(out["revenue_ngn_m"] > 0, out["ev_ngn_m"] / out["revenue_ngn_m"], np.nan)
    out["ps"] = np.where(out["revenue_ngn_m"] > 0, (out["market_cap_bn"] * 1000.0) / out["revenue_ngn_m"], np.nan)
    out["pb"] = np.where(out["total_equity_ngn_m"] > 0, (out["market_cap_bn"] * 1000.0) / out["total_equity_ngn_m"], np.nan)

    return out


def run_institutional_dcf(
    base_rev: float,
    rev_growth: float,
    gross_margin: float,
    opex_pct_rev: float,
    da_pct_rev: float,
    capex_pct_rev: float,
    nwc_pct_delta_rev: float,
    tax_rate: float,
    wacc: float,
    terminal_growth: float,
    exit_ev_ebitda: float,
    terminal_weight_ggm: float,
    cash_ngn_m: float,
    debt_ngn_m: float,
    shares_out_m: float,
    years: int = 5,
) -> dict:
    if wacc <= terminal_growth:
        raise ValueError("WACC must exceed terminal growth rate.")
    if shares_out_m <= 0:
        raise ValueError("Shares outstanding must be greater than zero.")

    rows = []
    raw_ufcfs = []
    raw_ebitdas = []
    pv_ufcf_total = 0.0

    rev = float(base_rev)
    for yr in range(1, years + 1):
        prev_rev = rev
        rev = prev_rev * (1.0 + rev_growth)
        delta_rev = rev - prev_rev

        gp = rev * gross_margin
        ebitda = max(gp - (rev * opex_pct_rev), 0.0)
        ebit = ebitda - (rev * da_pct_rev)
        nopat = ebit * (1.0 - tax_rate)
        capex = rev * capex_pct_rev
        delta_nwc = delta_rev * nwc_pct_delta_rev

        ufcf = nopat + (rev * da_pct_rev) - capex - delta_nwc
        disc_factor = (1.0 + wacc) ** yr
        pv_ufcf = ufcf / disc_factor
        pv_ufcf_total += pv_ufcf

        raw_ufcfs.append(ufcf)
        raw_ebitdas.append(ebitda)

        rows.append(
            {
                "Year": f"FY+{yr}",
                "Revenue (₦M)": round(rev, 1),
                "Gross Profit (₦M)": round(gp, 1),
                "EBITDA (₦M)": round(ebitda, 1),
                "EBIT (₦M)": round(ebit, 1),
                "NOPAT (₦M)": round(nopat, 1),
                "CapEx (₦M)": round(capex, 1),
                "ΔNWC (₦M)": round(delta_nwc, 1),
                "UFCF (₦M)": round(ufcf, 1),
                "PV of UFCF (₦M)": round(pv_ufcf, 1),
            }
        )

    disc_n = (1.0 + wacc) ** years
    tv_ggm = (raw_ufcfs[-1] * (1.0 + terminal_growth)) / (wacc - terminal_growth)
    pv_tv_ggm = tv_ggm / disc_n

    tv_mult = raw_ebitdas[-1] * exit_ev_ebitda
    pv_tv_mult = tv_mult / disc_n

    w_ggm = float(np.clip(terminal_weight_ggm, 0.0, 1.0))
    pv_tv_blended = (w_ggm * pv_tv_ggm) + ((1.0 - w_ggm) * pv_tv_mult)

    ev_blended = pv_ufcf_total + pv_tv_blended
    eq_val_blended = ev_blended + cash_ngn_m - debt_ngn_m
    px_blended = max(eq_val_blended / shares_out_m, 0.0)

    ev_ggm_only = pv_ufcf_total + pv_tv_ggm
    px_ggm_only = max((ev_ggm_only + cash_ngn_m - debt_ngn_m) / shares_out_m, 0.0)

    ev_mult_only = pv_ufcf_total + pv_tv_mult
    px_mult_only = max((ev_mult_only + cash_ngn_m - debt_ngn_m) / shares_out_m, 0.0)

    return {
        "schedule": pd.DataFrame(rows),
        "pv_ufcf_total": pv_ufcf_total,
        "pv_tv_ggm": pv_tv_ggm,
        "pv_tv_mult": pv_tv_mult,
        "pv_tv_blended": pv_tv_blended,
        "ev_blended": ev_blended,
        "eq_val_blended": eq_val_blended,
        "px_blended": px_blended,
        "px_ggm_only": px_ggm_only,
        "px_mult_only": px_mult_only,
    }


def sensitivity_grid_price(
    base_rev: float,
    rev_growth: float,
    gross_margin: float,
    opex_pct_rev: float,
    da_pct_rev: float,
    capex_pct_rev: float,
    nwc_pct_delta_rev: float,
    tax_rate: float,
    exit_ev_ebitda: float,
    terminal_weight_ggm: float,
    cash_ngn_m: float,
    debt_ngn_m: float,
    shares_out_m: float,
    center_wacc: float,
    center_tg: float,
) -> pd.DataFrame:
    wacc_vals = [max(0.10, center_wacc - 0.03), max(0.10, center_wacc - 0.015), center_wacc, center_wacc + 0.015, center_wacc + 0.03]
    tg_vals = [max(0.01, center_tg - 0.02), max(0.01, center_tg - 0.01), center_tg, center_tg + 0.01, center_tg + 0.02]

    grid = {}
    for tg in tg_vals:
        col = []
        for w in wacc_vals:
            if w <= tg:
                col.append(np.nan)
            else:
                res = run_institutional_dcf(
                    base_rev, rev_growth, gross_margin, opex_pct_rev, da_pct_rev,
                    capex_pct_rev, nwc_pct_delta_rev, tax_rate, w, tg,
                    exit_ev_ebitda, terminal_weight_ggm, cash_ngn_m, debt_ngn_m, shares_out_m,
                )
                col.append(round(res["px_blended"], 2))
        grid[f"g = {tg*100:.1f}%"] = col

    out = pd.DataFrame(grid, index=[f"WACC {w*100:.1f}%" for w in wacc_vals])
    out.index.name = "WACC \\ Terminal g"
    return out


def compute_peer_multiple_stats(fin_df: pd.DataFrame) -> pd.DataFrame:
    stat_rows = []
    for col_name, label in [("pe", "P/E"), ("ev_ebitda", "EV/EBITDA"), ("ev_sales", "EV/Sales"), ("ps", "P/S"), ("pb", "P/B")]:
        s = fin_df[col_name].dropna()
        s = s[s > 0]
        if len(s) > 0:
            stat_rows.append(
                {
                    "Multiple": label,
                    "25th Pct": f"{np.percentile(s, 25):.1f}x",
                    "Median": f"{np.median(s):.1f}x",
                    "Mean": f"{np.mean(s):.1f}x",
                    "75th Pct": f"{np.percentile(s, 75):.1f}x",
                }
            )
    return pd.DataFrame(stat_rows)
