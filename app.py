import numpy as np
import pandas as pd
import streamlit as st

from valuation_engine import (
    build_reference_price_panel,
    compute_financial_statements_metrics,
    compute_peer_multiple_stats,
    compute_price_analytics,
    compute_wacc,
    fmt_multiple,
    load_reference_fundamentals,
    run_institutional_dcf,
    sensitivity_grid_price,
)

st.set_page_config(
    page_title="NGX Pharma Wall Street Valuation Terminal",
    page_icon="🏦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 8px;
        padding: 12px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: Macro Assumptions, CAPM Inputs & CSV Data Ingestion
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Macro & CAPM Inputs")
    rf = st.slider("Risk-Free Rate, Rf (%)", 5.0, 30.0, 18.0, 0.5) / 100.0
    erp = st.slider("Equity Risk Premium, ERP (%)", 3.0, 15.0, 7.0, 0.5) / 100.0
    crp = st.slider("Country / Liquidity Risk Add-on (%)", 0.0, 10.0, 2.0, 0.5) / 100.0
    tax_rate = st.slider("Corporate Tax Rate (%)", 10.0, 40.0, 30.0, 1.0) / 100.0
    pre_tax_kd = st.slider("Pre-Tax Cost of Debt, Kd (%)", 8.0, 35.0, 21.0, 0.5) / 100.0

    st.divider()
    st.markdown("### 📂 Data Ingestion")
    fund_upload = st.file_uploader("Fundamentals CSV (Optional)", type=["csv"], key="fund_csv")
    price_upload = st.file_uploader("Historical Prices CSV (Optional)", type=["csv"], key="px_csv")

    raw_fund = pd.read_csv(fund_upload) if fund_upload else load_reference_fundamentals()
    prices_df = pd.read_csv(price_upload) if price_upload else build_reference_price_panel()

    st.download_button(
        label="⬇️ Download Fundamentals CSV",
        data=raw_fund.to_csv(index=False),
        file_name="ngx_pharma_fundamentals.csv",
        mime="text/csv",
        use_container_width=True,
    )

fin_df = compute_financial_statements_metrics(raw_fund, tax_rate=tax_rate)
risk_df = compute_price_analytics(prices_df, rf_annual=rf)
stats_df = compute_peer_multiple_stats(fin_df)

# ---------------------------------------------------------------------------
# Top Header & Executive KPI Strip
# ---------------------------------------------------------------------------
st.title("🏦 NGX Pharma Wall Street Valuation & Quant Terminal")
st.caption(
    "Institutional equity research workbench covering CAPM/WACC builds, 5-year Unlevered Free Cash Flow (UFCF) DCF (Gordon Growth + Exit Multiple), Bull/Base/Bear cases, working-capital cycle diagnostics, and peer comps."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Covered Universe", f"{len(fin_df)} NGX Tickers", "Custom CSV" if fund_upload else "Reference Dataset")
k2.metric("Peer Median EV/EBITDA", fmt_multiple(float(fin_df["ev_ebitda"].median())), f"Median P/S: {fmt_multiple(float(fin_df['ps'].median()))}")
k3.metric("Peer Median CCC", f"{float(fin_df['ccc'].median()):.0f} Days", f"Median ROIC: {float(fin_df['roic_pct'].median()):.1f}%")
k4.metric("Macro Anchor", f"Rf {rf*100:.1f}% | Tax {tax_rate*100:.0f}%", "Uploaded Prices" if price_upload else "252D Ref. Panel")

st.divider()

# ---------------------------------------------------------------------------
# Four Institutional Research Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. Market Risk & Return",
        "2. Operating & Working Capital",
        "3. Institutional DCF & Scenarios",
        "4. Peer Comps & Multiples",
    ]
)

with tab1:
    st.subheader("Indexed Price Performance (Base = 100)")
    px_indexed = prices_df.copy()
    px_indexed["Date"] = pd.to_datetime(px_indexed["Date"])
    px_indexed = px_indexed.sort_values("Date").set_index("Date")
    st.line_chart((px_indexed / px_indexed.iloc[0]) * 100.0, use_container_width=True)

    st.subheader("Volatility, Drawdown, Beta & Risk-Adjusted Return")
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Profitability, Return on Invested Capital & Cash Conversion Cycle")
    st.markdown(
        "Pharmaceutical cash generation depends on margin discipline, return on invested capital (**ROIC**), and working-capital velocity (**CCC = DIO + DSO - DPO**)."
    )

    op_table = pd.DataFrame(
        {
            "Ticker": fin_df["ticker"],
            "Company": fin_df["company"],
            "Gross %": fin_df["gross_margin_pct"].round(1),
            "EBITDA %": fin_df["ebitda_margin_pct"].round(1),
            "EBIT %": fin_df["ebit_margin_pct"].round(1),
            "Net %": fin_df["net_margin_pct"].round(1),
            "ROIC %": fin_df["roic_pct"].round(1),
            "ROE %": fin_df["roe_pct"].round(1),
            "Net Debt/EBITDA": fin_df["net_debt_to_ebitda"].map(fmt_multiple),
            "DIO": fin_df["dio"].round(0),
            "DSO": fin_df["dso"].round(0),
            "DPO": fin_df["dpo"].round(0),
            "CCC (Days)": fin_df["ccc"].round(0),
            "Basis": fin_df["reporting_basis"],
        }
    )
    st.dataframe(op_table, use_container_width=True, hide_index=True)

    st.subheader("Working Capital Cycle Comparison (Days)")
    st.bar_chart(op_table.set_index("Ticker")[["DIO", "DSO", "CCC (Days)"]], use_container_width=True)

with tab3:
    st.subheader("5-Year UFCF DCF, CAPM/WACC Build & Scenario Modeling")

    top_ctrl1, top_ctrl2 = st.columns([1, 2])
    with top_ctrl1:
        target_ticker = st.selectbox("Target Ticker", fin_df["ticker"].tolist())
    with top_ctrl2:
        scenario = st.radio(
            "Operating Scenario",
            ["Base Case", "Bull Case", "Bear Case", "Custom"],
            horizontal=True,
        )

    row = fin_df[fin_df["ticker"] == target_ticker].iloc[0]

    base_gm = float(np.clip(row["gross_margin_pct"] / 100.0, 0.15, 0.70))
    base_ebit_m = float(np.clip(row["ebit_margin_pct"] / 100.0, 0.04, 0.35))
    base_opex = max(base_gm - base_ebit_m, 0.03)
    base_da = float(np.clip(row["da_ngn_m"] / max(row["revenue_ngn_m"], 1.0), 0.01, 0.10))
    base_capex = float(np.clip(row["capex_ngn_m"] / max(row["revenue_ngn_m"], 1.0), 0.01, 0.15))

    if scenario == "Bull Case":
        def_g, def_gm, def_opex, def_nwc, def_wacc_adj = 0.24, min(base_gm + 0.03, 0.75), max(base_opex - 0.02, 0.02), 0.08, -0.015
    elif scenario == "Bear Case":
        def_g, def_gm, def_opex, def_nwc, def_wacc_adj = 0.07, max(base_gm - 0.04, 0.10), min(base_opex + 0.03, 0.60), 0.18, 0.02
    else:
        def_g, def_gm, def_opex, def_nwc, def_wacc_adj = 0.16, base_gm, base_opex, 0.12, 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Operating & Working Capital**")
        g_rate = st.slider("Revenue Growth (%)", 0.0, 45.0, round(def_g * 100, 1), 0.5) / 100.0
        gm_rate = st.slider("Gross Margin (%)", 10.0, 75.0, round(def_gm * 100, 1), 0.5) / 100.0
        opex_rate = st.slider("OpEx ex-D&A (% of Rev)", 2.0, 60.0, round(def_opex * 100, 1), 0.5) / 100.0
        nwc_delta_rate = st.slider("ΔNWC / ΔRevenue (%)", -5.0, 35.0, round(def_nwc * 100, 1), 0.5) / 100.0

    with c2:
        st.markdown("**CapEx, D&A & WACC**")
        beta_in = st.number_input("Beta (β)", min_value=0.30, max_value=2.50, value=float(row["beta_est"]), step=0.05)
        capex_rate = st.slider("CapEx (% of Revenue)", 1.0, 18.0, round(base_capex * 100, 1), 0.5) / 100.0
        da_rate = st.slider("D&A (% of Revenue)", 0.5, 12.0, round(base_da * 100, 1), 0.5) / 100.0
        wacc_override = st.slider("Discount Rate / WACC (%)", 12.0, 36.0, round((rf + beta_in * erp + crp + def_wacc_adj) * 100, 1), 0.5) / 100.0

    with c3:
        st.markdown("**Terminal Value Blending**")
        tg_rate = st.slider("Terminal Growth, g (%)", 1.0, 10.0, 5.0, 0.5) / 100.0
        exit_mult = st.slider("Exit EV / EBITDA (x)", 3.0, 20.0, 8.0, 0.5)
        ggm_wt = st.slider("Gordon Growth Weight (%)", 0, 100, 50, 5) / 100.0

    wacc_info = compute_wacc(
        rf=rf,
        erp=erp,
        beta=beta_in,
        country_risk_prem=crp,
        pre_tax_kd=pre_tax_kd,
        tax_rate=tax_rate,
        debt_val=float(row["total_debt_ngn_m"]),
        equity_val=float(row["market_cap_bn"]) * 1000.0,
    )
    st.caption(
        f"CAPM Ke: **{wacc_info['ke']*100:.2f}%** | After-Tax Kd: **{wacc_info['after_tax_kd']*100:.2f}%** | Formula WACC: **{wacc_info['wacc']*100:.2f}%** | Model WACC: **{wacc_override*100:.2f}%**"
    )

    if wacc_override <= tg_rate:
        st.error("WACC must exceed Terminal Growth Rate (g) for perpetuity convergence.")
    else:
        dcf = run_institutional_dcf(
            base_rev=float(row["revenue_ngn_m"]),
            rev_growth=g_rate,
            gross_margin=gm_rate,
            opex_pct_rev=opex_rate,
            da_pct_rev=da_rate,
            capex_pct_rev=capex_rate,
            nwc_pct_delta_rev=nwc_delta_rate,
            tax_rate=tax_rate,
            wacc=wacc_override,
            terminal_growth=tg_rate,
            exit_ev_ebitda=exit_mult,
            terminal_weight_ggm=ggm_wt,
            cash_ngn_m=float(row["cash_ngn_m"]),
            debt_ngn_m=float(row["total_debt_ngn_m"]),
            shares_out_m=float(row["shares_out_m"]),
        )

        spot = float(row["share_price_ngn"])
        px_b = dcf["px_blended"]
        upside_b = ((px_b / spot) - 1.0) * 100.0 if spot > 0 else 0.0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Current Spot", f"₦{spot:,.2f}")
        m2.metric("Blended Implied", f"₦{px_b:,.2f}", f"{upside_b:+.1f}%")
        m3.metric("GGM Implied", f"₦{dcf['px_ggm_only']:,.2f}")
        m4.metric("Exit-Mult Implied", f"₦{dcf['px_mult_only']:,.2f}")
        m5.metric("Implied EV", f"₦{dcf['ev_blended']/1000.0:,.2f}B")

        st.markdown("**5-Year UFCF Projection Bridge (₦ Millions)**")
        st.dataframe(dcf["schedule"], use_container_width=True, hide_index=True)

        st.markdown("**Sensitivity Matrix: Blended Implied Share Price (₦)**")
        sens = sensitivity_grid_price(
            base_rev=float(row["revenue_ngn_m"]),
            rev_growth=g_rate,
            gross_margin=gm_rate,
            opex_pct_rev=opex_rate,
            da_pct_rev=da_rate,
            capex_pct_rev=capex_rate,
            nwc_pct_delta_rev=nwc_delta_rate,
            tax_rate=tax_rate,
            exit_ev_ebitda=exit_mult,
            terminal_weight_ggm=ggm_wt,
            cash_ngn_m=float(row["cash_ngn_m"]),
            debt_ngn_m=float(row["total_debt_ngn_m"]),
            shares_out_m=float(row["shares_out_m"]),
            center_wacc=wacc_override,
            center_tg=tg_rate,
        )
        st.dataframe(sens, use_container_width=True)

with tab4:
    st.subheader("Comparable Companies Trading Multiples")
    comps_tbl = pd.DataFrame(
        {
            "Ticker": fin_df["ticker"],
            "Company": fin_df["company"],
            "Spot (₦)": fin_df["share_price_ngn"].map(lambda x: f"₦{x:,.2f}"),
            "Mkt Cap (₦B)": fin_df["market_cap_bn"].map(lambda x: f"₦{x:,.2f}B"),
            "EV (₦M)": fin_df["ev_ngn_m"].map(lambda x: f"₦{x:,.0f}M"),
            "P/E": fin_df["pe"].map(fmt_multiple),
            "EV/EBITDA": fin_df["ev_ebitda"].map(fmt_multiple),
            "EV/Sales": fin_df["ev_sales"].map(fmt_multiple),
            "P/S": fin_df["ps"].map(fmt_multiple),
            "P/B": fin_df["pb"].map(fmt_multiple),
        }
    )
    st.dataframe(comps_tbl, use_container_width=True, hide_index=True)

    st.subheader("Peer Multiple Distribution (25th / Median / Mean / 75th)")
    if not stats_df.empty:
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
