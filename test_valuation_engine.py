import pytest

from valuation_engine import (
    build_reference_price_panel,
    compute_financial_statements_metrics,
    compute_price_analytics,
    compute_wacc,
    load_reference_fundamentals,
    run_institutional_dcf,
)


def test_wacc_increases_with_beta_and_rf():
    low = compute_wacc(0.15, 0.07, 0.80, 0.02, 0.20, 0.30, 5000.0, 20000.0)
    high = compute_wacc(0.18, 0.07, 1.20, 0.02, 0.20, 0.30, 5000.0, 20000.0)
    assert high["ke"] > low["ke"]
    assert high["wacc"] > low["wacc"]


def test_dcf_price_falls_as_wacc_rises():
    low_wacc = run_institutional_dcf(
        base_rev=50000.0,
        rev_growth=0.15,
        gross_margin=0.45,
        opex_pct_rev=0.22,
        da_pct_rev=0.03,
        capex_pct_rev=0.04,
        nwc_pct_delta_rev=0.10,
        tax_rate=0.30,
        wacc=0.18,
        terminal_growth=0.05,
        exit_ev_ebitda=8.0,
        terminal_weight_ggm=0.50,
        cash_ngn_m=3000.0,
        debt_ngn_m=4000.0,
        shares_out_m=2000.0,
    )
    high_wacc = run_institutional_dcf(
        base_rev=50000.0,
        rev_growth=0.15,
        gross_margin=0.45,
        opex_pct_rev=0.22,
        da_pct_rev=0.03,
        capex_pct_rev=0.04,
        nwc_pct_delta_rev=0.10,
        tax_rate=0.30,
        wacc=0.25,
        terminal_growth=0.05,
        exit_ev_ebitda=8.0,
        terminal_weight_ggm=0.50,
        cash_ngn_m=3000.0,
        debt_ngn_m=4000.0,
        shares_out_m=2000.0,
    )
    assert low_wacc["px_blended"] > high_wacc["px_blended"]
    assert len(low_wacc["schedule"]) == 5


def test_dcf_rejects_wacc_below_terminal_growth():
    with pytest.raises(ValueError):
        run_institutional_dcf(
            base_rev=50000.0,
            rev_growth=0.15,
            gross_margin=0.45,
            opex_pct_rev=0.22,
            da_pct_rev=0.03,
            capex_pct_rev=0.04,
            nwc_pct_delta_rev=0.10,
            tax_rate=0.30,
            wacc=0.05,
            terminal_growth=0.06,
            exit_ev_ebitda=8.0,
            terminal_weight_ggm=0.50,
            cash_ngn_m=3000.0,
            debt_ngn_m=4000.0,
            shares_out_m=2000.0,
        )


def test_fundamentals_and_price_analytics_run():
    fund = load_reference_fundamentals()
    metrics = compute_financial_statements_metrics(fund, tax_rate=0.30)
    assert "ccc" in metrics.columns
    assert "roic_pct" in metrics.columns
    assert len(metrics) == len(fund)

    prices = build_reference_price_panel(periods=60, seed=7)
    risk = compute_price_analytics(prices, rf_annual=0.18)
    assert len(risk) == 5
