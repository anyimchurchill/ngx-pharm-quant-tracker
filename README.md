# NGX Pharma Wall Street Valuation & Quant Terminal

An institutional equity research and quantitative analytics workbench built in Python and Streamlit for Nigerian Exchange (NGX) pharmaceutical equities (**MeCure Industries**, **Fidson Healthcare**, **May & Baker Nigeria**, **Neimeth International Pharmaceuticals**, and **Morison Industries**).

## Architecture
- **`valuation_engine.py`**: Pure financial and quantitative functions for CAPM/WACC builds, 5-year Unlevered Free Cash Flow (UFCF) DCF modeling (blended Gordon Growth and Exit EV/EBITDA terminal value), 2D sensitivity grids, working-capital cash conversion metrics (`DIO`, `DSO`, `DPO`, `CCC`), `ROIC`, peer multiple distributions, and market risk/return metrics (`Beta`, `Sharpe`, `Sortino`, `Max Drawdown`).
- **`app.py`**: Interactive Streamlit terminal with Bull / Base / Bear / Custom scenarios, assumption sliders, custom CSV ingestion for fundamentals and historical prices, and clean tabular/chart views.
- **`tests/test_valuation_engine.py`**: Automated unit tests for WACC, DCF monotonicity, input validation, and analytical tables.

## Run Locally & Test
```bash
pip install -r requirements.txt
pytest
streamlit run app.py
