import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

st.set_page_config(page_title="SIM Portfolio Model", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F7F5EF;
        color: #002F30;
    }

    h1, h2, h3 {
        color: #002F30;
    }

    .piraeus-header {
        background-color: #FFD900;
        color: #002F30;
        padding: 22px;
        border-radius: 14px;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 20px;
    }

    .piraeus-subbox {
        background-color: #002F30;
        color: #FFD900;
        padding: 14px;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def run_sim_model(
    prices,
    benchmark_col="Benchmark",
    allow_shorting=False,
    use_cap=False,
    max_weight=0.25
):
    prices = prices.copy()

    prices["Date"] = pd.to_datetime(prices["Date"])
    prices.set_index("Date", inplace=True)

    if "Adjusted Risk Free" in prices.columns:
        rf_col = "Adjusted Risk Free"
    elif "Risk Free" in prices.columns:
        rf_col = "Risk Free"
    else:
        raise ValueError("No risk-free column found. Use 'Adjusted Risk Free' or 'Risk Free'.")

    rf = prices[rf_col]

    # Assumes RF is annual percentage, e.g. 3.5 = 3.5% annual
    rf = (rf / 100) / 12

    stock_cols = [
        col for col in prices.columns
        if col not in [benchmark_col, rf_col]
    ]

    if benchmark_col not in prices.columns:
        raise ValueError(f"Benchmark column '{benchmark_col}' not found.")

    stock_prices = prices[stock_cols]
    benchmark_prices = prices[benchmark_col]

    stock_returns = np.log(stock_prices / stock_prices.shift(1))
    benchmark_returns = np.log(benchmark_prices / benchmark_prices.shift(1))

    returns = pd.concat(
        [stock_returns, benchmark_returns.rename(benchmark_col)],
        axis=1
    ).dropna()

    rf = rf.loc[returns.index]

    risk_premia = returns.subtract(rf, axis=0)

    mean = risk_premia.mean()
    std = risk_premia.std()

    risk_premia_stats = pd.DataFrame({
        "Ticker": risk_premia.columns,
        "Average Risk Premia": mean.values,
        "Standard Deviation": std.values,
        "Variance": risk_premia.var().values,
        "Sharpe Ratio": (mean / std).values
    })

    sim_results = []

    for stock in stock_cols:
        Y = risk_premia[stock]
        X = risk_premia[benchmark_col]
        X = sm.add_constant(X)

        model = sm.OLS(Y, X).fit()

        sim_results.append({
            "Ticker": stock,
            "Alpha": model.params["const"],
            "Beta": model.params[benchmark_col],
            "Residual Variance": model.resid.var()
        })

    sim_table = pd.DataFrame(sim_results)

    sim_table["Alpha/Residual Variance"] = (
        sim_table["Alpha"] / sim_table["Residual Variance"]
    )

    sim_table = sim_table.sort_values(
        by="Alpha/Residual Variance",
        ascending=False
    ).reset_index(drop=True)

    market_variance = risk_premia[benchmark_col].var()

    sim_table["Cutoff Numerator Component"] = (
        sim_table["Beta"] * sim_table["Alpha"] / sim_table["Residual Variance"]
    )

    sim_table["Cutoff Denominator Component"] = (
        sim_table["Beta"] ** 2 / sim_table["Residual Variance"]
    )

    sim_table["Cumulative Numerator"] = (
        sim_table["Cutoff Numerator Component"].cumsum()
    )

    sim_table["Cumulative Denominator"] = (
        sim_table["Cutoff Denominator Component"].cumsum()
    )

    sim_table["Cutoff Rate"] = (
        (market_variance * sim_table["Cumulative Numerator"])
        /
        (1 + market_variance * sim_table["Cumulative Denominator"])
    )

    sim_table["Pass Cutoff Test"] = (
        sim_table["Alpha/Residual Variance"] > sim_table["Cutoff Rate"]
    )

    valid_rows = sim_table[sim_table["Pass Cutoff Test"] == True]

    if valid_rows.empty:
        raise ValueError("No stocks passed the cutoff rate.")

    last_included_index = valid_rows.index[-1]

    sim_table["Include"] = sim_table.index <= last_included_index

    included_stocks = sim_table[sim_table["Include"] == True].copy()

    c_star = included_stocks["Cutoff Rate"].iloc[-1]

    sim_table["Z"] = (
        (sim_table["Alpha"] - (sim_table["Beta"] * c_star))
        /
        sim_table["Residual Variance"]
    )

    if allow_shorting:
        included_stocks = sim_table[sim_table["Include"] == True].copy()
    else:
        included_stocks = sim_table[
            (sim_table["Include"] == True) & (sim_table["Z"] > 0)
        ].copy()

    if included_stocks.empty:
        raise ValueError("No stocks were available for weighting.")

    included_stocks["Weight"] = (
        included_stocks["Z"] / included_stocks["Z"].sum()
    )

    if use_cap:
        included_stocks["Weight"] = included_stocks["Weight"].clip(
            lower=-max_weight if allow_shorting else 0,
            upper=max_weight
        )

        included_stocks["Weight"] = (
            included_stocks["Weight"] / included_stocks["Weight"].sum()
        )

    return risk_premia_stats, sim_table, included_stocks, c_star


st.markdown(
    '<div class="piraeus-header">SIM Portfolio Model</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="piraeus-subbox">Piraeus-style Single Index Model portfolio tool</div>',
    unsafe_allow_html=True
)

st.write("Upload your Excel file to run the Single Index Model.")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "ods"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    if uploaded_file.name.endswith(".ods"):
        prices = pd.read_excel(uploaded_file, engine="odf")
    else:
        prices = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(prices)

    allow_shorting = st.checkbox("Allow shorting / Negative weights")

    use_cap = st.checkbox("Apply max weight cap")

    max_weight = 0.25
    if use_cap:
        max_weight = st.slider(
            "Max absolute weight per stock",
            min_value=0.05,
            max_value=0.50,
            value=0.25,
            step=0.05
        )

    try:
        risk_premia_stats, sim_table, included_stocks, c_star = run_sim_model(
            prices,
            allow_shorting=allow_shorting,
            use_cap=use_cap,
            max_weight=max_weight
        )

        st.subheader("Risk Premia Statistics")
        st.dataframe(risk_premia_stats)

        st.subheader("Full SIM Table")
        st.dataframe(sim_table)

        st.subheader("Final Cutoff Rate")
        st.write(c_star)

        st.subheader("Final Portfolio Weights")
        st.dataframe(included_stocks[["Ticker", "Z", "Weight"]])

        st.write("Total Weight:")
        st.write(included_stocks["Weight"].sum())

    except Exception as e:
        st.error(f"Error running SIM model: {e}")

# To run:
# python -m streamlit run app.py
