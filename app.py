import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm


def run_sim_model(prices, benchmark_col="Benchmark"):
    prices = prices.copy()

    # Set Date as index
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices.set_index("Date", inplace=True)

    # Choose risk-free column
    if "Adjusted Risk Free" in prices.columns:
        rf_col = "Adjusted Risk Free"
    elif "Risk Free" in prices.columns:
        rf_col = "Risk Free"
    else:
        raise ValueError("No risk-free column found. Use 'Adjusted Risk Free' or 'Risk Free'.")

    # Separate risk-free rate
    rf = prices[rf_col]

    # Identify stock columns automatically
    stock_cols = [
        col for col in prices.columns
        if col not in [benchmark_col, rf_col]
    ]

    # Separate stock and benchmark prices
    stock_prices = prices[stock_cols]
    benchmark_prices = prices[benchmark_col]

    # Calculate log returns
    stock_returns = np.log(stock_prices / stock_prices.shift(1))
    benchmark_returns = np.log(benchmark_prices / benchmark_prices.shift(1))

    # Combine stock and benchmark returns
    returns = pd.concat(
        [stock_returns, benchmark_returns.rename(benchmark_col)],
        axis=1
    )

    # Remove first NaN row
    returns = returns.dropna()

    # Align risk-free rate
    rf = rf.loc[returns.index]

    # Calculate risk premia
    risk_premia = returns.subtract(rf, axis=0)

    # Risk premia statistics
    mean = risk_premia.mean()
    std = risk_premia.std()

    risk_premia_stats = pd.DataFrame({
        "Ticker": risk_premia.columns,
        "Average Risk Premia": mean.values,
        "Standard Deviation": std.values,
        "Variance": risk_premia.var().values,
        "Sharpe Ratio": (mean / std).values
    })

    # Regression for each stock
    sim_results = []

    for stock in stock_cols:
        Y = risk_premia[stock]
        X = risk_premia[benchmark_col]
        X = sm.add_constant(X)

        model = sm.OLS(Y, X).fit()

        alpha = model.params["const"]
        beta = model.params[benchmark_col]
        residual_variance = model.resid.var()

        sim_results.append({
            "Ticker": stock,
            "Alpha": alpha,
            "Beta": beta,
            "Residual Variance": residual_variance
        })

    sim_table = pd.DataFrame(sim_results)

    # Ranking metric
    sim_table["Alpha/Residual Variance"] = (
        sim_table["Alpha"] / sim_table["Residual Variance"]
    )

    sim_table = sim_table.sort_values(
        by="Alpha/Residual Variance",
        ascending=False
    ).reset_index(drop=True)

    # Market variance
    market_variance = risk_premia[benchmark_col].var()

    # Cutoff calculation
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

    # Select included stocks
    sim_table["Include"] = (
        sim_table["Alpha/Residual Variance"] > sim_table["Cutoff Rate"]
    )

    included_stocks = sim_table[sim_table["Include"] == True].copy()

    if included_stocks.empty:
        raise ValueError("No stocks passed the cutoff rule.")

    # Final cutoff rate
    c_star = included_stocks["Cutoff Rate"].iloc[-1]

    # Raw weight score
    sim_table["Z"] = (
        (sim_table["Alpha"] - (sim_table["Beta"] * c_star))
        /
        sim_table["Residual Variance"]
    )

    # Normalize weights only for included stocks
    included_stocks = sim_table[
        (sim_table["Include"] == True) & (sim_table ["Z"]>0)
    ].copy()

    included_stocks["Weight"] = (
        included_stocks["Z"] / included_stocks["Z"].sum()
    )

    return risk_premia_stats, sim_table, included_stocks, c_star


# Streamlit App
st.title("SIM Portfolio Model")

st.write("Upload your Excel file to run the Single Index Model.")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "ods"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    # Read uploaded Excel file
    if uploaded_file.name.endswith(".ods"):
        prices = pd.read_excel(uploaded_file, engine="odf")
    else:
        prices = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(prices)

    try:
        risk_premia_stats, sim_table, included_stocks, c_star = run_sim_model(prices)

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
    # To run APP on vscode: python -m streamlit run app.py
