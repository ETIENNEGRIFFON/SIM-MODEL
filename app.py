import io

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================
# This controls the browser tab title, page icon, and layout.
# "wide" gives the app more horizontal space, which is better
# for financial dashboards and large tables.
st.set_page_config(
    page_title="SIM Portfolio Model | Piraeus Bank",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PIRAEUS BANK STYLE / CUSTOM CSS
# =========================================================
# This block controls the entire visual design of the app.
# The main colors are based on Piraeus Bank-style dark teal,
# yellow, cream, and white.
#
# IMPORTANT:
# The sidebar has a dark background, but input boxes need white
# backgrounds and black text so the user can actually read them.
st.markdown(
    """
    <style>
    :root {
        --piraeus-teal: #002F30;
        --piraeus-teal-soft: #06484A;
        --piraeus-yellow: #FFD900;
        --piraeus-cream: #F7F5EF;
        --piraeus-card: #FFFFFF;
        --piraeus-muted: #6D7A7A;
        --piraeus-border: #E4E0D5;
    }

    .stApp {
        background: linear-gradient(135deg, #F7F5EF 0%, #EFEBDD 100%);
        color: var(--piraeus-teal);
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Sidebar container */
    section[data-testid="stSidebar"] {
        background: var(--piraeus-teal);
        border-right: 4px solid var(--piraeus-yellow);
    }

    /* Sidebar labels and general text stay white */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] div {
        color: #FFFFFF !important;
    }

    /* Input fields must have black text and white background */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] select {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }

    /* Streamlit selectbox selected value */
    .stSelectbox div[data-baseweb="select"] > div {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }

    /* Streamlit number input text */
    .stNumberInput input {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }

    /* Streamlit text input text */
    .stTextInput input {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }

    /* Multiselect values */
    .stMultiSelect div[data-baseweb="select"] > div {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }

    /* File uploader text area */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] div {
        color: #FFFFFF !important;
    }

    /* Main hero banner */
    .hero {
        background: linear-gradient(135deg, var(--piraeus-teal) 0%, var(--piraeus-teal-soft) 100%);
        border-radius: 24px;
        padding: 34px 38px;
        box-shadow: 0 18px 45px rgba(0, 47, 48, 0.22);
        border-bottom: 8px solid var(--piraeus-yellow);
        margin-bottom: 24px;
    }

    .hero-kicker {
        color: var(--piraeus-yellow);
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        font-size: 13px;
        margin-bottom: 10px;
    }

    .hero-title {
        color: #FFFFFF;
        font-size: 44px;
        line-height: 1.05;
        font-weight: 900;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        color: #DDEAEA;
        font-size: 17px;
        max-width: 900px;
        line-height: 1.55;
    }

    /* Reusable white dashboard card */
    .section-card {
        background: rgba(255, 255, 255, 0.93);
        border: 1px solid var(--piraeus-border);
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 10px 28px rgba(0, 47, 48, 0.08);
        margin-bottom: 20px;
    }

    .section-title {
        color: var(--piraeus-teal);
        font-size: 22px;
        font-weight: 850;
        margin-bottom: 6px;
    }

    .section-subtitle {
        color: var(--piraeus-muted);
        font-size: 14px;
        margin-bottom: 18px;
    }

    .status-pill {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255, 217, 0, 0.25);
        border: 1px solid rgba(255, 217, 0, 0.8);
        color: var(--piraeus-teal);
        font-weight: 750;
        font-size: 13px;
        margin-bottom: 14px;
    }

    .info-box {
        background: #FFFFFF;
        border-left: 6px solid var(--piraeus-yellow);
        border-radius: 16px;
        padding: 18px 20px;
        color: var(--piraeus-teal);
        box-shadow: 0 8px 22px rgba(0, 47, 48, 0.07);
        margin-bottom: 20px;
    }

    .download-box {
        background: linear-gradient(135deg, var(--piraeus-teal) 0%, #053F41 100%);
        border-radius: 22px;
        padding: 22px;
        border: 1px solid rgba(255, 217, 0, 0.55);
        box-shadow: 0 12px 28px rgba(0, 47, 48, 0.14);
    }

    .download-box h3 {
        color: var(--piraeus-yellow) !important;
        margin-top: 0;
    }

    .download-box p {
        color: #EAF5F5;
        margin-bottom: 8px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--piraeus-border);
        border-radius: 20px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 8px 22px rgba(0, 47, 48, 0.07);
    }

    div[data-testid="stMetric"] label {
        color: var(--piraeus-muted) !important;
        font-weight: 700;
    }

    div[data-testid="stMetricValue"] {
        color: var(--piraeus-teal) !important;
        font-weight: 900;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background: var(--piraeus-yellow) !important;
        color: var(--piraeus-teal) !important;
        border: 0 !important;
        border-radius: 14px !important;
        font-weight: 850 !important;
        padding: 0.75rem 1.1rem !important;
        box-shadow: 0 8px 18px rgba(0, 47, 48, 0.16);
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(0, 47, 48, 0.22);
    }

    h1, h2, h3, h4 {
        color: var(--piraeus-teal) !important;
        letter-spacing: -0.02em;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid var(--piraeus-border);
    }

    .small-muted {
        color: #DDEAEA;
        font-size: 13px;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CORE SIM MODEL FUNCTION
# =========================================================
# This function does the real Single Index Model work.
# It:
# 1. Reads the Date column and sets it as the index.
# 2. Finds the risk-free rate column.
# 3. Converts prices into log returns.
# 4. Converts returns into risk premia by subtracting the risk-free rate.
# 5. Runs a regression for each stock against the benchmark.
# 6. Calculates Alpha, Beta, Residual Variance, cutoff rate, Z, and weights.
def run_sim_model(
    prices,
    benchmark_col="Benchmark",
    allow_shorting=False,
    use_cap=False,
    max_weight=0.25,
):
    prices = prices.copy()

    # Convert Date column into datetime format and make it the index.
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices.set_index("Date", inplace=True)

    # The app accepts either "Adjusted Risk Free" or "Risk Free".
    if "Adjusted Risk Free" in prices.columns:
        rf_col = "Adjusted Risk Free"
    elif "Risk Free" in prices.columns:
        rf_col = "Risk Free"
    else:
        raise ValueError("No risk-free column found. Use 'Adjusted Risk Free' or 'Risk Free'.")

    rf = prices[rf_col]

    # Assumes risk-free rate is annual percentage, for example 3.5 = 3.5%.
    # Dividing by 100 converts it to decimal, then dividing by 12 converts it to monthly.
    rf = (rf / 100) / 12

    # Stock columns are every column except benchmark and risk-free.
    stock_cols = [col for col in prices.columns if col not in [benchmark_col, rf_col]]

    if benchmark_col not in prices.columns:
        raise ValueError(f"Benchmark column '{benchmark_col}' not found.")

    stock_prices = prices[stock_cols]
    benchmark_prices = prices[benchmark_col]

    # Calculate log returns for stocks and benchmark.
    stock_returns = np.log(stock_prices / stock_prices.shift(1))
    benchmark_returns = np.log(benchmark_prices / benchmark_prices.shift(1))

    # Combine stock and benchmark returns into one dataframe.
    returns = pd.concat(
        [stock_returns, benchmark_returns.rename(benchmark_col)],
        axis=1,
    ).dropna()

    # Align the risk-free rate with the return dates.
    rf = rf.loc[returns.index]

    # Risk premium = return above the risk-free rate.
    risk_premia = returns.subtract(rf, axis=0)

    # Basic risk premia statistics.
    mean = risk_premia.mean()
    std = risk_premia.std()

    risk_premia_stats = pd.DataFrame(
        {
            "Ticker": risk_premia.columns,
            "Average Risk Premia": mean.values,
            "Standard Deviation": std.values,
            "Variance": risk_premia.var().values,
            "Sharpe Ratio": (mean / std).values,
        }
    )

    # Run regression for each stock:
    # Stock risk premium = alpha + beta * benchmark risk premium + residual error.
    sim_results = []

    for stock in stock_cols:
        y = risk_premia[stock]
        x = risk_premia[benchmark_col]
        x = sm.add_constant(x)

        model = sm.OLS(y, x).fit()

        sim_results.append(
            {
                "Ticker": stock,
                "Alpha": model.params["const"],
                "Beta": model.params[benchmark_col],
                "Residual Variance": model.resid.var(),
            }
        )

    sim_table = pd.DataFrame(sim_results)

    # Ranking metric used in the Single Index Model.
    sim_table["Alpha/Residual Variance"] = (
        sim_table["Alpha"] / sim_table["Residual Variance"]
    )

    # Sort stocks from highest alpha per unit of residual variance to lowest.
    sim_table = sim_table.sort_values(
        by="Alpha/Residual Variance",
        ascending=False,
    ).reset_index(drop=True)

    # Market variance is the variance of benchmark risk premia.
    market_variance = risk_premia[benchmark_col].var()

    # Components used to calculate cutoff rates.
    sim_table["Cutoff Numerator Component"] = (
        sim_table["Beta"] * sim_table["Alpha"] / sim_table["Residual Variance"]
    )

    sim_table["Cutoff Denominator Component"] = (
        sim_table["Beta"] ** 2 / sim_table["Residual Variance"]
    )

    sim_table["Cumulative Numerator"] = sim_table["Cutoff Numerator Component"].cumsum()
    sim_table["Cumulative Denominator"] = sim_table["Cutoff Denominator Component"].cumsum()

    # Cutoff rate calculation.
    sim_table["Cutoff Rate"] = (
        market_variance * sim_table["Cumulative Numerator"]
    ) / (1 + market_variance * sim_table["Cumulative Denominator"])

    # A stock passes if Alpha / Residual Variance is greater than the cutoff rate.
    sim_table["Pass Cutoff Test"] = (
        sim_table["Alpha/Residual Variance"] > sim_table["Cutoff Rate"]
    )

    valid_rows = sim_table[sim_table["Pass Cutoff Test"] == True]

    if valid_rows.empty:
        raise ValueError("No stocks passed the cutoff rate.")

    # The last stock that passes the test determines the final cutoff rate.
    last_included_index = valid_rows.index[-1]
    sim_table["Include"] = sim_table.index <= last_included_index

    included_stocks = sim_table[sim_table["Include"] == True].copy()
    c_star = included_stocks["Cutoff Rate"].iloc[-1]

    # Z is the raw score used to calculate portfolio weights.
    sim_table["Z"] = (
        sim_table["Alpha"] - (sim_table["Beta"] * c_star)
    ) / sim_table["Residual Variance"]

    # If shorting is not allowed, remove stocks with negative Z values.
    if allow_shorting:
        included_stocks = sim_table[sim_table["Include"] == True].copy()
    else:
        included_stocks = sim_table[
            (sim_table["Include"] == True) & (sim_table["Z"] > 0)
        ].copy()

    if included_stocks.empty:
        raise ValueError("No stocks were available for weighting.")

    # Normalize Z values so that weights sum to 1.
    included_stocks["Weight"] = included_stocks["Z"] / included_stocks["Z"].sum()

    # Optional cap: limit each position to max_weight.
    if use_cap:
        included_stocks["Weight"] = included_stocks["Weight"].clip(
            lower=-max_weight if allow_shorting else 0,
            upper=max_weight,
        )

        included_stocks["Weight"] = (
            included_stocks["Weight"] / included_stocks["Weight"].sum()
        )

    return risk_premia_stats, sim_table, included_stocks, c_star


# =========================================================
# DISPLAY FORMATTING HELPERS
# =========================================================
def pct_fmt(x):
    return f"{x:.2%}"


def num_fmt(x):
    return f"{x:,.6f}"


def style_numeric_table(df):
    # Formats output tables so decimals look cleaner.
    return df.style.format(
        {
            "Average Risk Premia": "{:.6f}",
            "Standard Deviation": "{:.6f}",
            "Variance": "{:.8f}",
            "Sharpe Ratio": "{:.4f}",
            "Alpha": "{:.6f}",
            "Beta": "{:.4f}",
            "Residual Variance": "{:.8f}",
            "Alpha/Residual Variance": "{:.4f}",
            "Cutoff Numerator Component": "{:.6f}",
            "Cutoff Denominator Component": "{:.6f}",
            "Cumulative Numerator": "{:.6f}",
            "Cumulative Denominator": "{:.6f}",
            "Cutoff Rate": "{:.6f}",
            "Z": "{:.6f}",
            "Weight": "{:.2%}",
        },
        na_rep="—",
    )


# =========================================================
# MAIN HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Piraeus Bank • Quantitative Portfolio Tool</div>
        <div class="hero-title">Single Index Model Portfolio Dashboard</div>
        <div class="hero-subtitle">
            Upload price data, estimate alpha and beta against a benchmark, calculate the cutoff rate,
            and generate optimized SIM portfolio weights in a clean professional dashboard.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR INPUTS
# =========================================================
# All model controls live in the sidebar so the main page can stay clean.
with st.sidebar:
    st.markdown("### SIM Model Controls")
    st.markdown(
        "<p class='small-muted'>Expected columns: <b>Date</b>, <b>Benchmark</b>, and either <b>Risk Free</b> or <b>Adjusted Risk Free</b>.</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload Excel or ODS file", type=["xlsx", "ods"])

    st.divider()

    benchmark_col = st.text_input("Benchmark column name", value="Benchmark")
    allow_shorting = st.checkbox("Allow shorting / negative weights", value=False)
    use_cap = st.checkbox("Apply max weight cap", value=False)

    max_weight = 0.25
    if use_cap:
        max_weight = st.slider(
            "Max absolute weight per stock",
            min_value=0.05,
            max_value=0.50,
            value=0.25,
            step=0.05,
        )

    st.divider()
    st.markdown("### Output")
    show_raw_data = st.toggle("Show uploaded data", value=True)
    show_full_table = st.toggle("Show full SIM table", value=True)


# =========================================================
# EMPTY STATE
# =========================================================
# This is what the user sees before uploading a file.
if uploaded_file is None:
    st.markdown(
        """
        <div class="info-box">
            <b>Start here:</b> upload your Excel or ODS file from the sidebar. Once uploaded, the app will calculate risk premia,
            rank securities by Alpha / Residual Variance, calculate the cutoff rate, and produce final portfolio weights.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", "SIM")
    with col2:
        st.metric("Benchmark", benchmark_col)
    with col3:
        st.metric("Weight Cap", "Off" if not use_cap else pct_fmt(max_weight))

    st.stop()


# =========================================================
# FILE READING + MODEL OUTPUT
# =========================================================
try:
    # Read the uploaded file.
    if uploaded_file.name.endswith(".ods"):
        prices = pd.read_excel(uploaded_file, engine="odf")
    else:
        prices = pd.read_excel(uploaded_file)

    st.markdown(
        "<span class='status-pill'>File uploaded successfully</span>",
        unsafe_allow_html=True,
    )

    # Optional raw data preview.
    if show_raw_data:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Uploaded Data</div>
                <div class="section-subtitle">Review the imported price, benchmark, and risk-free rate data before running the model.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(prices, use_container_width=True, height=320)

    # Run the Single Index Model.
    risk_premia_stats, sim_table, included_stocks, c_star = run_sim_model(
        prices,
        benchmark_col=benchmark_col,
        allow_shorting=allow_shorting,
        use_cap=use_cap,
        max_weight=max_weight,
    )

    # Dashboard metrics.
    st.markdown("### Executive Summary")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Included Securities", f"{len(included_stocks)}")
    with m2:
        st.metric("Final Cutoff Rate", num_fmt(c_star))
    with m3:
        st.metric("Total Weight", pct_fmt(included_stocks["Weight"].sum()))
    with m4:
        top_position = included_stocks.sort_values("Weight", ascending=False).iloc[0]
        st.metric(
            "Largest Position",
            f"{top_position['Ticker']} ({pct_fmt(top_position['Weight'])})",
        )

    # Final weights section.
    st.markdown("### Final Portfolio Weights")
    left, right = st.columns([1.1, 1])

    with left:
        st.dataframe(
            style_numeric_table(included_stocks[["Ticker", "Alpha", "Beta", "Z", "Weight"]]),
            use_container_width=True,
            height=360,
        )

    with right:
        chart_df = included_stocks[["Ticker", "Weight"]].sort_values("Weight", ascending=True)
        st.bar_chart(chart_df.set_index("Ticker"), height=360)

    # Tabs separate detailed outputs.
    tab1, tab2, tab3 = st.tabs(["Risk Premia", "SIM Ranking", "Export"])

    with tab1:
        st.markdown(
            """
            <div class="section-title">Risk Premia Statistics</div>
            <div class="section-subtitle">Average excess returns, volatility, variance, and Sharpe ratios for each security and the benchmark.</div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            style_numeric_table(risk_premia_stats),
            use_container_width=True,
            height=420,
        )

    with tab2:
        st.markdown(
            """
            <div class="section-title">Full SIM Table</div>
            <div class="section-subtitle">Securities are ranked by Alpha / Residual Variance, then tested against the cutoff rate.</div>
            """,
            unsafe_allow_html=True,
        )

        if show_full_table:
            st.dataframe(
                style_numeric_table(sim_table),
                use_container_width=True,
                height=520,
            )
        else:
            compact_cols = [
                "Ticker",
                "Alpha",
                "Beta",
                "Residual Variance",
                "Alpha/Residual Variance",
                "Cutoff Rate",
                "Pass Cutoff Test",
                "Include",
            ]
            st.dataframe(
                style_numeric_table(sim_table[compact_cols]),
                use_container_width=True,
                height=520,
            )

    with tab3:
        st.markdown(
            """
            <div class="download-box">
                <h3>Download SIM Model Output</h3>
                <p>Export the risk premia statistics, full SIM table, and final portfolio weights to Excel.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        output = io.BytesIO()

        # Create Excel workbook in memory.
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            risk_premia_stats.to_excel(writer, sheet_name="Risk Premia", index=False)
            sim_table.to_excel(writer, sheet_name="SIM Table", index=False)
            included_stocks.to_excel(writer, sheet_name="Portfolio Weights", index=False)

        excel_data = output.getvalue()

        st.download_button(
            label="Download Full SIM Model Output",
            data=excel_data,
            file_name="sim_model_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

except Exception as e:
    # Friendly error message if the file columns are wrong or the model fails.
    st.error(f"Error running SIM model: {e}")
    st.markdown(
        """
        <div class="info-box">
            <b>Check your file structure:</b> the model needs a Date column, a benchmark column, and a Risk Free or Adjusted Risk Free column.
        </div>
        """,
        unsafe_allow_html=True,
    )


# To run this app:


# python -m streamlit run app.py
