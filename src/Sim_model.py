# TO RUN CODE python src/sim_model.py 
import pandas as pd
import numpy as np

file_path = "data/Sim Model Portfolio For Code.ods"
sheet_name = "Prices"

# User-selected model settings
date_col = "Date"
benchmark_col = "SPX"

# Load prices
prices = pd.read_excel(file_path, sheet_name=sheet_name, engine="odf")

# Set Date as index
prices[date_col] = pd.to_datetime(prices[date_col])
prices.set_index(date_col, inplace=True)

# Choose Risk-Free column name
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

# Separate price data
stock_prices = prices[stock_cols]
benchmark_prices = prices[benchmark_col]

# Calculate log returns
stock_returns = np.log(stock_prices / stock_prices.shift(1))
benchmark_returns = np.log(benchmark_prices / benchmark_prices.shift(1))

# Combine stock and benchmark returns
returns = pd.concat([stock_returns, benchmark_returns.rename(benchmark_col)], axis=1)

# Remove first NaN row
returns = returns.dropna()

# Align risk-free rate with return dates
rf = rf.loc[returns.index]

# Calculate risk premia
risk_premia = returns.subtract(rf, axis=0)

# Display work
print("\nStock Columns:")
print(stock_cols)

print("\nStock Returns:")
print(stock_returns.dropna().head())

print("\nBenchmark Returns:")
print(benchmark_returns.dropna().head())

print("\nRisk-Free Rate:")
print(rf.head())

print("\nRisk Premia:")
print(risk_premia.head())

Mean = risk_premia.mean()
std = risk_premia.std()
#CALCULATE RISK PREMIA STATISTICS
risk_premia_stats = pd.DataFrame({
    "Ticker" : risk_premia.columns,
    "Average Risk Premia": risk_premia.mean().values,
    "Standard Deviation": risk_premia.std().values,
    "Variance": risk_premia.var().values,
    "Sharpe Ratio": (Mean/std).values
})
print ("\nRisk Premia Statistics:")
print(risk_premia_stats)

import statsmodels.api as sm 

sim_results = []
#This is the loop basically says for all the stock in stock columns
for stock in stock_cols:
    #Defines Y as your dependent variable
    Y = risk_premia[stock]
    #Defines X as your Independent Variable
    X = risk_premia[benchmark_col]
    #This adds the Alpha TERM
    X = sm.add_constant(X)
    #This runs the linear Regression
    model = sm.OLS(Y,X).fit()
    # This defines alpha as the intercept
    alpha= model.params["const"]
    #This defines beta as the slope 
    beta = model.params[benchmark_col]
    #This calculates the variance of the regression
    residual_variance = model.resid.var()
    #Saves the Results
    sim_results.append({
        "Ticker": stock,
        "Alpha": alpha,
        "Beta": beta,
        "Residual Variance": residual_variance
    })
#Turns Results into a table
sim_table = pd.DataFrame(sim_results)

print("\nSIM Regression Results:")
print(sim_table)


#Add Ranking Metric
sim_table ["Alpha/Residual Variance"] = ( 
    sim_table ["Alpha"] / sim_table ["Residual Variance"] 
)
sim_table = sim_table.sort_values (
    by = "Alpha/Residual Variance",
    ascending=False
)
sim_table = sim_table.reset_index(drop=True)

print("\nRanked SIM TABLE:")
print (sim_table)

#Market Variance Calculation TAKES VARIANCE OF SPX
market_variance = risk_premia[benchmark_col].var()

print("\nMarket Variance:")
print(market_variance)

#Create New Column in SIM Table for the numerator of the cutoff rate
sim_table["Cutoff Numerator Component"]= (
#This calculates (beta*alpha)/Residual Variance= The contribution of a stocks alpha adjusted for market exposure and firm specific risk
    sim_table["Beta"] * sim_table["Alpha"] / sim_table["Residual Variance"]
)
#Denominator Component of Cutoff Rate: Stocks Market Risk Contribution adjusted for residual risk
sim_table ["Cutoff Denominator Component"] = (
    sim_table["Beta"] ** 2 / sim_table["Residual Variance"]
)
 #Running total from top to bottom for numerator
sim_table["Cumulative Numerator"] = (
    sim_table["Cutoff Numerator Component"].cumsum()
)
#Running total from top to bottom for denominator
sim_table ["Cumulative Denominator"] = (
    sim_table["Cutoff Denominator Component"].cumsum()
)
sim_table["Cutoff Rate"] = (
    (market_variance * sim_table["Cumulative Numerator"])
    /
    (1 + market_variance * sim_table["Cumulative Denominator"])
)

print("\nCutoff Rate")
print( sim_table [["Ticker","Alpha/Residual Variance","Cutoff Rate"]])

sim_table["Incude"]= (
    sim_table ["Alpha/Residual Variance"]> sim_table["Cutoff Rate"]
)

#Select and Filter only included stocks that lie above the cutoff rate
included_stocks = sim_table[sim_table["Include"]==True]
c_star = included_stocks["Cutoff Rate"].iloc[-1]

#WEIGHTS Calculation in order to do this we take alpha - (Cutoff Rate *beta) 
#Which gives us how much extra alpha or additional return to the benchmark a stock gives above the cutoff rate
#Then we divide that by residual Variance 
#This gives excess alpha above the cutoff-adjusted required return per unit of unsystematic risk
sim_table["Z"] = (
   ( sim_table["Alpha"] - (sim_table["Beta"] * c_star))
    /
    sim_table["Residual Variance"]
)
#That gives us raw weights now we need to normalize the weights to equal 1
included_stocks = sim_table[sim_table["Include"]==True]
included_stocks["Weights"] = (
    included_stocks["Z"]/included_stocks["Z"].sum()
)

print("\nFinal Portfolio Weights:")
print(included_stocks[["Ticker", "Z", "Weights"]])

