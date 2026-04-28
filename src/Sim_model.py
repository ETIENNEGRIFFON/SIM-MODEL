# SIM model code will go here
import pandas as pd

file_path = "data/Sim Model Portfolio For Code.ods"

prices = pd.read_excel(file_path, sheet_name="Prices", engine="odf")

print(prices.head())
print(prices.columns)
print(prices.shape)
