# SIM model code will go here
import pandas as pd

file_path = "data/Sim Model Portfolio For Code.ods"

excel_file = pd.ExcelFile(file_path, engine="odf")

print("Sheets in this file:")
print(excel_file.sheet_names)
