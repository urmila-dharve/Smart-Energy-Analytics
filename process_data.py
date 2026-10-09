import pandas as pd
import os

input_file = "data/raw/LD2011_2014.txt"
output_file = "data/processed/energy_demand.csv"

print("Loading dataset...")

df = pd.read_csv(
    input_file,
    sep=";",
    decimal=","
    
)

print("Dataset loaded!")
#RENAME FIRST COLUMN

df.rename(columns={df.columns[0]: "DateTime"}, inplace=True)

#CONVERT DATETIME 
df["DateTime"] = pd.to_datetime(df["DateTime"])

#GET CLIENT COLUMNS
client_columns = df.columns[1:]

#calculate total onsumption
df["Total_Consumption"] = df[client_columns].sum(axis=1)

#keep only required columns
result = df[["DateTime", "Total_Consumption"]]

#Create processed folder
os.makedirs("data/processed",exist_ok=True)

#Save processed data
result.to_csv(output_file, index=False)

print("Processing completed!")
print("Rows:", len(result))
print("Columns:", result.columns.tolist())
print("Saved to:", output_file)