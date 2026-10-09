import pandas as pd

# Load processed data
df = pd.read_csv("data/processed/energy_demand.csv")

# Convert DateTime
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Time-based features
df["Hour"] = df["DateTime"].dt.hour
df["Day"] = df["DateTime"].dt.day
df["Day_of_Week"] = df["DateTime"].dt.dayofweek
df["Month"] = df["DateTime"].dt.month

# Weekend feature
df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)

# Lag features
df["Lag_1"] = df["Total_Consumption"].shift(1)
df["Lag_4"] = df["Total_Consumption"].shift(4)
df["Lag_96"] = df["Total_Consumption"].shift(96)

# Rolling average
df["Rolling_Mean_96"] = (
    df["Total_Consumption"]
    .shift(1)
    .rolling(window=96)
    .mean()
)

# Remove rows created by lag/rolling calculations
df.dropna(inplace=True)

# Save ML-ready dataset
output_file = "data/processed/ml_ready_energy.csv"
df.to_csv(output_file, index=False)

print("Feature engineering completed!")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Saved to:", output_file)