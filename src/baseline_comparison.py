import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load ML-ready data
df = pd.read_csv("data/processed/ml_ready_energy.csv")

# Convert DateTime
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Sort by time
df = df.sort_values("DateTime").reset_index(drop=True)

# 80% training, 20% testing
split_index = int(len(df) * 0.8)

test = df.iloc[split_index:].copy()

# Actual values
y_test = test["Total_Consumption"].to_numpy()

# Naive forecast:
# Next prediction = previous 15-minute consumption
naive_predictions = test["Lag_1"].to_numpy()

# Check shapes
print("Actual values:", y_test.shape)
print("Predictions:", naive_predictions.shape)

# Evaluation
mae = mean_absolute_error(y_test, naive_predictions)
rmse = np.sqrt(mean_squared_error(y_test, naive_predictions))
r2 = r2_score(y_test, naive_predictions)

print("\nNaive Baseline Performance")
print("-----------------------------")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))