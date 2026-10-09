import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------
# 1. Load ML-ready data
# ---------------------------------------

df = pd.read_csv("data/processed/ml_ready_energy.csv")

df["DateTime"] = pd.to_datetime(df["DateTime"])
df = df.sort_values("DateTime").reset_index(drop=True)

# ---------------------------------------
# 2. Features and target
# ---------------------------------------

features = [
    "Hour",
    "Day",
    "Day_of_Week",
    "Month",
    "Is_Weekend",
    "Lag_1",
    "Lag_4",
    "Lag_96",
    "Rolling_Mean_96"
]

target = "Total_Consumption"

X = df[features]
y = df[target]

# ---------------------------------------
# 3. Time-based train/test split
# ---------------------------------------

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_dates = df["DateTime"].iloc[split_index:].reset_index(drop=True)

# ---------------------------------------
# 4. Train Random Forest
# ---------------------------------------

print("Training anomaly detection model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model trained!")

# ---------------------------------------
# 5. Predict expected demand
# ---------------------------------------

predictions = model.predict(X_test)

# ---------------------------------------
# 6. Calculate prediction error
# ---------------------------------------

actual = y_test.to_numpy()

residual = actual - predictions
absolute_error = np.abs(residual)

# ---------------------------------------
# 7. Define anomaly threshold
# ---------------------------------------

threshold = np.percentile(absolute_error, 99)

print("\nAnomaly Threshold:", round(threshold, 2))

# ---------------------------------------
# 8. Create anomaly results
# ---------------------------------------

results = pd.DataFrame({
    "DateTime": test_dates,
    "Actual_Consumption": actual,
    "Expected_Consumption": predictions,
    "Error": residual,
    "Absolute_Error": absolute_error
})

results["Anomaly"] = results["Absolute_Error"] > threshold

# High or Low anomaly
results["Anomaly_Type"] = np.where(
    results["Anomaly"],
    np.where(
        results["Error"] > 0,
        "High Demand Anomaly",
        "Low Demand Anomaly"
    ),
    "Normal"
)

# ---------------------------------------
# 9. Save results
# ---------------------------------------

output_file = "data/processed/anomaly_results.csv"

results.to_csv(output_file, index=False)

# ---------------------------------------
# 10. Print summary
# ---------------------------------------

total_anomalies = results["Anomaly"].sum()

print("\nAnomaly Detection Summary")
print("-----------------------------")
print("Test observations:", len(results))
print("Anomalies detected:", total_anomalies)
print(
    "Anomaly percentage:",
    round(total_anomalies / len(results) * 100, 2),
    "%"
)

print("\nTop 20 anomalies:")
print(
    results[
        results["Anomaly"]
    ]
    .sort_values("Absolute_Error", ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\nSaved to:", output_file)