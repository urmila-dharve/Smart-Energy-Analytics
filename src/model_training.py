import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os
import joblib

# Load ML-ready data
df = pd.read_csv("data/processed/ml_ready_energy.csv")

# Convert DateTime
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Sort by time
df = df.sort_values("DateTime")

# Features
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

# Time-based train/test split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")

# Prediction
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("------------------------")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))



os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/demand_model.pkl")

print("Model saved successfully!")
print("Saved to: models/demand_model.pkl")