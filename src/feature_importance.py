import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("data/processed/ml_ready_energy.csv")

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

# Time-based split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# Plot
plt.figure(figsize=(10, 6))

importance.sort_values().plot(kind="barh")

plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()