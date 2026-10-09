import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/processed/energy_demand.csv")

# Convert DateTime
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Create day type
df["Day_Type"] = df["DateTime"].dt.dayofweek.map(
    lambda x: "Weekend" if x >= 5 else "Weekday"
)

# Average consumption by day type
day_type_consumption = (
    df.groupby("Day_Type")["Total_Consumption"]
    .mean()
)

print("Average Consumption:")
print(day_type_consumption)

# Plot
plt.figure(figsize=(8, 6))

plt.bar(
    day_type_consumption.index,
    day_type_consumption.values
)

plt.title("Average Electricity Consumption: Weekday vs Weekend")
plt.xlabel("Day Type")
plt.ylabel("Average Consumption")

plt.tight_layout()
plt.show()