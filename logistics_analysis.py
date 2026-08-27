import pandas as pd
import numpy as np

# Load public transaction data
df = pd.read_excel("Online Retail.xlsx")

# Basic preparation
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# Flag cancellations and remove invalid analytical records
df["IsCancelled"] = df["InvoiceNo"].astype(str).str.startswith("C")
clean = df[(~df["IsCancelled"]) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

# Time features
clean["Month"] = clean["InvoiceDate"].dt.to_period("M").astype(str)
clean["DayOfWeek"] = clean["InvoiceDate"].dt.day_name()

print(clean.shape)
print(clean[["Quantity", "UnitPrice", "Revenue"]].describe())
# Product-level demand summary
product_summary = (
    clean.groupby("StockCode")
    .agg(
        units=("Quantity", "sum"),
        revenue=("Revenue", "sum"),
        orders=("InvoiceNo", "nunique")
    )
    .reset_index()
)

# Example segmentation features
X = product_summary[["units", "revenue", "orders"]].fillna(0)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X_scaled = StandardScaler().fit_transform(X)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
product_summary["Cluster"] = model.fit_predict(X_scaled)
# Simple monthly demand baseline
monthly = (
    clean.groupby(["StockCode", "Month"])["Quantity"]
    .sum()
    .reset_index()
)

monthly["Lag_1"] = monthly.groupby("StockCode")["Quantity"].shift(1)
monthly["Rolling_3"] = (
    monthly.groupby("StockCode")["Quantity"]
    .transform(lambda s: s.shift(1).rolling(3).mean())
)

# Future stage: train a forecasting/regression model
# using Lag_1, Rolling_3, calendar features, price, etc.
# Evaluate with a chronological train/test split and MAE/RMSE.
# Future route-optimization concept using Google OR-Tools
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Inputs required:
# distance_matrix, customer_demands, vehicle_capacities, depot, num_vehicles

# The solver would minimize route cost while enforcing
# vehicle-capacity constraints. Time-window constraints
# can be added when delivery appointment data is available.
