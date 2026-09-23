"""
train_model.py
--------------
Loads Crop_data.csv, preprocesses it, trains a Random Forest Regressor
to predict crop Yield, and persists artefacts to the model/ directory.
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings("ignore")

# ── 1. Load data ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Crop_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# ── 2. Clean column names ────────────────────────────────────────────────────
df.columns = [c.strip() for c in df.columns]

# ── 3. Drop rows with missing target or key features ────────────────────────
df.dropna(subset=["Yield", "Area", "Production", "Crop_Year"], inplace=True)
df = df[df["Yield"] > 0]
df = df[df["Area"] > 0]

# Strip whitespace from string columns
for col in ["State", "District", "Crop", "Season"]:
    df[col] = df[col].astype(str).str.strip()

# ── 4. Encode categoricals ───────────────────────────────────────────────────
CATEGORICAL_COLS = ["State", "District", "Crop", "Season"]
encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

# ── 5. Feature / target split ────────────────────────────────────────────────
FEATURE_COLS = [c + "_enc" for c in CATEGORICAL_COLS] + ["Crop_Year", "Area", "Production"]
X = df[FEATURE_COLS]
y = df["Yield"]

# ── 6. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 7. Train model ───────────────────────────────────────────────────────────
print("Training Random Forest Regressor …")
model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── 8. Evaluate ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")

# ── 9. Persist artefacts ─────────────────────────────────────────────────────
joblib.dump(model,    os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))

# Save unique values for each categorical (used to populate dropdowns in UI)
unique_vals = {}
for col in CATEGORICAL_COLS:
    unique_vals[col] = sorted([str(v) for v in df[col].unique().tolist()])

with open(os.path.join(MODEL_DIR, "unique_values.json"), "w") as f:
    json.dump(unique_vals, f)

# Save metrics
metrics = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f)

# Save crop-year range
year_range = {"min": int(df["Crop_Year"].min()), "max": int(df["Crop_Year"].max())}
with open(os.path.join(MODEL_DIR, "year_range.json"), "w") as f:
    json.dump(year_range, f)

print("Artefacts saved to model/")
print("  rf_model.pkl")
print("  encoders.pkl")
print("  unique_values.json")
print("  metrics.json")
print("  year_range.json")
print("Done!")
