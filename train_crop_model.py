"""
train_crop_model.py
=====================================================
Trains a REAL scikit-learn RandomForestClassifier for crop recommendation.

Training data note (read this):
--------------------------------
This script generates training samples by sampling within documented
agronomic requirement RANGES for each crop (nitrogen/phosphorus/potassium
needs, ideal temperature, humidity, rainfall, and soil pH) drawn from
standard agronomy references. This is NOT the same as downloading a
pre-existing historical farm-records dataset (e.g. from Kaggle), which
this environment cannot access directly.

This is still a genuinely trained machine learning model — it learns
decision boundaries between crops from the data and makes real
probabilistic predictions (not hardcoded if/else rules). If you have
access to a real historical dataset (e.g. the public "Crop Recommendation
Dataset" CSV with N,P,K,temperature,humidity,ph,rainfall,label columns),
you can drop it in as `real_data.csv` in this folder and the script will
automatically prefer it over the synthetic generator — this will produce
an even more realistic model.

Run:
    python3 train_crop_model.py

Produces:
    crop_model.joblib   - the trained model
    crop_ranges.json     - per-crop condition ranges used for the "suitable
                            conditions" text shown to users
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

RANDOM_SEED = 42
SAMPLES_PER_CROP = 300

# Documented agronomic ranges per crop:
# (N kg/ha, P kg/ha, K kg/ha, temperature °C, humidity %, pH, rainfall mm)
CROP_RANGES = {
    "Rice":        {"N": (80, 120), "P": (40, 60), "K": (40, 60), "temp": (20, 35), "humidity": (70, 95), "ph": (5.5, 7.0), "rainfall": (150, 300)},
    "Maize":       {"N": (60, 100), "P": (30, 60), "K": (30, 50), "temp": (18, 30), "humidity": (50, 75), "ph": (5.5, 7.5), "rainfall": (60, 150)},
    "Wheat":       {"N": (50, 90),  "P": (25, 50), "K": (20, 40), "temp": (10, 25), "humidity": (40, 65), "ph": (6.0, 7.5), "rainfall": (40, 100)},
    "Cotton":      {"N": (40, 80),  "P": (20, 40), "K": (20, 40), "temp": (21, 35), "humidity": (40, 65), "ph": (5.8, 8.0), "rainfall": (50, 100)},
    "Sugarcane":   {"N": (100, 150),"P": (50, 80), "K": (50, 80), "temp": (21, 35), "humidity": (65, 90), "ph": (6.0, 7.5), "rainfall": (100, 250)},
    "Groundnut":   {"N": (10, 30),  "P": (20, 40), "K": (20, 40), "temp": (22, 33), "humidity": (50, 75), "ph": (6.0, 7.0), "rainfall": (50, 120)},
    "Soybean":     {"N": (15, 40),  "P": (30, 60), "K": (30, 50), "temp": (20, 30), "humidity": (55, 80), "ph": (6.0, 7.0), "rainfall": (60, 150)},
    "Chickpea":    {"N": (10, 30),  "P": (30, 60), "K": (15, 30), "temp": (10, 25), "humidity": (30, 55), "ph": (6.0, 7.5), "rainfall": (30, 65)},
    "Mustard":     {"N": (40, 70),  "P": (20, 40), "K": (15, 30), "temp": (10, 25), "humidity": (30, 60), "ph": (6.0, 7.5), "rainfall": (30, 60)},
    "Potato":      {"N": (80, 120), "P": (40, 60), "K": (60, 100),"temp": (10, 22), "humidity": (60, 85), "ph": (5.0, 6.5), "rainfall": (50, 100)},
    "Tomato":      {"N": (60, 100), "P": (40, 60), "K": (50, 90), "temp": (18, 28), "humidity": (55, 80), "ph": (6.0, 6.8), "rainfall": (40, 90)},
    "Onion":       {"N": (60, 100), "P": (30, 50), "K": (40, 70), "temp": (13, 28), "humidity": (50, 75), "ph": (6.0, 7.0), "rainfall": (35, 75)},
    "Banana":      {"N": (100, 150),"P": (30, 60), "K": (100, 150),"temp": (22, 32), "humidity": (70, 95), "ph": (5.5, 7.0), "rainfall": (120, 220)},
    "Coffee":      {"N": (60, 100), "P": (20, 40), "K": (30, 60), "temp": (15, 28), "humidity": (60, 85), "ph": (5.0, 6.5), "rainfall": (150, 250)},
    "Millet":      {"N": (20, 50),  "P": (15, 30), "K": (10, 25), "temp": (25, 35), "humidity": (30, 55), "ph": (5.5, 7.5), "rainfall": (30, 70)},
}


def generate_synthetic_dataset():
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for crop, ranges in CROP_RANGES.items():
        for _ in range(SAMPLES_PER_CROP):
            row = {
                "N": rng.uniform(*ranges["N"]),
                "P": rng.uniform(*ranges["P"]),
                "K": rng.uniform(*ranges["K"]),
                "temperature": rng.uniform(*ranges["temp"]),
                "humidity": rng.uniform(*ranges["humidity"]),
                "ph": rng.uniform(*ranges["ph"]),
                "rainfall": rng.uniform(*ranges["rainfall"]),
                "label": crop,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    real_csv = os.path.join(os.path.dirname(__file__), "real_data.csv")
    if os.path.exists(real_csv):
        print(f"Found {real_csv} — training on real historical data instead of synthetic ranges.")
        df = pd.read_csv(real_csv)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"n": "N", "p": "P", "k": "K"})
    else:
        print("No real_data.csv found — generating training data from documented agronomic ranges.")
        df = generate_synthetic_dataset()

    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation accuracy: {acc:.4f}")

    out_dir = os.path.dirname(__file__)
    joblib.dump({"model": model, "features": feature_cols, "classes": list(model.classes_)},
                os.path.join(out_dir, "crop_model.joblib"))

    # Save human-readable ranges for building "suitable conditions" text in API responses
    with open(os.path.join(out_dir, "crop_ranges.json"), "w") as f:
        json.dump(CROP_RANGES, f, indent=2)

    print("Saved crop_model.joblib and crop_ranges.json")


if __name__ == "__main__":
    main()
