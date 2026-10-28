"""Prédit le prix d'une voiture à partir de ses caractéristiques (JSON)."""
import argparse
import json

import joblib
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument(
    "--car",
    required=True,
    help='Ex: \'{"year": 2015, "km_driven": 70000, "fuel": "Diesel", '
         '"seller_type": "Individual", "transmission": "Manual", "owner": "First Owner"}\'',
)
args = p.parse_args()

model = joblib.load("models/best_model.joblib")
car = pd.DataFrame([json.loads(args.car)])
print(f"Prix estimé : {model.predict(car)[0]:,.0f}")
