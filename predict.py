"""Prédit le prix d'une voiture à partir de ses caractéristiques (JSON)."""
import argparse
import json

import joblib
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument(
    "--car",
    required=True,
    help='Ex: \'{"Year": 2015, "Present_Price": 8.5, "Kms_Driven": 70000, '
         '"Fuel_Type": "Diesel", "Seller_Type": "Dealer", "Transmission": "Manual", "Owner": 0}\'',
)
args = p.parse_args()

model = joblib.load("models/best_model.joblib")
car = pd.DataFrame([json.loads(args.car)])
print(f"Prix estimé : {model.predict(car)[0]:.2f}")
