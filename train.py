"""Entraîne et compare plusieurs modèles de régression pour prédire le prix de voitures."""
import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/cars.csv")
    p.add_argument("--target", default="selling_price", help="Nom de la colonne prix")
    p.add_argument("--drop", nargs="*", default=["name"], help="Colonnes à ignorer")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_preprocessor(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # valeurs manquantes
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),  # encodage
    ])
    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])


def make_model(regressor, preprocessor):
    # Le prix est très asymétrique : on apprend sur log(prix)
    pipe = Pipeline([("prep", preprocessor), ("reg", regressor)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def main():
    args = parse_args()
    df = pd.read_csv(args.data)
    print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Nettoyage
    df = df.drop_duplicates()
    df = df.dropna(subset=[args.target])
    df = df[df[args.target] > 0]
    df = df.drop(columns=[c for c in args.drop if c in df.columns])

    X = df.drop(columns=[args.target])
    y = df[args.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )

    candidates = {
        "Régression linéaire": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=args.seed
        ),
    }

    results, best_name, best_model, best_r2, best_pred = {}, None, None, -np.inf, None
    for name, reg in candidates.items():
        model = make_model(reg, build_preprocessor(X_train))
        cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2 test": r2_score(y_test, pred),
            "R2 CV (5 plis)": cv_r2,
        }
        if results[name]["R2 test"] > best_r2:
            best_name, best_model, best_r2 = name, model, results[name]["R2 test"]
            best_pred = pred

    print("\n=== Résultats ===")
    print(pd.DataFrame(results).T.round(3).to_string())
    print(f"\nMeilleur modèle : {best_name}")

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/best_model.joblib")

    # Graphique prix réels vs prédits
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best_pred, alpha=0.4)
    lim = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
    plt.plot(lim, lim, "r--")
    plt.xlabel("Prix réel")
    plt.ylabel("Prix prédit")
    plt.title(f"Réel vs prédit ({best_name})")
    plt.tight_layout()
    plt.savefig("reports/actual_vs_predicted.png", dpi=150)
    print("Modèle sauvegardé : models/best_model.joblib")
    print("Graphique : reports/actual_vs_predicted.png")


if __name__ == "__main__":
    main()
