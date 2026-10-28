# Prédiction du prix des voitures

Modèle de Machine Learning qui prédit le prix d'un véhicule à partir de ses
caractéristiques (année, kilométrage, carburant, puissance, etc.).

## Pipeline

1. **Nettoyage** : doublons, prix invalides
2. **Prétraitement** : imputation des valeurs manquantes (médiane / valeur la plus fréquente),
   standardisation des variables numériques, One-Hot Encoding des variables catégorielles
3. **Modélisation** : régression linéaire vs Random Forest, entraînés sur le log du prix
4. **Évaluation** : MAE, RMSE, R² sur jeu de test + validation croisée (5 plis)

## Technologies

Python, pandas, NumPy, scikit-learn, Matplotlib

## Structure

```
car-price-prediction/
├── data/            # jeu de données (cars.csv, non versionné)
├── models/          # modèle entraîné (généré)
├── reports/         # graphiques (généré)
├── train.py         # entraînement et évaluation
├── predict.py       # prédiction sur une voiture
└── requirements.txt
```

## Données

Télécharger un jeu de données de voitures d'occasion (ex. « Vehicle dataset » / CarDekho
sur Kaggle) et le placer dans `data/cars.csv`. Le script s'adapte aux colonnes présentes :
utiliser `--target` et `--drop` si les noms diffèrent.

## Utilisation

```bash
pip install -r requirements.txt
python train.py --data data/cars.csv --target selling_price --drop name
python predict.py --car '{"year": 2015, "km_driven": 70000, "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual", "owner": "First Owner"}'
```

## Résultats

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| Régression linéaire | _à compléter_ | _à compléter_ | _à compléter_ |
| Random Forest | _à compléter_ | _à compléter_ | _à compléter_ |

![Réel vs prédit](reports/actual_vs_predicted.png)

## Pistes d'amélioration

- Ajouter Gradient Boosting / XGBoost et un `GridSearchCV`
- Extraire la marque depuis le nom du véhicule
- Déployer une petite interface (Streamlit)
