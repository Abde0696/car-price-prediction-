# Prédiction du prix des voitures

Modèle de Machine Learning qui prédit le prix d'un véhicule à partir de ses
caractéristiques (année, kilométrage, carburant, puissance, etc.).

## Pipeline

1. **Nettoyage** : doublons, prix invalides
2. **Prétraitement** : imputation des valeurs manquantes (médiane / valeur la plus fréquente),
   standardisation des variables numériques, One-Hot Encoding des variables catégorielles
3. **Modélisation** : régression Ridge vs Random Forest
4. **Évaluation** : MAE, RMSE, R² sur jeu de test + validation croisée (5 plis), le meilleur modèle est choisi sur le R² en validation croisée

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

Jeu de données Kaggle « Vehicle dataset » (CarDekho, ~300 voitures d'occasion), à placer dans `data/cars.csv`.
Colonnes : Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner.

## Utilisation

```bash
pip install -r requirements.txt
python train.py
python predict.py --car '{"Year": 2015, "Present_Price": 8.5, "Kms_Driven": 70000, "Fuel_Type": "Diesel", "Seller_Type": "Dealer", "Transmission": "Manual", "Owner": 0}'
```

## Résultats

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| Régression Ridge | 1.48 | 2.52 | 0.753 |
| Random Forest | 1.44 | 3.39 | 0.555 |

Prix en lakhs de roupies. R² en validation croisée (5 plis) : Ridge 0.870, Random Forest 0.877 (modèle retenu).
Le R² sur le jeu de test est plus bas et variable car le jeu de données est petit (301 lignes) et contient quelques voitures très chères.

![Réel vs prédit](reports/actual_vs_predicted.png)

## Pistes d'amélioration

- Ajouter Gradient Boosting / XGBoost et un `GridSearchCV`
- Extraire la marque depuis le nom du véhicule
- Déployer une petite interface (Streamlit)
