from data_rea import *
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV, cross_val_score
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""Mise en place du modèle Gradient Boosting"""


# Initialisez le Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Entraînez le modèle
gb_clf.fit(features_train, death_train)

# Prédictions sur l'ensemble de test
death_pred = gb_clf.predict(features_test)

# Évaluation du modèle
score = roc_auc_score(death_test, death_pred)
print(f"Score: {score}")


""""Optimisation par GridSearchCV"""

#Définition de la grille d'hyperparamètres
param_grid = {'n_estimators': [115, 116, 117, 118, 119],
              'learning_rate': [0.019, 0.018, 0.02, 0.017],
              'max_depth': [3, 4, 2],
              'min_samples_split': [8, 9, 10],
              'min_samples_leaf': [8, 9, 10]}

# Initialisation de Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(random_state=42)

# Création d'une instance de GridSearchCV
grid_search = GridSearchCV(estimator=gb_clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Exécutetion de la recherche par grille sur les données d'entraînement
grid_search.fit(features_train, death_train)

# Les meilleurs paramètres
print(f"Meilleurs paramètres avec GridSearchCV: {grid_search.best_params_}")


"""Optimisation par RandomizedSearchCV"""

# Définissez la distribution des hyperparamètres à échantillonner
param_dist = {
    'n_estimators': sp_randint(100, 500),  # uniforme discret entre 100 et 500
    'learning_rate': uniform(0.01, 0.2),   # uniforme continu entre 0.01 et 0.21 (0.01 + 0.2)
    'max_depth': sp_randint(3, 10),        # uniforme discret entre 3 et 10
    'min_samples_split': sp_randint(2, 11),# uniforme discret entre 2 et 11
    'min_samples_leaf': sp_randint(1, 11)  # uniforme discret entre 1 et 11
}
# Initialisez le Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(random_state=42)

# Créez une instance de RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=gb_clf, param_distributions=param_dist, n_iter=1000, cv=5, n_jobs=-1, verbose=2, random_state=42)

# Exécutez la recherche aléatoire sur les données d'entraînement
random_search.fit(features_train, death_train)

# Affichez les meilleurs paramètres pour RandomizedSearchCV
print(f"Meilleurs paramètres avec RandomizedSearchCV: {random_search.best_params_}")


"""Optimisation par HalvingGridSearchCv"""

# Créez une instance de HalvingGridSearchCV
halving_grid_search = HalvingGridSearchCV(estimator=gb_clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, factor=2, resource='n_samples', min_resources='smallest', aggressive_elimination=False)

# Exécutez la recherche par grille avec diminution sur les données d'entraînement
halving_grid_search.fit(features_train, death_train)

# Affichez les meilleurs paramètres
print(f"Meilleurs paramètres avec HalvingGridSearchCV: {halving_grid_search.best_params_}")


"""Optimisation par optuna"""

def objective(trial):
    # Définition de l'espace de recherche des hyperparamètres
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 3, 9)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)

    # Création du modèle avec les hyperparamètres suggérés
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        learning_rate=learning_rate,
        random_state=42
    )

    # Calcul de la validation croisée
    score = cross_val_score(clf, features_train, death_train, n_jobs=-1, cv=3)

    # Optuna veut minimiser la fonction objectif donc on retourne le négatif de l'accuracy
    return -score.mean()

# Création de l'étude qui va contenir toutes les informations de l'optimisation
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Affichage des meilleurs hyperparamètres
print(study.best_params)


"""Création du modèle optimal"""

# Initialisez le Gradient Boosting Classifier
gb_clf_gs = GradientBoostingClassifier(**grid_search.best_params_)

# Entraînez le modèle
gb_clf_gs.fit(features_train, death_train)

# Prédictions sur l'ensemble de test
death_pred = gb_clf_gs.predict(features_test)

# Évaluation du modèle
score = roc_auc_score(death_test, death_pred)
print(f"Score: {score}")