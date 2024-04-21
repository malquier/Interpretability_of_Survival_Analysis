from data_ins import *
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

"""----------- Mise en place du modèle Gradient Boosting pour insurance -----------"""

# Optimsiation des hyperparamètres

#1ère approche en utilisant optuna :

def objective_ins(trial):
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
    score = cross_val_score(clf, ins_features_train, ins_credit_train, n_jobs=-1, cv=3)

    # Optuna veut minimiser la fonction objectif donc on retourne le négatif de l'accuracy
    return -score.mean()

# Création de l'étude qui va contenir toutes les informations de l'optimisation
study_ins = optuna.create_study(direction='minimize')
study_ins.optimize(objective_ins, n_trials=100)

# Affichage des meilleurs hyperparamètres
print(study_ins.best_params)