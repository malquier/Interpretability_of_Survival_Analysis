import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from data_rea import *
from grad_boost import * 
import lime
import shap



"""Courbes d'apprentissage"""

# Génération de la courbe d'apprentissage
train_sizes, train_scores, test_scores = learning_curve(
    gb_clf_gs, features, death, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

# Calcul des scores moyens et des écarts types
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Tracer de la courbe d'apprentissage
plt.figure(figsize=(10, 6))
plt.title('Courbe d\'apprentissage Gradient Boosting')
plt.xlabel('Taille de l\'ensemble d\'entraînement')
plt.ylabel('Score')
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Entraînement")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation")

plt.legend(loc="best")
plt.show()



"""Effets des hyperparamètres sur la prédiction"""

# Pour n_estimators : 
n_estimators = [5*i + 1 for i in range(100)]
AUR_ROC = []

for i in n_estimators:

    # Définir le modèle SVM
    gb_model = GradientBoostingClassifier(n_estimators = i)

    gb_model.fit(features_train,death_train)

    gb_death_pred = gb_model.predict(features_test)

    aur_roc = roc_auc_score(death_test,gb_death_pred)

    AUR_ROC.append(aur_roc)

plt.plot(n_estimators,AUR_ROC)
plt.ylabel('AUR_ROC')
plt.xlabel('n_estimators')
plt.grid()
plt.show()


# Pour learning_rate : 
learning_rate = [0.01*i for i in range(100)]
AUR_ROC = []

for i in learning_rate:

    # Définir le modèle SVM
    gb_model = GradientBoostingClassifier(learning_rate = i)

    gb_model.fit(features_train,death_train)

    gb_death_pred = gb_model.predict(features_test)

    aur_roc = roc_auc_score(death_test,gb_death_pred)

    AUR_ROC.append(aur_roc)

plt.plot(learning_rate,AUR_ROC)
plt.ylabel('AUR_ROC')
plt.xlabel('learning_rate')
plt.grid()
plt.show()


"""Influence relative des variables"""

# Récupération de l'importance des caractéristiques
feature_importances = gb_clf_gs.feature_importances_

# Noms des caractéristiques, ajustez en fonction de votre jeu de données
feature_names = features_train.columns
features = np.array(features_train.columns)
sorted_idx = np.argsort(feature_importances)[::-1]

# Création du graphique
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features[sorted_idx])
plt.gca().invert_yaxis()  # Inverser l'axe des ordonnées pour avoir la caractéristique la plus importante en haut
plt.xlabel('Importance')
plt.ylabel('Caractéristiques')
plt.title('Importance des Caractéristiques')
plt.show()


"""Méthode LIME"""
# Création de l'explainer LIME pour les données tabulaires
explainer = lime.lime_tabular.LimeTabularExplainer(features_train.values, feature_names=features_train.columns, class_names=['Mort', 'Vivant'], discretize_continuous=True)

# Sélection d'une instance à expliquer
instance = 3
exp = explainer.explain_instance(features_test.values[instance], gb_clf_gs.predict_proba, num_features=5)

# Affichage de l'explication
exp.show_in_notebook(show_table=True, show_all=False)



"""Méthode SHAP"""

# Pour le modèle XGBClassifier
explicateur_shap_xgbC = shap.TreeExplainer(gb_clf_gs)
shap_values_xgbC = explicateur_shap_xgbC.shap_values(features_test)

# Affichage des interprétations
shap.summary_plot(shap_values_xgbC, features_test)

# Diagramme de force pour une instance spécifique (ici la première)
shap.initjs()
shap.force_plot(explicateur_shap_xgbC.expected_value, shap_values_xgbC[0,:], features_test.iloc[0,:], matplotlib = True)

# Diagramme de dépendance pour une feature donnée
shap.dependence_plot("enum", shap_values_xgbC, features_test)