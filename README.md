# Etude des méthodes d’interprétabilité du Machine Learning en analyse de survie

## Introduction

Ce projet académique vise à explorer les méthodes d'interprétabilités de modèles de ML appliqués à l'analyse de survie. Il a été encadré par Juliette Murris et Sandrine Katsahian, sans qui nous n'aurions pas pour aboutir un tel projet dans un tel laps de temps.
Pour cela, nous avons fait le choix d'étudier trois modèles d'interprétabilité : Random Forest, Gradient Boosting et enfin Support Vector Machine. Nous avons ensuite choisi d'appliquer trois méthodes d'interprétabilités à ces modèles : Permutation feature importance, Shap et LIME.
Nous avons tout d'abord travailler sur un problème de classification afin s'approprier ces modèles et méthodes, puis nous avons adaptés ces modèles/méthodes à l'analyse de survie.

## Contenu

Ce projet contient : 
  - 1 classe Model permettant d'automatiser nos travaux. Cette classe permet, à partir d'une base de données traitée, d'entraîner un modèle, de l'optimiser et d'appliquer les méthodes d'interprétabilités. Elle a également d'autres méthodes comme "hyperparameters_importance" qui permet de connaître l'importance des hyperparamètres pour un modèle.
  - 1 exemple d'utilisation de la classe Model qui explique comment utiliser la classe et ces différentes méthodes.
  - 2 dossiers "Classification" et "Survival_Analysis" comportant les notebooks sur lesquels nous avons travaillé dans un premier temps que ce soit en classification et en survie. Il y a un notebook par modèle.
  - 4 datasets : 1 dataset Réadmission, 1 dataset Insurance avec X_train.csv et y_train.csv qui sont concatés pour former ce dataset, 1 dataset Insurance réduit en gardant que 30% aléatoirement.
  - 1 fichier python "tools.py" dans lequel on peut retrouver différentes fonctions utilisées pour la classe Model.
  - 1 fichier requirement.txt sur lequel est présent toutes les librairies et leurs versions utilisées.

## Utilisation

Ce rapport est destiné aux praticiens, chercheurs et décideurs impliqués dans le développement et l'utilisation de l'analyse de survie. Il fournit une vue d'ensemble de différentes méthodes d'interprétation des trois modèles présentés précèdemment. Les lecteurs pourront ainsi comprendre l'importance de l'interprétabilité, choisir les méthodes appropriées pour leurs besoins et mettre en œuvre des bonnes pratiques pour améliorer la transparence et la confiance dans les modèles de machine learning.

## Auteurs

Ce rapport a été rédigé par Mattéo Alquier, Emile Cassant et Marama Simoneau en collaboration avec Juliette Murris et Sandrine Katsahian. Pour toute question ou commentaire, veuillez nous contacter à alquier.matteo@gmail.com.

## Historique des versions

1.0 : 16/05/2024


## Note

Ce rapport est une ressource vivante et sera régulièrement mis à jour avec de nouvelles informations et des améliorations. Nous vous encourageons à consulter régulièrement cette ressource pour accéder aux dernières mises à jour et développements dans le domaine de l'interprétabilité des modèles de machine learning.
