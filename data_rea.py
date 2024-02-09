import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


""" Datasets' importation """ 

# Dataset Readmission
data_rea = pd.read_csv("C:/Users/alqui/Downloads/readmission.csv")
#data_rea.info()


""" Filtration du dataset"""

# On s'assure que le dataset est trié en fonction des "id" et ensuite de "enum"
rea_trie = data_rea.sort_values(by=['id', 'enum'], ascending=[True, True])

# On filtre ensuite le dataset en prenant seulement les données de la dernière admission pour un patient (on prend pour un même "id", "enum" le plus grand)
rea_filtre = rea_trie.groupby('id').last().reset_index()

# print(rea_filtre.head)


"""Préparation des données"""

#Encodage des variables catégorielles (One-Hot)
colonnes_categorielles = ['sex', 'chemo', 'dukes', 'charlson']
encoder = OneHotEncoder()
encoder.fit(rea_filtre[colonnes_categorielles])
rea_encodees = encoder.fit_transform(rea_filtre[colonnes_categorielles]).toarray()
nouveaux_noms_colonnes = encoder.get_feature_names_out(colonnes_categorielles)
rea_encodees_df = pd.DataFrame(rea_encodees, columns=nouveaux_noms_colonnes)
rea_final = rea_filtre.drop(columns=colonnes_categorielles).join(rea_encodees_df)

print(rea_final.head())


features = rea_final.drop('death', axis = 1).drop('id', axis=1).drop('t.stop', axis = 1).drop('t.start', axis = 1)
death = rea_final['death']
#print(features.head())
features_train, features_test, death_train, death_test = train_test_split(features, death, test_size=0.2, random_state=42)

#print(features_train.dtypes)