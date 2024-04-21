import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Dataset Insurance
data_ins = pd.read_csv("C:/Users/alqui/Downloads/Insurance_data.csv")
ins_credit = pd.read_csv("C:/Users/alqui/Downloads/Insurance_time_data.csv")

data_ins.rename(columns = {'0':'age','1':'sex', '2':'smoker', '3':'pren_prod', '4':'pren_comp', '5':'point_sales', '6': 'product_type', '7': 'dist_channel', '8': 'pay_freq', '9': 'pay_method', '10':'profession'}, inplace = True)
data_ins.info()

colonnes_categorielles = ['sex', 'smoker', 'point_sales', 'product_type', 'dist_channel', 'pay_freq', 'pay_method', 'profession']
encoder = OneHotEncoder()
encoder.fit(data_ins[colonnes_categorielles])
ins_encodees = encoder.fit_transform(data_ins[colonnes_categorielles]).toarray()
nouveaux_noms_colonnes = encoder.get_feature_names_out(colonnes_categorielles)
ins_encodees_df = pd.DataFrame(ins_encodees, columns=nouveaux_noms_colonnes)
ins_features = data_ins.drop(columns=colonnes_categorielles).join(ins_encodees_df)

print(ins_features.head())
print(ins_credit.head())

ins_features_train, ins_features_test, ins_credit_train, ins_credit_test = train_test_split(ins_features, ins_credit, test_size=0.2, random_state=42)