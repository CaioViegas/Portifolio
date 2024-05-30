import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv("C:/Users/caioc/OneDrive/ProjetosData/Cirrose/data/liver_cirrhosis.csv")

mapas_colunas = {'N_Days': 'Num_Dias', 'Drug': 'Medicamento', 'Age': 'Idade', 'Sex': 'Sexo', 'Ascites': 'Ascite', 'Hepatomegaly': 'Hepatomegalia', 
                 'Spiders': 'Aranhas_Vasculares', 'Bilirubin': 'Bilirrubina', 'Cholesterol': 'Colesterol', 'Albumin': 'Albumina', 'Copper': 'Cobre_Urina', 
                 'Alk_Phos': 'Fos_Alcalina', 'SGOT': 'AST', 'Tryglicerides': 'Triglicerideos', 'Platelets': 'Plaquetas', 'Prothrombin': 'Tempo_Protrombina',
                 'Stage': 'Estagio'}

df.rename(columns=mapas_colunas, inplace=True)

df.to_csv('data/liver_cirrhosis_translated.csv', index=False)

def utilizar_label_encoder(dataframe, coluna):
    le = LabelEncoder()
    dataframe[coluna] = le.fit_transform(dataframe[coluna])
    dataframe[coluna] = df[coluna].astype('int64')

colunas_label = ['Sexo', 'Ascite', 'Hepatomegalia', 'Aranhas_Vasculares']

for col in colunas_label:
    utilizar_label_encoder(df, col)

def utilizar_onehot_encoder(dataframe, coluna):
    ohe = OneHotEncoder()
    encoded = ohe.fit_transform(dataframe[coluna].values.reshape(-1, 1)).toarray()
    df_one_hot = pd.DataFrame(encoded, columns=[coluna + "_" + str(int(i)) for i in range(encoded.shape[1])])
    dataframe = pd.concat([dataframe, df_one_hot], axis=1)
    dataframe = dataframe.drop(coluna, axis=1)
    return dataframe

colunas_ohe = ['Status', 'Medicamento', 'Edema']

for col in colunas_ohe:
    df = utilizar_onehot_encoder(df, col)

df.to_csv('data/liver_cirrhosis_transformed.csv', index=False)
