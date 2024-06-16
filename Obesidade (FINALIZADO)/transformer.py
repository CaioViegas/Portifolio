import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

df = pd.read_csv('data/ObesityDataSet_raw_and_data_sinthetic.csv')

mapas_colunas = {
    'Age': 'Idade',
    'Gender': 'Genero',
    'Height': 'Altura',
    'Weight': 'Peso',
    'CALC': 'Freq_Alcool',
    'FAVC': 'Freq_Densidade_Calorica',
    'FCVC': 'Freq_Vegetais',
    'NCP': 'Freq_Refeicoes',
    'SCC': 'Monitoramento_Calorias_Diarias',
    'SMOKE': 'Fumante',
    'CH2O': 'Agua_Diaria',
    'family_history_with_overweight': 'Historico_Familiar',
    'FAF': 'Atividade_Fisica',
    'TUE': 'Freq_Tecnologia',
    'CAEC': 'Alimento_Entre_Refeicoes',
    'MTRANS': 'Meio_Transporte',
    'NObeyesdad': 'Nivel_Obesidade'
}

df.rename(columns=mapas_colunas, inplace=True)

traducoes_colunas = {
    'Genero': {
        'Male': 'Masculino',
        'Female': 'Feminino'
    },
    'Freq_Alcool': {
        'Sometimes': 'Ocasionalmente',
        'no': 'Negativo',
        'Frequently': 'Frequentemente',
        'Always': 'Sempre'
    },
    'Freq_Densidade_Calorica': {
        'yes': 'Sim',
        'no': 'Nao'
    },
    'Monitoramento_Calorias_Diarias': {
        'no': 'Nao',
        'yes': 'Sim'
    },
    'Fumante': {
        'no': 'Nao',
        'yes': 'Sim'
    },
    'Historico_Familiar': {
        'yes': 'Sim',
        'no': 'Nao'
    },
    'Alimento_Entre_Refeicoes': {
        'Sometimes': 'Ocasionalmente',
        'Frequently': 'Frequentemente',
        'Always': 'Sempre',
        'no': 'Negativo'
    },
    'Meio_Transporte': {
        'Public_Transportation': 'Transporte_Publico',
        'Automobile': 'Automovel',
        'Walking': 'Caminhada',
        'Motorbike': 'Moto',
        'Bike': 'Bicicleta'
    },
    'Nivel_Obesidade': {
        'Insufficient_Weight': 'Peso_Insuficiente',
        'Normal_Weight': 'Peso_Normal',
        'Overweight_Level_I': 'Sobrepeso_I',
        'Overweight_Level_II': 'Sobrepeso_II',
        'Obesity_Type_I': 'Obesidade_Tipo_I',
        'Obesity_Type_II': 'Obesidade_Tipo_II',
        'Obesity_Type_III': 'Obesidade_Tipo_III'
    }
}

for coluna, traducao in traducoes_colunas.items():
    df[coluna] = df[coluna].map(traducao)

df.to_csv("data/obesidade_traduzido.csv", index=False)

ordinal_cols = ['Freq_Alcool']
ordinal_mappings = {
    'Freq_Alcool': ['Negativo', 'Ocasionalmente', 'Frequentemente', 'Sempre']
}
ordinal_encoders = {col: OrdinalEncoder(categories=[ordinal_mappings[col]]) for col in ordinal_cols}
for col, encoder in ordinal_encoders.items():
    df[col] = encoder.fit_transform(df[[col]])

nominal_cols = ['Alimento_Entre_Refeicoes', 'Meio_Transporte']
df = pd.get_dummies(df, columns=nominal_cols)

bool_columns = df.select_dtypes(include=['bool']).columns
df[bool_columns] = df[bool_columns].astype('float64')

label_cols = ['Genero', 'Freq_Densidade_Calorica', 'Monitoramento_Calorias_Diarias', 'Fumante', 'Historico_Familiar']
label_encoders = {col: LabelEncoder() for col in label_cols}
for col, encoder in label_encoders.items():
    df[col] = encoder.fit_transform(df[col])

target_col = 'Nivel_Obesidade'
target_encoder = LabelEncoder()
df[target_col] = target_encoder.fit_transform(df[target_col])

int_columns = df.select_dtypes(include=['int32']).columns
df[int_columns] = df[int_columns].astype('float64')

df.to_csv('data/obesidade_encoded.csv', index=False)