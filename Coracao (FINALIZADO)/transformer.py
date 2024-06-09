import pandas as pd

df = pd.read_csv("C:/Users/caioc/Documents/ProjetosData/Coracao/data/heart.csv")

mapas_colunas = {
    'Age': 'Idade',
    'Sex': 'Sexo',
    'ChestPainType': 'TipoDorPeito',
    'RestingBP': 'PressaoArterialRepouso',
    'Cholesterol': 'Colesterol',
    'FastingBS': 'GlicoseJejum',
    'RestingECG': 'EletrocardiogramaRepouso',
    'MaxHR': 'FreqCardiacaMaxima',
    'ExerciseAngina': 'AnginaExercicio',
    'Oldpeak': 'DepressaoSTExercicioRepouso',
    'ST_Slope': 'InclinacaoST',
    'HeartDisease': 'DoencaCardiaca'
}

df.rename(columns=mapas_colunas, inplace=True)

traducoes = {
    'TipoDorPeito': {
        'ASY': 'Assintomatico',
        'NAP': 'ANA',
        'ATA': 'AT',
        'TA': 'AAT'
    },
    'EletrocardiogramaRepouso': {
        'Normal': 'Normal',
        'LVH': 'HVE',
        'ST': 'ASST'
    },
    'AnginaExercicio': {
        'N': 'Ausente',
        'Y': 'Presente'
    },
    'InclinacaoST': {
        'Flat': 'Plano',
        'Up': 'Ascendente',
        'Down': 'Descendente'
    },
    'GlicoseJejum': {
        0: 'Normal',
        1: 'Alto'
    }
}

for coluna, traducao in traducoes.items():
    df[coluna] = df[coluna].map(traducao)

df = df[df['Colesterol'] != 0]

df.to_csv("C:/Users/caioc/Documents/ProjetosData/Coracao/data/heart_translated.csv", index=False)

def transform_outliers_iqr(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[column] = df[column].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    
    return df

df = transform_outliers_iqr(df, ['PressaoArterialRepouso', 'Colesterol', 'FreqCardiacaMaxima', 'DepressaoSTExercicioRepouso'])

colunas_label = ['Sexo', 'GlicoseJejum', 'AnginaExercicio']

transform_manual = {
    'Sexo': {
        'M': 0,
        'F': 1
    },
    'GlicoseJejum': {
        'Normal': 0,
        'Alto': 1
    },
    'AnginaExercicio': {
        'Ausente': 0,
        'Presente': 1
    }
}

for coluna, transform in transform_manual.items():
    df[coluna] = df[coluna].map(transform)

df = pd.get_dummies(df, columns=['TipoDorPeito', 'EletrocardiogramaRepouso', 'InclinacaoST'])

bool_columns = df.select_dtypes(include=['bool']).columns
df[bool_columns] = df[bool_columns].astype('int64')

df.to_csv("C:/Users/caioc/Documents/ProjetosData/Coracao/data/heart_encoded.csv", index=False)