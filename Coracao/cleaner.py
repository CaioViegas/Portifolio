import pandas as pd

try:
    df = pd.read_csv("data/heart.csv")
except FileNotFoundError:
    print("Arquivo não encontrado.")
    exit()

mapeamento_colunas = {'Age': 'Idade', 'Sex': 'Sexo', 'ChestPainType': 'TipoDorPeito', 'RestingBP': 'PressaoArterialRepouso', 'Cholesterol': 'Colesterol', 
                      'FastingBS': 'GlicoseJejum', 'RestingECG': 'EletrocardiogramaRepouso', 'MaxHR': 'FreqCardiacaMaxima', 'ExerciseAngina': 'AnginaExercicio',
                      'Oldpeak': 'DepressaoSTExercicioRepouso', 'ST_Slope': 'InclinacaoST', 'HeartDisease': 'DoencaCardiaca'}

df.rename(columns=mapeamento_colunas, inplace=True)

df['GlicoseJejum'] = df['GlicoseJejum'].replace({0: 'Normal', 1: 'Alto'})

df['GlicoseJejum'] = df['GlicoseJejum'].astype('object')

df = df[df['Colesterol'] != 0]

df.to_csv("data/coracao.csv", index=False)