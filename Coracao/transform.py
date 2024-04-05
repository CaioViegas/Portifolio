import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

df = pd.read_csv("data/coracao.csv")

le = LabelEncoder()

df['Sexo'] = le.fit_transform(df['Sexo'])
df['AnginaExercicio'] = le.fit_transform(df['AnginaExercicio'])
df['InclinacaoST'] = le.fit_transform(df['InclinacaoST'])
df['TipoDorPeito'] = le.fit_transform(df['TipoDorPeito'])
df['EletrocardiogramaRepouso'] = le.fit_transform(df['EletrocardiogramaRepouso'])

ordinal_encoder = OrdinalEncoder(categories=[['Normal', 'Alto']])

df['GlicoseJejum'] = ordinal_encoder.fit_transform(df[['GlicoseJejum']])

df.to_csv("data/coracao_codificado.csv", index=False)
