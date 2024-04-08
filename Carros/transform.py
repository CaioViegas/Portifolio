import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/carros_limpo.csv")

le = LabelEncoder()

df['UsadoOuNovo'] = le.fit_transform(df['UsadoOuNovo'])
df['Transmissao'] = le.fit_transform(df['Transmissao'])
df['TipoDirecao'] = le.fit_transform(df['TipoDirecao'])
df['TipoCombustivel'] = le.fit_transform(df['TipoCombustivel'])
df['TipoCarroceria'] = le.fit_transform(df['TipoCarroceria'])

encoder = ce.TargetEncoder(smoothing=0.5)

df['Empresa'] = encoder.fit_transform(df['Empresa'], df['Preco'])
df['Modelo'] = encoder.fit_transform(df['Modelo'], df['Preco'])
df['Carro/SUV'] = encoder.fit_transform(df['Carro/SUV'], df['Preco'])
df['Titulo'] = encoder.fit_transform(df['Titulo'], df['Preco'])
df['Motor'] = encoder.fit_transform(df['Motor'], df['Preco'])
df['Quilometros'] = encoder.fit_transform(df['Quilometros'], df['Preco'])
df['EsquemaCores'] = encoder.fit_transform(df['EsquemaCores'], df['Preco'])

df.to_csv('data/carros_codificado.csv', index=False)