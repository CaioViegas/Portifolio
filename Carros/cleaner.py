import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

df = pd.read_csv('data/Australian Vehicle Prices.csv')

mapeamento_colunas = {'Brand': 'Empresa', 'Year': 'Ano', 'Model': 'Modelo', 'Car/Suv': 'Carro/SUV', 'Title': 'Titulo', 'UsedOrNew': 'UsadoOuNovo', 
                      'Transmission': 'Transmissao', 'Engine': 'Motor', 'DriveType': 'TipoDirecao', 'FuelType': 'TipoCombustivel', 
                      'FuelConsumption': 'ConsumoCombustivel', 'Kilometres': 'Quilometros', 'ColourExtInt': 'EsquemaCores', 'CylindersinEngine': 'CilindrosMotor',
                      'BodyType': 'TipoCarroceria', 'Doors': 'Portas', 'Seats': 'Bancos', 'Price': 'Preco'}

df.rename(columns=mapeamento_colunas, inplace=True)

df.drop('Location', axis=1, inplace=True)

df['Preco'] = df['Preco'].replace('POA', np.nan).astype('float64')

df['Portas'] = df['Portas'].fillna('0').astype(str).str.extract('(\\d+)').astype('int64')

df['Bancos'] = df['Bancos'].fillna('0').astype(str).str.extract('(\\d+)').astype('int64')

df['CilindrosMotor'] = df['CilindrosMotor'].fillna('0').astype(str).str.extract('(\\d+)').fillna(0).astype('int64')

df['ConsumoCombustivel'] = df['ConsumoCombustivel'].fillna('0').astype(str).str.extract('(\\d+\\.?\\d*)').astype('float64')

df = df.dropna(subset=['Portas', 'Bancos', 'ConsumoCombustivel'], how='all')

imputer = KNNImputer(n_neighbors=5)

df['ConsumoCombustivel'] = imputer.fit_transform(df[['ConsumoCombustivel']])

df[['Portas', 'Bancos', 'Preco']] = imputer.fit_transform(df[['Portas', 'Bancos', 'Preco']])

df[['Portas', 'Bancos']] = df[['Portas', 'Bancos']].astype('int64')

df = df.dropna()

df.to_csv('data/carros_limpo.csv', index=False)
