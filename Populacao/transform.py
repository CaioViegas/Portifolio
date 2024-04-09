import pandas as pd

df1 = pd.read_csv("C:/Users/caioc/OneDrive/ProjetosData/Populacao/data/world_country_stats.csv")
df2 = pd.read_csv("C:/Users/caioc/OneDrive/ProjetosData/Populacao/data/world_population_by_country_2023.csv")
df3 = pd.read_csv("C:/Users/caioc/OneDrive/ProjetosData/Populacao/data/world_population_by_year_1950_2023.csv")

paises_brics = ['Brazil', 'Russia', 'India', 'China', 'South Africa']

df1 = df1[df1['country'].isin(paises_brics)]
df2 = df2[df2['country'].isin(paises_brics)]
df3 = df3[df3['country'].isin(paises_brics)]

df1 = pd.merge(df1, df2, on='country', how='left')

df1 = df1.drop(columns=['land_area_y', 'fertility_rate_y', 'median_age_y'])

mapas_colunas = {'country': 'pais', 'region': 'continente', 'land_area_x': 'area', 'fertility_rate_x': 'indice_fecundidade', 'median_age_x': 'mediana_idade',
                 'population': 'populacao', 'yearly_change': 'mudanca_anual', 'net_change': 'mudanca_populacao', 'density': 'densidade', 
                 'net_migrants': 'taxa_migracao', 'population_urban': 'populacao_urbana', 'world_share': 'parcela_mundial'}

df1 = df1.rename(columns=mapas_colunas)

df3 = df3.rename(columns={'country': 'pais'})

df1.to_csv("data/dados_brics.csv", index=False)
df3.to_csv("data/anos_brics.csv", index=False)