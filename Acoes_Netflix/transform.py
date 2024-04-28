import pandas as pd

df = pd.read_csv("C:/Users/caioc/OneDrive/ProjetosData/Netflix/data/nflx_2014_2023.csv")

df['date'] = pd.to_datetime(df['date'])

df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

df['day'] = df['day'].astype('int64')
df['month'] = df['month'].astype('int64')
df['year'] = df['year'].astype('int64')

df.to_csv("data/netflix_2014_2023.csv", index=False)