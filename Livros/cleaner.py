import pandas as pd

try:
    df = pd.read_csv("data/Books_Data_Clean.csv")
except FileNotFoundError:
    print("Arquivo de entrada não encontrado.")
    exit()

df.loc[df['language_code'].isin(['en-US', 'en-GB', 'en-CA']), 'language_code'] = 'eng'
df.loc[df['genre'].isin(['genre fiction']), 'genre'] = 'fiction'
df.loc[df['Book Name'].isnull(), 'Book Name'] = 'Desconhecido'
df.loc[df['language_code'].isnull(), 'language_code'] = 'eng'

df.dropna(subset=['Publishing Year'], inplace=True)

mapeamento_colunas = {"Publishing Year": "ano_publicacao", "Book Name": "nome_livro", "Author": "autor", "language_code": "linguagem", 
                      "Author_Rating": "experiencia_autor", "Book_average_rating": "nota_media_livro", "Book_ratings_count": "contagem_votos", "genre": "genero",
                      "gross sales": "total_em_vendas", "publisher revenue": "receita_editora", "sale price": "preco_venda", "sales rank": "rank_vendas", 
                      "Publisher ": "editora", "units sold": "unidades_vendidas"}

df.rename(columns=mapeamento_colunas, inplace=True)

df.drop(columns=['index'], inplace=True)

df.to_csv("data/arquivo_livro_limpo.csv", index=False)