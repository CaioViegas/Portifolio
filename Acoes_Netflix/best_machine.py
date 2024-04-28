import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_data(file_path):  # Define uma função para carregar dados de um arquivo CSV
    df = pd.read_csv(file_path)  # Lê o arquivo CSV usando pandas e armazena os dados em um DataFrame
    return df  # Retorna o DataFrame 

def prepare_data(df, features, target, test_size=0.2, random_state=101):  # Define uma função para preparar os dados
    X = df[features]  # Seleciona as features do DataFrame
    y = df[target]  # Seleciona a coluna alvo do DataFrame
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)  # Divide os dados em conjuntos de treino e teste
    return X_train, X_test, y_train, y_test  # Retorna os conjuntos de dados de treino e teste

def evaluate_model(model, X_test, y_test):  # Define uma função para avaliar um modelo
    y_pred = model.predict(X_test)  # Faz previsões nos dados de teste

    r2 = r2_score(y_test, y_pred)  # Calcula o coeficiente de determinação nos dados de teste
    mse = mean_squared_error(y_test, y_pred)  # Calcula o erro quadrático médio nos dados de teste
    mae = mean_absolute_error(y_test, y_pred)  # Calcula o erro absoluto médio nos dados de teste
    return r2, mse, mae  # Retorna as métricas de avaliação do modelo

def save_model(model, filename):  # Define uma função para salvar um modelo treinado
    joblib.dump(model, filename)  # Salva o modelo em um arquivo

def load_model(filename):  # Define uma função para carregar um modelo salvo
    model = joblib.load(filename)  # Carrega o modelo do arquivo
    return model  # Retorna o modelo carregado

def main():  # Define a função principal
    file_path = "C:/Users/caioc/OneDrive/ProjetosData/Netflix/data/netflix_2014_2023.csv"  # Define o caminho do arquivo CSV
    features = ['open', 'high', 'low', 'close', 'volume', 'rsi_7', 'rsi_14', 'cci_7', 'cci_14',  # Define as features a serem utilizadas
                'sma_50', 'ema_50', 'sma_100', 'ema_100', 'macd', 'bollinger', 'TrueRange', 'atr_7', 'atr_14']
    target = 'next_day_close'  # Define o alvo da previsão

    df = load_data(file_path)  # Carrega os dados do arquivo CSV em um DataFrame

    X_train, X_test, y_train, y_test = prepare_data(df, features, target)  # Prepara os dados para modelagem

    pipeline = Pipeline([  # Define um pipeline de processamento de dados e modelo
        ('scaler', StandardScaler()),  # Adiciona um scaler ao pipeline
        ('model', LinearRegression())  # Adiciona um modelo de regressão linear ao pipeline
    ])

    param_grid = {  # Define um grid de hiperparâmetros para busca em grade
        'scaler__with_mean': [True, False],  # Define valores para o hiperparâmetro 'with_mean' do scaler
        'scaler__with_std': [True, False],  # Define valores para o hiperparâmetro 'with_std' do scaler
        'model__fit_intercept': [True, False]  # Define valores para o hiperparâmetro 'fit_intercept' do modelo
    }

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring='r2', cv=5, n_jobs=-1, verbose=1)  # Inicializa a busca em grade
    grid_search.fit(X_train, y_train)  # Executa a busca em grade para encontrar o melhor modelo

    best_model = grid_search.best_estimator_  # Obtém o melhor modelo encontrado pela busca em grade

    r2_train, mae_train, mse_train = evaluate_model(best_model, X_train, y_train)  # Avalia o melhor modelo nos dados de treino
    r2_test, mae_test, mse_test = evaluate_model(best_model, X_test, y_test)  # Avalia o melhor modelo nos dados de teste

    print("\nMelhores hiperparâmetros encontrados:")  # Imprime os melhores hiperparâmetros encontrados pela busca em grade
    print(grid_search.best_params_)
    print("\nResultados com melhor modelo encontrado:")  # Imprime os resultados da avaliação do melhor modelo
    print("R² Treino:", r2_train)
    print("MAE Treino:", mae_train)
    print("MSE Treino:", mse_train)
    print("R² Teste:", r2_test)
    print("MAE Teste:", mae_test)
    print("MSE Teste:", mse_test)

    save_model(best_model, 'modelo_linear_regression.joblib')  # Salva o modelo 

if __name__ == "__main__":  # Verifica se o script está sendo executado diretamente
    main()  # Chama a função principal se o script estiver sendo executado diretamente