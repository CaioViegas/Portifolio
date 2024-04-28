import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

def load_data(file_path):  # Define uma função para carregar dados de um arquivo CSV
    df = pd.read_csv(file_path)  # Lê o arquivo CSV usando pandas e armazena os dados em um DataFrame
    return df  # Retorna o DataFrame 

def prepare_data(data, features, target, test_size=0.2, scaler=None):  # Define uma função para preparar os dados para utilização no modelo
    X = data[features]  # Seleciona as features que serão usadas para a previsão
    y = data[target]  # Seleciona a coluna alvo para a previsão

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=101)  # Divide os dados em conjuntos de treino e teste

    if scaler:  # Se um scaler for fornecido
        X_train = scaler.fit_transform(X_train)  # Ajusta e transforma os dados de treino usando o scaler
        X_test = scaler.transform(X_test)  # Transforma os dados de teste usando o scaler ajustado

    return X_train, X_test, y_train, y_test  # Retorna os conjuntos de dados de treino e teste

def train_evaluate_model(model, X_train, X_test, y_train, y_test):  # Define uma função para treinar e avaliar um modelo
    model.fit(X_train, y_train)  # Treina o modelo com os dados de treino

    train_pred = model.predict(X_train)  # Faz previsões nos dados de treino
    test_pred = model.predict(X_test)  # Faz previsões nos dados de teste

    mse_train = mean_squared_error(y_train, train_pred)  # Calcula o erro quadrático médio nos dados de treino
    mse_test = mean_squared_error(y_test, test_pred)  # Calcula o erro quadrático médio nos dados de teste

    r2_train = r2_score(y_train, train_pred)  # Calcula o coeficiente de determinação nos dados de treino
    r2_test = r2_score(y_test, test_pred)  # Calcula o coeficiente de determinação nos dados de teste

    mae_train = mean_absolute_error(y_train, train_pred)  # Calcula o erro absoluto médio nos dados de treino
    mae_test = mean_absolute_error(y_test, test_pred)  # Calcula o erro absoluto médio nos dados de teste

    return {  # Retorna um dicionário contendo as métricas de avaliação do modelo
        "MSE_train": mse_train,
        "MSE_test": mse_test,
        "R²_train": r2_train,
        "R²_test": r2_test,
        "MAE_train": mae_train,
        "MAE_test": mae_test
    }

def main():  # Define a função principal
    file_path = "C:/Users/caioc/OneDrive/ProjetosData/Netflix/data/netflix_2014_2023.csv"  # Define o caminho do arquivo CSV
    data = load_data(file_path)  # Carrega os dados do arquivo CSV em um DataFrame

    features = ['open', 'high', 'low', 'close', 'volume', 'rsi_7', 'rsi_14', 'cci_7', 'cci_14',  # Define as features a serem utilizadas
                'sma_50', 'ema_50', 'sma_100', 'ema_100', 'macd', 'bollinger', 'TrueRange', 'atr_7', 'atr_14']
    target = 'next_day_close'  # Define a coluna alvo da previsão

    scalers = {  # Define diferentes scalers a serem testados
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "MaxAbsScaler": MaxAbsScaler()
    }

    results = {}  # Inicializa um dicionário para armazenar os resultados

    for scaler_name, scaler in scalers.items():  # Itera sobre os diferentes scalers
        print(f"\nUsando o Scaler: {scaler_name}")  # Imprime o nome do scaler atual
        X_train, X_test, y_train, y_test = prepare_data(data, features, target, test_size=0.2, scaler=scaler)  # Prepara os dados usando o scaler atual

        models = {  # Define os modelos de regressão a serem testados
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=101),
            "Random Forest Regressor": RandomForestRegressor(random_state=101),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=101),
            "SVR": SVR(),
            "KNeighbors Regressor": KNeighborsRegressor(),
            "MLP Regressor": MLPRegressor(random_state=101)
        }

        scaler_results = {}  # Inicializa um dicionário para armazenar os resultados dos modelos para o scaler atual

        for model_name, model in models.items():  # Itera sobre os diferentes modelos de regressão
            print(f"\nTreinando: {model_name}...")  # Imprime o nome do modelo atual
            model_results = train_evaluate_model(model, X_train, X_test, y_train, y_test)  # Treina e avalia o modelo atual
            scaler_results[model_name] = model_results  # Armazena os resultados do modelo atual

        results[scaler_name] = scaler_results  # Armazena os resultados dos modelos para o scaler atual

    for scaler_name, scaler_result in results.items():  # Itera sobre os resultados dos diferentes scalers
        print(f"\nResultados com {scaler_name}:")  # Imprime o nome do scaler atual
        for model_name, metrics in scaler_result.items():  # Itera sobre os resultados dos modelos para o scaler atual
            print(f"\n{model_name}:")  # Imprime o nome do modelo atual
            for metric_name, value in metrics.items():  # Itera sobre as métricas de avaliação do modelo atual
                print(f"{metric_name}: {value:.4f}")  # Imprime o nome da métrica e o valor correspondente

        example = X_test[[0]]  # Seleciona um exemplo dos dados de teste para fazer uma previsão
        for model_name, model in models.items():  # Itera sobre os diferentes modelos de regressão
            prediction = model.predict(example)  # Faz uma previsão usando o modelo atual
            actual = y_test.iloc[0]  # Obtém o valor real correspondente ao exemplo selecionado
            print(f"\n{model_name} - Exemplo de Previsão vs Realidade:")  # Imprime o nome do modelo atual e um título para os resultados
            print(f"Previsão: {prediction[0]:.4f}")  # Imprime a previsão do modelo atual
            print(f"Realidade: {actual:.4f}")  # Imprime o valor real correspondente ao exemplo selecionado

if __name__ == "__main__":  # Verifica se o script está sendo executado diretamente
    main()  # Chama a função principal se o script estiver sendo executado diretamente
