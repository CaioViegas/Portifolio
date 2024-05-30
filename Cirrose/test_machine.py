import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def prepare_data(data, features, target, test_size=0.2, scaler=None):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=101)

    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    accuracy_train = accuracy_score(y_train, train_pred)  # Acurácia nos dados de treino
    accuracy_test = accuracy_score(y_test, test_pred)  # Acurácia nos dados de teste

    precision_train = precision_score(y_train, train_pred, average=None)  # Precisão nos dados de treino
    precision_test = precision_score(y_test, test_pred, average=None)  # Precisão nos dados de teste

    recall_train = recall_score(y_train, train_pred, average=None)  # Recall nos dados de treino
    recall_test = recall_score(y_test, test_pred, average=None)  # Recall nos dados de teste

    f1_train = f1_score(y_train, train_pred, average=None)  # F1 score nos dados de treino
    f1_test = f1_score(y_test, test_pred, average=None)  # F1 score nos dados de teste

    return {  
        "Accuracy_train": accuracy_train,
        "Accuracy_test": accuracy_test,
        "Precision_train": precision_train,
        "Precision_test": precision_test,
        "Recall_train": recall_train,
        "Recall_test": recall_test,
        "F1_train": f1_train,
        "F1_test": f1_test,
    }

def main():
    file_path = os.path.join("C:/", "Users", "caioc", "OneDrive", "ProjetosData", "Cirrose", "data", "liver_cirrhosis_transformed.csv")

    # Verifica se o arquivo existe
    if os.path.exists(file_path):
        print(f"O arquivo existe no caminho: {file_path}")

        data = load_data(file_path=file_path)

        features = ['Num_Dias', 'Idade', 'Sexo', 'Ascite', 'Hepatomegalia', 'Aranhas_Vasculares', 'Bilirrubina', 'Colesterol', 'Albumina', 'Cobre_Urina', 'Fos_Alcalina', 'AST', 'Triglicerideos', 'Plaquetas', 'Tempo_Protrombina', 'Status_0', 'Status_1', 'Status_2', 'Medicamento_0', 'Medicamento_1', 'Edema_0', 'Edema_1', 'Edema_2'] # Define as features a serem utilizadas
        target = "Estagio" # Define a coluna alvo da previsão

        scalers = { # Define diferentes scalers a serem testados
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "MaxAbsScaler": MaxAbsScaler()
        }

        results = {}

        for scaler_name, scaler in scalers.items():  # Itera sobre os diferentes scalers
            print(f"\nUsando o Scaler: {scaler_name}")  # Imprime o nome do scaler atual
            X_train, X_test, y_train, y_test = prepare_data(data, features, target, test_size=0.2, scaler=scaler)  # Prepara os dados usando o scaler atual

            models = {  # Define os modelos de regressão a serem testados
                "Gradient Boosting": GradientBoostingClassifier(random_state=101),
                "Random Forest": RandomForestClassifier(random_state=101),
                "Ada Boost": AdaBoostClassifier(random_state=101),
                "Logistic Regression": LogisticRegression(random_state=101),
                "Decision Tree": DecisionTreeClassifier(random_state=101),
                "SVC": SVC(random_state=101),
                "KNeighbors": KNeighborsClassifier(),
                "MLP": MLPClassifier(max_iter=1500, random_state=101)
            }

            scaler_results = {}

            for model_name, model in models.items():
                print(f"\nTreinando: {model_name}...")  # Imprime o nome do modelo atual
                model_results = train_evaluate_model(model, X_train, X_test, y_train, y_test) # Treina e avalia o modelo atual
                scaler_results[model_name] = model_results  # Armazena os resultados do modelo atual

            results[scaler_name] = scaler_results  # Armazena os resultados do scaler atual

        for scaler_name, scaler_result in results.items():  # Itera sobre os resultados dos diferentes scalers
            print(f"\nResultados com {scaler_name}:")  # Imprime o nome do scaler atual
            for model_name, metrics in scaler_result.items():  # Itera sobre os resultados dos modelos para o scaler atual
                print(f"\n{model_name}:")  # Imprime o nome do modelo atual
                for metric_name, value in metrics.items():  # Itera sobre as métricas de avaliação do modelo atual
                    print(f"{metric_name}: {value}")  # Imprime o nome da métrica e o valor correspondente

        example = X_test[[0]]  # Seleciona um exemplo dos dados de teste para fazer uma previsão
        for model_name, model in models.items():  # Itera sobre os diferentes modelos de regressão
            prediction = model.predict(example)  # Faz uma previsão usando o modelo atual
            actual = y_test.iloc[0]  # Obtém o valor real correspondente ao exemplo selecionado
            print(f"\n{model_name} - Exemplo de Previsão vs Realidade:")  # Imprime o nome do modelo atual e um título para os resultados
            print(f"Previsão: {prediction[0]}")  # Imprime a previsão do modelo atual
            print(f"Realidade: {actual:.4f}")  # Imprime o valor real correspondente ao exemplo selecionado

    else:
        print(f"O arquivo não foi encontrado no caminho: {file_path}")

if __name__ == '__main__':
    main()
