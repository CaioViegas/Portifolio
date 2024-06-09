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
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    accuracy_train = accuracy_score(y_train, train_pred)
    accuracy_test = accuracy_score(y_test, test_pred)

    precision_train = precision_score(y_train, train_pred, average=None)
    precision_test = precision_score(y_test, test_pred, average=None)

    recall_train = recall_score(y_train, train_pred, average=None)
    recall_test = recall_score(y_test, test_pred, average=None)

    f1_train = f1_score(y_train, train_pred, average=None)
    f1_test = f1_score(y_test, test_pred, average=None)

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

def prepare_data(data, features, target, test_size=0.2, scaler=None, k_best=10):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=101)

    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    selector = SelectKBest(score_func=f_classif, k=k_best)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    sm = SMOTE(random_state=101)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test

def main():
    file_path = os.path.join("C:/", "Users", "caioc", "Documents", "ProjetosData", "Coracao", "data", "heart_encoded.csv") 
    if os.path.exists(file_path):
        print(f"O arquivo existe no caminho: {file_path}")

        data = load_data(file_path=file_path)

        features = ['Idade', 'Sexo', 'PressaoArterialRepouso', 'Colesterol', 'GlicoseJejum',
       'FreqCardiacaMaxima', 'AnginaExercicio', 'DepressaoSTExercicioRepouso',
        'TipoDorPeito_AAT', 'TipoDorPeito_ANA',
       'TipoDorPeito_AT', 'TipoDorPeito_Assintomatico',
       'EletrocardiogramaRepouso_ASST', 'EletrocardiogramaRepouso_HVE',
       'EletrocardiogramaRepouso_Normal', 'InclinacaoST_Ascendente',
       'InclinacaoST_Descendente', 'InclinacaoST_Plano']
        target = "DoencaCardiaca"

        scalers = {
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "MaxAbsScaler": MaxAbsScaler()
        }

        results = {}

        for scaler_name, scaler in scalers.items():
            print(f"\nUsando o Scaler: {scaler_name}")
            X_train, X_test, y_train, y_test = prepare_data(data, features, target, test_size=0.2, scaler=scaler, k_best=10)

            models = {
                "Gradient Boosting": GradientBoostingClassifier(random_state=101),
                "Random Forest": RandomForestClassifier(random_state=101),
                "Ada Boost": AdaBoostClassifier(random_state=101),
                "Logistic Regression": LogisticRegression(random_state=101),
                "Decision Tree": DecisionTreeClassifier(random_state=101),
                "SVC": SVC(random_state=101),
                "KNeighbors": KNeighborsClassifier(),
                "MLP": MLPClassifier(max_iter=1500, random_state=101),
                "XGB": XGBClassifier()
            }

            scaler_results = {}

            for model_name, model in models.items():
                print(f"\nTreinando: {model_name}...")
                model_results = train_evaluate_model(model, X_train, X_test, y_train, y_test)
                scaler_results[model_name] = model_results

            results[scaler_name] = scaler_results

        for scaler_name, scaler_result in results.items():
            print(f"\nResultados com {scaler_name}:")
            for model_name, metrics in scaler_result.items():
                print(f"\n{model_name}:")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value}")

        example = X_test[[0]]
        for model_name, model in models.items():
            prediction = model.predict(example)
            actual = y_test.iloc[0]
            print(f"\n{model_name} - Exemplo de Previsão vs Realidade:")
            print(f"Previsão: {prediction[0]}")
            print(f"Realidade: {actual:.4f}")

    else:
        print(f"O arquivo não foi encontrado no caminho: {file_path}")

if __name__ == '__main__':
    main()