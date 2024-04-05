import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

def main(file_path):
    if not os.path.isfile(file_path):
        print(f"O arquivo {file_path} não existe")
        return
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Erro ao ler o arquivo {file_path}: {e}")
        return

    df = pd.read_csv(file_path)

    X = df.drop('DoencaCardiaca', axis=1)
    y = df['DoencaCardiaca']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    scaler = MinMaxScaler()

    pipeline = Pipeline([
        ('scaler', scaler), 
    ])

    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    modelos = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'SVC': SVC(),
        'GaussianNB': GaussianNB(),
        'MLPClassifier': MLPClassifier(max_iter=4500),
        'GradientBoostingClassifier': GradientBoostingClassifier()
    }

    for modelo_name, modelo in modelos.items():
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"Modelo: {modelo_name}")
        print(f"Acurácia: {acc:.2f}")
        print(f"Precisão: {prec:.2f}")
        print(f"Revocação: {rec:.2f}")
        print(f"F1-Score: {f1:.2f}")
        print(f"Matriz de Confusão:\n{cm}")
        print()

    param_grid_gradient = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 5, 8],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.5, 0.7, 1.0],
        'loss': ['deviance', 'exponential'],
        'criterion': ['friedman_mse', 'squared_error'],
    } 
        
    grid_search = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_grid_gradient, cv=5)

    grid_search.fit(X_train_scaled, y_train)

    melhores_parametros = grid_search.best_params_

    print(f"Melhores paramêtros: {melhores_parametros}")

    melhor_modelo = GradientBoostingClassifier(**melhores_parametros)

    melhor_modelo.fit(X_train_scaled, y_train)

    y_pred = melhor_modelo.predict(X_test_scaled)

    print("Acurácia: ", accuracy_score(y_test, y_pred))
    print("Precisão: ", precision_score(y_test, y_pred))
    print("Revocação: ", recall_score(y_test, y_pred))
    print("F1-Score: ", f1_score(y_test, y_pred))

if __name__ == '__main__':
    file_path = "data/coracao_codificado.csv"
    main(file_path)