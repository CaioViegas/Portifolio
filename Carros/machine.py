import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
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
    
    X = df[['Motor', 'Empresa', 'Carro/SUV', 'EsquemaCores', 'Modelo', 'Ano', 'Quilometros', 'CilindrosMotor', 'Titulo', 'TipoCombustivel', 'Portas', 'Transmissao', 
            'Bancos']]
    y = df['Preco']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    scaler = MaxAbsScaler()

    pipeline = Pipeline([
        ('scaler', scaler)
    ])

    X_train_scaled = pipeline.fit_transform(X_train)
    X_test_scaled = pipeline.fit_transform(X_test)

    modelos = {
        'LinearRegression': LinearRegression(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'XGBRegressor': XGBRegressor()
    }

    for modelo_name, modelo in modelos.items():
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)

        print(f"Modelo: {modelo_name}")
        print(f"R2: {r2_score(y_pred, y_test)}")
        print(f"MSE: {mean_squared_error(y_pred, y_test)}")
        print(f"MAE: {mean_absolute_error(y_pred, y_test)}\n")

    param_grid_xgb = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'colsample_bytree': [0.7, 0.8],
        'min_child_weight': [1, 2, 3]
    }

    param_grid_rf = {
        'n_estimators': [100, 200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4,5,6,7,8],
        'criterion': ['squared_error', 'absolute_error']
    }

    grid_search_xgb = GridSearchCV(estimator=XGBRegressor(), param_grid=param_grid_xgb, cv=5)

    grid_search_xgb.fit(X_train_scaled, y_train)

    melhores_parametros_xgb = grid_search_xgb.best_params_

    melhor_modelo_xgb = XGBRegressor(**melhores_parametros_xgb)

    melhor_modelo_xgb.fit(X_train_scaled, y_train)

    y_pred_xgb = melhor_modelo_xgb.predict(X_test_scaled)

    print(f"R2: {r2_score(y_pred_xgb, y_test)}")
    print(f"MSE: {mean_squared_error(y_pred_xgb, y_test)}")
    print(f"MAE: {mean_absolute_error(y_pred_xgb, y_test)}\n")

    grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid_rf, cv=5)

    grid_search_rf.fit(X_train_scaled, y_train)

    melhores_parametros_rf = grid_search_rf.best_params_

    melhor_modelo_rf = RandomForestRegressor(**melhores_parametros_rf)

    melhor_modelo_rf.fit(X_train_scaled, y_train)

    y_pred_rf = melhor_modelo_rf.predict(X_test_scaled)

    print(f"R2: {r2_score(y_pred_rf, y_test)}")
    print(f"MSE: {mean_squared_error(y_pred_rf, y_test)}")
    print(f"MAE: {mean_absolute_error(y_pred_rf, y_test)}\n")

if __name__ == '__main__':
    file_path = "data/carros_codificado.csv"
    main(file_path)