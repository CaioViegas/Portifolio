import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def main(file_path):
    if not os.path.isfile(file_path):
        print(f"O arquivo {file_path} não existe.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Erro ao ler o arquivo {file_path}: {e}")
        return

    le = LabelEncoder()
    df['experiencia_autor'] = le.fit_transform(df['experiencia_autor'])
    df['editora'] = le.fit_transform(df['editora'])
    df['genero'] = le.fit_transform(df['genero'])
    df['linguagem'] = le.fit_transform(df['linguagem'])

    X = df.drop(columns=['unidades_vendidas', 'autor', 'nome_livro', 'ano_publicacao'], axis=1)
    y = df['unidades_vendidas']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  
    ])

    modelos = [
        DecisionTreeRegressor(),
        LinearRegression(),
        RandomForestRegressor(),
        SVR(),
        KNeighborsRegressor(),
        XGBRegressor(),
        CatBoostRegressor(),
    ]

    parameters = [
        {},  # Para DecisionTreeRegressor
        {},  # Para LinearRegression
        {
            'model__n_estimators': [100, 200, 300],
            'model__max_features': ['auto', 'sqrt', 'log2'],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        },  # Para RandomForestRegressor
        {
            'model__kernel': ['linear', 'poly', 'rbf'],
            'model__C': [0.1, 1, 10]
        },  # Para SVR
        {
            'model__n_neighbors': [3, 5, 7, 10],
            'model__weights': ['uniform', 'distance']
        },  # Para KNeighborsRegressor
        {
            'model__n_estimators': [100, 200, 300],
            'model__learning_rate': [0.01, 0.1, 0.3]
        },  # Para XGBRegressor
        {
            'model__iterations': [100, 200, 300],
            'model__learning_rate': [0.01, 0.1, 0.3]
        },  # Para CatBoostRegressor
    ]

    scaler = MinMaxScaler()

    best_model = None
    best_params = None
    best_score = float('-inf')
    for model, params in zip(modelos, parameters):
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        grid_search = GridSearchCV(pipeline, params, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X_train, y_train)
        if grid_search.best_score_ > best_score:
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

    if best_model:
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("Melhor Modelo:", best_model.named_steps['model'].__class__.__name__)
        print("Melhores Parâmetros:", best_params)
        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R^2 Score:", r2)
    else:
        print("Nenhum modelo encontrado.")

    modelo_xgb = XGBRegressor(learning_rate=0.3, n_estimators=100)

    modelo_xgb.fit(X_train, y_train)

    y_pred = modelo_xgb.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nValores do XGB:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)

if __name__ == "__main__":
    file_path = "data/arquivo_livro_limpo.csv"
    main(file_path)