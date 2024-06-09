import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(file_path, features: list, target):
    df = pd.read_csv(file_path)
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=101)

def criar_pipeline():
    pipeline = Pipeline([
        ('scaler', MaxAbsScaler()),
        ('classifier', GradientBoostingClassifier())
    ])

    return pipeline

def ajustar_hiperparametros(pipeline, X_train, y_train):
    parametros = {
        'classifier__learning_rate': [0.1, 0.01, 0.001],
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__min_samples_split': [2, 3, 4],
        'classifier__min_samples_leaf': [1, 2, 3],
    }

    grid_search = GridSearchCV(pipeline, parametros, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

def avaliacao(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    return metrics

def salvar_modelo(modelo, file_path):
    joblib.dump(modelo, file_path)

def avaliar_cross_val(modelo, X, y, cv=5):
    scores = cross_val_score(modelo, X, y, cv=cv)
    return scores

def main():
    X_train, X_test, y_train, y_test = load_data('C:/Users/caioc/Documents/ProjetosData/Coracao/data/heart_encoded.csv', 
        ['Idade', 'Sexo', 'PressaoArterialRepouso', 'Colesterol', 'GlicoseJejum',
       'FreqCardiacaMaxima', 'AnginaExercicio', 'DepressaoSTExercicioRepouso',
        'TipoDorPeito_AAT', 'TipoDorPeito_ANA',
       'TipoDorPeito_AT', 'TipoDorPeito_Assintomatico',
       'EletrocardiogramaRepouso_ASST', 'EletrocardiogramaRepouso_HVE',
       'EletrocardiogramaRepouso_Normal', 'InclinacaoST_Ascendente',
       'InclinacaoST_Descendente', 'InclinacaoST_Plano'], 'DoencaCardiaca')
    
    pipeline = criar_pipeline()
    modelo = ajustar_hiperparametros(pipeline, X_train, y_train)
    metricas = avaliacao(modelo, X_test, y_test)

    scores = avaliar_cross_val(modelo, X_train, y_train, cv=5)
    print(f'Scores da validação cruzada: {scores}')
    print(f'Média dos scores: {scores.mean()}')

    print(metricas)
    salvar_modelo(modelo, 'C:/Users/caioc/Documents/ProjetosData/Coracao/modelos/modelo.joblib')

if __name__ == '__main__':
    main()