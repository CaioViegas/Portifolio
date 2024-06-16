import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from scipy.stats import randint as sp_randint, uniform as sp_uniform

def load_data(file_path, features, target):
    df = pd.read_csv(file_path)
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=101)

def criar_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])
    return pipeline

def ajustar_hiperparametros(pipeline, X_train, y_train):
    parametros = {
        'classifier__n_estimators': sp_randint(100, 300),
        'classifier__learning_rate': sp_uniform(0.01, 0.2),
        'classifier__max_depth': sp_randint(3, 6),
        'classifier__min_child_weight': sp_randint(1, 7),
        'classifier__gamma': sp_uniform(0, 0.3),
        'classifier__subsample': sp_uniform(0.6, 0.4),
        'classifier__colsample_bytree': sp_uniform(0.6, 0.4)
    }   

    random_search = RandomizedSearchCV(pipeline, parametros, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=101)
    random_search.fit(X_train, y_train)
    return random_search

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
    file_path = 'data/obesidade_encoded.csv'
    features = ['Idade', 'Genero', 'Altura', 'Peso', 'Freq_Alcool',
                'Freq_Densidade_Calorica', 'Freq_Vegetais', 'Freq_Refeicoes',
                'Monitoramento_Calorias_Diarias', 'Fumante', 'Agua_Diaria',
                'Historico_Familiar', 'Atividade_Fisica', 'Freq_Tecnologia',
                'Alimento_Entre_Refeicoes_Frequentemente',
                'Alimento_Entre_Refeicoes_Negativo',
                'Alimento_Entre_Refeicoes_Ocasionalmente',
                'Alimento_Entre_Refeicoes_Sempre', 'Meio_Transporte_Automovel',
                'Meio_Transporte_Bicicleta', 'Meio_Transporte_Caminhada',
                'Meio_Transporte_Moto', 'Meio_Transporte_Transporte_Publico']
    target = 'Nivel_Obesidade'

    X_train, X_test, y_train, y_test = load_data(file_path, features, target)
    
    pipeline = criar_pipeline()
    modelo = ajustar_hiperparametros(pipeline, X_train, y_train)
    metricas = avaliacao(modelo, X_test, y_test)

    scores = avaliar_cross_val(modelo, X_train, y_train, cv=5)
    print(f'Scores da validação cruzada: {scores}')
    print(f'Média dos scores: {scores.mean()}')

    print(metricas)
    salvar_modelo(modelo, 'modelos/modelo_obesidade.joblib')

if __name__ == '__main__':
    main()
