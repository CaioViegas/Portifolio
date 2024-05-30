import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from joblib import dump

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def feature_selection(X, y):
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=101))
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    return list(selected_features)

def prepare_data(data, features, target, test_size=0.2):
    X = data[features]
    y = data[target]
    selected_features = feature_selection(X, y)
    X = X[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=101)
    return X_train, X_test, y_train, y_test

def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    results = {
        "Accuracy_train": accuracy_score(y_train, train_pred),
        "Accuracy_test": accuracy_score(y_test, test_pred),
        "Precision_train": precision_score(y_train, train_pred, average=None),
        "Precision_test": precision_score(y_test, test_pred, average=None),
        "Recall_train": recall_score(y_train, train_pred, average=None),
        "Recall_test": recall_score(y_test, test_pred, average=None),
        "F1_train": f1_score(y_train, train_pred, average=None),
        "F1_test": f1_score(y_test, test_pred, average=None)
    }

    print("\nClassification Report (Test):\n", classification_report(y_test, test_pred))
    return results

def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'clf__n_estimators': [100, 200, 300, 400, 500],
        'clf__max_depth': [10, 20, 30, 40, 50],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt', 'log2']
    }

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('clf', RandomForestClassifier(random_state=101))
    ])

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

def cross_validate(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)
    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision_macro')
    recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall_macro')
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')

    results = {
        "Cross-Validation Accuracy": accuracy_scores.mean(),
        "Cross-Validation Precision": precision_scores.mean(),
        "Cross-Validation Recall": recall_scores.mean(),
        "Cross-Validation F1": f1_scores.mean()
    }
    return results

def main():
    file_path = os.path.join("C:/", "Users", "caioc", "OneDrive", "ProjetosData", "Cirrose", "data", "liver_cirrhosis_transformed.csv")

    if os.path.exists(file_path):
        print(f"O arquivo existe em: {file_path}")

        data = load_data(file_path=file_path)

        features = ['Num_Dias', 'Idade', 'Sexo', 'Ascite', 'Hepatomegalia', 'Aranhas_Vasculares', 'Bilirrubina', 'Colesterol', 'Albumina', 'Cobre_Urina', 'Fos_Alcalina', 'AST', 'Triglicerideos', 'Plaquetas', 'Tempo_Protrombina', 'Status_0', 'Status_1', 'Status_2', 'Medicamento_0', 'Medicamento_1', 'Edema_0', 'Edema_1', 'Edema_2']
        target = "Estagio"

        X_train, X_test, y_train, y_test = prepare_data(data=data, features=features, target=target, test_size=0.2)

        best_rf_model = hyperparameter_tuning(X_train, y_train)

        results = train_evaluate_model(best_rf_model, X_train, X_test, y_train, y_test)

        for metric_name, value in results.items():
            print(f"{metric_name}: {value}")

        cv_results = cross_validate(best_rf_model, X_train, y_train)
        for metric_name, value in cv_results.items():
            print(f"{metric_name}: {value}")

        model_path = 'modelo_cirrose.joblib'
        dump(best_rf_model, model_path)
        print(f"Modelo salvo em: {model_path}")

    else:
        print(f"O arquivo não foi encontrado no caminho: {file_path}")

if __name__ == "__main__":
    main()