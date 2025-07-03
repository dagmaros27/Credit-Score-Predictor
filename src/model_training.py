import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv('../data/final_training_data_with_risk.csv')
    
    # Keep CustomerId separately for future use if needed
    if 'CustomerId' in df.columns:
        customer_ids = df['CustomerId']
        X = df.drop(columns=['is_high_risk', 'CustomerId'])
    else:
        customer_ids = None
        X = df.drop(columns=['is_high_risk'])
    y = df['is_high_risk']
    return X, y, customer_ids

def evaluate_model(y_true, y_pred, y_prob):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }
    return metrics

def train_and_log_model(X_train, X_test, y_train, y_test, model, model_name, params=None):
    with mlflow.start_run(run_name=model_name):
        if params:
            model.set_params(**params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_prob)

        print(f"\n{model_name} Performance:")
        print(classification_report(y_test, y_pred))

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.sklearn.log_model(model, model_name)

        return metrics

if __name__ == "__main__":
    mlflow.set_experiment("Credit Risk Modeling")

    X, y, customer_ids = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    lr = LogisticRegression(max_iter=500)
    lr_params = {'C': 1.0}
    train_and_log_model(X_train, X_test, y_train, y_test, lr, "Logistic_Regression", lr_params)

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_params = {'n_estimators': 100, 'max_depth': 5}
    train_and_log_model(X_train, X_test, y_train, y_test, rf, "Random_Forest", rf_params)

    print("Model training and logging completed.")