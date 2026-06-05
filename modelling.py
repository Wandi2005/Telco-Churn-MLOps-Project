import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Telco_Customer_Churn_Basic")

# Enable autolog
mlflow.sklearn.autolog(log_models=True)

# Load data
train_df = pd.read_csv('telco_churn_train_preprocessed.csv')
test_df = pd.read_csv('telco_churn_test_preprocessed.csv')

X_train = train_df.drop('Churn', axis=1)
y_train = train_df['Churn']
X_test = test_df.drop('Churn', axis=1)
y_test = test_df['Churn']

# Define models
models = {
    'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_f1 = 0
best_name = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        mlflow.log_param("model_type", name)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        
        mlflow.sklearn.log_model(model, f"{name}_model")
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model = model
            best_name = name
        
        print(f"{name}: F1={metrics['f1_score']:.4f}")

print(f"\nBest: {best_name} (F1={best_f1:.4f})")
joblib.dump(best_model, 'best_model.pkl')
