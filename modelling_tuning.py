import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score)
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Telco_Churn_Tuning_Skilled")

# Load data
train_df = pd.read_csv('telco_churn_train_preprocessed.csv')
test_df = pd.read_csv('telco_churn_test_preprocessed.csv')

X_train = train_df.drop('Churn', axis=1)
y_train = train_df['Churn']
X_test = test_df.drop('Churn', axis=1)
y_test = test_df['Churn']

param_distributions = {
    'Random_Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient_Boosting': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
}

best_models = {}
best_params_all = {}

for name, params in param_distributions.items():
    print(f"\nTuning {name}...")
    base = RandomForestClassifier(random_state=42) if name == 'Random_Forest' else GradientBoostingClassifier(random_state=42)
    
    rs = RandomizedSearchCV(base, params, n_iter=15, cv=3, scoring='f1', n_jobs=-1, random_state=42)
    
    with mlflow.start_run(run_name=f"{name}_Tuning"):
        rs.fit(X_train, y_train)
        
        mlflow.log_param("model_type", name)
        mlflow.log_param("tuning_method", "RandomizedSearchCV")
        mlflow.log_params(rs.best_params_)
        mlflow.log_metric("best_cv_f1", rs.best_score_)
        
        y_pred = rs.best_estimator_.predict(X_test)
        y_proba = rs.best_estimator_.predict_proba(X_test)[:, 1]
        
        mlflow.log_metric("test_f1", f1_score(y_test, y_pred))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_proba))
        
        mlflow.sklearn.log_model(rs.best_estimator_, f"{name}_tuned")
        
        best_models[name] = rs.best_estimator_
        best_params_all[name] = rs.best_params_
        
        print(f"Best F1: {f1_score(y_test, y_pred):.4f}")

# Save
with open('best_params.json', 'w') as f:
    json.dump(best_params_all, f, indent=2)

print("\nTuning selesai!")
