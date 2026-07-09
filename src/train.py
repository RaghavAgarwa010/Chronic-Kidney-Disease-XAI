import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

def train_and_evaluate(X_train, y_train, X_test, y_test):
    models = {
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
    }
    
    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'model': model,
            'cv_f1_mean': np.mean(cv_scores),
            'test_report': classification_report(y_test, preds, output_dict=True),
            'test_auc': roc_auc_score(y_test, probs)
        }
    return results
