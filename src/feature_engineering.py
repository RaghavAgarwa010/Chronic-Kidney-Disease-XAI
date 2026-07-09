from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

def select_features(X_train, y_train, n_features=12):
    """Applies Recursive Feature Elimination (RFE) to identify top predictors."""
    estimator = XGBClassifier(random_state=42, eval_metric='logloss')
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.support_].tolist()
    return selected_features
