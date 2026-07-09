import shap
import numpy as np
from lime import lime_tabular

def generate_shap_explanations(model, X_train, X_test):
    """Generates tree-based SHAP explainers and calculates shap values."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    return explainer, shap_values

def generate_lime_explanation(X_train, X_test, model, instance_index=0):
    """Generates a localized LIME explanation for a single clinical observation."""
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=['Not CKD', 'CKD'],
        mode='classification'
    )
    
    exp = explainer.explain_instance(
        data_row=X_test.iloc[instance_index],
        predict_fn=model.predict_proba
    )
    return exp
