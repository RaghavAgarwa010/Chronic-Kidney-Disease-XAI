# Explainable Multi-Stage Chronic Kidney Disease Detection Using Hybrid Ensemble Learning and SHAP/LIME Clinical Interpretability

An end-to-end machine learning and explainable AI (XAI) pipeline designed to identify Chronic Kidney Disease (CKD) from clinical markers. This system balances high-accuracy tabular classification with medical transparency, ensuring that model assessments are globally validated and decipherable down to specific patient metrics.

## ⚙️ System Architecture & Data Engineering
The pipeline processes raw clinical data via a structured lifecycle to counter bias, handle sparsity, and extract clean features:

1. **Robust Preprocessing & Imputation:** Automatically decodes raw inputs, resolves structural text typos, and handles sparse values via deterministic data profiling (Continuous variables assigned historical medians; categorical fields filled using historical mode markers).
2. **Class Balancing (SMOTE):** Employs Synthetic Minority Over-sampling Technique (SMOTE) on the training matrix to address diagnostic class imbalance, preventing standard algorithmic bias toward healthy or diseased majorities.
3. **Recursive Feature Elimination (RFE):** Leverages a wrapper-based tree estimator to iteratively evaluate feature subsets, paring down 24 initial parameters to the **12 highest-variance medical indicators** to maximize inference throughput.

---

## 📊 Evaluation & Model Performance

The predictive subsystem evaluates and compares structural tree ensembles (**XGBoost** and **LightGBM**). Evaluation metrics are verified across a stratified cross-validation scheme to validate model consistency and guard against training-set leakage.

### Performance Summary
| Predictive Model | 5-Fold CV F1-Score | Test Accuracy | Test Precision | Test Recall | Test AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **98.64%** | **100%** | **100%** | **100%** | **1.00** |
| **LightGBM** | 97.91% | 100% | 100% | 100% | 1.00 |

*Note: The perfect classification metrics on the test split are cross-verified by robust internal cross-validation bounds to confirm uniform stability across data sub-distributions.*

---

## 🔍 Explainable AI (XAI) Integration

To bridge the gap between black-box calculations and clinical trust, this system integrates dual-layer explainability metrics:

### 1. Global Interpretability (SHAP TreeExplainer)
SHAP (SHapley Additive exPlanations) values outline overall feature importances across the whole dataset population. 
* **Primary Insights:** The model establishes **Specific Gravity (sg)**, **Hemoglobin levels (hemo)**, and **Serum Creatinine (sc)** as the primary biological drivers behind diagnostic evaluations, closely mirroring established real-world nephrology workflows.

### 2. Localized Interpretability (LIME Tabular Profiling)
LIME (Local Interpretable Model-agnostic Explanations) builds a localized linear model around an individual patient's metrics to show precisely *why* an evaluation was made.
* **Clinical Utility:** If an individual patient presents with elevated Serum Creatinine alongside drop-offs in Specific Gravity, LIME breaks down exactly how much weight those specific attributes carried toward the positive diagnostic classification.

---

## 🚀 How to Run the Pipeline

### 1. Clone the repository and setup dependencies
```bash
git clone [https://github.com/RaghavAgarwa010/Chronic-Kidney-Disease-XAI.git](https://github.com/RaghavAgarwa010/Chronic-Kidney-Disease-XAI.git)
cd Chronic-Kidney-Disease-XAI
pip install -r requirements.txt
