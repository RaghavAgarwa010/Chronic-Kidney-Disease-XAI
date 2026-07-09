import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

class CKDDataPipeline:
    def __init__(self, filepath):
        self.filepath = filepath
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        
    def load_and_clean(self):
        data, meta = arff.loadarff(self.filepath)
        df = pd.DataFrame(data)
        
        for col in df.select_dtypes(include=[object]).columns:
            df[col] = df[col].str.decode('utf-8')
            
        df = df.replace(to_replace={'\t': '', '?': np.nan, ' ': np.nan})
        df['classification'] = df['classification'].replace({'ckd\t': 'ckd'})
        
        num_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df

    def preprocess(self, df):
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
                
        cat_cols = df.select_dtypes(include=['object']).columns.drop('classification')
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            
        df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
        
        X = df.drop(columns=['classification'])
        y = df['classification']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X.columns)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=X.columns)
        
        X_train_res, y_train_res = self.smote.fit_resample(X_train_scaled, y_train)
        
        return X_train_res, X_test_scaled, y_train_res, y_test
