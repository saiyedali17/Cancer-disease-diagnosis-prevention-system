import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE


def get_trained_pipeline():
    # 1. Load dataset — use absolute path relative to this file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(BASE_DIR, 'Dataset.csv.xlsx')
    df = pd.read_excel(dataset_path)

    # 2. Encoding
    df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

    # Fix binary columns
    binary_cols = df.columns[2:-1]
    for col in binary_cols:
        df[col] = df[col].map({1: 0, 2: 1})

    # 3. Split features
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']

    # 4. Apply SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    # 5. Add slight noise (data expansion)
    noise = np.random.normal(0, 0.02, X_res.shape)
    X_res = X_res + noise

    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    # 7. Pipeline with RandomForest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ))
    ])

    # 8. Train model
    pipeline.fit(X_train, y_train)

    return pipeline
