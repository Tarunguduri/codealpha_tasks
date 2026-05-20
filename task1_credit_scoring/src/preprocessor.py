"""
Data preprocessing and feature engineering pipeline.

Handles missing values, stratified train/test split, and feature scaling.
Scaling is fit ONLY on the training set to prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "default",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """Clean, split, and scale the credit scoring dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from data_generator.
    target_col : str
        Name of the binary target column.
    test_size : float
        Fraction reserved for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Keys: X_train, X_test, y_train, y_test, feature_names, scaler,
              X_train_raw, X_test_raw (unscaled copies for tree models).
    """
    df = df.copy()

    # --- handle missing values ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # --- separate features / target ---
    feature_names: List[str] = [c for c in df.columns if c != target_col]
    X = df[feature_names].values
    y = df[target_col].values

    # --- stratified split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # --- scale (fit on train only to prevent data leakage) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  Train set : {X_train_scaled.shape[0]} samples")
    print(f"  Test  set : {X_test_scaled.shape[0]} samples")
    print(f"  Features  : {len(feature_names)}")
    print(f"  Train default rate: {y_train.mean():.2%}")
    print(f"  Test  default rate: {y_test.mean():.2%}")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
    }


if __name__ == "__main__":
    from data_generator import generate_credit_data

    df = generate_credit_data()
    result = preprocess_data(df)
    print("Feature names:", result["feature_names"])
