"""Preprocessing pipeline for fraud detection datasets."""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

logger = logging.getLogger(__name__)


class FraudPreprocessor:
    """Preprocessing pipeline: cleaning, scaling, and feature engineering."""

    def __init__(self, scaler_type: str = "standard"):
        """Initialize preprocessor.

        Args:
            scaler_type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler.
        """
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")

        self.scaler_type = scaler_type
        self._fitted = False
        self._feature_names = None

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values by median imputation for numeric columns.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing values handled.
        """
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            logger.info(f"Found {null_count} missing values. Imputing with median.")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        else:
            logger.info("No missing values found.")
        return df

    def engineer_features_kaggle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for the Kaggle Credit Card dataset.

        Args:
            df: DataFrame with Time, Amount, V1-V28 columns.

        Returns:
            DataFrame with additional engineered features.
        """
        df = df.copy()

        # Hour of day (Time is seconds from first transaction)
        df["Hour"] = (df["Time"] % 86400) // 3600

        # Log-transform the skewed Amount
        df["Amount_Log"] = np.log1p(df["Amount"])

        # PCA feature statistics
        v_cols = [f"V{i}" for i in range(1, 29)]
        df["V_Mean"] = df[v_cols].mean(axis=1)
        df["V_Std"] = df[v_cols].std(axis=1)

        logger.info("Engineered 4 features for Kaggle CC: Hour, Amount_Log, V_Mean, V_Std")
        return df

    def engineer_features_paysim(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for the PaySim dataset.

        Args:
            df: PaySim DataFrame.

        Returns:
            DataFrame with additional engineered features.
        """
        df = df.copy()

        # Amount relative to origin balance
        df["Amount_Ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

        # Balance changes
        df["Balance_Change_Orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
        df["Balance_Change_Dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

        # Encode transaction type
        type_map = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}
        df["Type_Encoded"] = df["type"].map(type_map).fillna(-1).astype(int)

        # Log amount
        df["Amount_Log"] = np.log1p(df["amount"])

        logger.info(
            "Engineered 5 features for PaySim: Amount_Ratio, Balance_Change_Orig, "
            "Balance_Change_Dest, Type_Encoded, Amount_Log"
        )
        return df

    def fit_transform(self, X: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """Fit scaler on training data and transform.

        Args:
            X: Training feature DataFrame.
            columns: Columns to scale. If None, scales all numeric columns.

        Returns:
            Scaled DataFrame.
        """
        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()

        X_scaled = X.copy()
        X_scaled[columns] = self.scaler.fit_transform(X[columns])
        self._fitted = True
        self._feature_names = X_scaled.columns.tolist()

        logger.info(f"Fitted and transformed {len(columns)} columns with {self.scaler_type} scaler.")
        return X_scaled

    def transform(self, X: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """Transform data using fitted scaler.

        Args:
            X: Feature DataFrame to transform.
            columns: Columns to scale. Must match fit_transform columns.

        Returns:
            Scaled DataFrame.
        """
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform() first.")

        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()

        X_scaled = X.copy()
        X_scaled[columns] = self.scaler.transform(X[columns])
        return X_scaled

    def get_feature_names(self) -> list:
        """Return list of feature names after transformation."""
        if self._feature_names is None:
            raise RuntimeError("No features available. Call fit_transform() first.")
        return self._feature_names

    def save_scaler(self, path: str):
        """Save fitted scaler to disk.

        Args:
            path: File path for the scaler (.pkl).
        """
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform() first.")
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler saved to {path}")

    def load_scaler(self, path: str):
        """Load a previously fitted scaler.

        Args:
            path: Path to saved scaler file.
        """
        self.scaler = joblib.load(path)
        self._fitted = True
        logger.info(f"Scaler loaded from {path}")
