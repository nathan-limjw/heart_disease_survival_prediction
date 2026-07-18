# Standard library imports
import logging
from typing import Any, Dict

# Related third-party imports
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)


class DataPreparation:
    """
    A class used to clean and preprocess coronary artery disease survival data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numeric and ordinal features.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame. Written defensively so it degrades
        gracefully if a given data issue (e.g. negative ages, inconsistent
        capitalization, shorthand category codes) isn't present in a given
        batch of data, rather than assuming it always is.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Starting data cleaning.")
        df = df.copy()

        df.drop_duplicates(inplace=True)

        # Normalize binary/categorical string values that may vary in casing
        # or spelling across data pulls.
        if "Survive" in df.columns:
            df["Survive"] = self._standardize_binary(df["Survive"])

        if "Smoke" in df.columns:
            df["Smoke"] = df["Smoke"].astype(str).str.strip().str.capitalize()

        if "Ejection Fraction" in df.columns:
            df["Ejection Fraction"] = self._expand_shorthand(
                df["Ejection Fraction"],
                mapping={"L": "Low", "N": "Normal", "H": "High"},
            )

        # Age should be non-negative; flip sign errors rather than dropping rows.
        if "Age" in df.columns:
            n_negative = (df["Age"] < 0).sum()
            if n_negative:
                logging.info(f"Correcting {n_negative} negative Age values.")
            df["Age"] = df["Age"].abs()

        # Drop columns identified as uninformative/synthetic during EDA,
        # but don't error out if a column is already missing (e.g. if this
        # pipeline runs on a version of the data without it).
        drop_cols = [c for c in self.config.get("drop_features", []) if c in df.columns]
        df.drop(columns=drop_cols, inplace=True)

        logging.info("Data cleaning completed.")
        return df

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for numeric (with per-column skew
        correction) and ordinal features, based on config.

        Returns:
        --------
        sklearn.compose.ColumnTransformer
        """
        numeric_cfg = self.config["numeric_features"]
        transformers = []

        for col in numeric_cfg.get("boxcox_features", []):
            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("boxcox", PowerTransformer(method="box-cox")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append((f"boxcox_{col}", pipe, [col]))

        for col in numeric_cfg.get("sqrt_features", []):
            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "sqrt",
                        FunctionTransformer(np.sqrt, feature_names_out="one-to-one"),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append((f"sqrt_{col}", pipe, [col]))

        plain_cols = numeric_cfg.get("plain_features", [])
        if plain_cols:
            plain_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("plain_numeric", plain_pipe, plain_cols))

        for col, categories in self.config.get("ordinal_features", {}).items():
            ord_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ordinal",
                        OrdinalEncoder(
                            categories=[categories],
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )
            transformers.append((f"ordinal_{col}", ord_pipe, [col]))

        return ColumnTransformer(transformers=transformers, n_jobs=-1)

    @staticmethod
    def _standardize_binary(series: pd.Series) -> pd.Series:
        """
        Normalizes a binary target column that may contain a mix of string
        ('Yes'/'No') and numeric ('0'/'1') representations, case-insensitively.
        """
        mapping = {"yes": 1, "no": 0, "1": 1, "0": 0}
        cleaned = series.astype(str).str.strip().str.lower().map(mapping)
        if cleaned.isna().any():
            n_unmapped = cleaned.isna().sum()
            logging.warning(f"{n_unmapped} values in target column did not map to 0/1.")
        return cleaned.astype("Int64")

    @staticmethod
    def _expand_shorthand(series: pd.Series, mapping: Dict[str, str]) -> pd.Series:
        """
        Replaces shorthand category codes (e.g. 'L' -> 'Low') while leaving
        already-full values untouched. Applied via a dict.get fallback so
        unrecognized values pass through rather than becoming NaN.
        """
        return series.astype(str).apply(lambda v: mapping.get(v.strip(), v.strip()))
