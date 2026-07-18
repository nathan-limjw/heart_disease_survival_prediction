# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


class ModelTraining:
    """
    A class used to train and evaluate classification models for predicting
    patient survival, with ID-grouped, stratified splitting to prevent
    patient-level data leakage across repeated visits.

    Attributes:
    -----------
    config : Dict[str, Any]
    preprocessor : sklearn.compose.ColumnTransformer
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        self.config = config
        self.preprocessor = preprocessor

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Splits data into train/val/test sets, grouped by patient ID and
        stratified on the target, to prevent the same patient's records
        appearing across multiple sets.

        Returns:
        --------
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logging.info("Starting data splitting.")
        target_col = self.config["target_column"]
        id_col = self.config["id_column"]
        random_state = self.config.get("random_state", 42)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        sgkf1 = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=random_state
        )
        train_val_idx, test_idx = next(sgkf1.split(X, y, groups=X[id_col]))

        X_train_val, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
        y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]

        sgkf2 = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=random_state
        )
        train_idx, val_idx = next(
            sgkf2.split(X_train_val, y_train_val, groups=X_train_val[id_col])
        )

        X_train = X_train_val.iloc[train_idx].reset_index(drop=True)
        X_val = X_train_val.iloc[val_idx].reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train_val.iloc[train_idx].reset_index(drop=True)
        y_val = y_train_val.iloc[val_idx].reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        logging.info(
            f"Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Trains and evaluates baseline classification models at default
        hyperparameters.
        """
        logging.info("Training and evaluating baseline models.")
        random_state = self.config.get("random_state", 42)
        models = {
            "logistic_regression": LogisticRegression(
                random_state=random_state, max_iter=1000
            ),
            "random_forest": RandomForestClassifier(random_state=random_state),
            "xgboost": XGBClassifier(random_state=random_state, eval_metric="logloss"),
        }
        pipelines, metrics = {}, {}

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[("preprocessor", self.preprocessor), ("classifier", model)]
            )
            drop_cols = [self.config["id_column"]]
            pipeline.fit(X_train.drop(columns=drop_cols, errors="ignore"), y_train)
            pipelines[model_name] = pipeline
            metrics[model_name] = self._evaluate_model(
                pipeline,
                X_val.drop(columns=drop_cols, errors="ignore"),
                y_val,
                model_name,
            )

        return pipelines, metrics

    def train_and_evaluate_tuned_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_name: str = "xgboost",
    ) -> Tuple[Pipeline, Dict[str, float]]:
        """
        Hyperparameter-tunes the given model (defaults to XGBoost, the
        top baseline performer) using grouped, stratified cross-validation.
        """
        logging.info(f"Starting hyperparameter tuning for {model_name}.")
        random_state = self.config.get("random_state", 42)
        id_col = self.config["id_column"]

        model = XGBClassifier(random_state=random_state, eval_metric="logloss")
        pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("classifier", model)]
        )

        param_grid = {
            f"classifier__{k}": v for k, v in self.config["xgb_param_grid"].items()
        }
        cv = StratifiedGroupKFold(
            n_splits=self.config.get("cv_splits", 5),
            shuffle=True,
            random_state=random_state,
        )

        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=self.config.get("xgb_n_iter", 15),
            cv=cv,
            scoring=self.config.get("scoring", "average_precision"),
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(
            X_train.drop(columns=[id_col], errors="ignore"),
            y_train,
            groups=X_train[id_col],
        )

        best_pipeline = search.best_estimator_
        metrics = self._evaluate_model(
            best_pipeline,
            X_val.drop(columns=[id_col], errors="ignore"),
            y_val,
            f"{model_name}_tuned",
        )
        logging.info(f"Best params: {search.best_params_}")
        return best_pipeline, metrics

    def evaluate_final_model(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluates the final selected model once on the untouched test set.
        """
        id_col = self.config["id_column"]
        metrics = self._evaluate_model(
            model,
            X_test.drop(columns=[id_col], errors="ignore"),
            y_test,
            model_name,
            log_report=True,
        )
        return metrics

    @staticmethod
    def _evaluate_model(
        model: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        log_report: bool = False,
    ) -> Dict[str, float]:
        """
        Computes PR-AUC and ROC-AUC (primary metrics given class imbalance),
        with an optional full classification report for final evaluation.
        """
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        metrics = {
            "PR-AUC": average_precision_score(y, probs),
            "ROC-AUC": roc_auc_score(y, probs),
        }
        logging.info(f"{model_name} metrics: {metrics}")
        if log_report:
            logging.info(f"\n{classification_report(y, preds)}")
        return metrics
