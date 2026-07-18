# Standard library imports
import logging

# Third-party imports
import pandas as pd
import yaml
from sklearn.utils._testing import ignore_warnings

# Local application/library specific imports
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

logging.basicConfig(level=logging.INFO)


@ignore_warnings(category=Warning)
def main():
    config_path = "./src/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    df = pd.read_csv(config["file_path"])

    data_prep = DataPreparation(config)
    cleaned_df = data_prep.clean_data(df)

    model_training = ModelTraining(config, data_prep.preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(
        cleaned_df
    )

    baseline_models, baseline_metrics = (
        model_training.train_and_evaluate_baseline_models(
            X_train, y_train, X_val, y_val
        )
    )

    tuned_model, tuned_metrics = model_training.train_and_evaluate_tuned_model(
        X_train, y_train, X_val, y_val, model_name="xgboost"
    )

    all_metrics = {**baseline_metrics, "xgboost_tuned": tuned_metrics}
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["PR-AUC"])
    logging.info(f"Best Model Found: {best_model_name}")

    best_model = (
        tuned_model
        if best_model_name == "xgboost_tuned"
        else baseline_models[best_model_name]
    )

    final_metrics = model_training.evaluate_final_model(
        best_model, X_test, y_test, best_model_name
    )
    logging.info(f"Final Test Metrics: {final_metrics}")


if __name__ == "__main__":
    main()
