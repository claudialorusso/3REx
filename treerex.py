import pickle
import argparse
from pathlib import Path
from typing import Sequence
import csv
import logging
import time
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="3REx",
        description=("3REx extracts human-readable rules from pre-trained Random Forest models.")
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the Random Forest model."
    )

    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to the CSV file containing the ordered feature names."
    )

    arguments = parser.parse_args()
    return arguments

def validate_input(path: Path, description : str):
    """
    Verifies whether the input path exists and is referring to a real file. Otherwise, it raises and exception.
    """
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{description} does not a real file: {path}")

def validate_model_features(model: RandomForestClassifier, feature_names: Sequence[str]):
    """
    Verifies whether the total model's features number is the same as the feature_names one.
    """
    model_feature_count = int(model.n_features_in_)
    input_feature_count = len(feature_names)

    if model_feature_count != input_feature_count:
        raise ValueError(f"Inconsistency: model expects {model_feature_count} features, but the feature file contains {input_feature_count} rows.")

def load_model(model_path):
    """
    Loads the pre-trained Random Forest Classifier's model from a pickle file.
    Checks whether the model it is a RandomForestClassifier instance or not.
    """
    logger.info("\033[33mLoading model from %s\033[0m", model_path)
    with model_path.open("rb") as file:
        model = pickle.load(file)
    if not isinstance(model, RandomForestClassifier):
        raise TypeError(f"Expected a RandomForestClassifier, found {type(model).__name__}.")
    return model

def load_feature_names(features_path: Path):
    """
    Loads feature names in the correct order.
    """
    logger.info("\033[33mLoading feature names from %s\033[0m", features_path)

    feature_names = []

    with features_path.open(mode="r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required_columns = {"feature_position", "feature_name"}
        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                f"Feature file is missing required columns: {sorted(missing_columns)}"
            )
        rows = []
        for row in reader:
            rows.append(
                {
                    "feature_position": int(row["feature_position"]),
                    "feature_name": row["feature_name"]
                }
            )
    rows.sort(key=lambda row: row["feature_position"])

    expected_positions = list(range(len(rows)))

    actual_position = []
    for row in rows:
        actual_position.append(row["feature_position"])

    if actual_position != expected_positions:
        raise ValueError(
            "Feature positions are not contiguous or do not start from 0. "
            f"Expected {expected_positions[:10]}..., found {actual_position[:10]}..."
        )

    feature_names = []
    for row in rows:
        feature_name = row["feature_name"]
        feature_names.append(feature_name)

    return feature_names

def main():
    logger.info("\033[94mInitializing 3REx\033[0m")
    # read arguments
    args = parse_arguments()
    model_path = args.model
    features_path = args.features
    # control file existance
    validate_input(model_path, "Model file")
    validate_input(features_path, "Feature file")
    # load model
    model = load_model(model_path)
    feature_names = load_feature_names(features_path)
    # verify that model features' number == #features in features_names.csv
    validate_model_features(model=model, feature_names=feature_names)

    logger.info("\033[94mInput validation completed successfully.\033[0m")
    logger.info("\033[32mModel type: %s\033[0m", type(model).__name__)
    logger.info("\033[32mNumber of trees: %d\033[0m", len(model.estimators_))
    logger.info("\033[32mNumber of features: %d\033[0m", len(feature_names))
    logger.info("\033[32mFirst feature: %s\033[0m", feature_names[0])
    logger.info("\033[32mLast feature: %s\033[0m", feature_names[-1])

if __name__ == "__main__":
    start_time = time.time()

    try:
        main()
    except Exception:
        logger.exception("3REx execution failed.")
        raise
    finally:
        elapsed_seconds = time.time() - start_time
        logger.info("\033[33mTotal runtime: %.2fs\033[0m", elapsed_seconds)
        logger.info("\033[33mTotal runtime: %.2fmin\033[0m", elapsed_seconds / 60)
