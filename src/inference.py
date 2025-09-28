import logging
from joblib import load
import xgboost as xgb
import pandas as pd
from src.train import feature_label_split
from src.utils import (
    OUTPUT_PATH, build_dataframe, evaluate_model, save_predictions, get_model_path
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def main() -> None:
    model_path, model_name = get_model_path(directory=OUTPUT_PATH)
    model: xgb.Booster = load(model_path)
    logger.info(f"Loaded model {model_name}")

    df: pd.DataFrame = build_dataframe()
    X, y = feature_label_split(df)

    y_pred = model.predict(xgb.DMatrix(X))

    evaluate_model("Inference test", y, y_pred)

    save_predictions(y, y_pred, model_name, df)

if __name__ == "__main__":
    main()
