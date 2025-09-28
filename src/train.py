from typing import Tuple, Dict, Any
import pandas as pd
import os
import logging
import joblib
import datetime
import xgboost as xgb
from matplotlib.dates import relativedelta
from src.utils import (
    build_dataframe,
    OUTPUT_PATH,
    FEATURE_COLS,
    LABEL_COL,
    LAST_TRAINING_MONTH,
    PARAMS,
    N_ESTIMATORS,
    N_TRAINING_MONTHS,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def feature_label_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLS]
    y = df[LABEL_COL]
    return X, y


def split_train_by_period(
    data_set: pd.DataFrame,
    execution_date: datetime.datetime,
    n_training_months: int = N_TRAINING_MONTHS,
) -> pd.DataFrame:
    start_dt = execution_date - relativedelta(months=n_training_months)
    start = pd.Period(start_dt.strftime("%Y-%m"), freq="M")
    end = pd.Period(execution_date.strftime("%Y-%m"), freq="M")

    train = data_set[
        (data_set["month_period"] > start) & (data_set["month_period"] <= end)
    ]

    logger.info(
        f"Train period: {start_dt:%Y-%m-%d} → {execution_date:%Y-%m-%d}"
        f" ({n_training_months} months)"
    )
    months = sorted(train["month_period"].unique())
    logger.info(f"Distinct months: {[str(m) for m in months]}")

    return train


def train_model(
    train_set: pd.DataFrame,
    n_estimators: int = N_ESTIMATORS,
    params: Dict[str, Any] = PARAMS,
) -> Tuple[xgb.Booster, Dict[str, Any]]:
    X_train, y_train = feature_label_split(train_set)
    dtrain = xgb.DMatrix(X_train, label=y_train)

    evals_result: Dict[str, Any] = {}
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, "train")],
        verbose_eval=False,
        evals_result=evals_result,
    )
    return model, evals_result


def save_model(model: Any, model_name: str) -> None:
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(OUTPUT_PATH, f"{ts}_{model_name}.pkl")
    logger.info(f"Saving model to {path}")
    joblib.dump(model, path)


def train(df: pd.DataFrame) -> None:
    train_set = split_train_by_period(df, LAST_TRAINING_MONTH)
    model, _ = train_model(train_set)
    save_model(model, "xgboost")


def main():
    df = build_dataframe()
    train(df)


if __name__ == "__main__":
    main()