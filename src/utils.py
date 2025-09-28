import os
import datetime
import logging
from typing import Optional, Tuple
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

N_TRAINING_MONTHS = 6
LAST_TRAINING_MONTH = datetime.datetime(2024, 11, 1)
N_ESTIMATORS = 150

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.01,
    "max_depth": 3,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "nthread": 10,
    "random_state": 1,
}

OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/"))
STORAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
PREDICTIONS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../predictions/"))

NAN_FEATURES = [
    "monthly_distinct_ads",
    "monthly_avg_ad_price",
    "monthly_avg_ad_price_3_months_mean",
    "monthly_avg_ad_price_3_months_mean_delta",
]

IMPORTANT_FEATURES = [
    "months_since_last_contract",
    "tenure",
]

FEATURE_COLS = [
    "monthly_published_ads",
    "monthly_published_ads_3_months_mean",
    "monthly_published_ads_3_months_mean_delta",
    "monthly_unique_published_ads",
    "monthly_contracted_ads",
    "monthly_contracted_ads_3_months_mean",
    "monthly_contracted_ads_3_months_mean_delta",
    "monthly_leads",
    "monthly_leads_3_months_mean",
    "monthly_leads_3_months_mean_delta",
    "monthly_visits",
    "monthly_visits_3_months_mean",
    "monthly_visits_3_months_mean_delta",
    "monthly_oro_ads",
    "monthly_plata_ads",
    "monthly_destacados_ads",
    "monthly_pepitas_ads",
    "monthly_shows",
    "monthly_total_phone_views",
    "monthly_total_calls",
    "monthly_total_emails",
    "monthly_total_invoice",
    "monthly_total_invoice_3_months_mean",
    "monthly_total_invoice_3_months_mean_delta",
    "monthly_unique_calls",
    "monthly_unique_emails",
    "monthly_unique_leads",
    "has_renewed",
    "monthly_total_premium_ads",
    "ratio_published_contracted",
    "ratio_published_contracted_3_months_mean",
    "ratio_published_contracted_3_months_mean_delta",
    "ratio_unique_published",
    "ratio_unique_published_3_months_mean",
    "ratio_unique_published_3_months_mean_delta",
    "ratio_premium_ads",
    "ratio_premium_ads_3_months_mean",
    "ratio_premium_ads_3_months_mean_delta",
    "leads_per_published_ad",
    "leads_per_published_ad_3_months_mean",
    "leads_per_published_ad_3_months_mean_delta",
    "leads_per_premium_ad",
    "leads_per_premium_ad_3_months_mean",
    "leads_per_premium_ad_3_months_mean_delta",
    "visits_per_published_ad",
    "visits_per_published_ad_3_months_mean",
    "visits_per_published_ad_3_months_mean_delta",
    "leads_per_visit",
    "leads_per_visit_3_months_mean",
    "leads_per_visit_3_months_mean_delta",
    "leads_per_shows",
    "leads_per_shows_3_months_mean",
    "leads_per_shows_3_months_mean_delta",
    "invoice_per_published_ad",
    "invoice_per_published_ad_3_months_mean",
    "invoice_per_published_ad_3_months_mean_delta",
    "invoice_per_lead",
    "invoice_per_lead_3_months_mean",
    "invoice_per_lead_3_months_mean_delta",
]

LABEL_COL = "churn"


def convert_datetime_to_month_period(
    df: pd.DataFrame,
    datetime_col: str,
    new_col: str,
    drop_original: bool = True,
) -> pd.DataFrame:
    df[new_col] = pd.to_datetime(df[datetime_col]).dt.to_period("M")
    if drop_original:
        df = df.drop(columns=[datetime_col])
    return df


def convert_period_int_to_month_period(
    df: pd.DataFrame,
    period_col: str = "period_int",
    new_col: str = "month_period",
) -> pd.DataFrame:
    df[new_col] = (
        pd.to_datetime(df[period_col].astype(str) + "01", format="%Y%m%d")
        .dt.to_period("M")
    )
    return df


def load_dataset() -> pd.DataFrame:
    dataset_name = "full_data.parquet"
    loading_file = os.path.join(STORAGE_PATH, dataset_name)
    logger.info(f"Loading dataset from {loading_file}")
    return pd.read_parquet(loading_file)


def delete_nan_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(NAN_FEATURES, axis=1)


def delete_important_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(IMPORTANT_FEATURES, axis=1)


def build_dataframe() -> pd.DataFrame:
    logger.info("Building dataframe")
    return (
        load_dataset()
        .pipe(delete_nan_features)
        .pipe(delete_important_features)
    )


def evaluate_model(
    model_name: str,
    y_test: pd.Series,
    y_pred: pd.Series,
) -> tuple[float, float]:
    pr_auc = average_precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    logger.info(
        f"{model_name} results — PR AUC: {pr_auc:.2f}, ROC AUC: {roc_auc:.2f}"
    )
    return pr_auc, roc_auc


def save_predictions(
    y: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    df: pd.DataFrame,
):
    os.makedirs(PREDICTIONS_PATH, exist_ok=True)

    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_name = f"{model_name}_{date.replace(':', '-')}.csv"
    filename = os.path.join(PREDICTIONS_PATH, dataset_name)

    df_filtered = df[["advertiser_zrive_id", 'month_period']]
    df_predictions = df_filtered.assign(
        model_name=model_name,
        date=date,
        y=y,
        y_pred=y_pred,
    )

    df_predictions.to_csv(filename, index=False)
    logger.info(f"Saved predictions to {filename}")


def get_model_path(
    directory: str,
    model_name: Optional[str] = None
) -> Tuple[str, str]:
    if model_name:
        path = os.path.join(directory, model_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model not found in: {path}")
        return path

    files = [model for model in os.listdir(directory) if model.endswith('.pkl')]

    if not files:
        raise FileNotFoundError(f"No .pkl found in {directory}")

    files.sort(reverse=True)
    latest_model = files[0]

    return os.path.join(directory, latest_model), latest_model
