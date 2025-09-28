import logging
import pandas as pd
from pathlib import Path
from typing import Tuple
from src.utils import (
    convert_datetime_to_month_period, 
    convert_period_int_to_month_period,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()

DATA_PATH = SCRIPT_DIR.parent / "data"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_withdrawals = pd.read_parquet(DATA_PATH / "zrive_advertiser_withdrawals.parquet")
    df_advertiser = pd.read_parquet(DATA_PATH / "zrive_dim_advertiser.parquet")
    df_monthly = pd.read_parquet(DATA_PATH / "zrive_fct_monthly_snapshot_advertiser.parquet")

    return df_withdrawals, df_advertiser, df_monthly

def preprocess_withdrawals(df_withdrawals: pd.DataFrame) -> pd.DataFrame:
    df_withdrawals = convert_datetime_to_month_period(df_withdrawals, 'withdrawal_creation_date', 'withdrawal_month', True)
    df_withdrawals = add_predict_month(df_withdrawals)
    df_withdrawals = add_churn(df_withdrawals)

    return df_withdrawals

def preprocess_monthly(df_monthly: pd.DataFrame) -> pd.DataFrame:
    df_monthly = convert_period_int_to_month_period(df_monthly)

    return df_monthly

def process_target(
        df_monthly: pd.DataFrame, 
        df_withdrawals: pd.DataFrame, 
        df_advertiser: pd.DataFrame
) -> pd.DataFrame:
    return (
        df_monthly
        .pipe(add_churn_target, df_withdrawals)
        .pipe(remove_activity_after_first_churn)
        .pipe(add_churn_from_advertiser_data, df_advertiser)
        .pipe(remove_incomplete_users)
        .pipe(remove_inactive_periods_without_contract)
    )

def add_churn(df: pd.DataFrame) -> pd.DataFrame:
    CHURN_REASONS_EXCLUDED = [
        'Upselling-cambio de contrato',
        'Cambio a Bundle Online',
        'Cambio de Contrato/propuesta/producto'
    ]

    valid_entries = (
        df["withdrawal_type"].notna() & 
        df["withdrawal_status"].notna() & 
        df["withdrawal_reason"].notna()
    )

    df["churn"] = 0

    df.loc[valid_entries, "churn"] = (
        (df["withdrawal_type"] == "TOTAL") &
        (df["withdrawal_status"] != "Denegada") &
        (~df["withdrawal_reason"].isin(CHURN_REASONS_EXCLUDED))
    ).astype(int)

    return df

def add_predict_month(
        df: pd.DataFrame, 
        predict_col: str = "predict_month", 
        withdrawal_col: str = "withdrawal_month", 
        months_to_subtract: int = 1
) -> pd.DataFrame:
    df[predict_col] = df[withdrawal_col] - months_to_subtract
    return df

def add_churn_target(df_monthly: pd.DataFrame, df_withdrawals: pd.DataFrame) -> pd.DataFrame:
    df_target = df_monthly.merge(
        df_withdrawals[['advertiser_zrive_id', 'predict_month', 'churn']].rename(
            columns={'predict_month': 'month_period'}
        ),
        on=['advertiser_zrive_id', 'month_period'],
        how='left'
    )

    df_target['churn'] = df_target['churn'].fillna(0)

    return df_target

def remove_activity_after_first_churn(df: pd.DataFrame) -> pd.DataFrame:
    first_churn = (
        df[df['churn'] == 1]
        .groupby('advertiser_zrive_id')['period_int']
        .min()
        .reset_index()
        .rename(columns={'period_int': 'first_churn_period'})
    )

    df_with_churn_info = df.merge(
        first_churn,
        on='advertiser_zrive_id',
        how='left'
    )

    df_filtered = df_with_churn_info[
        (df_with_churn_info['first_churn_period'].isna()) |
        (df_with_churn_info['period_int'] <= df_with_churn_info['first_churn_period'])
    ]

    df_filtered = df_filtered.drop(columns=['first_churn_period'])

    return df_filtered

def add_churn_from_advertiser_data(df_target: pd.DataFrame, df_advertiser: pd.DataFrame) -> pd.DataFrame:
    """
    Add churns for users who did not churn explicitly through withdrawals,
    but have a 'contrato_churn_date' in the advertiser data.
    The churn is assigned to the previous month to the contract churn date,
    only if the user had activity during that month.
    """
    no_churn_users = (
        df_target.groupby('advertiser_zrive_id')['churn']
        .max()
        .loc[lambda x: x == 0]
        .index
    )

    churn_info = (
        df_advertiser
        .loc[
            df_advertiser['advertiser_zrive_id'].isin(no_churn_users) &
            df_advertiser['contrato_churn_date'].notna(),
            ['advertiser_zrive_id', 'contrato_churn_date']
        ]
        .copy()
    )

    if churn_info.empty:
        logger.info("No additional churn dates found in df_advertiser")
        return df_target
    
    churn_info = convert_datetime_to_month_period(
        churn_info, "contrato_churn_date", "churn_month"
    )

    updates = []
    for _, row in churn_info.iterrows():
        user_id = row['advertiser_zrive_id']
        churn_month = row['churn_month']

        churn_period_int = int((churn_month - 1).strftime('%Y%m'))

        #Check if there is an activity record for that user in the previous month
        user_data = df_target[df_target['advertiser_zrive_id'] == user_id]

        if churn_period_int in user_data['period_int'].values:
            updates.append((user_id, churn_period_int))

    for user_id, period in updates:
        df_target.loc[
            (df_target['advertiser_zrive_id'] == user_id) &
            (df_target['period_int'] == period),
            'churn'
        ] = 1

    logger.info(f"Updated {len(updates)} records with churn information from df_advertiser")

    df_target = remove_activity_after_first_churn(df_target)

    return df_target

def remove_incomplete_users(df_target: pd.DataFrame, latest_period=None) -> pd.DataFrame:
    """
    Remove users who do not have churn registered and finish before the last period.
    """
    if latest_period is None:
        latest_period = df_target['period_int'].max()
    
    users_without_churn = df_target.groupby('advertiser_zrive_id')['churn'].max()
    users_without_churn = users_without_churn[users_without_churn == 0].index.tolist()
    
    last_period_by_user = df_target.groupby('advertiser_zrive_id')['period_int'].max().reset_index()
    
    users_to_remove = last_period_by_user[
        (last_period_by_user['advertiser_zrive_id'].isin(users_without_churn)) & 
        (last_period_by_user['period_int'] < latest_period)
    ]['advertiser_zrive_id'].tolist()
    
    df_filtered = df_target[~df_target['advertiser_zrive_id'].isin(users_to_remove)]
    
    logger.info(f"Removed {len(users_to_remove)} users without complete churn information")
    
    return df_filtered

def remove_inactive_periods_without_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows where has_active_contract=False and no ads were published.

    Args:
        df: DataFrame with monthly advertiser data
            
    Returns:
        Filtered DataFrame
    """
    rows_before = df.shape[0]
    
    df_filtered = df[~(
        (df['has_active_contract'] == False) & 
        (df['monthly_published_ads'] == 0)
    )]
    
    rows_removed = rows_before - df_filtered.shape[0]
    
    logger.info(f"Removed {rows_removed} out of {rows_before} rows with no active contract and no activity")
    
    return df_filtered


def main():
    df_withdrawals, df_advertiser, df_monthly = load_data()
    logger.info("All datasets successfully loaded.")

    df_withdrawals = preprocess_withdrawals(df_withdrawals)
    logger.info("Withdrawals data preprocessed.")

    df_monthly = preprocess_monthly(df_monthly)
    logger.info("Monthly data preprocessed.")

    df_target = process_target(df_monthly, df_withdrawals, df_advertiser)
    logger.info("Target data processed.")

    df_target.to_parquet(DATA_PATH / "processed_data.parquet")


if __name__ == "__main__":
    main()