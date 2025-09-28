import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from typing import List


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()

DATA_PATH = SCRIPT_DIR.parent / "data"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(DATA_PATH / "processed_data.parquet")
    df_advertiser = pd.read_parquet(DATA_PATH / "zrive_dim_advertiser.parquet")

    return df, df_advertiser


def create_time_features(df: pd.DataFrame, df_advertiser: pd.DataFrame) -> pd.DataFrame:
    '''
    Adds the following features to the dataset: 
    tenure, months_since_last_contract, has_renewed

    Assume first month with activity as the start date in case there is activity before the first contract
    '''

    df_time_features = df.merge(df_advertiser, on='advertiser_zrive_id', how='left')
    
    #Convert to period in months
    df_time_features['min_start_contrato_date'] = pd.to_datetime(df_time_features['min_start_contrato_date'], errors='coerce').dt.to_period('M')
    df_time_features['max_start_contrato_nuevo_date'] = pd.to_datetime(df_time_features['max_start_contrato_nuevo_date'], errors='coerce').dt.to_period('M')
    df_time_features['contrato_churn_date'] = pd.to_datetime(df_time_features['contrato_churn_date'], errors='coerce').dt.to_period('M')
    
    #Take first month with activity as the start date in case there is activity before the first contract
    first_activity_date = df_time_features[~df_time_features['has_active_contract']].groupby('advertiser_zrive_id')['month_period'].min()
    df_time_features['first_activity_date'] = df_time_features['advertiser_zrive_id'].map(first_activity_date).fillna(df_time_features['min_start_contrato_date'])

    current_date = df_time_features['month_period']
    start_date =  df_time_features['first_activity_date']
    new_start_date = df_time_features['max_start_contrato_nuevo_date']

    #Compute features
    df_time_features['tenure'] = (current_date - start_date).apply(lambda x: x.n if pd.notnull(x) else None).apply(lambda x: None if x < 0 else x)
    df_time_features['months_since_last_contract'] = (current_date - new_start_date).apply(lambda x: x.n if pd.notnull(x) else None).apply(lambda x: None if x < 0 else x)
    df_time_features['has_renewed'] = df_time_features['months_since_last_contract'].notna().astype(int)
    df_time_features['months_since_last_contract'] =  df_time_features['months_since_last_contract'].fillna(df_time_features['tenure'])

    df_time_features = df_time_features.drop(['province_id', 'updated_at',
       'advertiser_province', 'advertiser_group_id', 'min_start_contrato_date',
       'max_start_contrato_nuevo_date', 'has_active_contract', 'contrato_churn_date',
       'first_activity_date'], axis=1)

    logger.info('Time features created')

    return df_time_features


def create_ratios(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Create ratios between features
    '''
    
    df_features = df.copy()

    def safe_divide(numerator: int, denominator: int) -> float:
        return np.where(denominator > 0, numerator / denominator, 0)

    df_features['monthly_total_premium_ads'] = (
        df_features['monthly_oro_ads'] + 
        df_features['monthly_plata_ads'] +
        df_features['monthly_destacados_ads'] +
        df_features['monthly_pepitas_ads']
    )

    # Ads ratios
    df_features['ratio_published_contracted'] = safe_divide(
        df_features['monthly_published_ads'], df_features['monthly_contracted_ads']
    )
    df_features['ratio_unique_published'] = safe_divide(
        df_features['monthly_unique_published_ads'], df_features['monthly_published_ads']
    )
    df_features['ratio_premium_ads'] = safe_divide(
        df_features['monthly_total_premium_ads'], df_features['monthly_published_ads']
    )

    # Engagement ratios
    df_features['leads_per_published_ad'] = safe_divide(
        df_features['monthly_leads'], df_features['monthly_published_ads']
    )

    df_features['leads_per_premium_ad'] = safe_divide(
        df_features['monthly_leads'], df_features['monthly_total_premium_ads']
    )

    df_features['visits_per_published_ad'] = safe_divide(
        df_features['monthly_visits'], df_features['monthly_published_ads']
    )
    df_features['leads_per_visit'] = safe_divide(
        df_features['monthly_leads'], df_features['monthly_visits']
    )
    df_features['leads_per_shows'] = safe_divide(
        df_features['monthly_leads'], df_features['monthly_shows']
    )

    # Economic ratios
    df_features['invoice_per_published_ad'] = safe_divide(
        df_features['monthly_total_invoice'], df_features['monthly_published_ads']
    )
    df_features['invoice_per_lead'] = safe_divide(
        df_features['monthly_total_invoice'], df_features['monthly_leads']
    )

    logger.info('Ratios created')

    return df_features


def create_agg_stats(
        df: pd.DataFrame, 
        features: List[str], 
        months = 3, 
        agg_funcs=['mean', 'std', 'min', 'max'], 
        add_deltas=True
) -> pd.DataFrame:
    '''
    Adds aggregate features over the last months for the features passed to the function 
    '''

    df_agg = df.copy().sort_values(by=['advertiser_zrive_id','month_period'], ascending=[True, True])

    for feature in features:
        for agg_func in agg_funcs:
            col_name = f'{feature}_{months}_months_{agg_func}'
            df_agg[col_name] = (
                df_agg.groupby('advertiser_zrive_id')[feature]
                .transform(lambda x: x.rolling(window=months, min_periods=1).agg(agg_func))
            )

            if add_deltas and agg_func == 'mean':
                delta_col = f'{feature}_{months}_months_mean_delta'
                df_agg[delta_col] = df_agg[feature] - df_agg[col_name]

    df_agg = df_agg.sort_values(by=['advertiser_zrive_id','month_period'])

    logger.info('Aggregate stats created')

    return df_agg


def engineer_features(df: pd.DataFrame, df_advertiser: pd.DataFrame) -> pd.DataFrame:
    df_time_features = create_time_features(df, df_advertiser)
    df_ratios = create_ratios(df_time_features)

    features_to_aggregate = [
        'monthly_leads', 
        'monthly_visits',
        'monthly_total_invoice',
        'monthly_avg_ad_price',
        'monthly_published_ads',
        'monthly_contracted_ads',
        'ratio_published_contracted',
        'ratio_unique_published', 
        'ratio_premium_ads', 
        'leads_per_published_ad',
        'leads_per_premium_ad', 
        'visits_per_published_ad', 
        'leads_per_visit',
        'leads_per_shows', 
        'invoice_per_published_ad', 
        'invoice_per_lead'
    ]
    
    agg_funcs = ['mean']

    full_engineered_dataset = create_agg_stats(
        df_ratios, features = features_to_aggregate, months = 3, agg_funcs=agg_funcs, add_deltas=True
    )
    
    return full_engineered_dataset



def main():
    df, df_advertiser = load_data()
    logger.info("All datasets successfully loaded.")

    full_dataset = engineer_features(df, df_advertiser)
    logger.info("Full dataset ready")

    full_dataset.to_parquet(DATA_PATH / "full_data.parquet")

 
if __name__ == "__main__":
    main()