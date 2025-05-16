import pandas as pd
from prefect import task, get_run_logger
from config import PATHS
import os

@task
def clean_and_transform_data(df: pd.DataFrame):
    logger = get_run_logger()
    
    # Convert claps
    def convert_claps(value):
        if isinstance(value, str) and 'k' in value.lower():
            return int(float(value.lower().replace('k', '')) * 1000)
        try:
            return int(value)
        except:
            return 0
    
    df['claps'] = df['claps'].apply(convert_claps)
    
    # Convert responses
    df['responses'] = pd.to_numeric(df['responses'], errors='coerce').fillna(0).astype(int)
    
    # Convert reading time
    df['reading_time_mins'] = pd.to_numeric(df['reading_time_mins'], errors='coerce').fillna(0).astype(int)
    
    # Drop unnecessary columns
    df = df.drop(columns=['followers'], errors='ignore')
    
    # Drop rows with missing values
    df = df.dropna(subset=['title', 'claps'])
    
    # Save processed data
    os.makedirs(PATHS['processed_data'], exist_ok=True)
    processed_path = os.path.join(PATHS['processed_data'], 'medium_articles_processed.csv')
    df.to_csv(processed_path, index=False)
    
    logger.info(f"Data cleaned and transformed. Saved to {processed_path}")
    return processed_path