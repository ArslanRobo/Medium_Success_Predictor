from prefect import flow, get_run_logger
from pipeline.data_ingestion import scrape_medium_articles
from pipeline.data_storage import create_database_tables, store_raw_data_to_db, load_data_from_db
from pipeline.data_transformation import clean_and_transform_data
from pipeline.model_training import train_model
from config import SCRAPING_CONFIG
import time
import argparse

@flow(name="Medium Author Success Prediction Pipeline")
def medium_success_pipeline(run_scraping: bool = False):
    logger = get_run_logger()
    logger.info("🚀 Starting Medium Author Success Pipeline")
    
    # Step 1: Setup database
    create_database_tables()
    
    if run_scraping:
        # Step 2: Data ingestion (optional)
        logger.info("Running data scraping...")
        for tag in SCRAPING_CONFIG['tags']:
            for year in SCRAPING_CONFIG['years']:
                csv_path = scrape_medium_articles(tag, year)
                store_raw_data_to_db(csv_path, tag)
                time.sleep(5)  # Be gentle with Medium's servers
    
    # Step 3: Data transformation
    df = load_data_from_db()
    if df.empty:
        logger.error("No data loaded from database!")
        return
    
    processed_path = clean_and_transform_data(df)
    
    # Step 4: Model training
    model_path = train_model(processed_path)
    
    logger.info("✅ Pipeline execution completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-scraping', action='store_true', help='Run data scraping')
    args = parser.parse_args()
    
    medium_success_pipeline(run_scraping=args.run_scraping)
