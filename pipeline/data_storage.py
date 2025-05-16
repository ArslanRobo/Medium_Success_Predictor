import sqlite3
import pandas as pd
from prefect import task, get_run_logger
from config import DB_CONFIG, PATHS
import os

@task
def create_database_tables():
    logger = get_run_logger()
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(DB_CONFIG['db_path']), exist_ok=True)
        
        conn = sqlite3.connect(DB_CONFIG['db_path'])
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            title TEXT,
            claps INTEGER,
            responses INTEGER,
            author_name TEXT,
            followers TEXT,
            reading_time_mins INTEGER,
            tag TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            prediction INTEGER,
            probability REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (article_id) REFERENCES articles(id)
        )
        """)
        
        conn.commit()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
    finally:
        if conn:
            conn.close()

@task
def store_raw_data_to_db(csv_path: str, tag: str):
    logger = get_run_logger()
    try:
        df = pd.read_csv(csv_path)
        df['tag'] = tag
        
        conn = sqlite3.connect(DB_CONFIG['db_path'])
        
        # Convert claps to integers
        df['claps'] = df['claps'].apply(lambda x: int(float(str(x).replace('k', '')) * 1000 if 'k' in str(x).lower() else int(x)))
        
        # Store to database
        df.to_sql(DB_CONFIG['table_name'], conn, if_exists='append', index=False)
        logger.info(f"Stored {len(df)} records from {csv_path} to database")
    except Exception as e:
        logger.error(f"Error storing data to database: {e}")
    finally:
        if conn:
            conn.close()

@task
def load_data_from_db():
    logger = get_run_logger()
    try:
        conn = sqlite3.connect(DB_CONFIG['db_path'])
        # UPDATE THIS QUERY TO MATCH YOUR ACTUAL COLUMNS:
        query = "SELECT date, title, claps, responses, author_name, followers, reading_time_mins FROM articles"
        df = pd.read_sql(query, conn)
        logger.info(f"Loaded {len(df)} records from database")
        return df
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        return pd.DataFrame()