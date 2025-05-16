import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Database configuration (SQLite)
DB_CONFIG = {
    'db_path':  'data/medium_articles.db',
    'table_name': 'articles'
}

# Scraping configuration
SCRAPING_CONFIG = {
    'base_url': 'https://medium.com/tag/',
    'tags': ['business', 'technology', 'ai'],
    'years': [2020, 2021, 2022, 2023, 2024, 2025],
    'max_stories_per_day': 20,
    'request_delay': 2  # seconds
}

# Model configuration
MODEL_CONFIG = {
    'clap_threshold': 500,
    'test_size': 0.2,
    'random_state': 42
}

# Path configuration
PATHS = {
    'raw_data': 'data/raw',
    'processed_data': 'data/processed',
    'models': 'models'
}

# Create directories if they don't exist
for path in PATHS.values():
    full_path = BASE_DIR / path
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured directory exists: {full_path}")  # Optional debug