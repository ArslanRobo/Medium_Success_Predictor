import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from prefect import task, get_run_logger
from config import MODEL_CONFIG, PATHS
import os

def preprocess(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

@task
def train_model(csv_path: str):
    logger = get_run_logger()
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
    
    # Create target variable
    df['high_engagement'] = (df['claps'] > MODEL_CONFIG['clap_threshold']).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['title'], 
        df['high_engagement'], 
        test_size=MODEL_CONFIG['test_size'], 
        random_state=MODEL_CONFIG['random_state']
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            preprocessor=preprocess,
            stop_words='english',
            token_pattern=r'\b\w+\b'
        )),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    logger.info("Model evaluation metrics:")
    logger.info(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    logger.info(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
    logger.info(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")
    
    # Save model
    os.makedirs(PATHS['models'], exist_ok=True)
    model_path = os.path.join(PATHS['models'], 'engagement_predictor.pkl')
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model_path