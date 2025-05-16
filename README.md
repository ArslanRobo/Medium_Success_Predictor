
# Medium Article Engagement Prediction System

## Project Overview
An end-to-end data engineering and machine learning pipeline that:
- **Collects** Medium article data through web scraping
- **Processes** and stores the data in a structured database
- **Trains** a machine learning model to predict article engagement
- **Serves** predictions through an interactive web interface
- **Monitors** the entire pipeline with workflow orchestration

## Technology Stack
| Component               | Technology Used          |
|-------------------------|--------------------------|
| Data Ingestion          | Python, BeautifulSoup, Requests |
| Data Storage            | SQLite                   |
| Data Processing         | Pandas, NumPy            |
| Machine Learning        | Scikit-learn, TF-IDF, Logistic Regression |
| Frontend                | Streamlit                |
| Workflow Orchestration  | Prefect                  |
| Visualization           | Matplotlib, Streamlit components |

## Repository Structure
```
medium-engagement-predictor/
├── data/
│   ├── raw/                    # Raw scraped data
│   ├── processed/              # Cleaned datasets
│   └── medium_articles.db      # SQLite database
├── frontend/
│   └── app.py                  # Streamlit application
├── pipeline/
│   ├── __init__.py
│   ├── data_ingestion.py       # Web scraping logic
│   ├── data_storage.py         # Database operations
│   ├── data_transformation.py  # Data cleaning
│   └── model_training.py       # ML model code
├── models/
│   └── engagement_predictor.pkl # Trained model
├── config.py                   # Project configuration
├── requirements.txt            # Dependencies
└── run_pipeline.py             # Main workflow
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medium-engagement-predictor.git
cd medium-engagement-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Pipeline
Execute the complete data pipeline:
```bash
python run_pipeline.py
```

### Starting the Web Interface
Launch the Streamlit dashboard:
```bash
streamlit run frontend/app.py
```

### Monitoring Workflows
Start Prefect server to monitor pipeline runs:
```bash
prefect server start
```
Access the UI at: [http://localhost:4200](http://localhost:4200)

---

## Key Features

### Machine Learning Model

- **Input Feature:** Article titles only
- **Target Variable:** Binary engagement (1 if claps > 500, else 0)

**Text Processing:**
- Convert to lowercase
- Remove punctuation
- Eliminate stopwords

**Feature Extraction:** TF-IDF vectorization  
**Classifier:** Logistic Regression

### Model Performance

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.83  |
| Precision  | 0.81  |
| Recall     | 0.85  |
| F1 Score   | 0.83  |

## Future Improvements

- Containerize application with Docker
- Add unit and integration tests
- Implement model versioning
- Add data quality monitoring
- Set up automated pipeline scheduling
