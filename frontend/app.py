import sys
import os
from pathlib import Path
import sqlite3
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

@st.cache_resource
def get_db_connection():
    db_path = project_root / "data" / "medium_articles.db"
    return sqlite3.connect(db_path, check_same_thread=False)

@st.cache_resource
def load_model():
    model_path = project_root / "models" / "engagement_predictor.pkl"
    if not model_path.exists():
        st.error(f"Model not found at: {model_path}\nPlease run the training pipeline first.")
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def convert_claps(clap_str):
    """Convert clap strings (like '1.2K') to integers"""
    if isinstance(clap_str, (int, float)):
        return int(clap_str)
    clap_str = str(clap_str).upper().replace(',', '')
    if 'K' in clap_str:
        return int(float(clap_str.replace('K', ''))) * 1000
    try:
        return int(clap_str)
    except:
        return 0

def show_data_visualizations(conn):
    """Display data visualizations including top articles and distributions"""
    try:
        # Load all data
        query = "SELECT title, claps, responses, author_name, reading_time_mins FROM articles"
        df = pd.read_sql(query, conn)
        
        # Convert claps and responses to numeric
        df['claps'] = df['claps'].apply(convert_claps)
        df['responses'] = pd.to_numeric(df['responses'], errors='coerce').fillna(0).astype(int)
        
        if df.empty:
            st.warning("No data found in database")
            return
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Top Articles", "Distributions"])
        
        with tab1:
            # Show top 10 articles by claps
            st.subheader("Top 10 Articles by Claps")
            top_claps = df.sort_values('claps', ascending=False).head(10)
            
            # Create display dataframe with proper formatting
            display_claps = top_claps.copy()
            display_claps['claps'] = display_claps['claps'].apply(lambda x: f"{x:,}")
            display_claps['responses'] = display_claps['responses'].apply(lambda x: f"{x:,}")
            
            # Rename columns for display
            display_claps = display_claps.rename(columns={
                'title': 'Title',
                'claps': 'Claps',
                'responses': 'Responses',
                'author_name': 'Author',
                'reading_time_mins': 'Reading Time (mins)'
            })
            
            # Display the table
            st.dataframe(display_claps[[
                'Title', 
                'Claps', 
                'Responses', 
                'Author', 
                'Reading Time (mins)'
            ]])
            
            # Show top 10 articles by responses
            st.subheader("Top 10 Articles by Responses")
            top_responses = df.sort_values('responses', ascending=False).head(10)
            
            # Create display dataframe with proper formatting
            display_responses = top_responses.copy()
            display_responses['claps'] = display_responses['claps'].apply(lambda x: f"{x:,}")
            display_responses['responses'] = display_responses['responses'].apply(lambda x: f"{x:,}")
            
            # Rename columns for display
            display_responses = display_responses.rename(columns={
                'title': 'Title',
                'claps': 'Claps',
                'responses': 'Responses',
                'author_name': 'Author',
                'reading_time_mins': 'Reading Time (mins)'
            })
            
            # Display the table
            st.dataframe(display_responses[[
                'Title', 
                'Responses', 
                'Claps', 
                'Author', 
                'Reading Time (mins)'
            ]])
        
        with tab2:
            # Claps distribution visualization
            st.subheader("Claps Distribution")
            
            # Create bins for better visualization
            clap_bins = [0, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, np.inf]
            clap_labels = ['0-100', '101-500', '501-1K', '1K-2K', '2K-5K', '5K-10K', '10K-20K', '20K-50K', '50K-100K', '100K+']
            df['clap_range'] = pd.cut(df['claps'], bins=clap_bins, labels=clap_labels, right=False)
            
            # Plot histogram
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histogram of clap ranges
            clap_counts = df['clap_range'].value_counts().sort_index()
            ax1.bar(clap_counts.index.astype(str), clap_counts.values)
            ax1.set_title("Clap Count Frequency")
            ax1.set_xlabel("Clap Range")
            ax1.set_ylabel("Number of Articles")
            ax1.tick_params(axis='x', rotation=45)
            
            # Box plot (log scale)
            ax2.boxplot(np.log1p(df['claps']))
            ax2.set_title("Clap Distribution (log scale)")
            ax2.set_ylabel("log(claps + 1)")
            
            st.pyplot(fig1)
            
            # Responses distribution visualization
            st.subheader("Responses Distribution")
            
            # Create bins for responses
            response_bins = [0, 5, 10, 20, 50, 100, 200, 500, 1000, np.inf]
            response_labels = ['0-5', '6-10', '11-20', '21-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
            df['response_range'] = pd.cut(df['responses'], bins=response_bins, labels=response_labels, right=False)
            
            # Plot histogram
            fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histogram of response ranges
            response_counts = df['response_range'].value_counts().sort_index()
            ax3.bar(response_counts.index.astype(str), response_counts.values)
            ax3.set_title("Response Count Frequency")
            ax3.set_xlabel("Response Range")
            ax3.set_ylabel("Number of Articles")
            ax3.tick_params(axis='x', rotation=45)
            
            # Box plot (log scale)
            ax4.boxplot(np.log1p(df['responses']))
            ax4.set_title("Response Distribution (log scale)")
            ax4.set_ylabel("log(responses + 1)")
            
            st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")

def main():
    st.title("Medium Author Success Predictor")
    
    # Load resources
    model = load_model()
    conn = get_db_connection()
    
    # Sidebar options
    st.sidebar.header("Options")
    show_visualizations = st.sidebar.checkbox("Data Visualizations")
    make_predictions = st.sidebar.checkbox("Make Predictions")
    
    # Data Visualizations
    if show_visualizations:
        st.header("Data Visualizations")
        show_data_visualizations(conn)
    
    # Prediction Section
    if make_predictions:
        st.header("Article Engagement Predictor")
        
        if not model:
            st.warning("Model not available - cannot make predictions")
            return
            
        # Prediction input
        title_input = st.text_area(
            "Enter article title:", 
            "How AI is transforming business in 2025",
            height=100
        )
        
        if st.button("Predict Engagement"):
            with st.spinner("Analyzing..."):
                try:
                    prediction = model.predict([title_input])[0]
                    proba = model.predict_proba([title_input])[0][1]
                    
                    if prediction == 1:
                        st.success(f"✅ High engagement predicted ({proba:.0%} confidence)")
                    else:
                        st.warning(f"⚠️ Low engagement predicted ({proba:.0%} confidence)")
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
        
        # Example predictions
        st.subheader("Example Predictions")
        examples = [
            "10 Business Trends That Will Dominate 2025",
            "My Personal Journey as a Startup Founder",
            "Why AI Won't Take Your Job (Probably)",
            "The Complete Guide to Machine Learning in 2025"
        ]
        
        for example in examples:
            try:
                pred = model.predict([example])[0]
                proba = model.predict_proba([example])[0][1]
                emoji = "🔥" if pred == 1 else "💤"
                st.write(f"{emoji} {example} → {'High' if pred == 1 else 'Low'} engagement ({proba:.0%})")
            except:
                st.write(f"⚠️ Could not process example: {example}")

if __name__ == "__main__":
    main()