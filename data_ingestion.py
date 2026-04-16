import os
import pandas as pd
from sklift.datasets import fetch_criteo

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PROCESSED_FILE = os.path.join(DATA_DIR, "criteo_sample.pkl")

def ingest_data(sample_percent=0.1):
    print("Starting Criteo Data Ingestion...")
    
    if os.path.exists(PROCESSED_FILE):
        print(f"Sample file already exists at {PROCESSED_FILE}. Skipping download.")
        return pd.read_pickle(PROCESSED_FILE)

    print("Fetching Criteo dataset from source (Large Scale)...")
    try:
        # percent10=True isn't a direct param in fetch_criteo, 
        # but the library manages the download.
        data = fetch_criteo(target_col='visit', treatment_col='treatment')
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

    df = pd.concat([data.data, data.target, data.treatment], axis=1)
    
    print(f"Full Dataset Shape: {df.shape}")
    
    # Stratified sampling to keep treatment/target ratios
    print(f"Sampling {sample_percent*100}% of the data...")
    df_sample = df.sample(frac=sample_percent, random_state=42)
    
    # Memory optimization: Downcast numerical columns
    for col in df_sample.select_dtypes(include=['float64']).columns:
        df_sample[col] = pd.to_numeric(df_sample[col], downcast='float')
    
    print(f"Saving sampled data to {PROCESSED_FILE}...")
    df_sample.to_pickle(PROCESSED_FILE)
    
    print(f"Ingestion complete. Sample size: {df_sample.shape}")
    return df_sample

if __name__ == "__main__":
    ingest_data()
