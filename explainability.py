import shap
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_uplift_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "criteo_sample.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "models", "shap_summary.joblib")

def generate_explanations():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("Error: Model or Data not found.")
        return

    print("Loading assets for explainability...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_pickle(DATA_PATH)
    
    # Take a small sample for SHAP (SHAP is slow)
    X = df.drop(['visit', 'treatment', 'exposure', 'conversion'], axis=1, errors='ignore').head(500)
    
    print("Calculating SHAP values (this may take a minute)...")
    # For ClassTransformation in sklift, we explain the underlying estimator
    # or the predict method. Since sklift wraps the model, we can explain the wrapper's predict.
    
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)
    
    print(f"Saving SHAP summary to {OUTPUT_PATH}...")
    joblib.dump(shap_values, OUTPUT_PATH)
    
    # Optional: Save a static plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title("Uplift Feature Importance (Global)")
    plt.savefig(os.path.join(BASE_DIR, "models", "shap_plot.png"), bbox_inches='tight')
    
    print("Explainability assets generated successfully.")

if __name__ == "__main__":
    generate_explanations()
