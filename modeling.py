import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklift.models import SoloModel, TwoModels, ClassTransformation
from sklift.metrics import uplift_at_k, qini_auc_score

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SAMPLE_FILE = os.path.join(DATA_DIR, "criteo_sample.pkl")

def train_and_evaluate():
    if not os.path.exists(SAMPLE_FILE):
        print("Error: Sample data not found. Run data_ingestion.py first.")
        return

    print("Loading sampled data...")
    df = pd.read_pickle(SAMPLE_FILE)
    
    # Features, Treatment, Target
    X = df.drop(['visit', 'treatment', 'exposure', 'conversion'], axis=1, errors='ignore')
    y = df['visit']
    treat = df['treatment']
    
    X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(
        X, y, treat, test_size=0.2, random_state=42, stratify=treat
    )
    
    print("Training Meta-Learners (Base: XGBoost)...")
    
    # 1. S-Learner (Solo Model)
    print("   Training S-Learner...")
    sm = SoloModel(XGBClassifier(n_estimators=30, max_depth=4, random_state=42))
    sm.fit(X_train, y_train, treat_train)
    
    # 2. T-Learner (Two Models)
    print("   Training T-Learner...")
    tm = TwoModels(
        estimator_trmnt=XGBClassifier(n_estimators=30, max_depth=4, random_state=42),
        estimator_ctrl=XGBClassifier(n_estimators=30, max_depth=4, random_state=42)
    )
    tm.fit(X_train, y_train, treat_train)
    
    # 3. Class Transformation (Advanced meta-learner)
    print("   Training ClassTransformation...")
    ct = ClassTransformation(XGBClassifier(n_estimators=30, max_depth=4, random_state=42))
    ct.fit(X_train, y_train, treat_train)
    
    print("Evaluating models on test set...")
    
    results = {}
    for name, model in [("S-Learner", sm), ("T-Learner", tm), ("ClassTransformation", ct)]:
        uplift_preds = model.predict(X_test)
        qini_score = qini_auc_score(y_test, uplift_preds, treat_test)
        uplift_k = uplift_at_k(y_test, uplift_preds, treat_test, strategy='overall', k=0.3)
        results[name] = {"Qini": qini_score, "Uplift@30%": uplift_k}
        print(f"Results for {name}: Qini={qini_score:.4f}, Uplift@30%={uplift_k:.4f}")
        
        # Save ClassTransformation as the production model
        if name == "ClassTransformation":
            model_path = os.path.join(MODEL_DIR, "best_uplift_model.pkl")
            joblib.dump(model, model_path)
            print(f"Production model saved to {model_path}")

    return results

if __name__ == "__main__":
    train_and_evaluate()
