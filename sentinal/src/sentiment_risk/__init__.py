import os
import sys

# Define the root of the project
PROJECT_ROOT = r"E:\sentinal"

# The Canonical Project Structure from the Base Spec
STRUCTURE = [
    "src/ingestion",
    "src/systemic_risk",
    "src/credit_risk",
    "src/sentiment_risk",
    "src/explainability",
    "src/aggregation",
    "src/schemas",
    "configs",
    "data/raw",
    "data/processed",
    "data/sample",
    "notebooks",
    "outputs/scores",
    "outputs/explanations",
    "outputs/logs",
]

# Files to create (with empty content or comments)
FILES = {
    "src/ingestion/__init__.py": "",
    "src/ingestion/loaders.py": "# Data loading logic here",
    "src/ingestion/validators.py": "# Data validation logic here",
    "src/ingestion/normalizers.py": "# Data normalization logic here",
    
    "src/systemic_risk/__init__.py": "",
    "src/systemic_risk/graph_builder.py": "# NetworkX graph construction",
    "src/systemic_risk/centrality.py": "# Eigenvector/PageRank logic",
    "src/systemic_risk/contagion.py": "# Network failure simulation",
    
    "src/credit_risk/__init__.py": "",
    "src/credit_risk/features.py": "# Financial feature engineering",
    "src/credit_risk/monotonic_xgb.py": "# Constrained XGBoost model",
    "src/credit_risk/calibration.py": "# Probability calibration",
    "src/credit_risk/scoring.py": "# Final credit scoring",
    
    "src/sentiment_risk/__init__.py": "",
    "src/sentiment_risk/news_loader.py": "# News text ingestion",
    "src/sentiment_risk/finbert_inference.py": "# HuggingFace model inference",
    "src/sentiment_risk/stress_overlay.py": "# Sentiment shock logic",
    
    "src/explainability/__init__.py": "",
    "src/explainability/shap_engine.py": "# SHAP value calculation",
    "src/explainability/audit_artifacts.py": "# Audit trail generation",
    
    "src/aggregation/__init__.py": "",
    "src/aggregation/risk_fusion.py": "# Weighted signal combination",
    
    "src/schemas/__init__.py": "",
    "src/schemas/risk_objects.py": "# dataclasses for RiskSignal",
    
    "configs/model_config.yaml": "# Hyperparameters",
    "configs/monotonic_constraints.yaml": "# Feature constraints",
    
    "notebooks/sentinal_runner.ipynb": "# Orchestrator Notebook",
}

def setup():
    print(f"🚀 Initializing SentinAL in {PROJECT_ROOT}...")
    
    # 1. Create Directories
    for folder in STRUCTURE:
        path = os.path.join(PROJECT_ROOT, folder)
        os.makedirs(path, exist_ok=True)
        # Create an __init__.py in every directory to make it a package
        init_path = os.path.join(path, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write("")
        print(f"   [OK] Created dir: {folder}")

    # 2. Create Files
    for file_path, content in FILES.items():
        full_path = os.path.join(PROJECT_ROOT, file_path)
        if not os.path.exists(full_path):
            with open(full_path, "w") as f:
                f.write(content)
            print(f"   [OK] Created file: {file_path}")
        else:
            print(f"   [SKIP] Exists: {file_path}")

    print("\n✅ Project Structure Ready! You can now start coding.")

if __name__ == "__main__":
    setup()