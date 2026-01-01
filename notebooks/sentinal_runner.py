import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import numpy as np

# ---------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aggregation.train_meta_model import FusionEngine

def print_header(title):
    print("\n" + "█"*60)
    print(f" {title}")
    print("█"*60)

print_header("SentinAL v3.1: Real-Data Simulation Mode")
fusion_system = FusionEngine()
print("Status: System Active.\n")

# ---------------------------------------------------------
# 2. DATA LOADING (The Realism Layer)
# ---------------------------------------------------------
def get_real_high_risk_company():
    """
    Loads the actual dataset and picks a REAL company that went bankrupt.
    """
    data_path = "data/processed/credit_clean.csv"
    if not os.path.exists(data_path):
        print("❌ Error: Processed data not found.")
        return None
        
    df = pd.read_csv(data_path)
    
    # Filter for companies that actually failed (target = 1)
    risky_companies = df[df['target'] == 1]
    
    if risky_companies.empty:
        print("⚠️ No bankrupt companies found in dataset. Using random row.")
        row = df.iloc[0]
    else:
        # Pick the first one (or random one)
        row = risky_companies.iloc[5] # Index 5 is usually a good distinct example
        
    print(f"✅ Loaded Real Entity from Row #{row.name}")
    
    # Separate features from target
    features = row.drop('target').to_dict()
    return features

# ---------------------------------------------------------
# 3. HYBRID LOGIC
# ---------------------------------------------------------
def enhanced_sentiment_analysis(text):
    signal = fusion_system.sentiment_engine.analyze(text)
    critical_keywords = ['scandal', 'fraud', 'breach', 'investigation', 'default']
    if any(k in text.lower() for k in critical_keywords):
        signal.normalized_score = 95.0
        signal.risk_level = "CRITICAL"
    return signal

# ---------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------
print_header("SCENARIO: Analyzing Real High-Risk Entity")

# A. LOAD REAL DATA
financials = get_real_high_risk_company()
entity_id = "REAL_COMP_005" # Placeholder ID for the real data row

# B. RUN ENGINES
# 1. Credit (Real Data)
credit_signal = fusion_system.credit_engine.analyze(entity_id, financials)

# 2. Systemic (Simulated for Demo - Graph data is separate)
systemic_signal = fusion_system.graph_engine.analyze(entity_id)
systemic_signal.normalized_score = 88.0 

# 3. Sentiment (Simulated context for this real financial case)
news_context = "Auditors raise doubts about ability to continue as going concern."
sentiment_signal = enhanced_sentiment_analysis(news_context)

# 4. FUSION
weights = {"credit": 0.5, "systemic": 0.3, "sentiment": 0.2}
composite_score = (
    (credit_signal.normalized_score * weights['credit']) +
    (systemic_signal.normalized_score * weights['systemic']) +
    (sentiment_signal.normalized_score * weights['sentiment'])
)

print(f"\n[-] Credit Risk (Real Data): {credit_signal.normalized_score:.1f}")
print(f"[-] Systemic Risk (Simulated): {systemic_signal.normalized_score:.1f}")
print(f"[-] Sentiment Risk (Context):  {sentiment_signal.normalized_score:.1f}")
print(f"[-] FINAL COMPOSITE SCORE:     {composite_score:.1f}/100")

# ---------------------------------------------------------
# 5. DASHBOARD 1: Risk Overview
# ---------------------------------------------------------
print_header("Generating Risk Dashboard...")
def plot_dashboard():
    data = {
        'Source': ['CREDIT', 'SYSTEMIC', 'SENTIMENT'],
        'Risk Score': [credit_signal.normalized_score, systemic_signal.normalized_score, sentiment_signal.normalized_score]
    }
    df_viz = pd.DataFrame(data)
    colors = ['#ff4d4d' if x > 70 else '#ffcc00' if x > 30 else '#66cc66' for x in df_viz['Risk Score']]
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    bars = plt.bar(df_viz['Source'], df_viz['Risk Score'], color=colors, edgecolor='black', alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.axhline(y=composite_score, color='#333333', linestyle='--', linewidth=2.5, label=f'Composite: {composite_score:.1f}')
    plt.title("SentinAL: Real-World Risk Analysis", fontsize=16)
    plt.ylabel("Risk Score (0-100)")
    plt.ylim(0, 110)
    plt.legend()
    plt.savefig("outputs/explanations/risk_dashboard_real.png", dpi=300)
    print("✅ Dashboard saved.")
    plt.show()

plot_dashboard()

# ---------------------------------------------------------
# 6. DASHBOARD 2: SHAP (The "Why")
# ---------------------------------------------------------
print_header("Generating SHAP Explainability Report...")

def explain_credit_decision(engine, features_dict):
    try:
        model = engine.model
        # Convert to DataFrame to match training format exactly
        X_instance = pd.DataFrame([features_dict])
        
        # Ensure columns match model expectation (crucial for SHAP)
        # We need to get the feature names from the model itself
        booster = model.get_booster()
        model_features = booster.feature_names
        
        # Reorder columns to match trained model
        X_instance = X_instance[model_features]
        
        print("   -> Calculating Shapley Values on Real Data...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_instance)
        
        print("   -> Rendering Waterfall Plot...")
        plt.figure()
        # This plot shows exactly which financial metrics pushed the risk UP (Red) or DOWN (Blue)
        shap.plots.waterfall(shap_values[0], show=False, max_display=12)
        
        plt.savefig("outputs/explanations/shap_explanation_real.png", bbox_inches='tight', dpi=300)
        print("✅ SHAP Explanation saved.")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")

explain_credit_decision(fusion_system.credit_engine, financials)