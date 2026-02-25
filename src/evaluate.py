import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_creditcard_data
from preprocess import preprocess_data
from train_model import train_model
from sklearn.metrics import precision_score, recall_score, confusion_matrix

if __name__ == "__main__":
    # Load and preprocess
    df = load_creditcard_data()
    X, y = preprocess_data(df)
    model = train_model(df)

    # ----------------------- BASELINE -----------------------
    baseline_pred = np.where(model.predict(X) == -1, 1, 0)
    tn, fp, fn, tp = confusion_matrix(y, baseline_pred).ravel()
    baseline_flagged = tp + fp
    baseline_precision = precision_score(y, baseline_pred)
    baseline_recall = recall_score(y, baseline_pred)

    print("========== BASELINE ==========")
    print(f"Fraud caught (TP): {tp}")
    print(f"False positives (FP): {fp}")
    print(f"Total flagged: {baseline_flagged}")
    print(f"Precision: {baseline_precision:.4f}")
    print(f"Recall: {baseline_recall:.4f}\n")

    # ----------------------- TIERED RISK -----------------------
    scores = model.decision_function(X)
    high_pct = 0.5
    med_pct = 1.5
    high_thresh = np.percentile(scores, high_pct)
    med_thresh = np.percentile(scores, med_pct)

    risk_tiers = []
    for s in scores:
        if s <= high_thresh:
            risk_tiers.append("High")
        elif s <= med_thresh:
            risk_tiers.append("Medium")
        else:
            risk_tiers.append("Low")

    df_results = pd.DataFrame({
        "y_true": y,
        "score": scores,
        "tier": risk_tiers
    })

    tier_counts = df_results['tier'].value_counts()
    fraud_per_tier = df_results.groupby('tier')['y_true'].sum()

    print("========== TIERED RISK ==========")
    print("Risk tier distribution:")
    print(tier_counts)
    print("\nFraud detected per tier:")
    print(fraud_per_tier)
    print("\nNote: High = auto-flag, Medium = manual review, Low = approve automatically\n")

    # ----------------------- OVERALL TIERED METRICS -----------------------
    tp_tiered = df_results[(df_results.tier.isin(["High","Medium"])) & (df_results.y_true==1)].shape[0]
    fp_tiered = df_results[(df_results.tier.isin(["High","Medium"])) & (df_results.y_true==0)].shape[0]
    total_flagged = tp_tiered + fp_tiered

    precision_tiered = tp_tiered / total_flagged if total_flagged > 0 else 0
    recall_tiered = tp_tiered / y.sum()

    print("========== OVERALL TIERED METRICS ==========")
    print(f"Fraud caught (TP): {tp_tiered}")
    print(f"False positives (FP): {fp_tiered}")
    print(f"Total flagged: {total_flagged}")
    print(f"Precision: {precision_tiered:.4f}")
    print(f"Recall: {recall_tiered:.4f}")
    print(f"Workload reduction: {(1 - total_flagged / baseline_flagged) * 100:.2f}%\n")

    # ----------------------- WORKLOAD VISUALIZATION -----------------------
    plt.figure(figsize=(6,4))
    plt.bar(
        ["Baseline Flagged", "Tiered Flagged"],
        [baseline_flagged, total_flagged],
        color=['gray', 'green']
    )
    plt.ylabel("Transactions Flagged")
    plt.title("Employee Workload: Baseline vs Tiered")
    plt.show()