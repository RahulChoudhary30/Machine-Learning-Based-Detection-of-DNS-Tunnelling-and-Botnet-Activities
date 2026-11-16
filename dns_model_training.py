import pandas as pd
import numpy as np
import os
import joblib
import logging
try:
    import shap
except ImportError as e:
    shap = None
    print(f"Warning: SHAP could not be imported. SHAP explainability will be skipped. Error: {e}")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

logger = logging.getLogger("BotnetDetection.DNSModelTraining")

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_cols):
    """
    Trains RF, GB, LR. Tunes threshold for higher recall.
    Evaluates and saves plots.
    Returns the best model and its name.
    """
    models = {
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic_Regression": LogisticRegression(random_state=42, max_iter=1000)
    }

    results = []
    best_f2_score = -1
    best_model = None
    best_model_name = ""
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Store data for ROC and PR curves
    curve_data = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predict probabilities
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # Threshold tuning (optimize for F2 score to favor recall)
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
        
        # F2 score: Beta = 2, favors recall
        f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
        best_idx = np.argmax(f2_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        logger.info(f"{name} best threshold for F2 (recall-focused): {best_threshold:.4f}")
        
        y_pred = (y_probs >= best_threshold).astype(int)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "Threshold": best_threshold,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1
        })
        
        logger.info(f"{name} -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
        
        # Save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"results/cm_{name}.png")
        plt.close()
        
        # Store curve data
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        curve_data[name] = {
            "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
            "precisions": precisions, "recalls": recalls
        }
        
        # Track best model (using F2 score index we found)
        if f2_scores[best_idx] > best_f2_score:
            best_f2_score = f2_scores[best_idx]
            best_model = model
            best_model_name = name

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/dns_model_comparison.csv", index=False)
    logger.info("Saved model comparison to 'results/dns_model_comparison.csv'")
    
    # Plot ROC curve for all models
    plt.figure(figsize=(8, 6))
    for name, data in curve_data.items():
        plt.plot(data['fpr'], data['tpr'], label=f"{name} (AUC = {data['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("results/roc_curve_comparison.png")
    plt.close()

    # Plot PR curve for all models
    plt.figure(figsize=(8, 6))
    for name, data in curve_data.items():
        plt.plot(data['recalls'], data['precisions'], label=name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig("results/pr_curve_comparison.png")
    plt.close()

    # Save best model
    model_path = "models/dns_model.joblib"
    joblib.dump(best_model, model_path)
    logger.info(f"Saved best model ({best_model_name}) to {model_path}")
    
    return best_model, best_model_name


def explain_model_shap(model, X_train, feature_cols, model_name):
    """
    Generate SHAP feature importance plot using a 1000-sample subset.
    """
    if shap is None:
        logger.warning("SHAP is not available. Skipping explainability.")
        return
        
    logger.info(f"Generating SHAP explanations for {model_name}...")
    try:
        # Sample subset for SHAP to avoid performance issues
        X_shap = shap.sample(X_train, 1000)
        
        # Depending on model type, we need different explainers
        if hasattr(model, "predict_proba") and "Logistic" not in model_name:
            # Tree-based
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            # Sometimes TreeExplainer returns a list (for classification), sometimes an array
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # Linear / general
            explainer = shap.KernelExplainer(model.predict, X_shap)
            shap_values = explainer.shap_values(X_shap)

        # Plot
        plt.figure()
        shap.summary_plot(shap_values, X_shap, feature_names=feature_cols, show=False)
        plt.title(f"SHAP Feature Importance ({model_name})")
        plt.tight_layout()
        plt.savefig(f"results/shap_summary_{model_name}.png")
        plt.close()
        
        logger.info(f"Saved SHAP plot to results/shap_summary_{model_name}.png")
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {e}")
