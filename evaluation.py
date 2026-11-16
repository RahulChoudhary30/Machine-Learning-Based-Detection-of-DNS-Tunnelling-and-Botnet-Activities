import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve, 
                             precision_recall_curve, auc)
import mlflow
from src.utils import setup_logger

logger = setup_logger()

def evaluate_models(models, X_test, y_test, config):
    """
    Evaluates models, logs metrics to MLflow, and exports results.
    X_test should be already transformed by the preprocessor.
    """
    logger.info("Evaluating models...")
    output_dir = config["paths"]["output_dir"]
    results = []
    
    for name, model in models.items():
        best_threshold = 0.5
        if name == "IsolationForest":
            # IF outputs -1 (anomaly), 1 (normal)
            preds = model.predict(X_test)
            # Map back to 1: Botnet(Anomaly), 0: Normal
            y_pred = np.where(preds == -1, 1, 0)
            y_prob = -model.decision_function(X_test) # Inverse score for AUC
        else:
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
            
            # Use default threshold since models use class_weight='balanced'
            best_threshold = 0.5
            y_pred = (y_prob >= best_threshold).astype(int)
            logger.info(f"[{name}] Using default threshold: {best_threshold}")
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # False Positive Rate
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            roc_auc = np.nan
            
        logger.info(f"[{name}] Threshold:{best_threshold} Acc:{acc:.4f} P:{prec:.4f} R:{rec:.4f} F1:{f1:.4f} FPR:{fpr_val:.4f} AUC:{roc_auc:.4f}")
        logger.info(f"[{name}] Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Plot Precision-Recall
        plot_pr_curve(y_test, y_prob, name, output_dir)
        
        # Plot ROC
        plot_roc_curve(y_test, y_prob, name, output_dir)
        
        # Plot Confusion matrix
        plot_confusion_matrix(y_test, y_pred, name, output_dir)
        
        # We need to link back to the MLFlow run
        runs = mlflow.search_runs(experiment_names=["Botnet_Detection_Experiment"], filter_string=f"tags.mlflow.runName = '{name}'")
        if not runs.empty:
            run_id = runs.iloc[0].run_id
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics({"threshold": best_threshold, "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "fpr": fpr_val, "roc_auc": roc_auc})
        
        results.append({
            "Model": name,
            "Threshold": best_threshold,
            "Recall": rec,
            "F1_Score": f1,
            "Precision": prec,
            "Accuracy": acc,
            "FPR": fpr_val,
            "ROC_AUC": roc_auc
        })
        
    results_df = pd.DataFrame(results)
    if config['evaluation'].get('export_csv', True):
        csv_path = os.path.join(output_dir, "model_comparison.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Results exported to {csv_path}")
        
    return results_df

def plot_pr_curve(y_true, y_prob, model_name, output_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, lw=2, color='darkorange', label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, f"{model_name}_pr_curve.png"))
    plt.close()

def plot_roc_curve(y_true, y_prob, model_name, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()
