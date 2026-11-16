import os
import shap
import matplotlib.pyplot as plt
import numpy as np
from src.utils import setup_logger

logger = setup_logger()

def explain_model(model, X_train, X_test, model_name, feature_names, config):
    """
    Generates SHAP values and plots to explain model predictions globally.
    X_train/X_test should be arrays or dataframes (preprocessed).
    """
    if not config['evaluation'].get('plot_shap', False):
        return

    logger.info(f"Generating SHAP explainability for {model_name}...")
    output_dir = config["paths"]["output_dir"]
    
    # We sample for performance because SHAP can be very slow
    sample_size = min(1000, X_test.shape[0])
    X_sample = X_test[:sample_size]
    
    try:
        if model_name in ["RandomForest", "GradientBoosting", "XGBoost", "IsolationForest"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # For tree explainers, binary classification may return a list of SHAP values
            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[1] # Choose positive class (Botnet)
            else:
                shap_values_to_plot = shap_values
                
            plt.figure()
            shap.summary_plot(shap_values_to_plot, X_sample, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary - {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_shap_summary.png"))
            plt.close()
            
        elif model_name in ["LogisticRegression"]:
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure()
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary - {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_shap_summary.png"))
            plt.close()
            
        logger.info(f"SHAP plots saved for {model_name}")
    except Exception as e:
        logger.error(f"Failed to generate SHAP for {model_name}. Error: {e}")
