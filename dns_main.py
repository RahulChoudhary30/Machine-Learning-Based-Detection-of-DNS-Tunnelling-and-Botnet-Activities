import os
import logging
from src.utils import setup_logger
from src.dns_data_loader import load_dns_data, detect_domain_column
from src.dns_preprocessing import prepare_and_balance_data, split_and_scale
from src.dns_feature_engineering import engineer_dns_features
from src.dns_model_training import train_and_evaluate_models, explain_model_shap

# Setup logger matching the main pipeline style
logger = setup_logger("pipeline.log")

def main():
    logger.info("===================================================")
    logger.info("Initializing DNS Tunneling Detection Pipeline...")
    logger.info("===================================================")
    
    # Define paths
    base_dir = "archive1"
    benign_path = os.path.join(base_dir, "benign", "benign.csv")
    dns2tcp_path = os.path.join(base_dir, "malicious", "dns2tcp.csv")
    iodine_path = os.path.join(base_dir, "malicious", "iodine.csv")
    
    try:
        # 1. Data Loading
        logger.info("Step 1: Loading Datasets")
        df_benign, df_dns2tcp, df_iodine = load_dns_data(benign_path, dns2tcp_path, iodine_path)
        
        # Detect domain column on benign (assuming all have same schema, but we can just use benign to detect)
        # However, to be safe we'll detect it on the combined or benign.
        domain_col = detect_domain_column(df_benign)
        
        # 2. Preprocessing & Balancing (Before Feature Engineering, to save time on useless rows)
        logger.info("Step 2: Preprocessing and Balancing")
        df_balanced = prepare_and_balance_data(df_benign, df_dns2tcp, df_iodine, domain_col)
        
        # 3. Feature Engineering
        logger.info("Step 3: Feature Engineering")
        df_features, feature_cols = engineer_dns_features(df_balanced, domain_col)
        logger.info(f"Features used for training: {feature_cols}")
        
        # 4. Splitting and Scaling
        logger.info("Step 4: Train-Test Split and Scaling")
        X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale(df_features, feature_cols)
        
        # 5. Model Training & Evaluation
        logger.info("Step 5: Model Training, Threshold Tuning, and Evaluation")
        best_model, best_model_name = train_and_evaluate_models(
            X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
        )
        
        # 6. SHAP Explainability
        logger.info("Step 6: Model Explainability (SHAP)")
        explain_model_shap(best_model, X_train_scaled, feature_cols, best_model_name)
        
        logger.info("===================================================")
        logger.info("DNS Pipeline Execution Completed Successfully.")
        logger.info("===================================================")

    except Exception as e:
        logger.error(f"DNS Pipeline crashed cleanly: {str(e)}", exc_info=True)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore') # Clean output
    main()
