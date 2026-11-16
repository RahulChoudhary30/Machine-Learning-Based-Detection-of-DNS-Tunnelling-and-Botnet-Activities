import os
import argparse
from src.utils import load_config, setup_logger
from src.data_loader import load_and_split_data
from src.feature_engineering import engineer_features
from src.preprocessing import clean_labels, get_preprocessor, apply_smote
from src.model_training import get_models, train_and_log_models
from src.evaluation import evaluate_models
from src.explainability import explain_model

logger = setup_logger()

def main():
    logger.info("Initializing Botnet Detection Pipeline...")
    config = load_config("config.yaml")
    
    # Ensure dirs exist
    os.makedirs(config["paths"]["data_dir"], exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. Data Loading & Splitting (File-based)
    logger.info("Step 1: Loading Data (File-based split)")
    train_df, test_df = load_and_split_data(config)
    
    # 2. Feature Engineering
    logger.info("Step 2: Feature Engineering")
    train_df = engineer_features(train_df, config)
    test_df = engineer_features(test_df, config)
    
    # 3. Label Cleaning
    logger.info("Step 3: Cleaning Labels")
    train_df = clean_labels(train_df, config)
    test_df = clean_labels(test_df, config)
    
    # Split features and target
    X_train_raw = train_df.drop(columns=['binary_label'])
    y_train = train_df['binary_label']
    
    X_test_raw = test_df.drop(columns=['binary_label'])
    y_test = test_df['binary_label']
    
    # 4. Preprocessing Pipeline
    logger.info("Step 4: Preprocessing (Imputation & Scaling)")
    preprocessor = get_preprocessor(config)
    
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    
    # Get feature names after preprocessing for SHAP
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features = list(cat_encoder.get_feature_names_out(config['features']['categorical']))
    num_features = config['features']['numerical']
    feature_names = num_features + cat_features
    
    # 5. Handling Class Imbalance
    logger.info("Step 5: Handling Imbalance")
    X_train_bal, y_train_bal = apply_smote(X_train_proc, y_train, config)
    
    # 6. Model Training
    logger.info("Step 6: Model Training")
    models = get_models(config)
    trained_models = train_and_log_models(models, preprocessor, X_train_bal, y_train_bal, config)
    
    # 7. Evaluation
    logger.info("Step 7: Evaluation")
    results_df = evaluate_models(trained_models, X_test_proc, y_test, config)
    print("\n==== Final Performance ====")
    print(results_df.to_string(index=False))
    
    # 8. Explainability
    logger.info("Step 8: Model Explainability (SHAP)")
    for name, model in trained_models.items():
        explain_model(model, X_train_bal, X_test_proc, name, feature_names, config)
        
    import sys
    if "--research" in sys.argv:
        from src.research_addon import run_research_addons
        X_test = X_test_proc
        df = test_df
        run_research_addons(model, X_test, y_test, df)

    logger.info("Pipeline Execution Completed Successfully.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore') # Disable general warnings for cleaner output
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline crashed cleanly: {str(e)}", exc_info=True)
