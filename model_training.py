import os
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.utils import setup_logger
from sklearn.pipeline import Pipeline

logger = setup_logger()

def get_models(config):
    """
    Returns a dictionary of instantiated models based on configuration.
    """
    random_state = config['training'].get('random_state', 42)
    models_to_train = config['training'].get('models_to_train', [])
    
    models = {}
    
    # Always enforce balanced class weight to prioritize minority class
    class_weight = 'balanced'
        
    if "RandomForest" in models_to_train:
        models["RandomForest"] = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight=class_weight, n_jobs=-1)
        
    if "GradientBoosting" in models_to_train:
        models["GradientBoosting"] = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        
    if "LogisticRegression" in models_to_train:
        models["LogisticRegression"] = LogisticRegression(max_iter=1000, random_state=random_state, class_weight=class_weight)
        
    return models

def train_and_log_models(models, preprocessor, X_train, y_train, config):
    """
    Trains models with MLFlow tracking and returns the trained pipelines.
    Expects X_train, y_train after SMOTE (if applicable).
    """
    mlflow.set_tracking_uri(config['training']['mlflow_tracking_uri'])
    mlflow.set_experiment("Botnet_Detection_Experiment")
    
    trained_pipelines = {}
    models_dir = config["paths"]["models_dir"]
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        with mlflow.start_run(run_name=name):
            # We don't include preprocessor in the pipeline here because SMOTE cannot be in standard Pipeline
            # We assume X_train is already preprocessed OR we create an imblearn Pipeline.
            # To keep things clean, we will train the model directly on the SMOTE'd features.
            # The full serving pipeline needs to be saved carefully.
            
            mlflow.log_param("imbalance_strategy", config['preprocessing'].get('imbalance_strategy'))
            mlflow.log_params(model.get_params())
            
            if name == "IsolationForest":
                # Unsupervised
                model.fit(X_train)
            else:
                model.fit(X_train, y_train)
                
            trained_pipelines[name] = model
            
            # Save the model
            model_path = os.path.join(models_dir, f"{name}.joblib")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)
            
            logger.info(f"{name} trained and saved locally.")
            
    return trained_pipelines

def load_model(name, config):
    """Utility to load a trained model from disk."""
    model_path = os.path.join(config["paths"]["models_dir"], f"{name}.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model {name} not found at {model_path}")
