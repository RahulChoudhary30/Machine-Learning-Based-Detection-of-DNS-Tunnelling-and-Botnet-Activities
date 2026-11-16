import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from src.utils import setup_logger

logger = setup_logger()

def clean_labels(df, config):
    """
    Converts labels to binary 1 (Botnet) and 0 (Normal).
    Drops unnecessary columns.
    """
    logger.info("Cleaning labels...")
    target_col = config['data']['target_col']
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
        
    # Example logic: if 'Botnet' is in the string label, it's 1, else 0
    # The CTU-13 dataset typically has labels like 'Flows_Background', 'Botnet', 'Normal'
    df['binary_label'] = df[target_col].astype(str).str.contains('Botnet', case=False, na=False).astype(int)
    
    # Drop old target and unused features
    cols_to_drop = config['features'].get('drop', []) + [target_col]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Remove rows with null labels (as a sanity check)
    df.dropna(subset=['binary_label'], inplace=True)
    
    return df

def get_preprocessor(config):
    """
    Creates and returns a scikit-learn preprocessing pipeline.
    """
    numeric_features = config['features']['numerical']
    categorical_features = config['features']['categorical']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    return preprocessor

def apply_smote(X, y, config):
    """
    Applies SMOTE if configured.
    Expects already transformed features because SMOTE doesn't handle string categoricals well without encoding.
    """
    strategy = config['preprocessing'].get('imbalance_strategy', 'none')
    
    if strategy == 'smote':
        logger.info("Applying SMOTE to handle class imbalance...")
        
        # Log distribution before SMOTE
        if isinstance(y, pd.Series):
            logger.info(f"Class distribution BEFORE SMOTE:\n{y.value_counts().to_dict()}")
        else:
            logger.info(f"Class distribution BEFORE SMOTE:\n{pd.Series(y).value_counts().to_dict()}")

        smote = SMOTE(random_state=config['training'].get('random_state', 42))
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(f"Original shape: {X.shape}. Resampled shape: {X_res.shape}")
        
        # Log distribution after SMOTE
        if isinstance(y_res, pd.Series):
            logger.info(f"Class distribution AFTER SMOTE:\n{y_res.value_counts().to_dict()}")
        else:
            logger.info(f"Class distribution AFTER SMOTE:\n{pd.Series(y_res).value_counts().to_dict()}")
            
        return X_res, y_res
        
    logger.info("No SMOTE applied.")
    return X, y
