import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger("BotnetDetection.DNSPreprocessing")

def prepare_and_balance_data(df_benign, df_dns2tcp, df_iodine, domain_col):
    """
    Drops nulls, duplicates, converts domain to string.
    Merges datasets and balances them (50k each).
    """
    try:
        # Pre-process individually before merging to save memory
        logger.info("Cleaning individual datasets...")
        
        for df in [df_benign, df_dns2tcp, df_iodine]:
            df.dropna(subset=[domain_col], inplace=True)
            df.drop_duplicates(subset=[domain_col], inplace=True)
            df[domain_col] = df[domain_col].astype(str)

        logger.info("Merging malicious datasets...")
        df_malicious = pd.concat([df_dns2tcp, df_iodine], ignore_index=True)
        # Drop duplicates again just in case overlap between malicious datasets
        df_malicious.drop_duplicates(subset=[domain_col], inplace=True)

        logger.info(f"Available Benign: {len(df_benign)}, Available Malicious: {len(df_malicious)}")
        
        # Sampling 50k from each
        sample_size = 50000
        
        # Ensure we don't sample more than available
        b_size = min(sample_size, len(df_benign))
        m_size = min(sample_size, len(df_malicious))
        
        if b_size < sample_size or m_size < sample_size:
            logger.warning(f"Could not reach {sample_size} samples. Using Benign: {b_size}, Malicious: {m_size}")
            
        df_benign_sampled = df_benign.sample(n=b_size, random_state=42)
        df_malicious_sampled = df_malicious.sample(n=m_size, random_state=42)
        
        df_final = pd.concat([df_benign_sampled, df_malicious_sampled], ignore_index=True)
        
        # Shuffle
        df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Final dataset size: {len(df_final)}. Class distribution:\n{df_final['label'].value_counts()}")
        
        return df_final

    except Exception as e:
        logger.error(f"Error in data balancing: {e}")
        raise

def split_and_scale(df_final, feature_cols):
    """
    Splits the data 80-20 and applies StandardScaler on feature columns.
    """
    try:
        X = df_final[feature_cols]
        y = df_final['label']
        
        logger.info("Performing 80-20 train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        logger.info("Applying StandardScaler to numerical features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling (e.g., SHAP)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error in split and scale: {e}")
        raise
