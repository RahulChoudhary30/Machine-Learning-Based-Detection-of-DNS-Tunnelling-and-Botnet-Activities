import pandas as pd
import os
import logging

logger = logging.getLogger("BotnetDetection.DNSDataLoader")

def load_dns_data(benign_path, dns2tcp_path, iodine_path):
    """
    Loads benign and malicious DNS datasets, assigns labels, and merges them.
    Returns: benign_df, malicious_df combined.
    """
    try:
        logger.info(f"Loading benign dataset from: {benign_path}")
        df_benign = pd.read_csv(benign_path)
        df_benign['label'] = 0

        logger.info(f"Loading malicious dataset (dns2tcp) from: {dns2tcp_path}")
        df_dns2tcp = pd.read_csv(dns2tcp_path)
        df_dns2tcp['label'] = 1

        logger.info(f"Loading malicious dataset (iodine) from: {iodine_path}")
        df_iodine = pd.read_csv(iodine_path)
        df_iodine['label'] = 1
        
        return df_benign, df_dns2tcp, df_iodine

    except Exception as e:
        logger.error(f"Error loading DNS datasets: {e}")
        raise

def detect_domain_column(df):
    """
    Detects the domain column from common names and validates it.
    """
    possible_names = ['query', 'qname', 'domain', 'fqdn', 'dns_domain_name']
    
    # Try to find a match (case-insensitive)
    selected_col = None
    df_cols_lower = [col.lower() for col in df.columns]
    
    # Exact match first
    for name in possible_names:
        if name in df_cols_lower:
            original_col_name = df.columns[df_cols_lower.index(name)]
            selected_col = original_col_name
            break
            
    # Substring match if exact fails
    if not selected_col:
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in possible_names):
                selected_col = col
                break
            
    if not selected_col:
        logger.error(f"Could not find domain column. Available columns: {list(df.columns)}")
        raise ValueError("Domain column not found.")
        
    logger.info(f"Selected Domain Column: '{selected_col}'")
    
    # Validation: Ensure it's not purely numeric
    if pd.api.types.is_numeric_dtype(df[selected_col]):
        logger.warning(f"Column '{selected_col}' appears to be strictly numeric. It might not be a valid domain column.")
        
    # Print first 5 values for verification
    first_5 = df[selected_col].dropna().astype(str).head(5).tolist()
    logger.info(f"First 5 values of '{selected_col}': {first_5}")
    print(f"Verified Domain Column: {selected_col} | Sample values: {first_5}")
    
    return selected_col
