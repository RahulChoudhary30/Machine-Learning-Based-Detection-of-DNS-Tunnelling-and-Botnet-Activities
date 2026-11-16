import pandas as pd
import numpy as np
import math
import logging

logger = logging.getLogger("BotnetDetection.DNSFeatureEngineering")

def calculate_entropy(s):
    """
    Manually calculates Shannon entropy of a string.
    Handles edge cases like empty strings safely.
    """
    if not isinstance(s, str) or len(s) == 0:
        return 0.0
        
    try:
        probabilities = [float(s.count(c)) / len(s) for c in set(s)]
        entropy = -sum(p * math.log(p) / math.log(2.0) for p in probabilities)
        return entropy
    except Exception as e:
        # In case of any strange math errors, return 0
        return 0.0

def engineer_dns_features(df, domain_col):
    """
    Extracts numerical features from the domain column.
    """
    try:
        logger.info(f"Extracting features from '{domain_col}'...")
        
        # 1. domain_length
        df['domain_length'] = df[domain_col].apply(lambda x: len(str(x)))
        
        # 2. num_subdomains
        # Assuming subdomains are separated by dots. E.g. a.b.com -> 3 parts -> 2 subdomains? Or just count dots.
        df['num_subdomains'] = df[domain_col].apply(lambda x: str(x).count('.'))
        
        # 3. entropy
        df['entropy'] = df[domain_col].apply(calculate_entropy)
        
        # 4. num_digits
        df['num_digits'] = df[domain_col].apply(lambda x: sum(c.isdigit() for c in str(x)))
        
        # 5. digit_ratio
        df['digit_ratio'] = df.apply(lambda row: row['num_digits'] / row['domain_length'] if row['domain_length'] > 0 else 0, axis=1)
        
        # 6. unique_char_count
        df['unique_char_count'] = df[domain_col].apply(lambda x: len(set(str(x))))
        
        # 7. longest_label_length
        # labels are separated by dots. e.g. www.google.com -> www, google, com
        df['longest_label_length'] = df[domain_col].apply(lambda x: max([len(label) for label in str(x).split('.')] + [0]))
        
        feature_cols = [
            'domain_length', 'num_subdomains', 'entropy', 
            'num_digits', 'digit_ratio', 'unique_char_count', 'longest_label_length'
        ]
        
        logger.info(f"Feature engineering complete. Extracted {len(feature_cols)} features.")
        return df, feature_cols
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise
