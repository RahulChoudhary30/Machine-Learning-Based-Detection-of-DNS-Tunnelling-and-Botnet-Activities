import os
import time
import math
from collections import Counter
import warnings

# Suppress sklearn warnings about feature names or unscaled data
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

# Configuration & Paths
BOTNET_MODEL_PATH = "models/RandomForest.joblib"
DNS_MODEL_PATH = "models/dns_model.joblib"

# Try to load models at startup
try:
    botnet_model = joblib.load(BOTNET_MODEL_PATH)
    dns_model = joblib.load(DNS_MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Please ensure you are in the correct directory and models are trained.")
    exit(1)

def calculate_entropy(s):
    """Calculates Shannon entropy of a string."""
    if not s:
        return 0.0
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log2(count/lns) for count in p.values())

def print_header(title, emoji):
    """Prints a styled header."""
    print("\n" + "=" * 55)
    print(f" {emoji}  {title}")
    print("=" * 55)

def dns_detection():
    print_header("DNS TUNNELING DETECTION", "🌐")
    domain = input("Enter domain name (e.g., google.com): ").strip()
    
    if not domain:
        print("❌ Invalid domain. Returning to menu.")
        time.sleep(1.5)
        return

    # Calculate features identical to the model's feature engineering step
    domain_length = len(domain)
    num_subdomains = domain.count('.')
    entropy = calculate_entropy(domain)
    num_digits = sum(c.isdigit() for c in domain)
    digit_ratio = num_digits / domain_length if domain_length > 0 else 0
    unique_char_count = len(set(domain))
    labels = domain.split('.')
    longest_label_length = max((len(label) for label in labels), default=0)

    # Feature DataFrame (required for models trained with feature names)
    feature_cols = [
        'domain_length', 'num_subdomains', 'entropy', 'num_digits', 
        'digit_ratio', 'unique_char_count', 'longest_label_length'
    ]
    df = pd.DataFrame([[
        domain_length, num_subdomains, entropy, num_digits, 
        digit_ratio, unique_char_count, longest_label_length
    ]], columns=feature_cols)

    # Predict probabilities
    try:
        probs = dns_model.predict_proba(df)[0]
        # Assuming index 1 is malicious (Tunneling) based on typical project structure
        tunneling_prob = probs[1] 
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        time.sleep(2)
        return

    is_malicious = tunneling_prob > 0.8

    # 🔥 Rule-based correction for obvious normal domains
    if entropy < 3.5 and num_digits == 0 and domain_length < 20:
        is_malicious = False
        confidence = probs[0]

    confidence = tunneling_prob if is_malicious else probs[0]
    prediction = "TUNNELING 🚨" if is_malicious else "NORMAL ✅"

    # Rule-based Explanations (Instead of heavy SHAP computation)
    reasons = []
    if entropy > 4.0:
        reasons.append("High entropy indicates possible data encoding.")
    if digit_ratio > 0.15:
        reasons.append(f"High digit ratio ({digit_ratio:.2f}) is suspicious for normal domains.")
    if domain_length > 30:
        reasons.append(f"Unusually long domain name ({domain_length} chars).")
    if num_subdomains >= 3:
        reasons.append(f"High number of subdomains ({num_subdomains}) suggests tunneling routing.")
    
    if not is_malicious and not reasons:
        reasons.append("Domain structure appears typical and benign.")
    elif is_malicious and not reasons:
        reasons.append("Model detected subtle malicious patterns in the domain structure.")
    elif not is_malicious and reasons:
        reasons = ["Domain exhibits some odd features but model determines it is overall safe."]

    # Output formatting
    print("\n====== RESULT ======")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence * 100:.1f}%")
    print("\nFeature Values:")
    print(f" 🔹 Domain Length : {domain_length}")
    print(f" 🔹 Entropy       : {entropy:.2f}")
    print(f" 🔹 Num Digits    : {num_digits}")
    print(f" 🔹 Subdomains    : {num_subdomains}")
    
    print("\nReason:")
    for reason in reasons:
        print(f" 🔸 {reason}")
    print("=" * 55)
    input("\nPress Enter to continue...")

def botnet_detection():
    print_header("BOTNET DETECTION", "🤖")
    try:
        duration = float(input("Enter duration (seconds): "))
        tot_pkts = float(input("Enter total packets: "))
        tot_bytes = float(input("Enter total bytes: "))
    except ValueError:
        print("❌ Invalid input. Please enter numbers.")
        time.sleep(1.5)
        return
        
    # Prevent division by zero
    if duration <= 0:
        duration = 0.0001
    if tot_pkts <= 0:
        tot_pkts = 0.0001
        
    # Derived Features
    bytes_per_sec = tot_bytes / duration
    packets_per_sec = tot_pkts / duration
    bytes_per_packet = tot_bytes / tot_pkts
    
    # Prepare array for the RF model.
    # The RF model was trained on 381 features (after preprocessing and OHE).
    # The first 7 numerical features correspond to our numeric columns.
    X = np.zeros((1, 381))
    X[0, 0] = duration
    X[0, 1] = tot_pkts
    X[0, 2] = tot_bytes
    X[0, 3] = 0 # src_bytes (dummy value)
    X[0, 4] = bytes_per_sec
    X[0, 5] = packets_per_sec
    X[0, 6] = bytes_per_packet

    try:
        probs = botnet_model.predict_proba(X)[0]
        # Assuming index 1 is Botnet
        botnet_prob = probs[1]
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        time.sleep(2)
        return

    is_botnet = botnet_prob > 0.5
    confidence = botnet_prob if is_botnet else probs[0]
    prediction = "BOTNET 🚨" if is_botnet else "NORMAL ✅"
    
    # Rule-based Explanations
    reasons = []
    if packets_per_sec > 100:
        reasons.append(f"High packets/sec ({packets_per_sec:,.1f}) indicates automated or burst traffic.")
    if duration < 1.0 and tot_pkts > 50:
        reasons.append(f"Low duration ({duration:.2f}s) combined with high packets suggests anomalous behavior.")
    if bytes_per_sec > 100000:
        reasons.append(f"High traffic volume ({bytes_per_sec:,.0f} B/s) is suspicious.")
    if bytes_per_packet < 50 and packets_per_sec > 50:
        reasons.append("High volume of very small packets suggests scanning or flooding.")
        
    if not is_botnet and not reasons:
        reasons.append("Traffic volume, rate, and duration are within normal parameters.")
    elif is_botnet and not reasons:
        reasons.append("Model detected suspicious flow characteristics based on multivariate analysis.")

    # Output formatting
    print("\n====== RESULT ======")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence * 100:.1f}%")
    print("\nDerived Features:")
    print(f" 🔹 Bytes/sec   : {bytes_per_sec:,.2f}")
    print(f" 🔹 Packets/sec : {packets_per_sec:,.2f}")
    
    print("\nReason:")
    for reason in reasons:
        print(f" 🔸 {reason}")
    print("=" * 55)
    input("\nPress Enter to continue...")

def main():
    while True:
        # Clear screen for clean UI
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" + "🛡️ " * 3 + " AI THREAT DETECTION SYSTEM " + "🛡️ " * 3)
        print("=" * 55)
        print(" 1. 🌐 DNS Tunneling Detection")
        print(" 2. 🤖 Botnet Detection")
        print(" 3. ❌ Exit")
        print("=" * 55)
        
        choice = input("Select module (1-3): ").strip()
        
        if choice == '1':
            dns_detection()
        elif choice == '2':
            botnet_detection()
        elif choice == '3':
            print("\nExiting System. Stay secure! 🔒\n")
            time.sleep(1)
            break
        else:
            print("\n❌ Invalid choice. Please select 1, 2, or 3.")
            time.sleep(1.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting System. Stay secure! 🔒\n")
