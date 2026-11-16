# 🛡️ DNS Tunneling & Botnet Detection — Complete Pipeline Flowchart

> **Project:** ML-Based Network Threat Detection System  
> **Modules:** DNS Tunneling Detection + Botnet Detection (CTU-13)

---

## 🔷 PART 1 — Botnet Detection Pipeline (CTU-13 Dataset)

```mermaid
flowchart TD
    A(["📂 START\nCTU-13 Dataset\n.binetflow Files"]):::start

    A --> B["🗂️ Data Loader\n`data_loader.py`\nLoad all scenario files\nRow-based stratified split"]

    B --> C["🔍 Preprocessing\n`preprocessor.py`\nHandle missing values\nEncode categorical cols\nLabel: Normal vs Botnet"]

    C --> D["⚙️ Feature Engineering\n`feature_engineering.py`\nFlow duration, byte ratios\nPacket stats, protocol flags\nPort entropy, IP diversity"]

    D --> E["⚖️ Class Imbalance Handling\nclass_weight='balanced'\nNO SMOTE — row-based split\nStratified by botnet family"]

    E --> F{{"🤖 Model Training\n`model_trainer.py`"}}

    F --> G["🌲 Random Forest\nclass_weight=balanced\nn_estimators=100"]
    F --> H["🚀 Gradient Boosting\nscale_pos_weight\nlearning_rate=0.1"]
    F --> I["📈 Logistic Regression\nclass_weight=balanced\nmax_iter=1000"]

    G --> J[["📊 Evaluation\n`evaluator.py`\nPrecision, Recall, F1\nROC-AUC, Confusion Matrix"]]
    H --> J
    I --> J

    J --> K["🏆 Best Model Selection\nF1-Score based\nThreshold Tuning on Val Set"]

    K --> L["🔬 SHAP Explainability\n`explainer.py`\nFeature importance\nForce plots, Summary plots"]

    L --> M["💾 Model Persistence\nmodels/botnet_model.pkl\nmodels/preprocessor.pkl"]

    M --> N(["✅ Botnet Detection\nReady for Inference"]):::done

    style A fill:#1a1a2e,color:#e94560,stroke:#e94560
    style N fill:#0f3460,color:#00ff88,stroke:#00ff88
```

---

## 🔶 PART 2 — DNS Tunneling Detection Pipeline

```mermaid
flowchart TD
    A2(["📂 START\nDNS Dataset\nMalicious + Benign CSVs"]):::start

    A2 --> B2["🗂️ DNS Data Loader\n`dns_data_loader.py`\nMerge malicious & benign\nAuto-detect domain column"]

    B2 --> C2["🔍 DNS Preprocessing\n`dns_preprocessor.py`\nHandle nulls\nEncode labels: 0=Benign, 1=Tunnel\nClean domain strings"]

    C2 --> D2["⚙️ DNS Feature Engineering\n`dns_feature_engineering.py`"]

    D2 --> D2a["📐 Shannon Entropy\nof domain string\nHigh entropy → suspicious"]
    D2 --> D2b["📏 Domain Length\nSubdomain depth\nUnique char ratio"]
    D2 --> D2c["🔤 N-gram Features\nBigram/Trigram freq\nVowel/Consonant ratio"]
    D2 --> D2d["🔢 Numeric Ratio\nDigit ratio in domain\nSpecial char count"]

    D2a --> E2["📦 Feature Matrix\nCombine all DNS features"]
    D2b --> E2
    D2c --> E2
    D2d --> E2

    E2 --> F2{{"🤖 Model Training\n`dns_model_trainer.py`"}}

    F2 --> G2["🌲 Random Forest\nclass_weight=balanced"]
    F2 --> H2["🚀 Gradient Boosting\nlearning_rate=0.05"]
    F2 --> I2["📈 Logistic Regression\nC=1.0"]

    G2 --> J2[["📊 Evaluation\nPrecision, Recall, F1\nROC-AUC Score\nClassification Report"]]
    H2 --> J2
    I2 --> J2

    J2 --> K2["🏆 Best Model\nF1-based selection\nSave threshold"]

    K2 --> L2["🔬 SHAP Explainability\nTop features for DNS tunneling\nEntropy most important"]

    L2 --> M2["💾 Save Model\nmodels/dns_model.pkl\nmodels/dns_preprocessor.pkl"]

    M2 --> N2(["✅ DNS Tunneling Detection\nReady for Inference"]):::done

    style A2 fill:#1a1a2e,color:#f5a623,stroke:#f5a623
    style N2 fill:#0f3460,color:#00ff88,stroke:#00ff88
```

---

## 🔁 PART 3 — Unified Inference Pipeline (Combined)

```mermaid
flowchart LR
    IN(["🌐 Live Network Traffic\nor PCAP / CSV Input"]):::input

    IN --> R["🚦 Router / Dispatcher\n`main.py`"]

    R --> DNS_IN["DNS Query Stream\n.dns / domain logs"]
    R --> NET_IN["NetFlow / Binetflow\nConnection records"]

    DNS_IN --> DNS_PIPE["🔶 DNS Tunneling\nDetection Module\n↓\nFeature Extraction\n↓\nModel Predict"]

    NET_IN --> BOT_PIPE["🔷 Botnet Detection\nModule\n↓\nFeature Extraction\n↓\nModel Predict"]

    DNS_PIPE --> AGG["📋 Aggregator\n`results_aggregator.py`\nCombine predictions\nCorrelate alerts"]

    BOT_PIPE --> AGG

    AGG --> OUT1["🟢 BENIGN\nNo threat detected"]
    AGG --> OUT2["🔴 THREAT DETECTED\nType: DNS Tunnel / Botnet\nConfidence Score\nSHAP Explanation"]

    OUT2 --> LOG["📝 Logger\nAlert with timestamp\nSource IP, domain\nModel confidence"]

    style IN fill:#1a1a2e,color:#a29bfe,stroke:#a29bfe
    style OUT2 fill:#c0392b,color:#fff,stroke:#e74c3c
    style OUT1 fill:#27ae60,color:#fff,stroke:#2ecc71
```

---

## 📁 PART 4 — Project File Structure

```mermaid
graph TD
    ROOT["📁 CourseProject/"]

    ROOT --> D1["📁 data/\n  ctu13/ → binetflow files\n  dns/ → malicious + benign CSVs"]
    ROOT --> D2["📁 src/\n  botnet/ → detection module\n  dns/ → dns tunnel module\n  shared/ → utils, logging"]
    ROOT --> D3["📁 models/\n  botnet_model.pkl\n  dns_model.pkl\n  preprocessors/"]
    ROOT --> D4["📁 notebooks/\n  EDA.ipynb\n  SHAP_analysis.ipynb"]
    ROOT --> D5["📁 results/\n  metrics.json\n  confusion_matrix.png\n  shap_plots/"]
    ROOT --> D6["📄 main.py\n  Unified entry point"]
    ROOT --> D7["📄 requirements.txt\n  config.yaml"]
```

---

## 🧠 Key Concepts Summary

| Step | Botnet Detection | DNS Tunneling Detection |
|------|-----------------|------------------------|
| **Dataset** | CTU-13 (binetflow) | Custom DNS CSVs |
| **Split Strategy** | Row-based stratified | Row-based stratified |
| **Key Features** | Flow duration, byte ratio, port entropy | Shannon entropy, domain length, n-grams |
| **Imbalance** | `class_weight='balanced'` | `class_weight='balanced'` |
| **Best Model** | Random Forest / GB | Random Forest / GB |
| **Explainability** | SHAP TreeExplainer | SHAP TreeExplainer |
| **Output** | Normal / Botnet family | Benign / Tunnel |

---

## 🔑 Most Important Feature — Shannon Entropy

```
Entropy(domain) = -Σ p(c) × log₂(p(c))

Normal domain:  "google.com"    → Entropy ≈ 2.8  ✅ Low
Tunnel domain:  "aB3xY9mK2pL.evil.com" → Entropy ≈ 4.9  🚨 HIGH
```

> **Rule of thumb:** Entropy > 3.5 for subdomain → suspicious DNS tunneling activity

---

*Generated for: ML-Based Network Threat Detection | CTU-13 + DNS Dataset*
