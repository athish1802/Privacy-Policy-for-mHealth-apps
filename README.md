# A Hybrid NLP Framework for Automated Scoring of Privacy Policies for mHealth Apps

> **CS 8395 – Special Topics in Computer Science**  
> Vanderbilt University | Fall 2025  
> **Authors:** Athish Suresh, Aryan Kalluri

---

## Overview

Privacy policies are the primary means by which organizations inform users about their data practices — yet these documents are consistently too long, legally complex, and impractical for users to read. This project proposes and implements a **hybrid machine learning framework** that automatically evaluates privacy policies across **5 quality dimensions (11 sub-dimensions)** by combining rule-based linguistic feature extraction with DistilBERT transformer embeddings.

The model achieves **72% average accuracy** (within 1 unit of human ratings on a 5-point scale) and a **mean absolute error (MAE) of 0.70**, with security dimensions reaching 100% accuracy and regulatory compliance presenting the most challenge.

---

## Repository Structure

```
Privacy-Policy-for-mHealth-apps/
├── Group_5_Privacy_Policy_Project.ipynb   # Full ML pipeline (feature extraction, training, evaluation)
├── Dataset_Train.csv                       # Training dataset (25 privacy policies, 11 dimensions)
├── Dataset_Test.csv                        # Test dataset (5 privacy policies, 11 dimensions)
├── Group_5_Privacy_Paper.pdf              # Full research paper (ACM format)
├── Pervasiv_Project_Presentation-1.pdf    # Project presentation slides
├── Instructions-for-Running-...pdf        # Step-by-step Colab setup guide
└── README.md
```

---

## Evaluation Dimensions

Each privacy policy is rated on a **5-point scale** across 11 dimensions organized into 5 categories:

| Category | Sub-Dimension | Description |
|----------|--------------|-------------|
| **Transparency** | Grade Level | Readability score (Flesch formula) |
| | Technical Term Density | Density of privacy/technical jargon |
| | Vagueness Index | Frequency of ambiguous language (may, might, etc.) |
| **Data Collection** | Specificity | Range of data types explicitly mentioned |
| | Purpose Clarity | Clarity of stated purposes for data use |
| **User Control** | Control Options | Opt-out, delete, access, modify rights offered |
| | Access Barriers | Terms indicating restrictions on user rights |
| **Security** | Security Detail | Encryption, authentication, firewall mentions |
| | Breach Notification | Incident response and notification procedures |
| **Regulatory Compliance** | Regulatory Coverage | References to GDPR, CCPA, HIPAA, etc. |
| | Rights Clarity | Clarity of user rights descriptions |

---

## Model Architecture

```
Raw Privacy Policy Text
        │
        ▼
┌─────────────────────────────────┐
│       Data Preprocessing        │
│  - Text normalization (UTF-8)   │
│  - Sentence segmentation        │
│  - Term standardization (120    │
│    privacy term mappings)       │
│  - Structure extraction         │
└────────────┬────────────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
┌─────────────┐  ┌──────────────────┐
│  Rule-Based │  │ Transformer       │
│  Feature    │  │ Embeddings        │
│  Extractor  │  │ (DistilBERT)      │
│             │  │                   │
│ ~64 features│  │ CLS token → PCA  │
│ per policy  │  │ → 5 components    │
└──────┬──────┘  └────────┬─────────┘
       │                  │
       └────────┬─────────┘
                ▼
     ┌─────────────────────┐
     │  Feature Concat +   │
     │  StandardScaler     │
     └──────────┬──────────┘
                ▼
     ┌─────────────────────┐
     │  ElasticNet          │
     │  Regression          │
     │  (α=0.5, L1=0.7)    │
     │  Per dimension       │
     └──────────┬──────────┘
                ▼
     Predicted Quality Score (1–5)
     per dimension
```

### Key Components

**1. `PrivacyPolicyRuleExtractor` (custom scikit-learn transformer)**  
Extracts ~64 linguistic and structural features per policy, organized by dimension:
- Transparency: Flesch readability, technical term density, vagueness indicators, structural clarity
- Data Collection: Data type enumeration (70-term dictionary), purpose statement detection
- User Control: Control option detection, access barrier terms, instruction clarity
- Security: Security terminology (encrypted, firewall, 2FA), breach notification language
- Regulatory: Regulation mentions (GDPR, CCPA, HIPAA), rights declarations, legal basis

**2. DistilBERT Transformer Embeddings**  
- Model: `distilbert-base-uncased` (66M parameters, 40% smaller than BERT-base)
- Sliding window (512 tokens, 128-token overlap) for long documents
- CLS token extraction → PCA (5 components, ~87% variance retained)

**3. ElasticNet Regression**  
- Combines L1 (feature selection) + L2 (correlated feature handling) regularization
- Trained independently per dimension
- Evaluated on MAE and accuracy within ±1 unit of human rating

---

## Results

### Performance by Category

| Category | MAE | Accuracy (within ±1) |
|----------|-----|----------------------|
| Security | 0.53 | **100%** |
| Data Collection | 0.67 | 75% |
| User Control | 0.71 | 75% |
| Transparency | 0.89 | 67% |
| Regulatory Compliance | 1.15 | 38% |
| **Overall** | **0.70** | **72%** |

### Top vs. Challenging Dimensions

| Dimension | MAE | Accuracy |
|-----------|-----|----------|
| Security Detail | 0.41 | 100% |
| Breach Notification | 0.65 | 100% |
| Regulatory Rights Clarity | 1.25 | 25% |
| Transparency Grade Level | 1.06 | 50% |

### Ablation Study
The hybrid approach outperforms either feature type alone:
- **+12%** accuracy over rule-based features only
- **+18%** accuracy over transformer embeddings only

---

## Dataset

- **Source:** 30 privacy policies from mobile health (mHealth) applications
- **Training set:** 25 policies × 11 dimensions = 275 ratings
- **Test set:** 5 policies × 11 dimensions = 55 ratings
- **Labeling:** Human expert ratings on a 5-point rubric per dimension
- **Domain diversity:** Health, fitness, telemedicine, wellness apps
- **Text length:** ~3,000 words per policy average

---

## Setup & Running Instructions

### Option 1: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `Group_5_Privacy_Policy_Project.ipynb` via **File → Upload notebook**
3. In the first cell, install dependencies:
   ```python
   !pip install pandas numpy nltk torch transformers scikit-learn matplotlib seaborn
   ```
4. Upload both dataset files using the **Files sidebar (↑ upload)**:
   - `Dataset_Train.csv`
   - `Dataset_Test.csv`
5. Run all cells (**Runtime → Run all**)
6. The notebook will automatically load from local files; if not found, it will prompt for upload

### Option 2: Local Environment

```bash
# Clone the repo
git clone https://github.com/athish1802/Privacy-Policy-for-mHealth-apps.git
cd Privacy-Policy-for-mHealth-apps

# Install dependencies
pip install pandas numpy nltk torch transformers scikit-learn matplotlib seaborn

# Launch notebook
jupyter notebook Group_5_Privacy_Policy_Project.ipynb
```

### Expected Outputs
- Performance bar charts (MAE + accuracy per dimension)
- Radar chart (accuracy distribution across all dimensions)
- Normalized MAE comparison chart
- Category-level performance summary
- Feature importance analysis per dimension
- Dashboard summary with overall metrics

---

## Dependencies

| Package | Purpose |
|---------|--------|
| `pandas`, `numpy` | Data manipulation |
| `nltk` | Sentence tokenization |
| `torch` | PyTorch backend for DistilBERT |
| `transformers` | DistilBERT model and tokenizer |
| `scikit-learn` | ElasticNet, PCA, StandardScaler, metrics |
| `matplotlib`, `seaborn` | Visualization |

---

## Key Findings

- **Security dimensions are most automatable** — explicit, standardized terminology makes them reliably predictable
- **Regulatory compliance is hardest** — complex legal language varies widely; transformer embeddings help more than rule-based features here
- **Hybrid approach wins** — combining rule-based features with DistilBERT embeddings outperforms either alone across all dimensions
- **Feature selection via L1** — many basic features (word count, sentence count) are eliminated in favor of targeted privacy-specific term matches

---

## Future Work

- Expand dataset beyond 30 policies for better generalization
- Fine-tune DistilBERT specifically on privacy policy text
- Build an interactive policy assessment dashboard for privacy professionals
- Extend to cross-policy benchmarking and historical policy change tracking
- Explore zero-shot/few-shot learning for new evaluation dimensions

---

## References

See `Group_5_Privacy_Paper.pdf` for full citations. Key works include:
- Harkous et al. (2018) — Polisis: Automated Analysis of Privacy Policies Using Deep Learning
- Wilson et al. (2016) — OPP-115 Corpus
- Devlin et al. (2019) — BERT
- McDonald & Cranor (2008) — The Cost of Reading Privacy Policies
