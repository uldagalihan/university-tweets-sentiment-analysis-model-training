# University Tweets Sentiment Analysis

## Contributors
- **Emre Çelik**
- **Alihan Uludağ**

> A sentiment analysis system for Turkish university-related tweets using deep learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/alihanuludag/turkish-universities-sentiment-analysis-dataset)

---

## About

This project focuses on **sentiment analysis for Turkish university-related tweets**, classifying posts as **positive** or **negative**. The dataset was collected from tweets about student life, academic processes, administrative services, and campus facilities across major Turkish universities between 2020 and 2026.

The study combines **real-world labeled tweets** with **synthetic augmented data (via ChatGPT 5.2 Thinking)** and evaluates multiple deep learning architectures for Turkish sentiment classification. To ensure reliable evaluation, the project uses **MinHash-based leakage-resistant data splitting**, helping reduce overlap between semantically similar tweets across train, validation, and test sets. Furthermore, the project includes an **interactive web-based decision support system** and a comprehensive **socio-temporal analysis** linking sentiment shifts to real-world events.

---

## Highlights

- **14 Turkish Universities** - Including YTU, ODTU, BOUN, ITU, Hacettepe, Marmara, and others.
- **17,188 Total Samples** - **11,020 real tweets** + **6,168 synthetic samples**.
- **6 Model Architectures** - BERTurk, Turkish ELECTRA, TabiBERT, CNN, BiLSTM, CNN-BiLSTM.
- **Leakage-Resistant Splitting** - MinHash LSH clustering used to strictly prevent train/test contamination.
- **Socio-Temporal Analysis** - Analyzing sentiment fluctuations corresponding to pandemic closures, university incidents, and socio-economic changes.
- **Interactive Web UI** - A Streamlit-based interface providing manual and batch inference for real-time monitoring.

---

## Interactive Web Interface

The theoretical models have been integrated into a comprehensive event monitoring and decision support system built with **Streamlit**. The interface offers:

1. **Manual Prediction Module:** Enter custom text to observe real-time predictions simultaneously from 6 different deep learning models.
2. **Batch Data Analysis:** Upload `.xlsx` or `.csv` files containing thousands of tweets, allowing the system to automatically analyze and produce aggregate sentiment insights.
3. **Analytic Dashboard:** View comprehensive performance metrics, socio-temporal graphs (e.g., hype graphs, heatmaps), and confusion matrices reflecting model training results.

---

## Dataset Availability

The dataset is also publicly available on Kaggle:

**Kaggle Dataset:**  
[Turkish Universities Sentiment Analysis Dataset](https://www.kaggle.com/datasets/alihanuludag/turkish-universities-sentiment-analysis-dataset)

---

## Quick Start

### Installation

```bash
git clone <repository-url>
cd university-tweets-sentiment-analysis-model-training

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Training

#### BERTurk
```bash
python model_training_codes/train_berturk_production.py \
  --splits_dir splits \
  --out outputs_berturk \
  --epochs 8 \
  --batch 16
```

#### Turkish ELECTRA
```bash
python model_training_codes/train_turkish_electra_from_splits.py \
  --splits_dir splits \
  --out outputs_electra \
  --epochs 12 \
  --batch 8
```

#### Classical Models
```bash
python model_training_codes/train_classical_spm_splits.py \
  --splits_dir splits \
  --arch cnn \
  --epochs 12
```

Available classical architectures:
- `cnn`
- `bilstm`
- `cnnbilstm`

### Inference

Use a trained model to predict sentiment for a new input text:

```bash
python predict.py \
  --model_path outputs_berturk \
  --text "Üniversitemizin kütüphanesi harika!"
```

---

## Dataset

### Data Collection

Tweets were collected from 14 Turkish universities between 2020 and 2026 using the Twitter/X API and keyword-based queries. The collected data reflects student opinions about:

- Academic life (Courses and exams)
- Administrative services
- Scholarships and student affairs
- Campus facilities (Cafeteria, library, dormitories, transportation)
- Complaints, dissatisfaction, praise, and appreciation

The sentiment distribution in the organically collected real tweets demonstrates a clear skew toward complaints, yielding roughly **74.9% Negative** and **25.1% Positive** samples. 

### Dataset Composition

| Split | Real Tweets | Synthetic Tweets | Total |
| :--- | :--- | :--- | :--- |
| Train | 8,800 | 6,168 | 14,968 |
| Validation | 1,110 | 0 | 1,110 |
| Test | 1,110 | 0 | 1,110 |
| **Total** | **11,020** | **6,168** | **17,188** |

Validation and test sets contain only real tweets to ensure fair and unbiased evaluation.

---

## Socio-Temporal Analysis (2020-2026)

The project explores sentiment changes not just intrinsically but by anchoring them to real-world sociological markers:

- **Pandemic and Remote Education (2020-2021):** Massive spikes in tweet volume regarding systemic remote education issues, exam anxieties, and administrative complaints.
- **Normalization Phase (2022-2026):** Return to face-to-face learning gradually stabilized and lowered digital interaction volumes as on-campus socialization resumed.
- **Boğaziçi University Anomaly:** A significant spike in interaction and positive sentiment unity surfaced in 2021 regarding the Boğaziçi rectorate protests, proving the power of digital social media acting as a modern "digital town square."

---

## Models & Evaluation

The project compares both transformer-based and classical deep learning models on the Turkish university tweet sentiment dataset. **Turkish ELECTRA** achieved the best overall performance among the compared models.

### Transformer Models

| Model | Base Model | Accuracy | Macro-F1 |
| :--- | :--- | :--- | :--- |
| Turkish ELECTRA | `dbmdz/electra-turkish-base-discriminator` | **93.34%** | **0.9094** |
| BERTurk | `dbmdz/bert-base-turkish-uncased` | 93.04% | 0.9052 |
| TabiBERT | `boun-tabilab/TabiBERT` | 91.77% | - |

*(Note: Turkish ELECTRA reached an exceptional 95.30% recall rate specifically for negative sentiment detection, making it highly robust for real-time crisis management.)*

### Classical Deep Learning Models

All classical models utilize **SentencePiece (BPE)** subword tokenization to combat the out-of-vocabulary (OOV) problem arising from Turkish's rich agglutinative morphology.

| Model | Architecture | Accuracy | Macro-F1 |
| :--- | :--- | :--- | :--- |
| TextCNNBiLSTM | Hybrid (BiLSTM → CNN) | 89.68% | 0.8590 |
| TextBiLSTM | Bidirectional LSTM | 89.47% | 0.8480 |
| TextCNN | Multi-kernel CNN | 88.74% | 0.8408 |

---

## Training Features

To improve robustness and generalization, the training pipeline embraces several specialized methods:

- **MinHash Leakage-Resistant Splits:** Uses Jaccard similarity and MinHash LSH clustering to eliminate data leakage. Near-duplicate templates and copied complaints are clustered together so the test set remains strictly distinct from the training set.
- **Weighted Loss & LLM Augmentation:** Addresses the 75-25 class imbalance by generating minority class synthetic data using ChatGPT 5.2 Thinking. These samples are integrated alongside a custom weighted loss function (with an optimized λ=1 coefficient) dynamically scaling their influence since MinHash ensures only high-quality unique synthetic data survives.
- **SentencePiece Tokenization:** Enhances robustness to Turkish morphology by breaking words into smaller sub-word units.
- **Early Stopping & Threshold Tuning:** Prevents overfitting optimally checking macro-F1 convergence.

---

## Requirements

```text
transformers>=4.44
datasets>=2.20
evaluate>=0.4
scikit-learn>=1.4
torch>=2.2
pandas>=2.0
numpy>=1.26
accelerate>=0.33
openpyxl>=3.1
sentencepiece
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Reproducibility Notes

For reproducible experiments:

- Use the provided split files under `splits/`
- Keep validation and test sets real-only
- Preserve MinHash-based grouping during re-splitting
- Use the same preprocessing pipeline before retraining

---

## Limitations

Although the dataset is built from real university-related tweets, sentiment analysis on social media remains challenging because of:

- Irony and sarcasm
- Slang and informal spelling
- Ambiguous context
- Evolving platform language
- Class imbalance in naturally collected data

---

## License

This project is provided for academic and research purposes.

For commercial or extended usage, please contact the contributors.

---

## Acknowledgments

- `dbmdz` for BERTurk and Turkish ELECTRA models
- Hugging Face for the Transformers ecosystem
- PyTorch, scikit-learn, and SentencePiece
- Everyone who contributed to data collection, annotation, and evaluation

---

## Contact

For questions, suggestions, or collaboration:

- Emre Çelik - GitHub: [@EmreCelik23](https://github.com/EmreCelik23)
- Alihan Uludağ - GitHub: [@uldagalihan](https://github.com/uldagalihan)
