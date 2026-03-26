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

This project focuses on **sentiment analysis for Turkish university-related tweets**, classifying posts as **positive** or **negative**. The dataset was collected from tweets about student life, academic processes, administrative services, and campus facilities across major Turkish universities.

The study combines **real-world labeled tweets** with **synthetic augmented data** and evaluates multiple deep learning architectures for Turkish sentiment classification. To ensure reliable evaluation, the project uses **MinHash-based leakage-resistant data splitting**, helping reduce overlap between semantically similar tweets across train, validation, and test sets.

---

## Highlights

- **14 Turkish Universities** - Including YTU, ODTU, BOUN, ITU, Hacettepe, Marmara, and others
- **17,188 Total Samples** - **11,020 real tweets** + **6,168 synthetic samples**
- **5 Model Architectures** - BERTurk, Turkish ELECTRA, CNN, BiLSTM, CNN-BiLSTM
- **Leakage-Resistant Splitting** - MinHash clustering used to reduce train/test contamination
- **Turkish Language Optimized** - SentencePiece tokenization for classical models, transformer-based encoders for contextual learning
- **Research + Practical Use** - Suitable for academic experiments and real-world sentiment inference

---

## Dataset Availability

The dataset is also publicly available on Kaggle:

**Kaggle Dataset:**  
[Turkish Universities Sentiment Analysis Dataset](https://www.kaggle.com/datasets/alihanuludag/turkish-universities-sentiment-analysis-dataset)

---

## Project Structure

```text
university-tweets-sentiment-analysis-model-training/
│
├── get_tweets.py                              # Collect tweets from Twitter API
├── cleaning.py                                # Remove duplicates and clean raw data
├── make_splits_minhash.py                     # Split data using MinHash clustering
├── predict.py                                 # Run inference on new tweets
│
├── model_training_codes/
│   ├── trainForLabeling.py                    # Semi-automated labeling helper
│   ├── train_berturk_production.py            # Train BERTurk model
│   ├── train_turkish_electra_from_splits.py   # Train Turkish ELECTRA model
│   └── train_classical_spm_splits.py          # Train CNN / BiLSTM / CNN-BiLSTM models
│
├── splits/
│   ├── train.xlsx                             # Training split (real + synthetic)
│   ├── val.xlsx                               # Validation split (real only)
│   └── test.xlsx                              # Test split (real only)
│
├── real-data.xlsx                             # Real labeled tweet dataset
├── tweetDataset.xlsx                          # Full combined dataset
└── requirements.txt                           # Project dependencies
```

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

- Academic life
- Courses and exams
- Administrative services
- Scholarships and student affairs
- Campus facilities
- Cafeteria, library, dormitories, and transportation
- Complaints, dissatisfaction, praise, and appreciation

### Query Categories

To improve coverage and class diversity, data collection included different query groups such as:

- Strong Negative
- Strong Positive
- Academic
- Administrative
- Campus

### Dataset Composition

| Split | Real Tweets | Synthetic Tweets | Total |
| :--- | :--- | :--- | :--- |
| Train | 8,800 | 6,168 | 14,968 |
| Validation | 1,110 | 0 | 1,110 |
| Test | 1,110 | 0 | 1,110 |
| **Total** | **11,020** | **6,168** | **17,188** |

Validation and test sets contain only real tweets to ensure fair and unbiased evaluation.

### Data Fields

Main columns used in the dataset:

- **text** - Tweet content
- **label** - Sentiment label (0 = negative, 1 = positive)
- **is_synth** - Indicates whether the sample is synthetic (1) or real (0)
- **url** - Original tweet URL for verification
- **university** - University code / source university
- **group** - Query category used during data collection

---

## Models

### Transformer Models

| Model | Base Model | Parameters | Max Length |
| :--- | :--- | :--- | :--- |
| BERTurk | `dbmdz/bert-base-turkish-uncased` | 110M | 128 |
| Turkish ELECTRA | `dbmdz/electra-turkish-base-discriminator` | 110M | 256 |
| TabiBERT | Turkish BERT variant used for comparison | - | 128 |

### Classical Deep Learning Models

All classical models use SentencePiece subword tokenization.

| Model | Architecture | Vocab Size |
| :--- | :--- | :--- |
| TextCNN | Multi-kernel CNN | 8,000 |
| TextBiLSTM | Bidirectional LSTM | 8,000 |
| TextCNNBiLSTM | Hybrid (BiLSTM → CNN) | 8,000 |

#### Why SentencePiece?

Turkish is an agglutinative language, meaning words can take many suffixes and surface forms. SentencePiece helps reduce sparsity by splitting text into subword units, making classical neural models more robust to Turkish morphology.

---

## Training Features

To improve robustness and generalization, the training pipeline includes:

- **Class Weighting:** Handles label imbalance using weights computed from the real training data.
- **Synthetic Sample Weighting:** Synthetic samples are assigned lower contribution in the loss function to reduce overfitting to generated data.
- **Early Stopping:** Stops training when validation macro-F1 does not improve.
- **Threshold Tuning:** Selects the optimal decision threshold on the validation set to maximize macro-F1.
- **Leakage-Resistant Splits:** Uses MinHash clustering to reduce near-duplicate and semantically similar tweets appearing across different splits.
- **Learning Curve Logging:** Tracks training/validation loss and evaluation metrics for better model analysis.

---

## Output Artifacts

Each training run produces:

- Best model checkpoint
- Evaluation metrics
- Learning curves
- Confusion matrix on the test set
- Production-ready inference bundle
- Saved threshold / hyperparameter configuration

---

## Evaluation Summary

The project compares both transformer-based and classical deep learning models on the Turkish university tweet sentiment dataset.

Key experimental goals:

- Measure performance on real-world university-related sentiment data
- Compare transformer and classical architectures
- Analyze the impact of synthetic augmentation
- Reduce evaluation bias with leakage-resistant splitting

In the study, **Turkish ELECTRA** achieved the best overall performance among the compared models.

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

## Data Verification

All real tweets can be verified via the `url` field in `real-data.xlsx`. Opening the URL allows inspection of the original tweet on Twitter/X, when still publicly accessible.

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

These remain important research directions for future work.

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
