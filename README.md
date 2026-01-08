# University Tweets Sentiment Analysis

## Contributors
- **Emre Çelik**
- **Alihan Uludağ**

> A sentiment analysis system for Turkish university student tweets using deep learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)

---

## About

This project analyzes sentiment in Turkish university student tweets, classifying them as **positive** or **negative**. We collected ~11,000 tweets from 14 major Turkish universities and trained multiple deep learning models to understand student opinions about academic life, campus facilities, and administrative services.

### Highlights

- 14 Turkish Universities - YTU, ODTU, BOUN, ITU, Hacettepe, Marmara, and more
- 11K Dataset - 5,043 real tweets + 6,150 synthetic augmented samples
- 5 Model Architectures - BERTurk, Turkish ELECTRA, CNN, BiLSTM, CNN-BiLSTM
- Smart Data Splitting - MinHash clustering prevents data leakage
- Turkish Language Optimized - SentencePiece tokenization for agglutinative morphology

---

## 📂 Project Structure

```
university-tweets-sentiment-analysis-model-training/
│
├── 📄 get_tweets.py                    # Collect tweets from Twitter API
├── 📄 cleaning.py                      # Remove duplicate tweets
├── 📄 make_splits_minhash.py           # Split data with MinHash clustering
├── 📄 predict.py                       # Run inference on new tweets
│
├── 📁 model_training_codes/
│   ├── trainForLabeling.py             # Semi-automated labeling helper
│   ├── train_berturk_production.py     # Train BERTurk model
│   ├── train_turkish_electra_from_splits.py  # Train Turkish ELECTRA
│   └── train_classical_spm_splits.py   # Train CNN/BiLSTM/CNN-BiLSTM
│
├── 📁 splits/                          # Train/validation/test splits
│   ├── train.xlsx                      # Training data (real + synthetic)
│   ├── val.xlsx                        # Validation data (real only)
│   └── test.xlsx                       # Test data (real only)
│
├── 📊 real-data.xlsx                   # Original labeled tweets
├── 📊 tweetDataset.xlsx                # Complete dataset
└── 📋 requirements.txt                 # Python dependencies
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd university-tweets-sentiment-analysis-model-training

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train a Model

**BERTurk (Recommended)**
```bash
python model_training_codes/train_berturk_production.py \
  --splits_dir splits \
  --out outputs_berturk \
  --epochs 8 \
  --batch 16
```

**Turkish ELECTRA**
```bash
python model_training_codes/train_turkish_electra_from_splits.py \
  --splits_dir splits \
  --out outputs_electra \
  --epochs 12 \
  --batch 8
```

**Classical Models (CNN/BiLSTM/CNN-BiLSTM)**
```bash
python model_training_codes/train_classical_spm_splits.py \
  --splits_dir splits \
  --arch cnn \
  --epochs 12
```

### Make Predictions

```bash
python predict.py \
  --model_path outputs_berturk \
  --text "Üniversitemizin kütüphanesi harika!"
```

---

## Dataset

### Data Collection

Tweets were collected from **14 Turkish universities** between **2020-2025** using the Twitter API.

**Query Categories:**
- Strong Negative - Complaints, dissatisfaction
- Strong Positive - Praise, gratitude
- Academic - Courses, professors, exams
- Administrative - Student affairs, scholarships
- Campus - Cafeteria, library, dormitories

### Dataset Composition

| Split | Real Tweets | Synthetic Tweets | Total |
|-------|-------------|------------------|-------|
| **Train** | 2,823 | 6,150 | 8,973 |
| **Validation** | 1,110 | 0 | 1,110 |
| **Test** | 1,110 | 0 | 1,110 |
| **Total** | 5,043 | 6,150 | 11,193 |

> Validation and test sets contain only real tweets to ensure unbiased evaluation

### Data Columns

- `text` - Tweet content
- `label` - Sentiment (0=negative, 1=positive)
- `is_synth` - Real (0) or synthetic (1)
- `url` - Tweet URL for verification
- `university` - University code
- `group` - Query category

---

## Models

### Transformer Models

| Model | Base | Parameters | Max Length |
|-------|------|------------|------------|
| **BERTurk** | `dbmdz/bert-base-turkish-uncased` | 110M | 128 |
| **Turkish ELECTRA** | `dbmdz/electra-turkish-base-discriminator` | 110M | 256 |

### Classical Models (with SentencePiece)

| Model | Architecture | Vocab Size |
|-------|--------------|------------|
| **TextCNN** | Multi-kernel CNN | 8,000 |
| **TextBiLSTM** | Bidirectional LSTM | 8,000 |
| **TextCNNBiLSTM** | Hybrid (BiLSTM → CNN) | 8,000 |

> **Why SentencePiece?** Turkish is an agglutinative language with complex morphology. SentencePiece's subword tokenization handles suffixes and inflections effectively.

---

## Training Features

All models are trained with advanced techniques to ensure robust performance:

- **Class Weighting** - Computed on real training data to handle label imbalance
- **Synthetic Sample Weighting** - Synthetic tweets have reduced loss weight (default: 0.3) to prevent overfitting to augmented data
- **Early Stopping** - Monitors validation macro-F1 with patience to prevent overfitting
- **Threshold Tuning** - Optimizes classification threshold on validation set to maximize macro-F1
- **Learning Curves** - Visualizes training/validation loss and F1 scores for model analysis

### Output Artifacts

Each training run produces:
- Best model checkpoint (selected by validation macro-F1)
- Learning curves (loss & F1 plots)
- Confusion matrix on test set
- Production bundle (hyperparameters + optimal threshold)

---

## Requirements

```
transformers>=4.44
datasets>=2.20
evaluate>=0.4
scikit-learn>=1.4
torch>=2.2
pandas>=2.0
numpy>=1.26
accelerate>=0.33
openpyxl>=3.1
sentencepiece  # For classical models
```

---

## Data Verification

All real tweets can be verified using the `url` column in `real-data.xlsx`. Simply open the URL to see the original tweet on Twitter.

---

## License

This project is available for academic and research purposes.

---

## Acknowledgments

- **Models** - [dbmdz](https://huggingface.co/dbmdz) for BERTurk and Turkish ELECTRA
- **Libraries** - Hugging Face, PyTorch, scikit-learn, SentencePiece

## Contact

For questions, suggestions, or collaboration:
- **Emre Çelik**: GitHub [@EmreCelik23](https://github.com/EmreCelik23)
- **Alihan Uludağ**: GitHub [@uldagalihan](https://github.com/uldagalihan)

