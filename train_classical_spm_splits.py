# train_classical_spm_splits.py
# Usage:
#   python train_classical_spm_splits.py --splits_dir splits --arch cnn
#   python train_classical_spm_splits.py --splits_dir splits --arch bilstm
#   python train_classical_spm_splits.py --splits_dir splits --arch cnn_bilstm
#
# Notlar:
# - splits/train.xlsx val.xlsx test.xlsx kolonları: text,label,is_synth
# - Val/Test script tarafından is_synth=0 zorlanır (REAL only)
# - SentencePiece vocab sadece TRAIN textlerinden eğitilir (leak yok)
# - class weight sadece REAL train'den hesaplanır
# - synth örneklere loss çarpanı uygulanır (default 0.3)
# - best model val macro-f1 ile seçilir + early stopping

import argparse
import os
import random
import re
import json
import numpy as np
import pandas as pd
import sentencepiece as spm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

import matplotlib.pyplot as plt


# ---------- Utils ----------
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def normalize_tweet(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower().strip()
    t = re.sub(r"http\S+|www\.\S+", "<url>", t)
    t = re.sub(r"@\w+", "<user>", t)
    t = re.sub(r"#", "", t)
    t = re.sub(r"\d+([\.,]\d+)?", "<num>", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def pad_seq(seq, maxlen, pad=0):
    if len(seq) < maxlen:
        seq = seq + [pad] * (maxlen - len(seq))
    return seq[:maxlen]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_curves(history, out_dir, arch):
    """
    history: list of dict(epoch, train_loss, val_loss, val_f1)
    """
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(plots_dir)

    epochs = [h["epoch"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"] for h in history]
    va_f1 = [h["val_f1"] for h in history]

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, tr_loss, label="Train Loss")
    plt.plot(epochs, va_loss, label="Val Loss", linestyle="--")
    plt.title(f"Learning Curve - Loss ({arch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{arch}_learning_curve_loss.png"), dpi=160)
    plt.close()

    # F1
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, va_f1, label="Val Macro-F1")
    plt.title(f"Learning Curve - Val Macro F1 ({arch})")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{arch}_learning_curve_macro_f1.png"), dpi=160)
    plt.close()

def save_confusion_matrix(y_true, y_pred, out_dir, arch):
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(plots_dir)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix (REAL) ({arch})")
    plt.colorbar()
    labels = ["negatif(0)", "pozitif(1)"]
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, labels, rotation=20)
    plt.yticks(tick_marks, labels)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{arch}_confusion_matrix_test.png"), dpi=160)
    plt.close()


# ---------- Args ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", type=str, required=True, help="splits/ içinde train.xlsx val.xlsx test.xlsx")
    ap.add_argument("--arch", type=str, default="cnn", choices=["cnn", "bilstm", "cnn_bilstm"])

    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--early_patience", type=int, default=2)

    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--maxlen", type=int, default=50)  # SPM token length

    ap.add_argument("--embed_dim", type=int, default=200)
    ap.add_argument("--hidden_dim", type=int, default=128)

    ap.add_argument("--out", type=str, default="outputs_spm_prod")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--spm_model", type=str, default="", help="Varolan .model yolu (verilirse eğitmez)")

    # SPM
    ap.add_argument("--spm_prefix", type=str, default="spm_bpe")
    ap.add_argument("--spm_vocab_size", type=int, default=8000)
    ap.add_argument("--character_coverage", type=float, default=1.0)

    # synth handling
    ap.add_argument("--synth_weight", type=float, default=0.3, help="sentetik örnekler loss çarpanı (0.2-0.6)")
    ap.add_argument("--truncate_chars", type=int, default=256, help="text karakter bazında kes")

    return ap.parse_args()


# ---------- Dataset ----------
class TweetDataset(Dataset):
    def __init__(self, X, y, is_synth):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        self.is_synth = torch.tensor(is_synth, dtype=torch.float32)  # 0/1

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.is_synth[idx]


# ---------- Models ----------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels=2,
                 filter_sizes=(3, 4, 5), num_filters=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(self, x):
        x = self.embed(x)          # [B, L, D]
        x = x.permute(0, 2, 1)     # [B, D, L]
        xs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            c = torch.max(c, dim=2).values
            xs.append(c)
        x = torch.cat(xs, dim=1)
        x = self.dropout(x)
        return self.fc(x)

class TextBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(2 * hidden_dim, num_labels)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        x, _ = torch.max(out, dim=1)
        x = self.dropout(x)
        return self.fc(x)

class TextCNNBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels=2,
                 filter_sizes=(3, 4, 5), num_filters=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        conv_in = 2 * hidden_dim
        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)        # [B, L, 2H]
        x = out.permute(0, 2, 1)     # [B, 2H, L]
        xs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            c = torch.max(c, dim=2).values
            xs.append(c)
        x = torch.cat(xs, dim=1)
        x = self.dropout(x)
        return self.fc(x)

def build_model(arch, vocab_size, embed_dim, hidden_dim):
    if arch == "cnn":
        return TextCNN(vocab_size, embed_dim)
    if arch == "bilstm":
        return TextBiLSTM(vocab_size, embed_dim, hidden_dim)
    if arch == "cnn_bilstm":
        return TextCNNBiLSTM(vocab_size, embed_dim, hidden_dim)
    raise ValueError("Unknown arch")


# ---------- SentencePiece ----------
def train_spm_from_train_texts(train_texts, out_dir, prefix, vocab_size, character_coverage):
    ensure_dir(out_dir)
    spm_input = os.path.join(out_dir, "spm_train_input.txt")

    with open(spm_input, "w", encoding="utf-8") as f:
        for t in train_texts:
            if isinstance(t, str) and t.strip():
                f.write(t.strip() + "\n")

    model_prefix = os.path.join(out_dir, prefix)

    cmd = (
        f"--input={spm_input} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type=bpe "
        f"--character_coverage={character_coverage} "
        "--pad_id=0 --unk_id=1 --bos_id=-1 --eos_id=-1"
    )

    print("[INFO] Training SentencePiece on TRAIN only (no leakage)...")
    spm.SentencePieceTrainer.Train(cmd)

    model_path = model_prefix + ".model"
    vocab_path = model_prefix + ".vocab"
    print(f"[OK] SPM saved: {model_path} / {vocab_path}")
    return model_path


# ---------- Train / Eval ----------
def compute_loss(logits, y, is_synth, class_weights, synth_weight):
    """
    CE(reduction=none) * (1.0 if real else synth_weight), sonra mean
    class_weights: tensor [2]
    """
    ce = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    per = ce(logits, y)  # [B]
    sw = torch.where(is_synth > 0.5, torch.tensor(synth_weight, device=per.device), torch.tensor(1.0, device=per.device))
    per = per * sw
    return per.mean()

def train_one_epoch(model, loader, optimizer, class_weights, synth_weight, device):
    model.train()
    total_loss = 0.0
    for Xb, yb, sb in loader:
        Xb, yb, sb = Xb.to(device), yb.to(device), sb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = compute_loss(logits, yb, sb, class_weights, synth_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_model(model, loader, class_weights, synth_weight, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for Xb, yb, sb in loader:
        Xb, yb, sb = Xb.to(device), yb.to(device), sb.to(device)
        logits = model(Xb)
        loss = compute_loss(logits, yb, sb, class_weights, synth_weight)
        total_loss += loss.item() * Xb.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(yb.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1, np.array(all_labels), np.array(all_preds)


# ---------- Main ----------
def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] DEVICE:", device)
    print("[INFO] ARCH:", args.arch)

    train_path = os.path.join(args.splits_dir, "train.xlsx")
    val_path = os.path.join(args.splits_dir, "val.xlsx")
    test_path = os.path.join(args.splits_dir, "test.xlsx")

    train_df = pd.read_excel(train_path)
    val_df = pd.read_excel(val_path)
    test_df = pd.read_excel(test_path)

    needed = {"text", "label", "is_synth"}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        miss = needed - set(df.columns)
        if miss:
            raise SystemExit(f"[{name}] Missing columns: {miss} (need text,label,is_synth)")

    def prep(df):
        df = df.dropna(subset=["text", "label", "is_synth"]).copy()
        df["text"] = df["text"].astype(str).apply(normalize_tweet)
        if args.truncate_chars and args.truncate_chars > 0:
            df["text"] = df["text"].str.slice(0, args.truncate_chars)
        df["label"] = df["label"].astype(int)
        df["is_synth"] = df["is_synth"].astype(int).clip(0, 1)
        df = df[df["label"].isin([0, 1])].reset_index(drop=True)
        return df

    train_df = prep(train_df)
    val_df = prep(val_df)
    test_df = prep(test_df)

    # REAL-only val/test
    val_df["is_synth"] = 0
    test_df["is_synth"] = 0

    real_train = train_df[train_df["is_synth"] == 0]
    synth_train = train_df[train_df["is_synth"] == 1]

    print("[INFO] ===== SPLITS LOADED =====")
    print(f"[INFO] Train total: {len(train_df)} | real: {len(real_train)} | synth: {len(synth_train)}")
    print(f"[INFO] Val (REAL): {len(val_df)} | Test (REAL): {len(test_df)}")
    print(f"[INFO] synth_weight(loss): {args.synth_weight}")
    print(f"[INFO] maxlen(spm tokens): {args.maxlen}, spm_vocab: {args.spm_vocab_size}\n")

    # class weight on REAL train only
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=real_train["label"].values
    )
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)
    print("[INFO] Class weights (REAL train):", cw)

    # train SPM ONLY on train texts (no leakage)
        # ---------- SentencePiece (TRAIN only, no leakage) ----------
    if args.spm_model and os.path.exists(args.spm_model):
        spm_model_path = args.spm_model
        print("[INFO] Using existing SentencePiece model:", spm_model_path)
    else:
        spm_model_path = train_spm_from_train_texts(
            train_texts=train_df["text"].tolist(),
            out_dir=args.out,
            prefix=f"{args.spm_prefix}_{args.spm_vocab_size}",
            vocab_size=args.spm_vocab_size,
            character_coverage=args.character_coverage
        )


    sp = spm.SentencePieceProcessor()
    sp.load(spm_model_path)
    vocab_size = sp.vocab_size()
    pad_id = 0

    def encode_text(t):
        ids = sp.encode(t, out_type=int)
        ids = pad_seq(ids, args.maxlen, pad=pad_id)
        return ids

    # encode all splits with the SAME tokenizer
    X_train = np.vstack(train_df["text"].apply(encode_text).values)
    y_train = train_df["label"].values
    s_train = train_df["is_synth"].values

    X_val = np.vstack(val_df["text"].apply(encode_text).values)
    y_val = val_df["label"].values
    s_val = val_df["is_synth"].values  # all 0

    X_test = np.vstack(test_df["text"].apply(encode_text).values)
    y_test = test_df["label"].values
    s_test = test_df["is_synth"].values  # all 0

    train_ds = TweetDataset(X_train, y_train, s_train)
    val_ds = TweetDataset(X_val, y_val, s_val)
    test_ds = TweetDataset(X_test, y_test, s_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    model = build_model(args.arch, vocab_size, args.embed_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    best_state = None
    patience_left = args.early_patience
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, class_weights, args.synth_weight, device)
        va_loss, va_acc, va_f1, _, _ = eval_model(model, val_loader, class_weights, args.synth_weight, device)

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_f1": va_f1})

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | "
              f"Val Acc: {va_acc:.4f} | Val Macro F1: {va_f1:.4f}")

        if va_f1 > best_val_f1 + 1e-4:
            best_val_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.early_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[INFO] EarlyStopping triggered at epoch {epoch}. Best val macro-F1: {best_val_f1:.4f}")
                break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # plots
    plot_curves(history, args.out, args.arch)

    # test (REAL)
    te_loss, te_acc, te_f1, y_true, y_pred = eval_model(model, test_loader, class_weights, args.synth_weight, device)
    print("\n=== TEST (REAL ONLY) ===")
    print(f"Test Loss: {te_loss:.4f} | Test Acc: {te_acc:.4f} | Test Macro F1: {te_f1:.4f}\n")

    print(classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["0_olumsuz", "1_olumlu"],
        digits=4
    ))

    save_confusion_matrix(y_true, y_pred, args.out, args.arch)

    # save model
    model_path = os.path.join(args.out, f"{args.arch}_spm_best.pt")
    torch.save(model.state_dict(), model_path)

    cfg = {
        "arch": args.arch,
        "spm_model": spm_model_path,
        "spm_vocab_size": args.spm_vocab_size,
        "maxlen": args.maxlen,
        "pad_id": 0,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "class_weights_real_train": [float(cw[0]), float(cw[1])],
        "synth_weight": float(args.synth_weight),
        "best_val_macro_f1": float(best_val_f1),
    }
    cfg_path = os.path.join(args.out, f"{args.arch}_spm_bundle.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Saved model: {model_path}")
    print(f"[INFO] Saved bundle: {cfg_path}")
    print(f"[INFO] Plots: {os.path.join(args.out, 'plots')}")

if __name__ == "__main__":
    main()
