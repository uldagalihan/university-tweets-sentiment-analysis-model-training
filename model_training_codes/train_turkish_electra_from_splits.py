# train_turkish_electra_from_splits.py
# Turkish ELECTRA FULL fine-tune (binary 0/1) - SENİN BERTweet/BERTürk mantığının birebiri
#
# SPLITS FORMAT:
#   splits/
#     - train.xlsx  (text, label, is_synth)
#     - val.xlsx    (text, label, is_synth=0 only)
#     - test.xlsx   (text, label, is_synth=0 only)
#
# Usage:
# python train_turkish_electra_from_splits.py \
#   --splits_dir splits \
#   --out outputs_electra_prod \
#   --epochs 12 \
#   --batch 8 \
#   --lr 1e-5 \
#   --maxlen 256 \
#   --synth_weight 0.3 \
#   --patience 2 \
#   --warmup_ratio 0.1 \
#   --weight_decay 0.05 \
#   --dropout 0.3
#
# Notlar:
# - EarlyStopping + load_best_model_at_end ile 12 epoch'a kadar dener,
#   en iyi VAL macro_f1'e göre "en optimum + en az overfit" modeli seçer.
# - Train'de REAL+SYNTH var; loss içinde SYNTH örneklerine synth_weight uygulanır.
# - Val/Test sadece REAL olmalı (script bunu kontrol ediyor).
# - Plotlar: train/val loss + val macro-f1 + overfit gap grafiği + confusion matrix.

import argparse
import os
import random
import json
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import evaluate

from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

# ---------------- Utils ----------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_split_xlsx(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"[ERROR] Split dosyası yok: {path}")
    return pd.read_excel(path)

def ensure_cols(df: pd.DataFrame, name: str):
    needed = {"text", "label", "is_synth"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"[ERROR] {name} split eksik kolon(lar): {missing} | gerekli: {needed}")

def clean_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    ensure_cols(df, name)
    df = df.dropna(subset=["text", "label", "is_synth"]).copy()
    df["label"] = df["label"].astype(int)
    df["is_synth"] = df["is_synth"].astype(int)

    bad = set(df["label"].unique()) - {0, 1}
    if bad:
        raise SystemExit(f"[ERROR] {name} split label hatalı: {bad} (sadece 0/1 olmalı)")
    bads = set(df["is_synth"].unique()) - {0, 1}
    if bads:
        raise SystemExit(f"[ERROR] {name} split is_synth hatalı: {bads} (sadece 0/1 olmalı)")

    df["text"] = df["text"].astype(str)
    return df.reset_index(drop=True)

def plot_learning_curves(log_history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    train_epochs, train_losses = [], []
    val_epochs, val_losses = [], []
    f1_epochs, val_f1s = [], []

    for e in log_history:
        # train loss
        if "loss" in e and "epoch" in e and "eval_loss" not in e:
            train_epochs.append(float(e["epoch"]))
            train_losses.append(float(e["loss"]))

        # eval loss + macro_f1
        if "eval_loss" in e and "epoch" in e:
            val_epochs.append(float(e["epoch"]))
            val_losses.append(float(e["eval_loss"]))
            if "eval_macro_f1" in e:
                f1_epochs.append(float(e["epoch"]))
                val_f1s.append(float(e["eval_macro_f1"]))

    # Loss curve
    plt.figure(figsize=(10, 5))
    if train_losses:
        plt.plot(train_epochs, train_losses, label="Train Loss")
    if val_losses:
        plt.plot(val_epochs, val_losses, label="Val Loss", linestyle="--")
    plt.title("Learning Curve - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curve_loss.png"), dpi=160)
    plt.close()

    # Val macro-F1 curve
    if val_f1s:
        plt.figure(figsize=(10, 5))
        plt.plot(f1_epochs, val_f1s, label="Val Macro-F1")
        plt.title("Learning Curve - Val Macro F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "learning_curve_val_macro_f1.png"), dpi=160)
        plt.close()

    # Overfit gap curve (val_loss - train_loss) approx by epoch matching
    if train_losses and val_losses:
        # basit eşleştirme: en yakın epoch değerleriyle
        gap_x, gap_y = [], []
        for ve, vl in zip(val_epochs, val_losses):
            # find nearest train epoch
            idx = int(np.argmin([abs(te - ve) for te in train_epochs]))
            tl = train_losses[idx]
            gap_x.append(ve)
            gap_y.append(vl - tl)

        plt.figure(figsize=(10, 5))
        plt.plot(gap_x, gap_y, label="(Val Loss - Train Loss)")
        plt.axhline(0.0, linewidth=1)
        plt.title("Overfitting Gap (Val Loss - Train Loss)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Gap")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "overfit_gap.png"), dpi=160)
        plt.close()

def tune_threshold_on_val(probs_pos: np.ndarray, y_true: np.ndarray):
    """
    VAL üzerinde Macro-F1 maksimize eden threshold bul.
    """
    best_thr = 0.5
    best_f1 = -1.0

    def macro_f1_at(thr: float):
        y_pred = (probs_pos >= thr).astype(int)
        return f1_score(y_true, y_pred, average="macro")

    for thr in np.linspace(0.05, 0.95, 91):
        score = macro_f1_at(float(thr))
        if score > best_f1:
            best_f1 = float(score)
            best_thr = float(thr)

    return best_thr, best_f1

def plot_confusion(cm, out_path, title="Confusion Matrix (REAL)"):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["negatif(0)", "pozitif(1)"], rotation=20)
    plt.yticks([0, 1], ["negatif(0)", "pozitif(1)"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# ---------------- Collator: keep is_synth ----------------
class DataCollatorWithPaddingAndSynth:
    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        is_synth = [int(f["is_synth"]) for f in features]
        # string kolonlar pad'e girmesin
        for f in features:
            f.pop("text", None)
        batch = self.base(features)
        batch["is_synth"] = torch.tensor(is_synth, dtype=torch.long)
        return batch

# ---------------- Custom Trainer: class weight + synth weight ----------------
class WeightedSynthTrainer(Trainer):
    """
    - class weights: sadece REAL train üzerinden
    - synth_weight: sentetik örneklerin loss çarpanı (0.2-0.6 mantıklı)
    """
    def __init__(self, *args, class_weights=None, synth_weight=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.synth_weight = float(synth_weight)
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        is_synth = inputs.pop("is_synth")  # [B]
        outputs = model(**inputs)
        logits = outputs.get("logits")     # [B,2]

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce = nn.CrossEntropyLoss(weight=weight, reduction="none")
        per_sample = ce(logits, labels)  # [B]

        sw = torch.ones_like(per_sample)
        sw = torch.where(is_synth == 1, sw * self.synth_weight, sw)
        loss = (per_sample * sw).mean()

        return (loss, outputs) if return_outputs else loss

# ---------------- Args ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", type=str, required=True)

    # ELECTRA modeli: drop-in değiştirilebilir
    ap.add_argument("--model", type=str, default="dbmdz/electra-turkish-base-discriminator")
    ap.add_argument("--out", type=str, default="outputs_electra_prod")

    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--grad_accum", type=int, default=1)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=2)

    ap.add_argument("--synth_weight", type=float, default=0.3)
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()

# ---------------- Main ----------------
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_path = os.path.join(args.splits_dir, "train.xlsx")
    val_path = os.path.join(args.splits_dir, "val.xlsx")
    test_path = os.path.join(args.splits_dir, "test.xlsx")

    train_df = clean_df(read_split_xlsx(train_path), "train")
    val_df = clean_df(read_split_xlsx(val_path), "val")
    test_df = clean_df(read_split_xlsx(test_path), "test")

    # Val/Test real-only kontrol
    if val_df["is_synth"].sum() != 0:
        raise SystemExit("[ERROR] val.xlsx içinde is_synth=1 var! Val sadece REAL olmalı.")
    if test_df["is_synth"].sum() != 0:
        raise SystemExit("[ERROR] test.xlsx içinde is_synth=1 var! Test sadece REAL olmalı.")

    print("[INFO] ===== SPLITS LOADED =====")
    print(f"[INFO] Train total: {len(train_df)} | real: {(train_df.is_synth==0).sum()} | synth: {(train_df.is_synth==1).sum()}")
    print(f"[INFO] Val (REAL): {len(val_df)} | Test (REAL): {len(test_df)}")
    print(f"[INFO] model: {args.model}")
    print(f"[INFO] synth_weight(loss): {args.synth_weight}")
    print(f"[INFO] Overfit guard: dropout={args.dropout}, weight_decay={args.weight_decay}, lr={args.lr}, maxlen={args.maxlen}")
    print(f"[INFO] epochs={args.epochs} (early patience={args.patience}) | batch={args.batch} | grad_accum={args.grad_accum}")
    print(f"[INFO] device={device}\n")

    # class weights sadece REAL train üzerinden
    real_train = train_df[train_df["is_synth"] == 0]
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=real_train["label"].values
    )
    print(f"[INFO] Class weights (REAL train): {cw}\n")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def to_ds(df: pd.DataFrame) -> Dataset:
        tmp = df[["text", "label", "is_synth"]].copy()
        tmp = tmp.rename(columns={"label": "labels"})
        return Dataset.from_pandas(tmp, preserve_index=False)

    ds_train = to_ds(train_df)
    ds_val = to_ds(val_df)
    ds_test = to_ds(test_df)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, padding=False, max_length=args.maxlen)

    # text'i dataset'ten kaldır (string kolonu batch'e girmesin)
    ds_train = ds_train.map(tok_fn, batched=True, remove_columns=["text"])
    ds_val   = ds_val.map(tok_fn, batched=True, remove_columns=["text"])
    ds_test  = ds_test.map(tok_fn, batched=True, remove_columns=["text"])

    config = AutoConfig.from_pretrained(
        args.model,
        num_labels=2,
        id2label={0: "negatif", 1: "pozitif"},
        label2id={"negatif": 0, "pozitif": 1},
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config).to(device)

    # Metrics
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "macro_f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    # TrainingArguments uyumluluk (transformers sürüm farkı: eval_strategy vs evaluation_strategy)
    ta_kwargs = dict(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,

        save_strategy="epoch",
        logging_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=2,
        report_to="none",
        fp16=(torch.cuda.is_available() and not args.cpu),
        disable_tqdm=False,

        # is_synth'i korumak için false kalmalı
        remove_unused_columns=False,
        seed=args.seed,
    )

    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in sig:
        ta_kwargs["eval_strategy"] = "epoch"
    else:
        ta_kwargs["evaluation_strategy"] = "epoch"

    tr_args = TrainingArguments(**ta_kwargs)

    collator = DataCollatorWithPaddingAndSynth(tok)

    trainer = WeightedSynthTrainer(
        model=model,
        args=tr_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=cw,
        synth_weight=args.synth_weight,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    print("[INFO] Training...")
    trainer.train()

    # Plotlar
    plots_dir = os.path.join(args.out, "plots")
    plot_learning_curves(trainer.state.log_history, plots_dir)
    print(f"[INFO] Plots saved -> {plots_dir}")

    # Threshold tuning on VAL (REAL)
    print("\n[INFO] Threshold tuning on REAL val (maximize MACRO-F1)...")
    val_out = trainer.predict(ds_val)
    val_logits = val_out.predictions
    val_probs = torch.softmax(torch.tensor(val_logits), dim=1).numpy()
    val_prob_pos = val_probs[:, 1]
    y_val = val_df["label"].to_numpy()

    best_thr, best_val_f1 = tune_threshold_on_val(val_prob_pos, y_val)
    print(f"[INFO] Best threshold (val, macro-F1): {best_thr:.3f} | val macro-F1@thr: {best_val_f1:.4f}")

    thr_path = os.path.join(args.out, "inference_threshold.json")
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": best_thr, "val_macro_f1": best_val_f1}, f, ensure_ascii=False, indent=2)

    # TEST (REAL)
    print("\n=== TEST RAPORU (REAL ONLY, macro-F1 tuned threshold) ===")
    test_out = trainer.predict(ds_test)
    test_logits = test_out.predictions
    test_probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
    test_prob_pos = test_probs[:, 1]
    y_test = test_df["label"].to_numpy()

    y_pred_thr = (test_prob_pos >= best_thr).astype(int)

    print(classification_report(
        y_test, y_pred_thr,
        labels=[0, 1],
        target_names=["0_olumsuz", "1_olumlu"],
        digits=4
    ))

    cm = confusion_matrix(y_test, y_pred_thr, labels=[0, 1])
    plot_confusion(cm, os.path.join(plots_dir, "confusion_matrix_real.png"), title="Confusion Matrix (REAL)")

    # Save model + tokenizer
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    bundle = {
        "model_name": args.model,
        "out_dir": args.out,
        "maxlen": args.maxlen,
        "threshold": best_thr,
        "synth_weight": args.synth_weight,
        "class_weights_real_train": [float(cw[0]), float(cw[1])],
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "early_stopping_patience": args.patience
    }

    bundle_path = os.path.join(args.out, "production_bundle.json")
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Model saved to: {args.out}")
    print(f"[INFO] Threshold saved to: {thr_path}")
    print(f"[INFO] Bundle saved to: {bundle_path}")
    print(f"[INFO] Confusion matrix saved to: {os.path.join(plots_dir, 'confusion_matrix_real.png')}")

if __name__ == "__main__":
    main()
