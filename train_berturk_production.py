# train_berturk_production.py
# Kullanım:
# python train_berturk_production.py --splits_dir splits --out outputs_prod --epochs 8 --batch 16 --lr 1e-5 --maxlen 128 --synth_weight 0.3
#
# splits/ içinde:
#   - train.xlsx  (text,label,is_synth)
#   - val.xlsx    (text,label,is_synth)  -> script is_synth'i 0'a zorlayacak
#   - test.xlsx   (text,label,is_synth)  -> script is_synth'i 0'a zorlayacak

import argparse
import os
import json
import re
import random
import inspect
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)

import evaluate
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
def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def basic_clean(t: str) -> str:
    t = str(t)
    t = re.sub(r"http\S+", "URL", t)
    t = re.sub(r"@\w+", "@USER", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def plot_learning_curves(log_history, out_dir):
    train_epochs, train_losses = [], []
    val_epochs, val_losses = [], []
    f1_epochs, f1_scores = [], []

    for e in log_history:
        if "loss" in e and "epoch" in e and "eval_loss" not in e:
            train_epochs.append(float(e["epoch"]))
            train_losses.append(float(e["loss"]))

        if "eval_loss" in e and "epoch" in e:
            val_epochs.append(float(e["epoch"]))
            val_losses.append(float(e["eval_loss"]))
            if "eval_macro_f1" in e:
                f1_epochs.append(float(e["epoch"]))
                f1_scores.append(float(e["eval_macro_f1"]))

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

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
    plt.savefig(os.path.join(plots_dir, "learning_curve_loss.png"), dpi=160)
    plt.close()

    if f1_scores:
        plt.figure(figsize=(10, 5))
        plt.plot(f1_epochs, f1_scores, label="Val Macro-F1")
        plt.title("Learning Curve - Val Macro F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "learning_curve_macro_f1.png"), dpi=160)
        plt.close()

    print(f"[INFO] Plots saved -> {plots_dir}")

def save_confusion_matrix_png(y_true, y_pred, out_path, labels=("negatif(0)", "pozitif(1)")):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (REAL)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, labels, rotation=20)
    plt.yticks(tick_marks, labels)

    # annotate
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def best_threshold_by_macro_f1(y_true, prob_pos, step=0.005):
    """
    Production için en iyi threshold'u VAL üzerinde macro-F1 maksimize ederek seç.
    Bu, "iki sınıfı da iyi yakala" hedefiyle en uyumlu yöntem.
    """
    best_thr = 0.5
    best_score = -1.0
    thr = 0.0
    while thr <= 1.0:
        y_hat = (prob_pos >= thr).astype(int)
        score = f1_score(y_true, y_hat, average="macro")
        if score > best_score:
            best_score = score
            best_thr = thr
        thr += step
    return float(best_thr), float(best_score)

# ---------------- Custom Trainer ----------------
class RealWorldTrainer(Trainer):
    """
    class weight + sentetik sample weight:
    - class_weights: REAL train üzerinden
    - synth_weight: sentetik örneklere loss çarpanı (0.2-0.6)
    """
    def __init__(self, *args, class_weights=None, synth_weight=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.synth_weight = float(synth_weight)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        is_synth = inputs.pop("is_synth", None)  # 0 real, 1 synth

        outputs = model(**inputs)
        logits = outputs.get("logits")

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight, reduction="none")
        per_sample_loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        if is_synth is not None:
            is_synth = is_synth.to(logits.device).float().view(-1)
            sw = torch.where(
                is_synth > 0.5,
                torch.tensor(self.synth_weight, device=logits.device),
                torch.tensor(1.0, device=logits.device),
            )
            per_sample_loss = per_sample_loss * sw

        loss = per_sample_loss.mean()
        return (loss, outputs) if return_outputs else loss

# ---------------- Args ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--model", type=str, default="dbmdz/bert-base-turkish-uncased")
    ap.add_argument("--out", type=str, default="outputs_prod")

    ap.add_argument("--epochs", type=int, default=8)          # senin grafiğe göre sweet spot
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--maxlen", type=int, default=128)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--synth_weight", type=float, default=0.3)
    ap.add_argument("--truncate_chars", type=int, default=256)

    ap.add_argument("--early_patience", type=int, default=2)
    ap.add_argument("--thr_step", type=float, default=0.005)  # threshold grid step
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    train_path = os.path.join(args.splits_dir, "train.xlsx")
    val_path   = os.path.join(args.splits_dir, "val.xlsx")
    test_path  = os.path.join(args.splits_dir, "test.xlsx")

    train_df = pd.read_excel(train_path)
    val_df   = pd.read_excel(val_path)
    test_df  = pd.read_excel(test_path)

    needed = {"text", "label", "is_synth"}
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = needed - set(d.columns)
        if missing:
            raise ValueError(f"[{name}] Eksik kolonlar: {missing}. Gerekli: text,label,is_synth")

    def prep_df(df):
        df = df.dropna(subset=["text", "label", "is_synth"]).copy()
        df["text"] = df["text"].astype(str).apply(basic_clean)
        if args.truncate_chars and args.truncate_chars > 0:
            df["text"] = df["text"].str.slice(0, args.truncate_chars)
        df["label"] = df["label"].astype(int)
        df["is_synth"] = df["is_synth"].astype(int).clip(0, 1)
        df = df[df["label"].isin([0, 1])].reset_index(drop=True)
        return df

    train_df = prep_df(train_df)
    val_df   = prep_df(val_df)
    test_df  = prep_df(test_df)

    # Production kuralı: val/test real-only
    val_df["is_synth"] = 0
    test_df["is_synth"] = 0

    real_train = train_df[train_df["is_synth"] == 0].reset_index(drop=True)
    synth_train = train_df[train_df["is_synth"] == 1].reset_index(drop=True)

    print("[INFO] ===== SPLITS LOADED =====")
    print(f"[INFO] Train total: {len(train_df)} | real: {len(real_train)} | synth: {len(synth_train)}")
    print(f"[INFO] Val (REAL): {len(val_df)} | Test (REAL): {len(test_df)}")
    print(f"[INFO] synth_weight(loss): {args.synth_weight}")
    print(f"[INFO] epochs={args.epochs} (early_patience={args.early_patience})")
    print(f"[INFO] Overfit guard: dropout={args.dropout}, weight_decay={args.weight_decay}, lr={args.lr}, maxlen={args.maxlen}\n")

    # Class weights -> REAL train
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=real_train["label"].values
    )
    print(f"[INFO] Class weights (REAL train): {cw}")

    def to_ds(df):
        x = df[["text", "label", "is_synth"]].rename(columns={"text": "input", "label": "labels"}).copy()
        return Dataset.from_pandas(x, preserve_index=False)

    ds_train = to_ds(train_df)
    ds_val   = to_ds(val_df)
    ds_test  = to_ds(test_df)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tok_fn(batch):
        return tok(batch["input"], truncation=True, padding=False, max_length=args.maxlen)

    ds_train = ds_train.map(tok_fn, batched=True)
    ds_val   = ds_val.map(tok_fn, batched=True)
    ds_test  = ds_test.map(tok_fn, batched=True)

    config = AutoConfig.from_pretrained(
        args.model,
        num_labels=2,
        id2label={0: "negatif", 1: "pozitif"},
        label2id={"negatif": 0, "pozitif": 1},
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config)

    # Metrics (daha production-odaklı: sınıf bazlı da bas)
    acc_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
        macro_f1 = f1_score(labels, preds, average="macro")
        f1_neg = f1_score(labels, preds, average=None, labels=[0, 1])[0]
        f1_pos = f1_score(labels, preds, average=None, labels=[0, 1])[1]

        # ekstra: precision/recall pos
        p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=[1], average="binary", zero_division=0)

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "f1_neg": f1_neg,
            "f1_pos": f1_pos,
            "pos_precision": float(p),
            "pos_recall": float(r),
        }

    # TrainingArguments uyumluluk (transformers sürüm farkı)
    ta_kwargs = dict(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,

        save_strategy="epoch",
        logging_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",

        warmup_ratio=0.10,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        seed=args.seed,
    )

    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in sig:
        ta_kwargs["eval_strategy"] = "epoch"
    else:
        ta_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**ta_kwargs)

    trainer = RealWorldTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,  # REAL validation
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        class_weights=cw,
        synth_weight=args.synth_weight,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_patience)],
    )

    print("\n[INFO] Training starts...")
    trainer.train()

    # plots
    plot_learning_curves(trainer.state.log_history, args.out)

    # ---- Threshold: VAL üzerinde MACRO-F1 optimize et ----
    print("\n[INFO] Threshold tuning on REAL val (maximize MACRO-F1)...")
    val_pred = trainer.predict(ds_val)
    val_probs = torch.softmax(torch.tensor(val_pred.predictions), dim=1).numpy()
    val_prob_pos = val_probs[:, 1]
    y_val = val_df["label"].values

    best_thr, best_macro = best_threshold_by_macro_f1(y_val, val_prob_pos, step=float(args.thr_step))
    print(f"[INFO] Best threshold (val, macro-F1): {best_thr:.3f} | val macro-F1@thr: {best_macro:.4f}")

    # ---- TEST (REAL) ----
    print("\n=== TEST REPORT (REAL ONLY, macro-F1 tuned threshold) ===")
    test_pred = trainer.predict(ds_test)
    test_probs = torch.softmax(torch.tensor(test_pred.predictions), dim=1).numpy()
    test_prob_pos = test_probs[:, 1]
    y_test = test_df["label"].values
    y_hat = (test_prob_pos >= best_thr).astype(int)

    print(classification_report(
        y_test, y_hat,
        target_names=["0_olumsuz", "1_olumlu"],
        digits=4
    ))

    # confusion matrix png
    plots_dir = os.path.join(args.out, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    cm_path = os.path.join(plots_dir, "confusion_matrix_test.png")
    save_confusion_matrix_png(y_test, y_hat, cm_path)

    # save model + threshold
    trainer.save_model(args.out)
    with open(os.path.join(args.out, "production_bundle.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": best_thr,
                "threshold_objective": "maximize_val_macro_f1",
                "thr_step": float(args.thr_step),
                "synth_weight": float(args.synth_weight),
                "class_weights_real_train": [float(cw[0]), float(cw[1])],
                "train_real_count": int(len(real_train)),
                "train_synth_count": int(len(synth_train)),
                "epochs": int(args.epochs),
                "early_patience": int(args.early_patience),
                "maxlen": int(args.maxlen),
                "dropout": float(args.dropout),
                "weight_decay": float(args.weight_decay),
                "lr": float(args.lr),
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"[INFO] Model saved to: {args.out}")
    print(f"[INFO] Plots saved to: {plots_dir}")
    print(f"[INFO] Bundle saved to: {os.path.join(args.out, 'production_bundle.json')}")

if __name__ == "__main__":
    main()
