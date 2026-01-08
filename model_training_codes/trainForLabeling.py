# train.py
# usage:
#   & ".\.venv\Scripts\python.exe" trainForLabeling.py --csv data\train.xlsx --epochs 4 --batch 16 --lr 2e-5 --maxlen 256 --out outputs_multi
#   & ".\.venv\Scripts\python.exe" trainForLabeling.py --csv data\train.csv  --epochs 4 --batch 16 --lr 2e-5 --maxlen 256 --out outputs_multi

import argparse, os, random
import numpy as np
import pandas as pd
import torch, evaluate
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)

# ---------------- Utils ----------------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/train.xlsx")  # xlsx/xls/csv
    ap.add_argument("--model", type=str, default="dbmdz/bert-base-turkish-uncased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--out", type=str, default="outputs_multi")
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

# ------------ Custom Trainer (weighted CE) ------------
class WeightedCELossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    # HF Trainer bazı ek paramlar geçer -> **kwargs koyduk
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # labels'ı biz işleyeceğiz
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ---------------- Main ----------------
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # 1) Veri yükle ve kolonları kontrol et
    df = load_df(args.csv)
    needed = {"text","universite","label"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Dosyada eksik kolon(lar): {missing}. 'text, universite, label' olmalı.")
    df = df.dropna(subset=list(needed)).reset_index(drop=True)

    # 2) 0/1/9 -> iç indeks 0/1/2 (CUDA hatasını önlemek için)
    df["labels_raw"] = df["label"].astype(int)
    allowed = {0, 1, 9}
    bad = set(df["labels_raw"].unique()) - allowed
    if bad:
        raise SystemExit(f"Beklenmeyen label(lar): {bad}. Sadece 0/1/9 olmalı.")

    to_internal = {0: 0, 1: 1, 9: 2}   # eğitimde kullanılan
    to_external = {0: 0, 1: 1, 2: 9}   # rapor için (gerekirse)
    df["labels"] = df["labels_raw"].map(to_internal)

    # hedef-farkındalıklı giriş
    df["input"] = "Üniversite: " + df["universite"].astype(str) + " || " + df["text"].astype(str)

    # 3) Stratified split (labels: 0/1/2)
    X_temp, X_test = train_test_split(df, test_size=args.test_size, stratify=df["labels"], random_state=args.seed)
    val_ratio = args.val_size / (1 - args.test_size)
    X_train, X_val = train_test_split(X_temp, test_size=val_ratio, stratify=X_temp["labels"], random_state=args.seed)

    # 4) HF Datasets
    def to_ds(x): return Dataset.from_pandas(x[["input","labels"]].copy(), preserve_index=False)
    ds_train, ds_val, ds_test = map(to_ds, [X_train, X_val, X_test])

    # 5) Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    def tok_fn(batch):
        return tok(batch["input"], truncation=True, padding=False, max_length=args.maxlen)
    print("Tokenizing TRAIN...")
    ds_train = ds_train.map(tok_fn, batched=True)
    print("Tokenizing VAL...")
    ds_val   = ds_val.map(tok_fn, batched=True)
    print("Tokenizing TEST...")
    ds_test  = ds_test.map(tok_fn, batched=True)
    print("Tokenization done.")

    # 6) Model (3 sınıf; iç etiket 0/1/2)
    id2label = {0: "negatif", 1: "pozitif", 2: "alakasiz"}
    label2id = {v:k for k,v in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    # 7) Class weights (0/1/2)
    classes_arr = np.array([0, 1, 2])
    cw = compute_class_weight(class_weight="balanced", classes=classes_arr, y=X_train["labels"].values)
    print(f"Class weights (internal 0,1,2): {cw}")

    # 8) Metrikler
    acc = evaluate.load("accuracy")
    f1  = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "macro_f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"]
        }

    # 9) TrainingArguments (sende sorunsuz çalışan minimal sürüm)
    args_tr = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,

        logging_steps=25,
        disable_tqdm=False,
        report_to="none",

        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        # İstersen:
        # max_grad_norm=1.0,
        # dataloader_num_workers=0,
    )

    trainer = WeightedCELossTrainer(
        model=model,
        args=args_tr,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,  # HF 5.0'da processing_class'a geçilecek; şimdilik uyarı gelebilir
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        class_weights=cw
    )

    # 10) Train
    trainer.train()

    # 11) Test raporu (iç etiket 0/1/2; adları 0/1/9 olarak yazıyoruz)
    print("\n=== TEST RAPORU ===")
    test_pred = trainer.predict(ds_test)
    preds_internal = np.argmax(test_pred.predictions, axis=1)
    y_true_internal = X_test["labels"].to_numpy()

    print(classification_report(
        y_true_internal, preds_internal,
        labels=[0,1,2],
        target_names=["0_olumsuz","1_olumlu","9_alakasiz"],
        digits=4
    ))

    # 12) Kaydet
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"\nModel ve tokenizer '{args.out}' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()
