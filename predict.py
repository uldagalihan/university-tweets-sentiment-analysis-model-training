# predict.py
# usage (CSV):
#   & ".\.venv\Scripts\python.exe" predict.py --model_dir outputs_multi --inp data\raw.csv --out data\raw_labeled.csv --maxlen 256 --thresh 0.58
# usage (XLSX):
#   & ".\.venv\Scripts\python.exe" predict.py --model_dir outputs_multi --inp data\raw.xlsx --out data\raw_labeled.xlsx --maxlen 256 --thresh 0.58

import argparse, os, re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

OUTPUT_COL_ORDER = [
    "type", "id", "url", "tags", "text",
    "createdAt", "location", "authorUserName",
    "university", "group"
]

def load_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def save_df(df: pd.DataFrame, path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)

def clean_text_minimal(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r'http\S+|www\.\S+', ' <url> ', s)
    s = re.sub(r'@\w+', '@user', s)
    s = re.sub(r'#([0-9a-z_çğıöşü]+)', r'\1', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="train.py çıktısı klasörü (örn: outputs_multi)")
    ap.add_argument("--inp", required=True, help="xlsx/csv giriş (kolonlar içinde en az: text, university)")
    ap.add_argument("--out", required=True, help="xlsx/csv çıkış (şema sabit: type,id,url,tags,text,createdAt,location,authorUserName,university,group)")
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--thresh", type=float, default=0.58, help="failsafe: max_prob<thresh ise 0/1 -> 9 yapılır")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    # model + tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # data yükle
    df = load_df(args.inp)

    # Gerekli alanlar
    needed_for_pred = {"text", "university"}
    missing = needed_for_pred - set(df.columns)
    if missing:
        raise SystemExit(f"Eksik kolon(lar): {missing}. Tahmin için gerekli: {needed_for_pred}")

    # input hazırlama (sadece text + university kullanılır)
    texts = [clean_text_minimal(x) for x in df["text"].astype(str).tolist()]
    unis  = df["university"].astype(str).tolist()
    inputs = [f"Üniversite: {u} || {t}" for u, t in zip(unis, texts)]

    # inference
    preds_internal, prob_max_list = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), args.batch), desc="Predict"):
            chunk = inputs[i:i+args.batch]
            enc = tok(
                chunk,
                truncation=True,
                padding=True,
                max_length=args.maxlen,
                return_tensors="pt"
            ).to(device)

            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()  # shape: [B, 3]
            pred_i = probs.argmax(axis=1)                            # 0/1/2 (internal)
            mx     = probs.max(axis=1)

            preds_internal.extend(pred_i.tolist())
            prob_max_list.extend(mx.tolist())

    # internal 0/1/2 -> external 0/1/9
    to_external = {0: 0, 1: 1, 2: 9}
    preds_external = [to_external[p] for p in preds_internal]

    # failsafe: düşük güvenli 0/1'i 9'a çevir
    final_preds = []
    for lab, mx in zip(preds_external, prob_max_list):
        if lab in (0, 1) and mx < args.thresh:
            final_preds.append(9)
        else:
            final_preds.append(lab)

    # ÇIKTI: sadece istenen sütunlar, 'tags' = tahmin (0/1/9)
    out_df = df.copy()

    # 'tags' sütununu oluştur/üzerine yaz (int 0/1/9)
    out_df["tags"] = final_preds

    # Şema garantisi: eksik olan sütunları boş oluştur
    for col in OUTPUT_COL_ORDER:
        if col not in out_df.columns:
            out_df[col] = pd.NA

    # Son olarak kolonları istenen sıraya diz
    out_df = out_df[OUTPUT_COL_ORDER]

    save_df(out_df, args.out)
    print(f"Saved -> {args.out}")

if __name__ == "__main__":
    main()
