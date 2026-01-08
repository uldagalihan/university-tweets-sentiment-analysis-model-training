# make_splits_minhash_excel_separate.py
# Amaç:
# - TEST ve VAL: SADECE REAL (is_synth=0) olacak
# - TEST/VAL büyüklüğü: TOPLAM veri sayısının oranı (örn 11k -> %10 test=1100, %10 val=1100)
# - TRAIN: kalan REAL + (isteğe bağlı) SYNTH
#
# ÇIKTI (ayrı dosyalar):
#   out_dir/
#     - train.xlsx
#     - val.xlsx
#     - test.xlsx
#
# Kullanım:
# python make_splits_minhash_excel_separate.py --input tweetVeriseti.xlsx --out splits --test 0.10 --val 0.10
# python make_splits_minhash_excel_separate.py --input tweetVeriseti.xlsx --out splits --synth_keep 0.5

import argparse
import os
import re
import random
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import GroupShuffleSplit


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="xlsx/csv: text,label,is_synth")
    ap.add_argument("--out", type=str, default="splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test", type=float, default=0.10, help="TOPLAM veri üzerinden oran (REAL'den seçilecek)")
    ap.add_argument("--val", type=float, default=0.10, help="TOPLAM veri üzerinden oran (REAL'den seçilecek)")
    ap.add_argument("--threshold", type=float, default=0.82)
    ap.add_argument("--num_perm", type=int, default=64)
    ap.add_argument("--max_chars", type=int, default=256)
    ap.add_argument("--synth_keep", type=float, default=1.0, help="0-1: train'e alınacak sentetik oranı")
    return ap.parse_args()


def normalize_for_minhash(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"http\S+", "URL", t)
    t = re.sub(r"@\w+", "@USER", t)
    t = re.sub(r"\d+", "NUM", t)
    t = re.sub(r"[^\w\sçğıöşü]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def shingles_words(t: str, k: int = 4):
    words = t.split()
    if len(words) <= k:
        return [" ".join(words)]
    return [" ".join(words[i:i + k]) for i in range(len(words) - k + 1)]


def build_minhash_clusters(texts, threshold=0.82, num_perm=64):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []

    print("[INFO] MinHash oluşturuluyor (REAL üzerinde)...")
    for i, txt in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        nt = normalize_for_minhash(txt)
        for sh in shingles_words(nt, k=4):
            m.update(sh.encode("utf-8"))
        lsh.insert(str(i), m)
        minhashes.append(m)
        if i % 1000 == 0 and i > 0:
            print(f"  -> {i} real tweet işlendi")

    print("[INFO] Cluster'lar birleştiriliyor (Union-Find)...")

    parent = list(range(len(texts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, m in enumerate(minhashes):
        hits = lsh.query(m)
        for h in hits:
            j = int(h)
            if i != j:
                union(i, j)

    root_to_id = {}
    cluster_ids = []
    cid = 0
    for i in range(len(texts)):
        r = find(i)
        if r not in root_to_id:
            root_to_id[r] = cid
            cid += 1
        cluster_ids.append(root_to_id[r])

    print(f"[INFO] Real üzerinde cluster sayısı: {cid}")
    return cluster_ids, cid


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.input.endswith(".xlsx"):
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input)

    # basic checks
    needed = {"text", "label", "is_synth"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}. Gerekli: text,label,is_synth")

    df = df.dropna(subset=["text", "label", "is_synth"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["is_synth"] = df["is_synth"].astype(int)

    # truncate
    df["text"] = df["text"].astype(str)
    if args.max_chars and args.max_chars > 0:
        df["text"] = df["text"].str.slice(0, args.max_chars)

    df_real = df[df["is_synth"] == 0].reset_index(drop=True)
    df_syn = df[df["is_synth"] == 1].reset_index(drop=True)

    total_n = len(df)
    real_n = len(df_real)
    syn_n = len(df_syn)

    print(f"[INFO] Total: {total_n} | Real: {real_n} | Synth: {syn_n}")

    # Hedef: TOPLAM üzerinden sayıya çevir, ama REAL'den seçilecek
    test_n = int(round(total_n * args.test))
    val_n = int(round(total_n * args.val))

    # En az 1 olsun (istersen bu kısmı kaldırabilirsin)
    test_n = max(1, test_n) if args.test > 0 else 0
    val_n = max(1, val_n) if args.val > 0 else 0

    print(f"[INFO] Hedef sayılar (TOPLAM üzerinden): test={test_n}, val={val_n} (ikisi de REAL'den seçilecek)")

    if test_n + val_n > real_n:
        raise ValueError(
            f"REAL veri yetersiz! test_n({test_n}) + val_n({val_n}) = {test_n + val_n} "
            f"ama real_n={real_n}. (Test/Val sadece real olacak şekilde istedin.)"
        )

    # cluster only on real
    clusters, n_clusters = build_minhash_clusters(
        df_real["text"].tolist(),
        threshold=args.threshold,
        num_perm=args.num_perm
    )
    df_real["cluster_id"] = clusters

    # 1) REAL -> TEST (sayıyla)
    if test_n > 0:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_n, random_state=args.seed)
        trval_idx, test_idx = next(
            gss1.split(df_real, df_real["label"], groups=df_real["cluster_id"])
        )
        real_trval = df_real.iloc[trval_idx].reset_index(drop=True)
        real_test = df_real.iloc[test_idx].reset_index(drop=True)
    else:
        real_trval = df_real.copy()
        real_test = df_real.iloc[0:0].copy()

    # 2) REAL_TRVAL -> VAL (sayıyla)
    if val_n > 0:
        if val_n > len(real_trval):
            raise ValueError(f"VAL için yeterli real_trval yok: val_n={val_n}, real_trval={len(real_trval)}")
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_n, random_state=args.seed)
        train_idx, val_idx = next(
            gss2.split(real_trval, real_trval["label"], groups=real_trval["cluster_id"])
        )
        real_train = real_trval.iloc[train_idx].reset_index(drop=True)
        real_val = real_trval.iloc[val_idx].reset_index(drop=True)
    else:
        real_train = real_trval.copy()
        real_val = real_trval.iloc[0:0].copy()

    # synth keep ratio (opsiyonel)
    if syn_n > 0 and args.synth_keep < 1.0:
        keep_n = int(round(syn_n * max(0.0, min(1.0, args.synth_keep))))
        df_syn = df_syn.sample(n=keep_n, random_state=args.seed).reset_index(drop=True)

    # final train = real_train + synth
    train = pd.concat([real_train.drop(columns=["cluster_id"]), df_syn], ignore_index=True)
    val = real_val.drop(columns=["cluster_id"]).copy()
    test = real_test.drop(columns=["cluster_id"]).copy()

    # güvenlik: val/test real-only
    if len(val) > 0:
        val["is_synth"] = 0
    if len(test) > 0:
        test["is_synth"] = 0

    # report
    print("\n===== SPLIT RAPORU =====")
    print(f"Real clusters: {n_clusters}")
    print(f"Train: {len(train)} (real_train={len(real_train)} + synth_used={len(df_syn)})")
    print(f"Val  : {len(val)} (REAL only hedef ~{val_n})")
    print(f"Test : {len(test)} (REAL only hedef ~{test_n})")

    if len(train) > 0:
        print("\nLabel oranları:")
        print("Train:\n", train["label"].value_counts(normalize=True))
    if len(val) > 0:
        print("Val:\n", val["label"].value_counts(normalize=True))
    if len(test) > 0:
        print("Test:\n", test["label"].value_counts(normalize=True))

    # save (AYRI EXCEL)
    os.makedirs(args.out, exist_ok=True)
    train_path = os.path.join(args.out, "train.xlsx")
    val_path = os.path.join(args.out, "val.xlsx")
    test_path = os.path.join(args.out, "test.xlsx")

    train.to_excel(train_path, index=False)
    val.to_excel(val_path, index=False)
    test.to_excel(test_path, index=False)

    print("\n[INFO] Dosyalar yazıldı:")
    print(f" - {train_path}")
    print(f" - {val_path}")
    print(f" - {test_path}")


if __name__ == "__main__":
    main()
