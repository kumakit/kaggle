"""
005_name-that-llm: Exp #005 - Colab L4 GPU Two-Stage Open Pool + 5-Fold ModernBERT-base
Competition: Name That LLM
Target Metric: Balanced Accuracy (18 Classes)
Architecture:
  - Stage 1: Domain fine-tuning on curated crtproduction/name-that-llm-open-pool
             (kimi-k3 filtered to gen_reasoning=='high', excluding prvsiyan,
              match_weight typography sampling capped per label, balanced class weights)
  - Stage 2: 5-Fold StratifiedGroupKFold (prompt_id grouped, random_state=42) on train.csv
  - Export : OOF probabilities (Base64 + CSV) and 5-fold float16 weights -> Kaggle Private Dataset
"""

import os
import sys
import subprocess
import json
import base64
import time
import math
import random
import zipfile
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
MODEL_NAME = "answerdotai/ModernBERT-base"
DATASET_SLUG = "nobuyukikumagai/name-that-llm-exp005-weights"
MAX_LEN = 416
CAP_PER_LABEL = 4000

BATCH_SIZE_S1 = 48
EPOCHS_S1 = 3
LR_S1 = 4e-5

BATCH_SIZE_S2 = 16
EPOCHS_S2 = 6
LR_S2 = 1.5e-5


def seed_everything(seed=SEED):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_environment():
    print("[INIT] Setting up environment and packages...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"], check=False)
    packages = ["transformers>=4.48.0", "accelerate", "kaggle"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U"] + packages, check=True)

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(kaggle_json):
        for drive_cand in [
            "/content/drive/MyDrive/.kaggle/kaggle.json",
            "/content/drive/MyDrive/kaggle.json",
        ]:
            if os.path.exists(drive_cand):
                import shutil
                shutil.copy2(drive_cand, kaggle_json)
                os.chmod(kaggle_json, 0o600)
                print(f"  - Copied kaggle credentials from {drive_cand}", flush=True)
                break
    if not os.path.exists(kaggle_json):
        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")
        if not (username and key):
            try:
                from google.colab import userdata
                username = username or userdata.get("KAGGLE_USERNAME")
                key = key or userdata.get("KAGGLE_KEY")
            except Exception:
                pass
        if not (username and key):
            raise RuntimeError(
                "Missing Kaggle credentials. Configure ~/.kaggle/kaggle.json, "
                "Google Drive kaggle.json, Colab Secrets, or KAGGLE_USERNAME/KAGGLE_KEY env vars."
            )
        with open(kaggle_json, "w") as f:
            json.dump({"username": username.strip(), "key": key.strip()}, f)
        os.chmod(kaggle_json, 0o600)
        print(f"  - Configured kaggle credentials for {username}", flush=True)


def download_data():
    data_dir = "/content/data"
    pool_dir = "/content/open_pool"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(pool_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        print("[DATA] Downloading name-that-llm competition dataset...")
        subprocess.run(["kaggle", "competitions", "download", "-c", "name-that-llm", "-p", data_dir], check=True)
        zip_path = os.path.join(data_dir, "name-that-llm.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(data_dir)
            os.remove(zip_path)

    pool_parquet = os.path.join(pool_dir, "ntl_open_pool.parquet")
    if not os.path.exists(pool_parquet):
        print("[DATA] Downloading Open Pool (ntl_open_pool.parquet)...")
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "crtproduction/name-that-llm-open-pool",
                "-f", "ntl_open_pool.parquet",
                "-p", pool_dir,
            ],
            check=True,
        )
        for fname in os.listdir(pool_dir):
            if fname.endswith(".zip"):
                zpath = os.path.join(pool_dir, fname)
                with zipfile.ZipFile(zpath, "r") as z:
                    z.extractall(pool_dir)
                os.remove(zpath)

    assert os.path.exists(train_path), f"Missing {train_path}"
    assert os.path.exists(pool_parquet), f"Missing {pool_parquet}"
    return data_dir, pool_parquet


def get_cosine_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps):
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def curate_open_pool(pool_path, classes, cap_per_label=CAP_PER_LABEL):
    print(f"[POOL] Loading {pool_path}...")
    cols = [
        "id", "text", "model", "source_dataset", "prompt_canon",
        "match_weight", "gen_reasoning", "flag_near_train_prompt", "flag_train_prompt_regen"
    ]
    pool = pd.read_parquet(pool_path, columns=cols)
    print(f"  - Raw clean pool rows: {len(pool):,}")

    # Safety filters against train leakage & nulls
    pool = pool.dropna(subset=["text", "model"]).copy()
    pool = pool[pool["model"].isin(classes)]
    if "flag_near_train_prompt" in pool.columns:
        pool = pool[~pool["flag_near_train_prompt"].fillna(False)]
    if "flag_train_prompt_regen" in pool.columns:
        pool = pool[~pool["flag_train_prompt_regen"].fillna(False)]

    # Fidelity-informed filters from Open Pool v14 audit
    # 1) Drop prvsiyan's low-fidelity 266 gemini-3.7-flash rows (0.3342 fidelity)
    pool = pool[pool["source_dataset"] != "prvsiyan/prvsiyans-extended-name-that-llm-dataset"]
    # 2) For kimi-k3, keep gen_reasoning == 'high' (0.8092 fidelity vs 0.11-0.36 otherwise)
    is_kimi = pool["model"] == "kimi-k3"
    pool = pool[(~is_kimi) | (pool["gen_reasoning"] == "high")]
    # 3) For qwen3.8-max, drop gen_reasoning == 'low' (70 rows, 0.0 fidelity)
    is_qwen_low = (pool["model"] == "qwen3.8-max") & (pool["gen_reasoning"] == "low")
    pool = pool[~is_qwen_low]

    pool["match_weight"] = pd.to_numeric(pool["match_weight"], errors="coerce").fillna(1.0).clip(lower=1e-4)

    sampled_parts = []
    for cls in classes:
        sub = pool[pool["model"] == cls]
        if len(sub) > cap_per_label:
            weights = sub["match_weight"].values
            weights = weights / weights.sum()
            sub = sub.sample(n=cap_per_label, replace=False, weights=weights, random_state=SEED)
        sampled_parts.append(sub)

    curated = pd.concat(sampled_parts, axis=0).reset_index(drop=True)
    # Assign group for internal Stage 1 holdout split
    curated["group_key"] = np.where(
        curated["prompt_canon"].fillna("").astype(str).str.len() > 0,
        curated["prompt_canon"].astype(str),
        curated["id"].astype(str),
    )
    print(f"  - Curated Stage 1 rows: {len(curated):,}")
    print("  - Class min/max counts:", curated["model"].value_counts().min(), "/", curated["model"].value_counts().max())
    return curated


def build_tokenized_dataset(texts, labels, tokenizer, max_len=MAX_LEN):
    import torch
    from torch.utils.data import Dataset

    enc = tokenizer(
        [str(t) for t in texts],
        truncation=True,
        max_length=max_len,
        padding=False,
    )

    class PreTokenizedDataset(Dataset):
        def __init__(self, input_ids, attention_mask, y=None):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.y = y

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            item = {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
            }
            if self.y is not None:
                item["labels"] = int(self.y[idx])
            return item

    return PreTokenizedDataset(enc["input_ids"], enc["attention_mask"], labels)


def evaluate_model(model, loader, device, autocast_dtype, use_bf16):
    import torch
    model.eval()
    probs_list = []
    targets_list = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits.float(), dim=-1)
            probs_list.append(probs.cpu().numpy())
            if "labels" in batch:
                targets_list.append(batch["labels"].cpu().numpy())
    probs_arr = np.concatenate(probs_list, axis=0)
    targets_arr = np.concatenate(targets_list, axis=0) if targets_list else None
    return probs_arr, targets_arr


def upload_private_dataset(export_dir, dataset_slug, message):
    meta = {
        "title": "Name That LLM Exp005 ModernBERT Weights",
        "id": dataset_slug,
        "licenses": [{"name": "CC-BY-SA-4.0"}],
    }
    meta_path = os.path.join(export_dir, "dataset-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[UPLOAD] Uploading {export_dir} to Kaggle Dataset {dataset_slug}...")
    check = subprocess.run(["kaggle", "datasets", "status", dataset_slug], capture_output=True, text=True)
    if check.returncode == 0:
        cmd = ["kaggle", "datasets", "version", "-p", export_dir, "-m", message, "-r", "zip", "-t"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", export_dir, "-r", "zip", "-t"]

    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"[UPLOAD WARN] stderr: {res.stderr}")
        # Retry with version in case status check failed on newly existing slug
        if "create" in cmd:
            res2 = subprocess.run(
                ["kaggle", "datasets", "version", "-p", export_dir, "-m", message, "-r", "zip", "-t"],
                capture_output=True, text=True
            )
            print(res2.stdout)
            if res2.returncode != 0:
                print(f"[UPLOAD ERR] stderr: {res2.stderr}")


def train_and_evaluate():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, DataCollatorWithPadding
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import balanced_accuracy_score

    seed_everything(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"[DEVICE] {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | bf16={use_bf16}")

    data_dir, pool_path = download_data()
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    classes = sorted(train_df["model"].unique().tolist())
    n_classes = len(classes)
    label2idx = {cls: i for i, cls in enumerate(classes)}
    idx2label = {i: cls for i, cls in enumerate(classes)}
    train_df["label"] = train_df["model"].map(label2idx)

    print(f"[MODEL] Tokenizer & Config: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    # =========================================================
    # STAGE 1: Open Pool Pretraining / Domain Fine-Tuning
    # =========================================================
    pool_df = curate_open_pool(pool_path, classes, cap_per_label=CAP_PER_LABEL)
    pool_df["label"] = pool_df["model"].map(label2idx)

    # Grouped 95% train / 5% internal validation on Open Pool
    s1_splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    s1_tr_idx, s1_val_idx = next(s1_splitter.split(pool_df, pool_df["label"].values, groups=pool_df["group_key"].values))
    pool_tr = pool_df.iloc[s1_tr_idx].reset_index(drop=True)
    pool_val = pool_df.iloc[s1_val_idx].reset_index(drop=True)
    print(f"[STAGE 1] Train={len(pool_tr):,}, Pool-Val={len(pool_val):,}")

    # Balanced class weights for slight differences (<4000 classes like kimi-k3, gemini-3.7)
    counts = np.bincount(pool_tr["label"].values, minlength=n_classes).astype(np.float32)
    cls_weights = len(pool_tr) / (n_classes * np.maximum(counts, 1.0))
    cls_weights_tensor = torch.tensor(cls_weights, dtype=torch.float32, device=device)

    print("[STAGE 1] Pre-tokenizing Open Pool passages...")
    t0 = time.time()
    s1_tr_ds = build_tokenized_dataset(pool_tr["text"].values, pool_tr["label"].values, tokenizer, MAX_LEN)
    s1_val_ds = build_tokenized_dataset(pool_val["text"].values, pool_val["label"].values, tokenizer, MAX_LEN)
    print(f"  - Pre-tokenized in {time.time() - t0:.1f}s")

    s1_tr_loader = DataLoader(s1_tr_ds, batch_size=BATCH_SIZE_S1, shuffle=True, drop_last=True, collate_fn=collator, num_workers=2, pin_memory=True)
    s1_val_loader = DataLoader(s1_val_ds, batch_size=BATCH_SIZE_S1 * 2, shuffle=False, collate_fn=collator, num_workers=2, pin_memory=True)

    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=n_classes, id2label=idx2label, label2id=label2idx)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    opt_params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(opt_params, lr=LR_S1)
    total_steps_s1 = len(s1_tr_loader) * EPOCHS_S1
    scheduler = get_cosine_warmup_scheduler(optimizer, int(total_steps_s1 * 0.06), total_steps_s1)
    criterion_s1 = nn.CrossEntropyLoss(weight=cls_weights_tensor, label_smoothing=0.05)

    stage1_ckpt_path = "/content/stage1_best.pt"
    best_s1_score = -1.0

    for epoch in range(EPOCHS_S1):
        t_ep = time.time()
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(s1_tr_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion_s1(outputs.logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        train_loss = running_loss / len(s1_tr_loader)
        val_probs, val_targets = evaluate_model(model, s1_val_loader, device, autocast_dtype, use_bf16)
        val_bal_acc = balanced_accuracy_score(val_targets, np.argmax(val_probs, axis=1))
        print(f"  [S1 Epoch {epoch + 1}/{EPOCHS_S1}] ({time.time() - t_ep:.1f}s) Loss: {train_loss:.4f} | Pool-Val BalAcc: {val_bal_acc:.5f}")
        if val_bal_acc > best_s1_score:
            best_s1_score = val_bal_acc
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, stage1_ckpt_path)

    del model, optimizer, scheduler, s1_tr_loader, s1_val_loader, s1_tr_ds, s1_val_ds
    torch.cuda.empty_cache()

    # =========================================================
    # STAGE 2: Zero-Shot Check + 5-Fold Fine-Tuning on train.csv
    # =========================================================
    print("\n[STAGE 2] Pre-tokenizing competition train.csv and test.csv...")
    train_ds_all = build_tokenized_dataset(train_df["text"].values, train_df["label"].values, tokenizer, MAX_LEN)
    train_loader_all = DataLoader(train_ds_all, batch_size=64, shuffle=False, collate_fn=collator)
    test_ds = build_tokenized_dataset(test_df["text"].values, None, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collator)

    stage1_state = torch.load(stage1_ckpt_path, map_location="cpu")

    # Zero-shot diagnostic on train.csv (strictly observational)
    diag_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    diag_model.load_state_dict(stage1_state)
    diag_model.to(device)
    zs_probs, zs_targets = evaluate_model(diag_model, train_loader_all, device, autocast_dtype, use_bf16)
    zs_score = balanced_accuracy_score(zs_targets, np.argmax(zs_probs, axis=1))
    print(f"[STAGE 1 DIAGNOSTIC] Zero-Shot train.csv Balanced Accuracy: {zs_score:.5f}")
    del diag_model
    torch.cuda.empty_cache()

    export_dir = "/content/export_weights"
    os.makedirs(export_dir, exist_ok=True)
    tokenizer.save_pretrained(export_dir)
    config.save_pretrained(export_dir)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros((len(train_df), n_classes), dtype=np.float32)
    test_probs = np.zeros((len(test_df), n_classes), dtype=np.float32)
    fold_scores = []

    y = train_df["label"].values
    groups = train_df["prompt_id"].values
    criterion_s2 = nn.CrossEntropyLoss(label_smoothing=0.02)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df, y, groups=groups)):
        print(f"\n{'='*20} Stage 2 Fold {fold + 1}/5 {'='*20}")
        assert len(set(train_df.iloc[train_idx]["prompt_id"]).intersection(set(train_df.iloc[val_idx]["prompt_id"]))) == 0

        tr_sub = train_df.iloc[train_idx]
        va_sub = train_df.iloc[val_idx]

        tr_ds = build_tokenized_dataset(tr_sub["text"].values, tr_sub["label"].values, tokenizer, MAX_LEN)
        va_ds = build_tokenized_dataset(va_sub["text"].values, va_sub["label"].values, tokenizer, MAX_LEN)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE_S2, shuffle=True, drop_last=True, collate_fn=collator, num_workers=2, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE_S2 * 2, shuffle=False, collate_fn=collator, num_workers=2, pin_memory=True)

        fold_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
        fold_model.load_state_dict(stage1_state)
        fold_model.to(device)

        opt_params_s2 = [
            {"params": [p for n, p in fold_model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.02},
            {"params": [p for n, p in fold_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(opt_params_s2, lr=LR_S2)
        total_steps_s2 = len(tr_loader) * EPOCHS_S2
        scheduler = get_cosine_warmup_scheduler(optimizer, int(total_steps_s2 * 0.1), total_steps_s2)

        best_fold_score = -1.0
        best_fold_probs = None
        fold_ckpt_path = os.path.join(export_dir, f"fold_{fold}.pt")

        for epoch in range(EPOCHS_S2):
            t_ep = time.time()
            fold_model.train()
            tr_loss = 0.0
            for batch in tr_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16):
                    outputs = fold_model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion_s2(outputs.logits.float(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fold_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                tr_loss += loss.item()

            tr_loss /= len(tr_loader)
            va_probs, va_targets = evaluate_model(fold_model, va_loader, device, autocast_dtype, use_bf16)
            va_score = balanced_accuracy_score(va_targets, np.argmax(va_probs, axis=1))
            print(f"  Epoch {epoch + 1}/{EPOCHS_S2} ({time.time() - t_ep:.1f}s) - Loss: {tr_loss:.4f} - Val BalAcc: {va_score:.5f}")

            if va_score > best_fold_score:
                best_fold_score = va_score
                best_fold_probs = va_probs
                # Save compact float16 weights for Kaggle T4/P100/L4 compatibility
                torch.save({k: v.half().cpu() for k, v in fold_model.state_dict().items()}, fold_ckpt_path)

        oof_probs[val_idx] = best_fold_probs
        fold_scores.append(best_fold_score)
        print(f"Fold {fold + 1} Best BalAcc: {best_fold_score:.5f}")

        # Test inference from saved float16 weights cast back to float32
        fold_state = {k: v.float() for k, v in torch.load(fold_ckpt_path, map_location="cpu").items()}
        fold_model.load_state_dict(fold_state)
        fold_test_probs, _ = evaluate_model(fold_model, test_loader, device, autocast_dtype, use_bf16)
        test_probs += fold_test_probs / 5.0

        del fold_model, optimizer, scheduler, tr_loader, va_loader
        torch.cuda.empty_cache()

    overall_preds = np.argmax(oof_probs, axis=1)
    overall_score = balanced_accuracy_score(y, overall_preds)
    mean_fold = float(np.mean(fold_scores))

    print("\n" + "=" * 60)
    print(f"Exp #005 Two-Stage OpenPool+train ModernBERT-base")
    print(f"Stage 1 Zero-Shot train.csv BalAcc : {zs_score:.5f}")
    print(f"Stage 2 5-Fold OOF Balanced Acc    : {overall_score:.5f}")
    print(f"Fold Scores                        : {[round(s, 5) for s in fold_scores]}")
    print(f"Mean Fold (+/- std)                : {mean_fold:.5f} +/- {np.std(fold_scores):.5f}")
    print("=" * 60)

    meta_info = {
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "classes": classes,
        "stage1_zeroshot_bal_acc": float(zs_score),
        "stage2_oof_bal_acc": float(overall_score),
        "fold_scores": [float(s) for s in fold_scores],
    }
    with open(os.path.join(export_dir, "classes.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    oof_df = pd.DataFrame(oof_probs, columns=classes)
    oof_df["id"] = train_df["id"]
    oof_df["true_model"] = train_df["model"]
    oof_df["pred_model"] = [idx2label[p] for p in overall_preds]
    oof_csv_path = "/content/oof_exp005_openpool_modernbert.csv"
    oof_df.to_csv(oof_csv_path, index=False)

    sub = pd.DataFrame({
        "id": test_df["id"],
        "model": [idx2label[p] for p in np.argmax(test_probs, axis=1)],
    })
    sub_parquet_path = "/content/submission_exp005_localtest.parquet"
    sub.to_parquet(sub_parquet_path, index=False)

    # Upload 5-fold weights + config + tokenizer to Kaggle Private Dataset
    upload_private_dataset(
        export_dir,
        DATASET_SLUG,
        f"Exp005 OpenPool+5Fold ModernBERT-base weights (OOF {overall_score:.5f})",
    )

    with open(oof_csv_path, "rb") as f:
        b64_oof = base64.b64encode(f.read()).decode("utf-8")
    print(f"\n===BASE64_OOF_START===\n{b64_oof}\n===BASE64_OOF_END===")

    with open(sub_parquet_path, "rb") as f:
        b64_sub = base64.b64encode(f.read()).decode("utf-8")
    print(f"\n===BASE64_SUBMISSION_START===\n{b64_sub}\n===BASE64_SUBMISSION_END===")


if __name__ == "__main__":
    ensure_environment()
    train_and_evaluate()
