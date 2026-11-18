"""
005_name-that-llm: Exp #004 - Colab L4 GPU DeBERTa-v3-base 5-Fold Grouped-CV Fine-Tuning (bfloat16 Optimized)
Competition: Name That LLM
Target Metric: Balanced Accuracy (18 Classes)
Validation: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on prompt_id
"""

import os
import sys
import subprocess
import json
import base64
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

def ensure_environment():
    print("[INIT] Setting up environment and packages...")
    packages = ["transformers", "torch", "accelerate", "scikit-learn", "kaggle"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=True)
    
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
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        print("[DATA] Downloading name-that-llm dataset from Kaggle...")
        subprocess.run(["kaggle", "competitions", "download", "-c", "name-that-llm", "-p", data_dir], check=True)
        zip_path = os.path.join(data_dir, "name-that-llm.zip")
        if os.path.exists(zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(data_dir)
            os.remove(zip_path)
            print("  - Extracted competition data successfully.")
    return data_dir

def train_and_evaluate():
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, get_cosine_schedule_with_warmup
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import balanced_accuracy_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using compute device: {device}")
    if torch.cuda.is_available():
        print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  - Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        use_bf16 = torch.cuda.is_bf16_supported()
        print(f"  - Native bfloat16 supported: {use_bf16}")
    else:
        use_bf16 = False

    data_dir = download_data()
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    print(f"[DATA] Train: {train_df.shape}, Test: {test_df.shape}")

    target_col = "model"
    group_col = "prompt_id"
    classes = sorted(train_df[target_col].unique())
    n_classes = len(classes)
    label2idx = {cls: i for i, cls in enumerate(classes)}
    idx2label = {i: cls for i, cls in enumerate(classes)}

    train_df["label"] = train_df[target_col].map(label2idx)
    groups = train_df[group_col].values
    y = train_df["label"].values

    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 512
    BATCH_SIZE = 16
    EPOCHS = 4
    LR = 2e-5
    WARMUP_RATIO = 0.1

    print(f"[MODEL] Loading tokenizer for: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class LLMDataset(Dataset):
        def __init__(self, texts, labels=None):
            self.texts = texts
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=MAX_LEN,
                padding="max_length",
                return_tensors="pt"
            )
            item = {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0)
            }
            if self.labels is not None:
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(train_df), n_classes))
    test_probs = np.zeros((len(test_df), n_classes))
    fold_scores = []

    test_dataset = LLMDataset(test_df["text"].values)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df, y, groups=groups)):
        print(f"\n{'='*20} Fold {fold + 1}/5 {'='*20}")
        train_prompts = set(train_df.iloc[train_idx][group_col].unique())
        val_prompts = set(train_df.iloc[val_idx][group_col].unique())
        assert len(train_prompts.intersection(val_prompts)) == 0

        train_sub = train_df.iloc[train_idx]
        val_sub = train_df.iloc[val_idx]

        train_ds = LLMDataset(train_sub["text"].values, train_sub["label"].values)
        val_ds = LLMDataset(val_sub["text"].values, val_sub["label"].values)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False)

        config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=n_classes)
        config.attention_probs_dropout_prob = 0.1
        config.hidden_dropout_prob = 0.1
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
        model.to(device)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LR)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps)
        criterion = nn.CrossEntropyLoss()

        best_val_score = 0.0
        best_val_probs = None

        autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

        for epoch in range(EPOCHS):
            t_epoch_start = time.time()
            model.train()
            train_loss = 0.0
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_preds_list = []
            val_probs_list = []
            val_targets_list = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        probs = torch.softmax(outputs.logits, dim=-1)

                    val_probs_list.append(probs.cpu().numpy())
                    val_preds_list.append(torch.argmax(probs, dim=-1).cpu().numpy())
                    val_targets_list.append(labels.cpu().numpy())

            val_probs_cat = np.concatenate(val_probs_list, axis=0)
            val_preds_cat = np.concatenate(val_preds_list, axis=0)
            val_targets_cat = np.concatenate(val_targets_list, axis=0)

            val_score = balanced_accuracy_score(val_targets_cat, val_preds_cat)
            print(f"  Epoch {epoch + 1}/{EPOCHS} ({time.time() - t_epoch_start:.1f}s) - Train Loss: {train_loss:.4f} - Val Balanced Acc: {val_score:.5f}")

            if val_score > best_val_score:
                best_val_score = val_score
                best_val_probs = val_probs_cat
                torch.save(model.state_dict(), f"/content/deberta_fold{fold}.pt")

        oof_probs[val_idx] = best_val_probs
        fold_scores.append(best_val_score)
        print(f"Fold {fold + 1} Best Balanced Acc: {best_val_score:.5f}")

        # Test inference with best fold model
        model.load_state_dict(torch.load(f"/content/deberta_fold{fold}.pt"))
        model.eval()
        fold_test_probs = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=-1)
                fold_test_probs.append(probs.cpu().numpy())
        test_probs += np.concatenate(fold_test_probs, axis=0) / 5

        # Cleanup memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    overall_preds = np.argmax(oof_probs, axis=1)
    overall_score = balanced_accuracy_score(y, overall_preds)
    mean_fold_score = np.mean(fold_scores)

    print("\n" + "="*50)
    print(f"DeBERTa-v3-base 5-Fold OOF Balanced Accuracy: {overall_score:.5f}")
    print(f"Mean Fold Score: {mean_fold_score:.5f} (+/- {np.std(fold_scores):.5f})")
    print("="*50)

    # Export OOF
    oof_df = pd.DataFrame(oof_probs, columns=classes)
    oof_df["id"] = train_df["id"]
    oof_df["true_model"] = train_df[target_col]
    oof_df["pred_model"] = [idx2label[p] for p in overall_preds]
    oof_csv_path = "/content/oof_exp004_deberta.csv"
    oof_df.to_csv(oof_csv_path, index=False)

    # Export Submission
    test_pred_indices = np.argmax(test_probs, axis=1)
    sub = pd.DataFrame({
        "id": test_df["id"],
        "model": [idx2label[p] for p in test_pred_indices]
    })
    sub_parquet_path = "/content/submission_exp004_deberta.parquet"
    sub.to_parquet(sub_parquet_path, index=False)

    # Google Drive export
    drive_dir = "/content/drive/MyDrive/kaggle_name_that_llm"
    try:
        os.makedirs(drive_dir, exist_ok=True)
        oof_df.to_csv(os.path.join(drive_dir, "oof_exp004_deberta.csv"), index=False)
        sub.to_parquet(os.path.join(drive_dir, "submission_exp004_deberta.parquet"), index=False)
        print(f"[EXPORT] Successfully saved outputs to Google Drive: {drive_dir}")
    except Exception as e:
        print(f"[EXPORT] Drive export skipped/failed: {e}")

    # Base64 export for fallback retrieval
    with open(oof_csv_path, "rb") as f:
        b64_oof = base64.b64encode(f.read()).decode("utf-8")
    with open(sub_parquet_path, "rb") as f:
        b64_sub = base64.b64encode(f.read()).decode("utf-8")

    print("\n===BASE64_OOF_START===")
    print(b64_oof)
    print("===BASE64_OOF_END===")

    print("\n===BASE64_SUB_START===")
    print(b64_sub)
    print("===BASE64_SUB_END===")

if __name__ == "__main__":
    ensure_environment()
    train_and_evaluate()
