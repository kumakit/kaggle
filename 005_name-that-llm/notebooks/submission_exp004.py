"""
005_name-that-llm: Exp #004 - 3-Pillar Ensemble (ModernBERT 52% + LR 45% + LGBM 3%)
Self-contained Kaggle Submission Script (GPU enabled, offline mode)
"""

import os
import re
import sys
import time
import math
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

def find_file(filename):
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                found_path = os.path.join(root, filename)
                print(f"[FOUND] {filename} at: {found_path}")
                return found_path
    candidates = [
        os.path.join("005_name-that-llm", "data", filename),
        os.path.join("data", filename),
        filename
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"[FOUND] {filename} locally at: {c}")
            return c
    raise FileNotFoundError(f"File {filename} not found!")

def find_model_dir():
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files and ("modernbert" in root.lower() or "answer" in root.lower()):
                print(f"[FOUND] ModernBERT directory at: {root}")
                return root
        # Fallback search for any config.json
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files:
                print(f"[FOUND] Fallback model directory at: {root}")
                return root
    return "answerdotai/ModernBERT-base"

def extract_stylometry_features(df, text_col="text"):
    feats = pd.DataFrame(index=df.index)
    texts = df[text_col].astype(str)
    
    char_lens = texts.str.len()
    words_list = texts.apply(lambda t: t.split())
    word_counts = words_list.apply(len)
    
    feats["char_len"] = char_lens
    feats["word_count"] = word_counts
    feats["mean_word_len"] = char_lens / (word_counts + 1e-5)
    feats["ttr"] = words_list.apply(lambda ws: len(set(ws)) / (len(ws) + 1e-5))
    
    feats["upper_count"] = texts.apply(lambda t: sum(1 for c in t if c.isupper()))
    feats["upper_ratio"] = feats["upper_count"] / (char_lens + 1e-5)
    feats["digit_count"] = texts.apply(lambda t: sum(1 for c in t if c.isdigit()))
    feats["digit_ratio"] = feats["digit_count"] / (char_lens + 1e-5)
    
    feats["cnt_straight_single"] = texts.apply(lambda t: t.count("'"))
    feats["cnt_curly_single_open"] = texts.apply(lambda t: t.count("‘"))
    feats["cnt_curly_single_close"] = texts.apply(lambda t: t.count("’"))
    feats["cnt_curly_single"] = feats["cnt_curly_single_open"] + feats["cnt_curly_single_close"]
    feats["ratio_curly_single"] = feats["cnt_curly_single"] / (feats["cnt_straight_single"] + feats["cnt_curly_single"] + 1e-5)
    
    feats["cnt_straight_double"] = texts.apply(lambda t: t.count('"'))
    feats["cnt_curly_double_open"] = texts.apply(lambda t: t.count("“"))
    feats["cnt_curly_double_close"] = texts.apply(lambda t: t.count("”"))
    feats["cnt_curly_double"] = feats["cnt_curly_double_open"] + feats["cnt_curly_double_close"]
    feats["ratio_curly_double"] = feats["cnt_curly_double"] / (feats["cnt_straight_double"] + feats["cnt_curly_double"] + 1e-5)
    
    feats["cnt_hyphen"] = texts.apply(lambda t: t.count("-"))
    feats["cnt_en_dash"] = texts.apply(lambda t: t.count("–"))
    feats["cnt_em_dash"] = texts.apply(lambda t: t.count("—"))
    feats["cnt_unspaced_em_dash"] = texts.apply(lambda t: len(re.findall(r"[^\s]—[^\s]", t)))
    feats["cnt_spaced_em_dash"] = texts.apply(lambda t: len(re.findall(r"\s—\s", t)))
    
    puncts = {
        "comma": ",", "period": ".", "colon": ":",
        "semicolon": ";", "exclamation": "!", "question": "?"
    }
    for name, p in puncts.items():
        feats[f"cnt_{name}"] = texts.apply(lambda t, char=p: t.count(char))
        feats[f"ratio_{name}"] = feats[f"cnt_{name}"] / (char_lens + 1e-5) * 1000
        
    feats["cnt_ellipsis"] = texts.apply(lambda t: len(re.findall(r"\.\.\.", t)))
    feats["cnt_brackets"] = texts.apply(lambda t: t.count("(") + t.count(")"))
    feats["cnt_square_brackets"] = texts.apply(lambda t: t.count("[") + t.count("]"))
    
    feats["cnt_newlines"] = texts.apply(lambda t: t.count("\n"))
    feats["cnt_paragraphs"] = texts.apply(lambda t: len([p for p in t.split("\n\n") if p.strip()]))
    feats["has_bullet"] = texts.apply(lambda t: int(bool(re.search(r"(?:^|\n)\s*[-*•]\s", t))))
    feats["has_numbered_list"] = texts.apply(lambda t: int(bool(re.search(r"(?:^|\n)\s*\d+\.\s", t))))
    feats["cnt_backticks"] = texts.apply(lambda t: t.count("`"))
    feats["cnt_asterisks"] = texts.apply(lambda t: t.count("*"))
    feats["has_bold_markdown"] = texts.apply(lambda t: int(bool(re.search(r"\*\*[^*]+\*\*", t))))
    
    def sentence_stats(text):
        sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sents:
            return 0, 0.0, 0.0
        lens = [len(s.split()) for s in sents]
        return len(sents), np.mean(lens), np.std(lens)
        
    s_stats = texts.apply(sentence_stats)
    feats["sent_count"] = s_stats.apply(lambda x: x[0])
    feats["mean_sent_word_len"] = s_stats.apply(lambda x: x[1])
    feats["std_sent_word_len"] = s_stats.apply(lambda x: x[2])
    
    markers = [
        "however", "moreover", "furthermore", "thus", "therefore",
        "additionally", "overall", "in summary", "crucially", "notably",
        "indeed", "first", "second", "finally", "specifically"
    ]
    for m in markers:
        col = f"dm_{m.replace(' ', '_')}"
        feats[col] = texts.apply(lambda t, pat=m: len(re.findall(r"\b" + pat + r"\b", t, re.IGNORECASE)))
        
    return feats

def get_cosine_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def main():
    print("=== Name That LLM: Exp004 3-Pillar Ensemble Submission ===")
    t0 = time.time()
    
    train_path = find_file("train.csv")
    test_path = find_file("test.csv")
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Loaded Train: {train.shape}, Test: {test.shape}")
    
    target_col = "model"
    classes = sorted(train[target_col].unique())
    n_classes = len(classes)
    label2idx = {cls: i for i, cls in enumerate(classes)}
    idx2label = {i: cls for i, cls in enumerate(classes)}
    y_train = train[target_col].map(label2idx).values
    groups = train["prompt_id"].values
    
    # -------------------------------------------------------------------------
    # PART 1: Stylometry & TF-IDF Extraction
    # -------------------------------------------------------------------------
    print("\n--- Part 1: Stylometry & TF-IDF Feature Engineering ---")
    train_stylo = extract_stylometry_features(train)
    test_stylo = extract_stylometry_features(test)
    
    scaler = StandardScaler()
    X_train_stylo_scaled = scaler.fit_transform(train_stylo.fillna(0).values)
    X_test_stylo_scaled = scaler.transform(test_stylo.fillna(0).values)
    
    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=12000, sublinear_tf=True, strip_accents="unicode")
    char_vec = TfidfVectorizer(ngram_range=(3, 5), max_features=25000, sublinear_tf=True, analyzer="char")
    
    X_train_word = word_vec.fit_transform(train["text"].values)
    X_test_word = word_vec.transform(test["text"].values)
    X_train_char = char_vec.fit_transform(train["text"].values)
    X_test_char = char_vec.transform(test["text"].values)
    
    X_train_tfidf = hstack([X_train_word, X_train_char])
    X_test_tfidf = hstack([X_test_word, X_test_char])
    
    # -------------------------------------------------------------------------
    # PART 2: Model 1 - Logistic Regression (C=2.0)
    # -------------------------------------------------------------------------
    print("\n--- Part 2: Training Logistic Regression (C=2.0) ---")
    X_train_lr = hstack([X_train_tfidf, X_train_stylo_scaled])
    X_test_lr = hstack([X_test_tfidf, X_test_stylo_scaled])
    
    lr_model = LogisticRegression(C=2.0, max_iter=1000, solver="lbfgs", random_state=42)
    lr_model.fit(X_train_lr, y_train)
    test_probs_lr = lr_model.predict_proba(X_test_lr)
    print("Logistic Regression inference complete.")
    
    # -------------------------------------------------------------------------
    # PART 3: Model 2 - LightGBM (500 Trees)
    # -------------------------------------------------------------------------
    print("\n--- Part 3: Training LightGBM ---")
    svd = TruncatedSVD(n_components=128, random_state=42)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)
    
    X_train_lgb = np.hstack([train_stylo.fillna(0).values, X_train_svd])
    X_test_lgb = np.hstack([test_stylo.fillna(0).values, X_test_svd])
    
    lgb_model = LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        learning_rate=0.04,
        num_leaves=24,
        max_depth=5,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    lgb_model.fit(X_train_lgb, y_train)
    test_probs_lgb = lgb_model.predict_proba(X_test_lgb)
    print("LightGBM inference complete.")
    
    # -------------------------------------------------------------------------
    # PART 4: Model 3 - ModernBERT-base 5-Fold Fine-Tuning
    # -------------------------------------------------------------------------
    print("\n--- Part 4: Fine-Tuning ModernBERT-base (5-Fold) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model_dir = find_model_dir()
    print(f"Loading ModernBERT from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    class LLMDataset(Dataset):
        def __init__(self, texts, labels=None):
            self.texts = texts
            self.labels = labels
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            encoding = tokenizer(text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
            item = {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0)
            }
            if self.labels is not None:
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    test_dataset = LLMDataset(test["text"].values)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_probs_bert = np.zeros((len(test), n_classes))
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train, y_train, groups=groups)):
        print(f"  Fold {fold + 1}/5 training...")
        train_sub = train.iloc[train_idx]
        train_ds = LLMDataset(train_sub["text"].values, y_train[train_idx])
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        
        config = AutoConfig.from_pretrained(model_dir, num_labels=n_classes)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
        model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
        total_steps = len(train_loader) * 4
        scheduler = get_cosine_warmup_scheduler(optimizer, int(total_steps * 0.1), total_steps)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(4):
            model.train()
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                optimizer.zero_grad()
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=autocast_dtype, enabled=torch.cuda.is_available()):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
        # Test inference for fold
        model.eval()
        fold_probs = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=autocast_dtype, enabled=torch.cuda.is_available()):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=-1)
                fold_probs.append(probs.cpu().numpy())
        test_probs_bert += np.concatenate(fold_probs, axis=0) / 5
        del model, optimizer, scheduler, train_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print("ModernBERT 5-Fold training and inference complete.")
    
    # -------------------------------------------------------------------------
    # PART 5: 3-Pillar Ensemble (ModernBERT 52% + LR 45% + LGBM 3%)
    # -------------------------------------------------------------------------
    print("\n--- Part 5: Blending 3-Pillar Predictions (52% BERT + 45% LR + 3% LGBM) ---")
    final_probs = 0.52 * test_probs_bert + 0.45 * test_probs_lr + 0.03 * test_probs_lgb
    pred_indices = np.argmax(final_probs, axis=1)
    
    sub = pd.DataFrame({
        "id": test["id"],
        "model": [idx2label[p] for p in pred_indices]
    })
    
    out_parquet = "submission.parquet"
    sub.to_parquet(out_parquet, index=False)
    print(f"Generated submission to: {out_parquet}")
    print(f"Submission shape: {sub.shape}")
    print(sub.head())
    print(f"Total time elapsed: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
