"""
005_name-that-llm: Exp #005 Inference-Only Kaggle Submission Script
Models:
  - 90% OpenPool Stage 1 + 5-Fold ModernBERT-base (loaded offline from Kaggle Dataset)
  - 10% Stylometry + Word/Char TF-IDF LogisticRegression (C=2.0)
OOF Balanced Accuracy: 0.51787 (Single Exp005: 0.51675)
"""

import os
import re
import json
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

MAX_LEN = 416
BATCH_SIZE = 32
WEIGHT_BERT = 0.90
WEIGHT_LR = 0.10


def find_file(filename):
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if filename in files:
                path = os.path.join(root, filename)
                print(f"[FOUND] {filename}: {path}")
                return path
    for cand in [os.path.join("005_name-that-llm", "data", filename), os.path.join("data", filename), filename]:
        if os.path.exists(cand):
            print(f"[FOUND] {filename} locally: {cand}")
            return cand
    raise FileNotFoundError(f"Could not locate {filename}")


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
        return len(sents), float(np.mean(lens)), float(np.std(lens))

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


class PreTokenizedDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def main():
    t0 = time.time()
    print("=== Name That LLM: Exp #005 Inference-Only Submission ===")

    train = pd.read_csv(find_file("train.csv"))
    test = pd.read_csv(find_file("test.csv"))
    classes_path = find_file("classes.json")
    weights_dir = os.path.dirname(classes_path)

    with open(classes_path, "r") as f:
        meta_info = json.load(f)

    classes = meta_info["classes"]
    assert classes == sorted(train["model"].unique().tolist()), "Class list mismatch!"
    label2idx = {c: i for i, c in enumerate(classes)}
    idx2label = {i: c for i, c in enumerate(classes)}
    y_train = train["model"].map(label2idx).values

    # ---------------------------------------------------------
    # Part 1: Fast Stylometry + TF-IDF Logistic Regression (10%)
    # ---------------------------------------------------------
    print("[LR] Extracting stylometry & TF-IDF features...")
    tr_stylo = extract_stylometry_features(train)
    te_stylo = extract_stylometry_features(test)

    scaler = StandardScaler()
    X_tr_stylo = scaler.fit_transform(tr_stylo.fillna(0).values)
    X_te_stylo = scaler.transform(te_stylo.fillna(0).values)

    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=12000, sublinear_tf=True, strip_accents="unicode")
    char_vec = TfidfVectorizer(ngram_range=(3, 5), max_features=25000, sublinear_tf=True, analyzer="char")
    X_tr_lr = hstack([word_vec.fit_transform(train["text"].values), char_vec.fit_transform(train["text"].values), X_tr_stylo])
    X_te_lr = hstack([word_vec.transform(test["text"].values), char_vec.transform(test["text"].values), X_te_stylo])

    lr_model = LogisticRegression(C=2.0, max_iter=1000, solver="lbfgs", random_state=42)
    lr_model.fit(X_tr_lr, y_train)
    test_probs_lr = lr_model.predict_proba(X_te_lr)
    print(f"[LR] Complete ({time.time() - t0:.1f}s)")

    # ---------------------------------------------------------
    # Part 2: Offline 5-Fold ModernBERT Inference (90%)
    # ---------------------------------------------------------
    t_bert = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[BERT] Device={device} | AMP={use_amp} ({amp_dtype}) | Weights={weights_dir}")

    tokenizer = AutoTokenizer.from_pretrained(weights_dir)
    config = AutoConfig.from_pretrained(weights_dir)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    test_enc = tokenizer(
        [str(t) for t in test["text"].values],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )
    test_loader = DataLoader(
        PreTokenizedDataset(test_enc),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    test_probs_bert = np.zeros((len(test), len(classes)), dtype=np.float64)
    model = AutoModelForSequenceClassification.from_config(config)
    model.to(device)
    model.eval()

    for fold in range(5):
        ckpt_path = os.path.join(weights_dir, f"fold_{fold}.pt")
        state = {k: v.float() for k, v in torch.load(ckpt_path, map_location="cpu").items()}
        model.load_state_dict(state)

        fold_probs = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    probs = torch.softmax(logits.float(), dim=-1)
                fold_probs.append(probs.cpu().numpy())

        test_probs_bert += np.concatenate(fold_probs, axis=0) / 5.0
        print(f"  - Fold {fold + 1}/5 inference done.")

    print(f"[BERT] Complete ({time.time() - t_bert:.1f}s)")

    # ---------------------------------------------------------
    # Part 3: Blend & Write submission.parquet
    # ---------------------------------------------------------
    final_probs = WEIGHT_BERT * test_probs_bert + WEIGHT_LR * test_probs_lr
    preds = [idx2label[p] for p in np.argmax(final_probs, axis=1)]

    sub = pd.DataFrame({"id": test["id"], "model": preds})
    sub.to_parquet("submission.parquet", index=False)
    print(f"[DONE] Saved submission.parquet shape={sub.shape} in {time.time() - t0:.1f}s")
    print(sub.head())


if __name__ == "__main__":
    main()
