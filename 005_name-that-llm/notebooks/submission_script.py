import os
import re
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

def find_file(filename):
    # 1. Search in /kaggle/input if on Kaggle
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                found_path = os.path.join(root, filename)
                print(f"Found {filename} at: {found_path}")
                return found_path
                
    # 2. Local candidates
    candidates = [
        os.path.join("005_name-that-llm", "data", filename),
        os.path.join("data", filename),
        filename
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"Found {filename} locally at: {c}")
            return c
            
    # List /kaggle/input if available to aid debugging
    if os.path.exists("/kaggle/input"):
        print("Listing contents of /kaggle/input:")
        for root, dirs, files in os.walk("/kaggle/input"):
            print(f"  {root}: {files}")
            
    raise FileNotFoundError(f"File {filename} not found!")

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

def main():
    print("=== Name That LLM: Exp003 Submission Pipeline ===")
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
    
    # 1. Stylometry features
    print("Extracting stylometry features...")
    train_stylo = extract_stylometry_features(train)
    test_stylo = extract_stylometry_features(test)
    
    scaler = StandardScaler()
    X_train_stylo_scaled = scaler.fit_transform(train_stylo.fillna(0).values)
    X_test_stylo_scaled = scaler.transform(test_stylo.fillna(0).values)
    
    # 2. TF-IDF features
    print("Extracting TF-IDF features...")
    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=12000, sublinear_tf=True, strip_accents="unicode")
    char_vec = TfidfVectorizer(ngram_range=(3, 5), max_features=25000, sublinear_tf=True, analyzer="char")
    
    X_train_word = word_vec.fit_transform(train["text"].values)
    X_test_word = word_vec.transform(test["text"].values)
    
    X_train_char = char_vec.fit_transform(train["text"].values)
    X_test_char = char_vec.transform(test["text"].values)
    
    X_train_tfidf = hstack([X_train_word, X_train_char])
    X_test_tfidf = hstack([X_test_word, X_test_char])
    
    # Combined sparse for Logistic Regression
    X_train_lr = hstack([X_train_tfidf, X_train_stylo_scaled])
    X_test_lr = hstack([X_test_tfidf, X_test_stylo_scaled])
    
    # SVD for LightGBM
    print("Running TruncatedSVD for LightGBM...")
    svd = TruncatedSVD(n_components=128, random_state=42)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)
    
    X_train_lgb = np.hstack([train_stylo.fillna(0).values, X_train_svd])
    X_test_lgb = np.hstack([test_stylo.fillna(0).values, X_test_svd])
    
    # 3. Model 1: Logistic Regression (C=2.0)
    print("Training Logistic Regression (C=2.0)...")
    lr_model = LogisticRegression(C=2.0, max_iter=1000, solver="lbfgs", random_state=42)
    lr_model.fit(X_train_lr, y_train)
    test_probs_lr = lr_model.predict_proba(X_test_lr)
    
    # 4. Model 2: LightGBM
    print("Training LightGBM...")
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
    
    # 5. Ensemble: 75% LR + 25% LGBM
    print("Blending predictions (75% LR + 25% LGBM)...")
    final_probs = 0.75 * test_probs_lr + 0.25 * test_probs_lgb
    pred_indices = np.argmax(final_probs, axis=1)
    
    # 6. Generate submission.parquet
    sub = pd.DataFrame({
        "id": test["id"],
        "model": [idx2label[p] for p in pred_indices]
    })
    
    out_parquet = "submission.parquet"
    sub.to_parquet(out_parquet, index=False)
    print(f"Generated submission to current working directory: {out_parquet}")
    print(f"Submission shape: {sub.shape}")
    print(sub.head())
    print(f"Total time elapsed: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
