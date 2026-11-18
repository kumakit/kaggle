import re
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

def extract_stylometry_features(df, text_col="text"):
    feats = pd.DataFrame(index=df.index)
    texts = df[text_col].astype(str)
    
    # 1. Length & Basic Counts
    char_lens = texts.str.len()
    words_list = texts.apply(lambda t: t.split())
    word_counts = words_list.apply(len)
    
    feats["char_len"] = char_lens
    feats["word_count"] = word_counts
    feats["mean_word_len"] = char_lens / (word_counts + 1e-5)
    feats["ttr"] = words_list.apply(lambda ws: len(set(ws)) / (len(ws) + 1e-5))
    
    # Uppercase & Digits
    feats["upper_count"] = texts.apply(lambda t: sum(1 for c in t if c.isupper()))
    feats["upper_ratio"] = feats["upper_count"] / (char_lens + 1e-5)
    feats["digit_count"] = texts.apply(lambda t: sum(1 for c in t if c.isdigit()))
    feats["digit_ratio"] = feats["digit_count"] / (char_lens + 1e-5)
    
    # 2. Typography: Quotes
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
    
    # 3. Typography: Dashes & Hyphens
    feats["cnt_hyphen"] = texts.apply(lambda t: t.count("-"))
    feats["cnt_en_dash"] = texts.apply(lambda t: t.count("–"))
    feats["cnt_em_dash"] = texts.apply(lambda t: t.count("—"))
    feats["cnt_unspaced_em_dash"] = texts.apply(lambda t: len(re.findall(r"[^\s]—[^\s]", t)))
    feats["cnt_spaced_em_dash"] = texts.apply(lambda t: len(re.findall(r"\s—\s", t)))
    
    # 4. Punctuations & Ratios
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
    
    # 5. Structure & Layout
    feats["cnt_newlines"] = texts.apply(lambda t: t.count("\n"))
    feats["cnt_paragraphs"] = texts.apply(lambda t: len([p for p in t.split("\n\n") if p.strip()]))
    feats["has_bullet"] = texts.apply(lambda t: int(bool(re.search(r"(?:^|\n)\s*[-*•]\s", t))))
    feats["has_numbered_list"] = texts.apply(lambda t: int(bool(re.search(r"(?:^|\n)\s*\d+\.\s", t))))
    feats["cnt_backticks"] = texts.apply(lambda t: t.count("`"))
    feats["cnt_asterisks"] = texts.apply(lambda t: t.count("*"))
    feats["has_bold_markdown"] = texts.apply(lambda t: int(bool(re.search(r"\*\*[^*]+\*\*", t))))
    
    # 6. Sentence-level statistics
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
    
    # 7. Discourse Markers
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
    print("=== Exp #002: TF-IDF + Stylometry Features + Logistic Regression ===")
    start_time = time.time()
    
    # 1. Load data
    train = pd.read_csv("005_name-that-llm/data/train.csv")
    test = pd.read_csv("005_name-that-llm/data/test.csv")
    print(f"Train: {train.shape}, Test: {test.shape}")
    
    target_col = "model"
    group_col = "prompt_id"
    classes = sorted(train[target_col].unique())
    n_classes = len(classes)
    label2idx = {cls: i for i, cls in enumerate(classes)}
    idx2label = {i: cls for i, cls in enumerate(classes)}
    
    y = train[target_col].map(label2idx).values
    groups = train[group_col].values
    
    # Extract Stylometry Features
    print("Extracting stylometry features for Train and Test...")
    train_dense = extract_stylometry_features(train)
    test_dense = extract_stylometry_features(test)
    dense_cols = train_dense.columns.tolist()
    print(f"Extracted {len(dense_cols)} stylometry features.")
    
    # 2. StratifiedGroupKFold
    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Test multiple C parameters to find optimal regularization
    for C_val in [0.5, 1.0, 2.0]:
        print(f"\n--- Testing Logistic Regression (C={C_val}) ---")
        oof_probs = np.zeros((len(train), n_classes))
        test_probs = np.zeros((len(test), n_classes))
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(train, y, groups=groups)):
            train_prompts = set(train.iloc[train_idx][group_col].unique())
            val_prompts = set(train.iloc[val_idx][group_col].unique())
            assert len(train_prompts.intersection(val_prompts)) == 0
            
            X_train_text = train.iloc[train_idx]["text"].values
            X_val_text = train.iloc[val_idx]["text"].values
            X_test_text = test["text"].values
            
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            # TF-IDF
            word_vec = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=12000,
                sublinear_tf=True,
                strip_accents="unicode"
            )
            char_vec = TfidfVectorizer(
                ngram_range=(3, 5),
                max_features=25000,
                sublinear_tf=True,
                analyzer="char"
            )
            
            X_train_word = word_vec.fit_transform(X_train_text)
            X_val_word = word_vec.transform(X_val_text)
            X_test_word = word_vec.transform(X_test_text)
            
            X_train_char = char_vec.fit_transform(X_train_text)
            X_val_char = char_vec.transform(X_val_text)
            X_test_char = char_vec.transform(X_test_text)
            
            # Standardize Dense Stylometry Features on Train fold
            scaler = StandardScaler()
            X_train_dense_scaled = scaler.fit_transform(train_dense.iloc[train_idx].fillna(0).values)
            X_val_dense_scaled = scaler.transform(train_dense.iloc[val_idx].fillna(0).values)
            X_test_dense_scaled = scaler.transform(test_dense.fillna(0).values)
            
            # Combine Sparse TF-IDF + Dense Stylometry
            X_train_combined = hstack([X_train_word, X_train_char, X_train_dense_scaled])
            X_val_combined = hstack([X_val_word, X_val_char, X_val_dense_scaled])
            X_test_combined = hstack([X_test_word, X_test_char, X_test_dense_scaled])
            
            model = LogisticRegression(
                C=C_val,
                max_iter=1000,
                solver="lbfgs",
                random_state=42
            )
            model.fit(X_train_combined, y_train)
            
            val_preds_prob = model.predict_proba(X_val_combined)
            oof_probs[val_idx] = val_preds_prob
            
            test_preds_prob = model.predict_proba(X_test_combined)
            test_probs += test_preds_prob / n_splits
            
            val_preds = np.argmax(val_preds_prob, axis=1)
            score = balanced_accuracy_score(y_val, val_preds)
            fold_scores.append(score)
            
        overall_preds = np.argmax(oof_probs, axis=1)
        overall_score = balanced_accuracy_score(y, overall_preds)
        print(f"C={C_val} -> OOF Balanced Accuracy: {overall_score:.5f} (Mean Fold: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f})")
        
        # Save best results
        if C_val == 1.0:
            oof_df = pd.DataFrame(oof_probs, columns=classes)
            oof_df["id"] = train["id"]
            oof_df["true_model"] = train[target_col]
            oof_df["pred_model"] = [idx2label[p] for p in overall_preds]
            oof_df.to_csv("005_name-that-llm/data/oof_exp002.csv", index=False)
            
            test_pred_indices = np.argmax(test_probs, axis=1)
            sub = pd.DataFrame({
                "id": test["id"],
                "model": [idx2label[p] for p in test_pred_indices]
            })
            sub.to_parquet("005_name-that-llm/data/submission_exp002.parquet", index=False)
            print("Saved oof_exp002.csv and submission_exp002.parquet")
            
            print("\n=== Per-Class Performance (C=1.0) ===")
            print(classification_report(y, overall_preds, target_names=classes, digits=4))
            
    print(f"\nTotal Elapsed Time: {time.time() - start_time:.2f} s")

if __name__ == "__main__":
    main()
