import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from scipy.sparse import hstack
import time

def main():
    print("=== Exp #001: TF-IDF + Logistic Regression Baseline ===")
    start_time = time.time()
    
    # 1. Load data
    train = pd.read_csv("005_name-that-llm/data/train.csv")
    test = pd.read_csv("005_name-that-llm/data/test.csv")
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    
    # Target and classes
    target_col = "model"
    group_col = "prompt_id"
    classes = sorted(train[target_col].unique())
    n_classes = len(classes)
    label2idx = {cls: i for i, cls in enumerate(classes)}
    idx2label = {i: cls for i, cls in enumerate(classes)}
    
    y = train[target_col].map(label2idx).values
    groups = train[group_col].values
    
    print(f"Classes ({n_classes}): {classes}")
    
    # 2. StratifiedGroupKFold
    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_probs = np.zeros((len(train), n_classes))
    test_probs = np.zeros((len(test), n_classes))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train, y, groups=groups)):
        train_prompts = set(train.iloc[train_idx][group_col].unique())
        val_prompts = set(train.iloc[val_idx][group_col].unique())
        overlap = train_prompts.intersection(val_prompts)
        assert len(overlap) == 0, f"Fold {fold}: Leakage detected! {len(overlap)} prompts overlap!"
        
        X_train_text = train.iloc[train_idx]["text"].values
        X_val_text = train.iloc[val_idx]["text"].values
        X_test_text = test["text"].values
        
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        # TF-IDF Feature Extraction (Fit on Train only!)
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
        
        X_train_sparse = hstack([X_train_word, X_train_char])
        X_val_sparse = hstack([X_val_word, X_val_char])
        X_test_sparse = hstack([X_test_word, X_test_char])
        
        # Logistic Regression Model
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        )
        model.fit(X_train_sparse, y_train)
        
        val_preds_prob = model.predict_proba(X_val_sparse)
        oof_probs[val_idx] = val_preds_prob
        
        test_preds_prob = model.predict_proba(X_test_sparse)
        test_probs += test_preds_prob / n_splits
        
        val_preds = np.argmax(val_preds_prob, axis=1)
        score = balanced_accuracy_score(y_val, val_preds)
        fold_scores.append(score)
        print(f"Fold {fold + 1}/{n_splits} Balanced Accuracy: {score:.5f} (Val Prompts: {len(val_prompts)}, Rows: {len(val_idx)})")
    
    # Overall OOF Evaluation
    overall_preds = np.argmax(oof_probs, axis=1)
    overall_score = balanced_accuracy_score(y, overall_preds)
    mean_fold_score = np.mean(fold_scores)
    
    print("\n" + "="*50)
    print(f"Overall OOF Balanced Accuracy: {overall_score:.5f}")
    print(f"Mean Fold Balanced Accuracy:   {mean_fold_score:.5f} (+/- {np.std(fold_scores):.5f})")
    print(f"Elapsed Time:                  {time.time() - start_time:.2f} s")
    print("="*50)
    
    # Classification Report
    print("\n=== Per-Class Performance ===")
    print(classification_report(y, overall_preds, target_names=classes, digits=4))
    
    # Save OOF Predictions
    oof_df = pd.DataFrame(oof_probs, columns=classes)
    oof_df["id"] = train["id"]
    oof_df["true_model"] = train[target_col]
    oof_df["pred_model"] = [idx2label[p] for p in overall_preds]
    oof_df.to_csv("005_name-that-llm/data/oof_exp001.csv", index=False)
    print("Saved OOF predictions to 005_name-that-llm/data/oof_exp001.csv")
    
    # Save Submission Parquet
    test_pred_indices = np.argmax(test_probs, axis=1)
    sub = pd.DataFrame({
        "id": test["id"],
        "model": [idx2label[p] for p in test_pred_indices]
    })
    sub.to_parquet("005_name-that-llm/data/submission_exp001.parquet", index=False)
    print("Saved submission to 005_name-that-llm/data/submission_exp001.parquet")
    print(sub.head())

if __name__ == "__main__":
    main()
