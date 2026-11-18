import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier

sys.path.append(os.path.dirname(__file__))
from analyze_features import extract_stylometry_features

def main():
    print("=== Exp #003: Stylometry + TruncatedSVD(TF-IDF) + LightGBM ===")
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
    
    # Stylometry Features
    print("Extracting stylometry features...")
    train_stylo = extract_stylometry_features(train)
    test_stylo = extract_stylometry_features(test)
    print(f"Stylometry features: {train_stylo.shape[1]}")
    
    # 2. StratifiedGroupKFold
    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_probs_lgb = np.zeros((len(train), n_classes))
    test_probs_lgb = np.zeros((len(test), n_classes))
    fold_scores = []
    
    n_svd = 128
    print(f"Setting up 5-Fold with SVD={n_svd} and LightGBM...")
    
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
        word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, strip_accents="unicode")
        char_vec = TfidfVectorizer(ngram_range=(3, 5), max_features=15000, sublinear_tf=True, analyzer="char")
        
        X_train_word = word_vec.fit_transform(X_train_text)
        X_val_word = word_vec.transform(X_val_text)
        X_test_word = word_vec.transform(X_test_text)
        
        X_train_char = char_vec.fit_transform(X_train_text)
        X_val_char = char_vec.transform(X_val_text)
        X_test_char = char_vec.transform(X_test_text)
        
        X_train_tfidf = hstack([X_train_word, X_train_char])
        X_val_tfidf = hstack([X_val_word, X_val_char])
        X_test_tfidf = hstack([X_test_word, X_test_char])
        
        # TruncatedSVD for TF-IDF dimensionality reduction
        svd = TruncatedSVD(n_components=n_svd, random_state=42)
        X_train_svd = svd.fit_transform(X_train_tfidf)
        X_val_svd = svd.transform(X_val_tfidf)
        X_test_svd = svd.transform(X_test_tfidf)
        
        # Combine Stylometry + SVD features
        X_train_dense = np.hstack([train_stylo.iloc[train_idx].fillna(0).values, X_train_svd])
        X_val_dense = np.hstack([train_stylo.iloc[val_idx].fillna(0).values, X_val_svd])
        X_test_dense = np.hstack([test_stylo.fillna(0).values, X_test_svd])
        
        # LightGBM Classifier
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
        lgb_model.fit(
            X_train_dense, y_train,
            eval_set=[(X_val_dense, y_val)],
            callbacks=[]
        )
        
        val_probs = lgb_model.predict_proba(X_val_dense)
        oof_probs_lgb[val_idx] = val_probs
        
        test_probs = lgb_model.predict_proba(X_test_dense)
        test_probs_lgb += test_probs / n_splits
        
        val_preds = np.argmax(val_probs, axis=1)
        score = balanced_accuracy_score(y_val, val_preds)
        fold_scores.append(score)
        print(f"Fold {fold + 1}/5 Balanced Accuracy: {score:.5f}")
        
    overall_lgb_preds = np.argmax(oof_probs_lgb, axis=1)
    overall_lgb_score = balanced_accuracy_score(y, overall_lgb_preds)
    print("\n" + "="*50)
    print(f"LightGBM Single Model OOF Balanced Accuracy: {overall_lgb_score:.5f}")
    print(f"Mean Fold Score: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    print("="*50)
    
    # Save LGBM OOF
    oof_lgb_df = pd.DataFrame(oof_probs_lgb, columns=classes)
    oof_lgb_df["id"] = train["id"]
    oof_lgb_df["true_model"] = train[target_col]
    oof_lgb_df["pred_model"] = [idx2label[p] for p in overall_lgb_preds]
    oof_lgb_df.to_csv("005_name-that-llm/data/oof_exp003_lgb.csv", index=False)
    
    # 3. Ensemble with Exp #002 (Logistic Regression)
    if os.path.exists("005_name-that-llm/data/oof_exp002_c2.csv"):
        oof_exp002 = pd.read_csv("005_name-that-llm/data/oof_exp002_c2.csv")
        oof_probs_lr = oof_exp002[classes].values
        lr_score = balanced_accuracy_score(y, np.argmax(oof_probs_lr, axis=1))
        print(f"\nExp #002 Logistic Regression OOF: {lr_score:.5f}")
        
        print("\n--- Exploring Ensemble Blend Weights (Linear LR + Tree LGBM) ---")
        best_w = 0.5
        best_blend_score = 0.0
        
        for w in np.linspace(0.0, 1.0, 21):
            # Weighted probability blend
            blend_probs = w * oof_probs_lr + (1 - w) * oof_probs_lgb
            blend_preds = np.argmax(blend_probs, axis=1)
            score = balanced_accuracy_score(y, blend_preds)
            print(f"Weight (LR={w:.2f}, LGB={1-w:.2f}) -> Balanced Accuracy: {score:.5f}")
            if score > best_blend_score:
                best_blend_score = score
                best_w = w
                
        print(f"\n🏆 Best Ensemble (LR={best_w:.2f}, LGB={1-best_w:.2f}) -> {best_blend_score:.5f}")
        if best_blend_score > lr_score:
            print(f"Improvement over Exp #002: +{best_blend_score - lr_score:.5f}!")
            
        # Create blended test submission
        if os.path.exists("005_name-that-llm/data/submission_exp002_c2.parquet"):
            sub_lr = pd.read_parquet("005_name-that-llm/data/submission_exp002_c2.parquet")
            # If we had test probabilities from LR:
            # For now, let's also save the LGBM submission
            test_pred_indices = np.argmax(test_probs_lgb, axis=1)
            sub_lgb = pd.DataFrame({
                "id": test["id"],
                "model": [idx2label[p] for p in test_pred_indices]
            })
            sub_lgb.to_parquet("005_name-that-llm/data/submission_exp003_lgb.parquet", index=False)
            print("Saved submission_exp003_lgb.parquet")

    print(f"\nTotal Elapsed Time: {time.time() - start_time:.2f} s")

if __name__ == "__main__":
    main()
