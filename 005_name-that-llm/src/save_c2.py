import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from exp002_stylometry import extract_stylometry_features
import os

os.chdir("005_name-that-llm")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

target_col = "model"
group_col = "prompt_id"
classes = sorted(train[target_col].unique())
n_classes = len(classes)
label2idx = {cls: i for i, cls in enumerate(classes)}
idx2label = {i: cls for i, cls in enumerate(classes)}

y = train[target_col].map(label2idx).values
groups = train[group_col].values

train_dense = extract_stylometry_features(train)
test_dense = extract_stylometry_features(test)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros((len(train), n_classes))
test_probs = np.zeros((len(test), n_classes))

for fold, (train_idx, val_idx) in enumerate(sgkf.split(train, y, groups=groups)):
    X_train_text = train.iloc[train_idx]["text"].values
    X_val_text = train.iloc[val_idx]["text"].values
    X_test_text = test["text"].values
    
    y_train = y[train_idx]
    y_val = y[val_idx]
    
    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=12000, sublinear_tf=True, strip_accents="unicode")
    char_vec = TfidfVectorizer(ngram_range=(3, 5), max_features=25000, sublinear_tf=True, analyzer="char")
    
    X_train_word = word_vec.fit_transform(X_train_text)
    X_val_word = word_vec.transform(X_val_text)
    X_test_word = word_vec.transform(X_test_text)
    
    X_train_char = char_vec.fit_transform(X_train_text)
    X_val_char = char_vec.transform(X_val_text)
    X_test_char = char_vec.transform(X_test_text)
    
    scaler = StandardScaler()
    X_train_dense_scaled = scaler.fit_transform(train_dense.iloc[train_idx].fillna(0).values)
    X_val_dense_scaled = scaler.transform(train_dense.iloc[val_idx].fillna(0).values)
    X_test_dense_scaled = scaler.transform(test_dense.fillna(0).values)
    
    X_train_combined = hstack([X_train_word, X_train_char, X_train_dense_scaled])
    X_val_combined = hstack([X_val_word, X_val_char, X_val_dense_scaled])
    X_test_combined = hstack([X_test_word, X_test_char, X_test_dense_scaled])
    
    model = LogisticRegression(C=2.0, max_iter=1000, solver="lbfgs", random_state=42)
    model.fit(X_train_combined, y_train)
    
    oof_probs[val_idx] = model.predict_proba(X_val_combined)
    test_probs += model.predict_proba(X_test_combined) / 5

overall_preds = np.argmax(oof_probs, axis=1)
overall_score = balanced_accuracy_score(y, overall_preds)
print(f"Verified C=2.0 OOF Balanced Accuracy: {overall_score:.5f}")

# Save
oof_df = pd.DataFrame(oof_probs, columns=classes)
oof_df["id"] = train["id"]
oof_df["true_model"] = train[target_col]
oof_df["pred_model"] = [idx2label[p] for p in overall_preds]
oof_df.to_csv("data/oof_exp002_c2.csv", index=False)

test_pred_indices = np.argmax(test_probs, axis=1)
sub = pd.DataFrame({
    "id": test["id"],
    "model": [idx2label[p] for p in test_pred_indices]
})
sub.to_parquet("data/submission_exp002_c2.parquet", index=False)
print("Saved oof_exp002_c2.csv and submission_exp002_c2.parquet")
print(classification_report(y, overall_preds, target_names=classes, digits=4))
