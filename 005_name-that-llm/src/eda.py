import pandas as pd
import numpy as np

train = pd.read_csv("005_name-that-llm/data/train.csv")
test = pd.read_csv("005_name-that-llm/data/test.csv")
sub = pd.read_parquet("005_name-that-llm/data/sample_submission.parquet")

print("=== Train Shape ===")
print(train.shape)
print("\n=== Train Columns & Dtypes ===")
print(train.dtypes)
print("\n=== Train Missing Values ===")
print(train.isnull().sum())

print("\n=== Train Head (First 2 rows) ===")
for col in train.columns:
    print(f"[{col}]: {train[col].iloc[0][:100] if isinstance(train[col].iloc[0], str) else train[col].iloc[0]}")

print("\n=== Test Shape ===")
print(test.shape)
print("\n=== Test Columns ===")
print(test.columns.tolist())

print("\n=== Sample Submission Shape & Head ===")
print(sub.shape)
print(sub.head())
print(sub.dtypes)

target_candidates = [c for c in train.columns if c not in test.columns]
print(f"\nTarget candidates: {target_candidates}")
target_col = target_candidates[0] if target_candidates else "model"
print(f"Target column: {target_col}")

print("\n=== Target Class Distribution ===")
print(train[target_col].value_counts())
print(f"Number of classes: {train[target_col].nunique()}")

prompt_cols = [c for c in train.columns if "prompt" in c.lower()]
print(f"\nPrompt related columns: {prompt_cols}")
for pc in prompt_cols:
    print(f"Unique {pc}: {train[pc].nunique()}")
    counts = train[pc].value_counts()
    print(f"  Count per {pc}: min={counts.min()}, median={counts.median()}, max={counts.max()}")

print("\n=== Text Length Statistics ===")
text_col = [c for c in ["text", "passage", "response", "output"] if c in train.columns][0]
train["char_len"] = train[text_col].astype(str).str.len()
train["word_len"] = train[text_col].astype(str).str.split().str.len()
print(f"Char length: mean={train['char_len'].mean():.1f}, min={train['char_len'].min()}, max={train['char_len'].max()}")
print(f"Word length: mean={train['word_len'].mean():.1f}, min={train['word_len'].min()}, max={train['word_len'].max()}")
