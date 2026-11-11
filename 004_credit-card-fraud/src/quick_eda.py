import pandas as pd
import numpy as np

train_path = "004_credit-card-fraud/data/train.csv"
test_path = "004_credit-card-fraud/data/test.csv"
sub_path = "004_credit-card-fraud/data/sample_submission.csv"

train_df = pd.read_csv(train_path, nrows=1000)
print("=== Train Shape Preview (first 1000) ===")
print("Columns:", train_df.columns.tolist())
print("\n=== Sample Submission Head ===")
sub_df = pd.read_csv(sub_path, nrows=5)
print(sub_df)

# Full count and target check
print("\n=== Full Train / Test Row Counts ===")
train_full = pd.read_csv(train_path)
test_full = pd.read_csv(test_path)
print(f"Train shape: {train_full.shape}")
print(f"Test shape: {test_full.shape}")

target_col = [col for col in train_full.columns if col not in test_full.columns]
print(f"Target column(s): {target_col}")

if target_col:
    t_col = target_col[0]
    print(f"\nTarget ({t_col}) Value Counts:")
    print(train_full[t_col].value_counts())
    print(f"Target ({t_col}) Normalized:")
    print(train_full[t_col].value_counts(normalize=True))

print("\n=== Missing Values in Train ===")
missing_train = train_full.isnull().sum()
print(missing_train[missing_train > 0])

print("\n=== Missing Values in Test ===")
missing_test = test_full.isnull().sum()
print(missing_test[missing_test > 0])

print("\n=== Column Data Types ===")
print(train_full.dtypes)

print("\n=== Train First 3 Rows ===")
print(train_full.head(3).T)
