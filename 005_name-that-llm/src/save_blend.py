import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

train = pd.read_csv("005_name-that-llm/data/train.csv")
test = pd.read_csv("005_name-that-llm/data/test.csv")

target_col = "model"
classes = sorted(train[target_col].unique())
label2idx = {cls: i for i, cls in enumerate(classes)}
idx2label = {i: cls for i, cls in enumerate(classes)}
y = train[target_col].map(label2idx).values

oof_lr = pd.read_csv("005_name-that-llm/data/oof_exp002_c2.csv")
oof_lgb = pd.read_csv("005_name-that-llm/data/oof_exp003_lgb.csv")

probs_lr = oof_lr[classes].values
probs_lgb = oof_lgb[classes].values

# 75% LR + 25% LGBM
blend_probs = 0.75 * probs_lr + 0.25 * probs_lgb
blend_preds = np.argmax(blend_probs, axis=1)

score = balanced_accuracy_score(y, blend_preds)
print(f"Verified Exp #003 Blend Balanced Accuracy: {score:.5f}")

# Save OOF
oof_blend = pd.DataFrame(blend_probs, columns=classes)
oof_blend["id"] = train["id"]
oof_blend["true_model"] = train[target_col]
oof_blend["pred_model"] = [idx2label[p] for p in blend_preds]
oof_blend.to_csv("005_name-that-llm/data/oof_exp003_blend.csv", index=False)

# Submission test inference
# In test, we can load test probs from both models
# For LR:
# We recreate the test prediction
print("Saved oof_exp003_blend.csv")
