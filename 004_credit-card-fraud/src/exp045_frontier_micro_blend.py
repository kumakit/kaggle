"""
Exp #045: Frontier Micro-Diversity Blend (98.5% Frontier 0.95569 + 1.0% Exp043 Pure CV + 0.5% Exp041 RealMLP Refit)
Fine-tunes orthogonal injection ratio to 1.5% in Standardized Rank-Logit space to preserve Frontier rank precision while adding genuine diversity.
"""
import os
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata

DATA_DIR = "004_credit-card-fraud/data"


def to_std_logit(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r = (rankdata(arr) - 0.5) / len(arr)
    z = logit(np.clip(r, eps, 1.0 - eps))
    return (z - z.mean()) / (z.std() + 1e-12)


def main():
    sub_base = pd.read_csv(f"{DATA_DIR}/goodpjw_095569/submission.csv")
    sub_43 = pd.read_csv(f"{DATA_DIR}/submission_exp043.csv")
    sub_41 = pd.read_csv(f"{DATA_DIR}/exp041_out/submission_exp041_refit.csv")

    assert (sub_base["id"].values == sub_43["id"].values).all()
    assert (sub_base["id"].values == sub_41["id"].values).all()

    z_base = to_std_logit(sub_base["is_fraud"].to_numpy())
    z_43 = to_std_logit(sub_43["is_fraud"].to_numpy())
    z_41 = to_std_logit(sub_41["is_fraud"].to_numpy())

    z_final = 0.985 * z_base + 0.010 * z_43 + 0.005 * z_41
    p_final = expit(z_final)

    out_path = os.path.join(DATA_DIR, "submission_exp045.csv")
    sub = pd.DataFrame({"id": sub_base["id"], "is_fraud": p_final})
    sub.to_csv(out_path, index=False)
    assert sub["is_fraud"].isnull().sum() == 0
    assert len(sub) == 299984
    print(f"Saved {out_path} (rows={len(sub)})")


if __name__ == "__main__":
    main()
