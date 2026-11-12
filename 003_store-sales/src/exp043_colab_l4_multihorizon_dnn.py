"""
003_store-sales: Exp #043 - Direct Multi-Horizon Deep Neural Network (DNN) with Entity Embeddings
Kaggle Competition: Store Sales - Time Series Forecasting
Designed for Google Colab L4 GPU (VRAM 24GB) Ephemeral Execution

Core Architecture:
  - Orthogonal Model to GBDT (Continuous Function Approximation breaking Extrapolation Saturation)
  - Direct 16-Horizon Vector Output (No error accumulation from recursive roll-forward)
  - Entity Embeddings for Store (16d), Family (16d), Cluster (8d), City (8d), State (8d), Type (4d)
  - Continuous Features (68d): Past 30-day Lags + Future 16-day Promotions + YoY 16-day Lags + Calendar/Earthquake
  - Deep ResNet Structure: 2 Residual Blocks with LayerNorm, SiLU activations, and Dropout
  - 3-Seed Averaging: [42, 2026, 777] for variance reduction
  - Automatic Synchronization: Google Drive for Desktop + Base64 stdout stream
"""

import os
import sys
import subprocess
import glob
import json
import base64
import time
import threading
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def start_heartbeat(interval_sec=15):
    """Keep WebSocket connection alive by periodic flushing"""
    def beat():
        while True:
            time.sleep(interval_sec)
            print("[HEARTBEAT] L4 GPU computation active...", flush=True)
    t = threading.Thread(target=beat, daemon=True)
    t.start()


def ensure_environment_and_data():
    """Verify packages, Kaggle auth, and dataset files"""
    print("[INIT] Verifying packages and datasets...")

    # Kaggle auth
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    username = os.environ.get("KAGGLE_USERNAME", "nobuyukikumagai")
    key = os.environ.get("KAGGLE_KEY", "757b1a8d9ce71f3b129fbae7cc34b1b2")
    if not os.path.exists(kaggle_json_path):
        with open(kaggle_json_path, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json_path, 0o600)
        print(f"  - Configured kaggle.json for user: {username}")

    # Data files
    data_dir = "data"
    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        candidates = ["/content/data", "003_store-sales/data", "/content/kaggle/003_store-sales/data"]
        for c in candidates:
            if os.path.exists(os.path.join(c, "train.csv")):
                data_dir = c
                break

    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        print("  - Downloading dataset from Kaggle...")
        os.makedirs("data", exist_ok=True)
        subprocess.run(["kaggle", "competitions", "download", "-c", "store-sales-time-series-forecasting", "-p", "data"], check=True)
        for zip_file in glob.glob("data/*.zip"):
            import zipfile
            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall("data")
        data_dir = "data"

    print(f"  - Datasets ready at: {data_dir}")
    return data_dir


def build_datasets(data_dir):
    """Build multi-horizon dataset using fast matrix operations"""
    print("\n[DATA] Building Multi-Horizon dataset...")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["date"])
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=["date"])
    stores = pd.read_csv(os.path.join(data_dir, "stores.csv"))

    full_dates = pd.date_range("2013-01-01", "2017-08-31")

    # Pivoting sales and promotion matrices
    piv_sales = train.pivot(index="date", columns=["store_nbr", "family"], values="sales")
    piv_sales = piv_sales.reindex(full_dates).fillna(0.0)

    piv_promo_tr = train.pivot(index="date", columns=["store_nbr", "family"], values="onpromotion")
    piv_promo_te = test.pivot(index="date", columns=["store_nbr", "family"], values="onpromotion")
    piv_promo = pd.concat([piv_promo_tr, piv_promo_te]).reindex(full_dates).fillna(0.0)

    series_cols = piv_sales.columns # MultiIndex [(store_nbr, family)] (1782 pairs)
    n_series = len(series_cols)

    # Static metadata mapping
    stores_map = stores.set_index("store_nbr")
    cities = sorted(stores["city"].unique())
    states = sorted(stores["state"].unique())
    types = sorted(stores["type"].unique())
    families = sorted(train["family"].unique())

    city2idx = {c: i for i, c in enumerate(cities)}
    state2idx = {s: i for i, s in enumerate(states)}
    type2idx = {t: i for i, t in enumerate(types)}
    fam2idx = {f: i for i, f in enumerate(families)}

    static_cats = []
    for s, f in series_cols:
        st_row = stores_map.loc[s]
        static_cats.append([
            s - 1, # store 0-53
            fam2idx[f], # family 0-32
            st_row["cluster"] - 1, # cluster 0-16
            city2idx[st_row["city"]], # city
            state2idx[st_row["state"]], # state
            type2idx[st_row["type"]], # type
        ])
    static_cats = np.array(static_cats, dtype=np.int64) # shape (1782, 6)

    # Sliding windows: weekly cutoffs from 2015-06-01 to 2017-07-15
    train_cutoffs = pd.date_range("2015-06-01", "2017-07-15", freq="7D")
    val_cutoff = pd.Timestamp("2017-07-31") # Validation: 2017-08-01 to 2017-08-15 (15 days, pad to 16)
    test_cutoff = pd.Timestamp("2017-08-15") # Test: 2017-08-16 to 2017-08-31 (16 days)

    def extract_features_for_cutoff(T, is_test=False):
        t_target = pd.date_range(T + pd.Timedelta(days=1), periods=16)
        t_past = pd.date_range(T - pd.Timedelta(days=29), periods=30)
        t_yoy = pd.date_range(T + pd.Timedelta(days=1) - pd.Timedelta(days=364), periods=16)

        x_lags = np.log1p(piv_sales.loc[t_past].values.T) # (1782, 30)
        x_promo = np.log1p(piv_promo.loc[t_target].values.T) # (1782, 16)
        x_yoy = np.log1p(piv_sales.loc[t_yoy].values.T) # (1782, 16)

        # Calendar features (start of horizon)
        start_date = t_target[0]
        month = start_date.month / 12.0
        day = start_date.day / 31.0
        dow = start_date.day_of_week / 6.0
        
        # Earthquake decay
        eq_days = max(0, (start_date - pd.Timestamp("2016-04-16")).days)
        eq_decay = np.exp(-eq_days / 30.0)

        cal_feat = np.array([month, day, dow, eq_decay], dtype=np.float32)
        cal_feats = np.tile(cal_feat, (n_series, 1)) # (1782, 4)

        x_cont = np.hstack([x_lags, x_promo, x_yoy, cal_feats]).astype(np.float32) # (1782, 66)

        if not is_test:
            y = np.log1p(piv_sales.loc[t_target].values.T).astype(np.float32) # (1782, 16)
            return x_cont, static_cats, y
        return x_cont, static_cats

    # Train dataset
    X_cont_tr, X_cat_tr, y_tr = [], [], []
    for T in train_cutoffs:
        xc, xcat, y = extract_features_for_cutoff(T)
        X_cont_tr.append(xc)
        X_cat_tr.append(xcat)
        y_tr.append(y)

    X_cont_tr = np.vstack(X_cont_tr)
    X_cat_tr = np.vstack(X_cat_tr)
    y_tr = np.vstack(y_tr)

    # Validation dataset (2017-08-01 to 2017-08-15)
    X_cont_va, X_cat_va, y_va = extract_features_for_cutoff(val_cutoff)

    # Test dataset (2017-08-16 to 2017-08-31)
    X_cont_te, X_cat_te = extract_features_for_cutoff(test_cutoff, is_test=True)

    print(f"  - Train samples: {len(y_tr)} | Cont dim: {X_cont_tr.shape[1]}")
    print(f"  - Val samples:   {len(y_va)}")
    print(f"  - Test samples:  {len(X_cont_te)} (1782 series x 16 horizons = 28512 predictions)")

    return (X_cont_tr, X_cat_tr, y_tr), (X_cont_va, X_cat_va, y_va), (X_cont_te, X_cat_te), test, series_cols


def train_and_predict(train_data, val_data, test_data, test_df, series_cols):
    """PyTorch Direct Multi-Horizon Deep ResNet Training with GPU acceleration"""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[GPU] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    X_cont_tr, X_cat_tr, y_tr = train_data
    X_cont_va, X_cat_va, y_va = val_data
    X_cont_te, X_cat_te = test_data

    # Standardize continuous features
    mean_cont = np.mean(X_cont_tr, axis=0)
    std_cont = np.std(X_cont_tr, axis=0) + 1e-5

    X_cont_tr = (X_cont_tr - mean_cont) / std_cont
    X_cont_va = (X_cont_va - mean_cont) / std_cont
    X_cont_te = (X_cont_te - mean_cont) / std_cont

    # Convert to Tensors
    ds_tr = TensorDataset(torch.tensor(X_cont_tr), torch.tensor(X_cat_tr), torch.tensor(y_tr))
    ds_va = TensorDataset(torch.tensor(X_cont_va), torch.tensor(X_cat_va), torch.tensor(y_va))
    ds_te = TensorDataset(torch.tensor(X_cont_te), torch.tensor(X_cat_te))

    batch_size = 1024
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)
    loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    class MultiHorizonResNet(nn.Module):
        def __init__(self, cont_dim=66, hidden_dim=512, dropout=0.15):
            super().__init__()
            # Entity Embeddings
            self.emb_store = nn.Embedding(54, 16)
            self.emb_fam = nn.Embedding(33, 16)
            self.emb_cluster = nn.Embedding(17, 8)
            self.emb_city = nn.Embedding(22, 8)
            self.emb_state = nn.Embedding(16, 8)
            self.emb_type = nn.Embedding(5, 4)

            total_emb_dim = 16 + 16 + 8 + 8 + 8 + 4 # 60 dims
            in_dim = total_emb_dim + cont_dim # 126 dims

            self.input_layer = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            )

            # Residual Block 1
            self.res1_fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.res1_norm1 = nn.LayerNorm(hidden_dim)
            self.res1_fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.res1_norm2 = nn.LayerNorm(hidden_dim)
            self.act1 = nn.SiLU()
            self.drop1 = nn.Dropout(dropout)

            # Residual Block 2
            self.res2_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.res2_norm1 = nn.LayerNorm(hidden_dim // 2)
            self.res2_fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
            self.res2_norm2 = nn.LayerNorm(hidden_dim // 2)
            self.skip2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.act2 = nn.SiLU()
            self.drop2 = nn.Dropout(dropout)

            # Direct 16-Horizon Head
            self.head = nn.Linear(hidden_dim // 2, 16)

        def forward(self, x_cont, x_cat):
            # Embeddings
            e_st = self.emb_store(x_cat[:, 0])
            e_fam = self.emb_fam(x_cat[:, 1])
            e_cl = self.emb_cluster(x_cat[:, 2])
            e_ci = self.emb_city(x_cat[:, 3])
            e_sta = self.emb_state(x_cat[:, 4])
            e_ty = self.emb_type(x_cat[:, 5])

            x_emb = torch.cat([e_st, e_fam, e_cl, e_ci, e_sta, e_ty], dim=-1)
            x = torch.cat([x_emb, x_cont], dim=-1)

            h = self.input_layer(x)

            # Block 1
            res1 = self.drop1(self.act1(self.res1_norm1(self.res1_fc1(h))))
            res1 = self.res1_norm2(self.res1_fc2(res1))
            h = self.act1(h + res1)

            # Block 2
            res2 = self.drop2(self.act2(self.res2_norm1(self.res2_fc1(h))))
            res2 = self.res2_norm2(self.res2_fc2(res2))
            h = self.act2(self.skip2(h) + res2)

            out = self.head(h)
            return out

    seeds = [42, 2026, 777]
    seed_preds_te = []
    seed_val_scores = []

    for seed in seeds:
        print(f"\n--- Training Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = MultiHorizonResNet(cont_dim=X_cont_tr.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience = 12
        patience_cnt = 0

        for epoch in range(1, 61):
            model.train()
            total_tr_loss = 0.0
            for xc, xcat, y in loader_tr:
                xc, xcat, y = xc.to(device), xcat.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(xc, xcat)
                loss = criterion(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_tr_loss += loss.item() * len(y)
            scheduler.step()
            tr_loss = total_tr_loss / len(ds_tr)

            # Validation
            model.eval()
            total_va_loss = 0.0
            with torch.no_grad():
                for xc, xcat, y in loader_va:
                    xc, xcat, y = xc.to(device), xcat.to(device), y.to(device)
                    pred = model(xc, xcat)
                    # evaluate on first 15 days (day 1-15 match validation holdout)
                    loss = criterion(pred[:, :15], y[:, :15])
                    total_va_loss += loss.item() * len(y)
            va_loss = total_va_loss / len(ds_va)
            va_rmsle = np.sqrt(va_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:2d}/60 | Train Loss: {tr_loss:.4f} | Val RMSLE: {va_rmsle:.5f}", flush=True)

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"  [Early Stopping] Epoch {epoch} - Best Val RMSLE: {np.sqrt(best_val_loss):.5f}")
                    break

        seed_val_scores.append(np.sqrt(best_val_loss))
        print(f"  Seed {seed} Best Val RMSLE: {seed_val_scores[-1]:.5f}")

        # Predict Test
        model.load_state_dict(best_state)
        model.to(device)
        model.eval()

        test_preds = []
        with torch.no_grad():
            for xc, xcat in loader_te:
                xc, xcat = xc.to(device), xcat.to(device)
                pred = model(xc, xcat) # (1782, 16)
                test_preds.append(pred.cpu().numpy())
        test_preds = np.vstack(test_preds) # (1782, 16)
        seed_preds_te.append(test_preds)

    print(f"\n[EVAL] 3-Seed Mean Val RMSLE: {np.mean(seed_val_scores):.5f}")

    # Average 3 seeds in log-space
    avg_log_preds = np.mean(seed_preds_te, axis=0) # (1782, 16)

    # Reconstruct test.csv row order (test.csv is ordered by date, then store, then family)
    # series_cols is [(store_nbr, family)] for 1782 series.
    # We create a DataFrame of shape (16 dates, 1782 series)
    test_dates = pd.date_range("2017-08-16", "2017-08-31")
    df_pred_matrix = pd.DataFrame(avg_log_preds.T, index=test_dates, columns=series_cols)

    # Unstack / melt into long format
    long_preds = df_pred_matrix.stack(["store_nbr", "family"]).reset_index()
    long_preds.columns = ["date", "store_nbr", "family", "log_sales"]
    long_preds["sales"] = np.clip(np.expm1(long_preds["log_sales"]), 0, None)

    # Merge with original test.csv to preserve exact id order
    test_sub = pd.merge(test_df[["id", "date", "store_nbr", "family"]], long_preds, on=["date", "store_nbr", "family"], how="left")

    out_path = "003_store-sales/submission_exp043_colab_l4_dnn.csv"
    if not os.path.exists("003_store-sales"):
        os.makedirs("003_store-sales", exist_ok=True)

    sub = pd.DataFrame({
        "id": test_sub["id"],
        "sales": test_sub["sales"]
    })
    sub.to_csv(out_path, index=False)
    print(f"\n[SAVE] Single DNN submission saved to: {out_path} ({len(sub)} rows)")

    # Save to Google Drive if available
    drive_dir = "/content/drive/MyDrive"
    if os.path.exists(drive_dir):
        drive_path = os.path.join(drive_dir, "submission_exp043_colab_l4_dnn.csv")
        sub.to_csv(drive_path, index=False)
        print(f"  - Synchronized to Google Drive: {drive_path}")

    # Base64 fallback stream
    with open(out_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    print(f"\n===BASE64_SUBMISSION_START===\n{b64}\n===BASE64_SUBMISSION_END===")

    return sub


def main():
    print("=" * 80)
    print("  003 Store Sales: Exp #043 - Direct Multi-Horizon Deep ResNet (Colab L4 GPU)")
    print("=" * 80)

    start_heartbeat(15)
    data_dir = ensure_environment_and_data()
    train_data, val_data, test_data, test_df, series_cols = build_datasets(data_dir)
    train_and_predict(train_data, val_data, test_data, test_df, series_cols)
    print("\n[COMPLETE] Exp #043 Multi-Horizon DNN finished successfully!")


if __name__ == "__main__":
    main()
