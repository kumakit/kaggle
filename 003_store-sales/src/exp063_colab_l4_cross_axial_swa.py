"""
003_store-sales: Exp #063 - Cross-Series Axial Attention + Multi-Window Stats + Top-3 SWA + Pseudo-Huber (Colab L4 GPU)
Kaggle Competition: Store Sales - Time Series Forecasting
Designed for Google Colab L4 GPU (VRAM 24GB) Ephemeral Execution

Combines the stability & record Val accuracy of Exp #057 (0.38439 Val / 0.38219 LB)
with the best non-overfitting cross-series & long-range enhancements from Exp #060:
  - Stable 56d (8w x 7 DOW) Conv + Axial Backbone augmented with analytic 112d/28d level,
    trend-delta, and intermittency (activity rate) channels (zero high-dim linear overfitting).
  - Pre-Holiday & Pre-Payday Eve covariates (national + city/state local holiday eves).
  - Pseudo-Huber (Charbonnier) smooth C^inf loss: sqrt(r^2 + beta^2) - beta (beta=0.30).
  - Within-seed Top-3 Checkpoint SWA (Stochastic Weight Averaging / prediction averaging)
    + 3-epoch linear warmup + CosineAnnealing across 5 seeds [42, 2026, 777, 1234, 9999].
  - Exports BOTH Test submission CSV and exact Val (2017-08-01..15) predictions (.npy/Base64).
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import subprocess
import threading
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

N_STORES = 54
N_FAMS = 33
N_SERIES = N_STORES * N_FAMS


def start_heartbeat(interval_sec: int = 20) -> None:
    """Keep Colab WebSocket connection alive via periodic stdout flush."""
    def beat() -> None:
        while True:
            time.sleep(interval_sec)
            print("[HEARTBEAT] L4 GPU Exp063 Cross-Series SWA active...", flush=True)

    threading.Thread(target=beat, daemon=True).start()


def ensure_environment_and_data() -> str:
    """Verify Kaggle credentials (without hardcoding secrets) and dataset CSV path."""
    print("[INIT] Verifying packages and datasets...")
    data_dir = "data"
    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        candidates = ["003_store-sales/data", "/content/data", "/content/kaggle/003_store-sales/data"]
        for c in candidates:
            if os.path.exists(os.path.join(c, "train.csv")):
                data_dir = c
                break

    if os.path.exists(os.path.join(data_dir, "train.csv")):
        print(f"  - Datasets ready at: {data_dir}")
        return data_dir

    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(kaggle_json_path):
        username = os.environ.get("KAGGLE_USERNAME", "")
        key = os.environ.get("KAGGLE_KEY", "")
        if not username or not key:
            try:
                from google.colab import userdata  # type: ignore
                username = username or userdata.get("KAGGLE_USERNAME")
                key = key or userdata.get("KAGGLE_KEY")
            except Exception:
                pass
        if username and key:
            os.makedirs(kaggle_dir, exist_ok=True)
            with open(kaggle_json_path, "w") as f:
                json.dump({"username": username, "key": key}, f)
            os.chmod(kaggle_json_path, 0o600)

    print("  - Downloading dataset from Kaggle...")
    os.makedirs("data", exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "store-sales-time-series-forecasting", "-p", "data"],
        check=True,
    )
    for zip_file in glob.glob("data/*.zip"):
        import zipfile
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall("data")
    data_dir = "data"
    print(f"  - Datasets ready at: {data_dir}")
    return data_dir


def build_tensor_store(data_dir: str) -> dict:
    """Precompute dense aligned (T, 54, 33) tensors plus analytic 112d/28d summary maps."""
    print("\n[DATA] Building aligned (T, 54, 33) tensors for Exp063...")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["date"])
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=["date"])
    stores = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    oil = pd.read_csv(os.path.join(data_dir, "oil.csv"), parse_dates=["date"])
    holidays = pd.read_csv(os.path.join(data_dir, "holidays_events.csv"), parse_dates=["date"])
    transactions = pd.read_csv(os.path.join(data_dir, "transactions.csv"), parse_dates=["date"])

    full_dates = pd.date_range("2013-01-01", "2017-08-31", freq="1D")
    n_dates = len(full_dates)
    date2idx = {d: i for i, d in enumerate(full_dates)}

    sorted_stores = sorted(stores["store_nbr"].unique())
    sorted_fams = sorted(train["family"].unique())
    assert len(sorted_stores) == N_STORES and sorted_stores == list(range(1, N_STORES + 1))
    assert len(sorted_fams) == N_FAMS
    series_cols = pd.MultiIndex.from_product([sorted_stores, sorted_fams], names=["store_nbr", "family"])

    piv_sales = (
        train.pivot(index="date", columns=["store_nbr", "family"], values="sales")
        .reindex(index=full_dates, columns=series_cols)
        .fillna(0.0)
    )
    piv_promo = (
        pd.concat([
            train.pivot(index="date", columns=["store_nbr", "family"], values="onpromotion"),
            test.pivot(index="date", columns=["store_nbr", "family"], values="onpromotion"),
        ])
        .reindex(index=full_dates, columns=series_cols)
        .fillna(0.0)
    )

    sales_log = np.log1p(np.clip(piv_sales.values, 0.0, None)).astype(np.float32).reshape(n_dates, N_STORES, N_FAMS)
    promo_log = np.log1p(np.clip(piv_promo.values, 0.0, None)).astype(np.float32).reshape(n_dates, N_STORES, N_FAMS)
    tr_end = date2idx[pd.Timestamp("2017-07-31")] + 1

    # Store transactions (normalized strictly <= 2017-07-31)
    piv_tx = (
        transactions.pivot(index="date", columns="store_nbr", values="transactions")
        .reindex(index=full_dates, columns=sorted_stores)
        .fillna(0.0)
    )
    tx_raw = np.log1p(np.clip(piv_tx.values, 0.0, None)).astype(np.float32)
    tx_norm_store = (tx_raw - tx_raw[:tr_end].mean()) / (tx_raw[:tr_end].std() + 1e-5)
    tx_norm = np.repeat(tx_norm_store[:, :, None], N_FAMS, axis=2)

    # Oil macro series
    oil_s = oil.set_index("date")["dcoilwtico"].reindex(full_dates).interpolate().ffill().bfill()
    oil_vals = oil_s.values.astype(np.float32)
    oil_norm = (oil_vals - oil_vals[:tr_end].mean()) / (oil_vals[:tr_end].std() + 1e-5)
    oil_diff14 = np.zeros_like(oil_norm)
    oil_diff14[14:] = oil_norm[14:] - oil_norm[:-14]

    # Holidays + Holiday Eve
    stores_map = stores.set_index("store_nbr")
    active_hol = holidays[~holidays["transferred"]].copy()
    nat_dates = set(active_hol[(active_hol["locale"] == "National") & (active_hol["type"] != "Work Day")]["date"])
    work_dates = set(active_hol[active_hol["type"] == "Work Day"]["date"])
    is_nat_hol = np.array([1.0 if d in nat_dates else 0.0 for d in full_dates], dtype=np.float32)
    is_nat_eve = np.array([1.0 if (d + pd.Timedelta(days=1)) in nat_dates else 0.0 for d in full_dates], dtype=np.float32)
    is_work_day = np.array([1.0 if d in work_dates else 0.0 for d in full_dates], dtype=np.float32)
    is_jan1 = np.array([1.0 if (d.month == 1 and d.day == 1) else 0.0 for d in full_dates], dtype=np.float32)

    reg_map = active_hol[active_hol["locale"] == "Regional"].groupby("locale_name")["date"].apply(set).to_dict()
    loc_map = active_hol[active_hol["locale"] == "Local"].groupby("locale_name")["date"].apply(set).to_dict()
    is_local_hol_store = np.zeros((n_dates, N_STORES), dtype=np.float32)
    is_local_eve_store = np.zeros((n_dates, N_STORES), dtype=np.float32)
    for s_idx, s in enumerate(sorted_stores):
        matched_dates = reg_map.get(stores_map.loc[s, "state"], set()) | loc_map.get(stores_map.loc[s, "city"], set())
        for d in matched_dates:
            if d in date2idx:
                is_local_hol_store[date2idx[d], s_idx] = 1.0
            d_eve = d - pd.Timedelta(days=1)
            if d_eve in date2idx:
                is_local_eve_store[date2idx[d_eve], s_idx] = 1.0

    is_local_hol = np.repeat(is_local_hol_store[:, :, None], N_FAMS, axis=2)
    is_local_eve = np.repeat(is_local_eve_store[:, :, None], N_FAMS, axis=2)

    dow = full_dates.dayofweek.values
    dom = full_dates.day.values
    month = full_dates.month.values
    dow_sin = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    dow_cos = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    dom_sin = np.sin(2 * np.pi * (dom - 1) / 31.0).astype(np.float32)
    dom_cos = np.cos(2 * np.pi * (dom - 1) / 31.0).astype(np.float32)
    mon_sin = np.sin(2 * np.pi * (month - 1) / 12.0).astype(np.float32)
    mon_cos = np.cos(2 * np.pi * (month - 1) / 12.0).astype(np.float32)
    is_payday = np.array([1.0 if (d.day == 15 or d.is_month_end) else 0.0 for d in full_dates], dtype=np.float32)
    is_payday_eve = np.array(
        [1.0 if ((d + pd.Timedelta(days=1)).day == 15 or (d + pd.Timedelta(days=1)).is_month_end) else 0.0 for d in full_dates],
        dtype=np.float32,
    )
    eq_start = pd.Timestamp("2016-04-16")
    eq_decay = np.array(
        [np.exp(-max(0, (d - eq_start).days) / 30.0) if d >= eq_start else 0.0 for d in full_dates],
        dtype=np.float32,
    )

    date_cov = np.stack(
        [
            oil_norm,
            oil_diff14,
            is_nat_hol,
            is_nat_eve,
            is_work_day,
            is_jan1,
            is_payday,
            is_payday_eve,
            dow_sin,
            dow_cos,
            dom_sin,
            dom_cos,
            mon_sin,
            mon_cos,
            eq_decay,
        ],
        axis=-1,
    ).astype(np.float32)  # (T, 15)

    # Analytic rolling stats per cutoff t (causal [t-W+1 .. t]) using prefix sums -> O(1) lookup at runtime
    cs_sales = np.concatenate([np.zeros((1, N_STORES, N_FAMS), dtype=np.float64), np.cumsum(sales_log, axis=0, dtype=np.float64)], axis=0)
    active_bin = (sales_log > 0).astype(np.float64)
    cs_active = np.concatenate([np.zeros((1, N_STORES, N_FAMS), dtype=np.float64), np.cumsum(active_bin, axis=0, dtype=np.float64)], axis=0)

    def rolling_causal_mean(cs: np.ndarray, win: int) -> np.ndarray:
        idx = np.arange(n_dates)
        lo = np.clip(idx + 1 - win, 0, n_dates)
        cnt = np.maximum(1.0, (idx + 1 - lo)[:, None, None])
        return ((cs[idx + 1] - cs[lo]) / cnt).astype(np.float32)

    roll28_sales = rolling_causal_mean(cs_sales, 28)
    roll112_sales = rolling_causal_mean(cs_sales, 112)
    roll28_act = rolling_causal_mean(cs_active, 28)
    roll112_act = rolling_causal_mean(cs_active, 112)

    cities = sorted(stores["city"].unique())
    states = sorted(stores["state"].unique())
    types = sorted(stores["type"].unique())
    city2idx = {c: i for i, c in enumerate(cities)}
    state2idx = {st: i for i, st in enumerate(states)}
    type2idx = {tp: i for i, tp in enumerate(types)}

    store_meta = np.array(
        [
            [
                s - 1,
                stores_map.loc[s, "cluster"] - 1,
                city2idx[stores_map.loc[s, "city"]],
                state2idx[stores_map.loc[s, "state"]],
                type2idx[stores_map.loc[s, "type"]],
            ]
            for s in sorted_stores
        ],
        dtype=np.int64,
    )

    train_cutoffs = np.array(
        [date2idx[d] for d in pd.date_range("2015-05-01", "2017-07-15", freq="1D")],
        dtype=np.int64,
    )
    val_cutoff = date2idx[pd.Timestamp("2017-07-31")]
    test_cutoff = date2idx[pd.Timestamp("2017-08-15")]

    print(
        f"  - Tensors ready: T={n_dates}, Shape=({N_STORES}, {N_FAMS}), DateCov={date_cov.shape[-1]} | "
        f"Train cutoff snapshots={len(train_cutoffs)} ({len(train_cutoffs) * N_SERIES:,} series-windows)"
    )

    return {
        "sales_log": sales_log,
        "promo_log": promo_log,
        "tx_norm": tx_norm,
        "is_local_hol": is_local_hol,
        "is_local_eve": is_local_eve,
        "date_cov": date_cov,
        "roll28_sales": roll28_sales,
        "roll112_sales": roll112_sales,
        "roll28_act": roll28_act,
        "roll112_act": roll112_act,
        "store_meta": store_meta,
        "train_cutoffs": train_cutoffs,
        "val_cutoff": val_cutoff,
        "test_cutoff": test_cutoff,
        "series_cols": series_cols,
        "test_df": test,
    }


def train_and_predict(tensors: dict, smoke_test: bool = False) -> pd.DataFrame:
    """Train 5-Seed Cross-Series Axial Attention + Top-3 SWA + Pseudo-Huber on GPU."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    print(
        f"\n[GPU] Device: {device} ({torch.cuda.get_device_name(0) if use_amp else 'CPU'}) "
        f"| AMP={use_amp} ({amp_dtype})"
    )

    PAST_LEN = 56  # 8 exact weeks x 7 DOW phases
    HORIZON = 16

    sales_t = torch.tensor(tensors["sales_log"], device=device)
    promo_t = torch.tensor(tensors["promo_log"], device=device)
    tx_t = torch.tensor(tensors["tx_norm"], device=device)
    loc_hol_t = torch.tensor(tensors["is_local_hol"], device=device)
    loc_eve_t = torch.tensor(tensors["is_local_eve"], device=device)
    date_cov_t = torch.tensor(tensors["date_cov"], device=device)
    r28_s_t = torch.tensor(tensors["roll28_sales"], device=device)
    r112_s_t = torch.tensor(tensors["roll112_sales"], device=device)
    r28_a_t = torch.tensor(tensors["roll28_act"], device=device)
    r112_a_t = torch.tensor(tensors["roll112_act"], device=device)
    store_meta_t = torch.tensor(tensors["store_meta"], device=device)
    fam_idx_base = torch.arange(N_FAMS, device=device, dtype=torch.long)

    tr_end = tensors["val_cutoff"]
    s_mean = sales_t[:tr_end].mean()
    s_std = sales_t[:tr_end].std() + 1e-5
    p_mean = promo_t[:tr_end].mean()
    p_std = promo_t[:tr_end].std() + 1e-5

    past_offsets = torch.arange(-PAST_LEN + 1, 1, device=device)
    fut_offsets = torch.arange(1, HORIZON + 1, device=device)
    dow_gather = torch.tensor([(h - 1) % 7 for h in range(1, HORIZON + 1)], device=device, dtype=torch.long)
    h_frac_base = torch.arange(1, HORIZON + 1, device=device, dtype=torch.float32) / HORIZON

    def gather_snapshots(cutoffs_idx: torch.Tensor, is_test: bool = False):
        B = len(cutoffs_idx)
        t_past = cutoffs_idx.unsqueeze(1) + past_offsets.unsqueeze(0)
        t_fut = cutoffs_idx.unsqueeze(1) + fut_offsets.unsqueeze(0)

        p_sales = sales_t[t_past].permute(0, 2, 3, 1).contiguous()
        p_promo = promo_t[t_past].permute(0, 2, 3, 1).contiguous()
        p_tx = tx_t[t_past].permute(0, 2, 3, 1).contiguous()
        p_lhol = loc_hol_t[t_past].permute(0, 2, 3, 1).contiguous()
        p_leve = loc_eve_t[t_past].permute(0, 2, 3, 1).contiguous()
        p_dcov = date_cov_t[t_past][:, None, None, :, :].expand(B, N_STORES, N_FAMS, PAST_LEN, 15)

        revin_mean = p_sales.mean(dim=-1, keepdim=True)
        revin_std = torch.clamp(p_sales.std(dim=-1, keepdim=True), min=0.25)
        p_sales_norm = (p_sales - revin_mean) / revin_std

        p_sales_store_rel = p_sales - p_sales.mean(dim=2, keepdim=True)
        p_sales_fam_rel = p_sales - p_sales.mean(dim=1, keepdim=True)
        p_promo_store_b = ((p_promo.mean(dim=2, keepdim=True) - p_mean) / p_std).expand(-1, -1, N_FAMS, -1)
        p_promo_fam_b = ((p_promo.mean(dim=1, keepdim=True) - p_mean) / p_std).expand(-1, N_STORES, -1, -1)

        p_weekly = p_sales.view(B, N_STORES, N_FAMS, 8, 7)
        sdow_med8 = p_weekly.median(dim=-2).values[..., dow_gather]
        sdow_mean8 = p_weekly.mean(dim=-2)[..., dow_gather]

        # Long-range analytic causal summary features at cutoff t -> broadcast across horizon 16
        c_r28 = r28_s_t[cutoffs_idx].unsqueeze(-1).expand(-1, -1, -1, HORIZON)
        c_r112 = r112_s_t[cutoffs_idx].unsqueeze(-1).expand(-1, -1, -1, HORIZON)
        c_trend = c_r28 - c_r112
        c_a28 = r28_a_t[cutoffs_idx].unsqueeze(-1).expand(-1, -1, -1, HORIZON)
        c_a112 = r112_a_t[cutoffs_idx].unsqueeze(-1).expand(-1, -1, -1, HORIZON)

        f_promo = promo_t[t_fut].permute(0, 2, 3, 1).contiguous()
        f_lhol = loc_hol_t[t_fut].permute(0, 2, 3, 1).contiguous()
        f_leve = loc_eve_t[t_fut].permute(0, 2, 3, 1).contiguous()
        f_yoy1 = sales_t[t_fut - 364].permute(0, 2, 3, 1).contiguous()
        f_yoy2 = sales_t[t_fut - 728].permute(0, 2, 3, 1).contiguous()
        f_dcov = date_cov_t[t_fut][:, None, None, :, :].expand(B, N_STORES, N_FAMS, HORIZON, 15)

        p_promo_mean = p_promo.mean(dim=-1, keepdim=True)
        f_promo_store_b = ((f_promo.mean(dim=2, keepdim=True) - p_mean) / p_std).expand(-1, -1, N_FAMS, -1)
        f_promo_fam_b = ((f_promo.mean(dim=1, keepdim=True) - p_mean) / p_std).expand(-1, N_STORES, -1, -1)
        h_frac = h_frac_base.view(1, 1, 1, HORIZON, 1).expand(B, N_STORES, N_FAMS, -1, -1)

        x_past = torch.cat(
            [
                p_sales_norm.unsqueeze(-1),
                ((p_sales - s_mean) / s_std).unsqueeze(-1),
                p_sales_store_rel.unsqueeze(-1),
                p_sales_fam_rel.unsqueeze(-1),
                ((p_promo - p_mean) / p_std).unsqueeze(-1),
                (p_promo - p_promo_mean).unsqueeze(-1),
                p_promo_store_b.unsqueeze(-1),
                p_promo_fam_b.unsqueeze(-1),
                p_tx.unsqueeze(-1),
                p_lhol.unsqueeze(-1),
                p_leve.unsqueeze(-1),
                (p_sales > 0).float().unsqueeze(-1),
                p_dcov,  # 15 cols -> 27 total
            ],
            dim=-1,
        )

        x_fut = torch.cat(
            [
                ((f_promo - p_mean) / p_std).unsqueeze(-1),
                (f_promo - p_promo_mean).unsqueeze(-1),
                f_promo_store_b.unsqueeze(-1),
                f_promo_fam_b.unsqueeze(-1),
                ((f_yoy1 - revin_mean) / revin_std).unsqueeze(-1),
                ((f_yoy2 - revin_mean) / revin_std).unsqueeze(-1),
                ((f_yoy1 - s_mean) / s_std).unsqueeze(-1),
                ((sdow_med8 - revin_mean) / revin_std).unsqueeze(-1),
                ((sdow_mean8 - revin_mean) / revin_std).unsqueeze(-1),
                ((c_r28 - revin_mean) / revin_std).unsqueeze(-1),
                ((c_r112 - revin_mean) / revin_std).unsqueeze(-1),
                c_trend.unsqueeze(-1),
                c_a28.unsqueeze(-1),
                c_a112.unsqueeze(-1),
                f_lhol.unsqueeze(-1),
                f_leve.unsqueeze(-1),
                h_frac,
                f_dcov,  # 15 cols -> 32 total
            ],
            dim=-1,
        )

        if is_test:
            return x_past, x_fut, p_sales_norm, revin_mean, revin_std, sdow_med8, sdow_mean8
        y_fut = sales_t[t_fut].permute(0, 2, 3, 1).contiguous()
        return x_past, x_fut, p_sales_norm, revin_mean, revin_std, sdow_med8, sdow_mean8, y_fut

    class AxialAttentionBlock(nn.Module):
        """Factorized Cross-Family (intra-store) + Cross-Store (intra-family) Transformer block."""
        def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.16):
            super().__init__()
            self.norm_f1 = nn.LayerNorm(d_model)
            self.attn_f = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm_f2 = nn.LayerNorm(d_model)
            self.mlp_f = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout),
            )

            self.norm_s1 = nn.LayerNorm(d_model)
            self.attn_s = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.norm_s2 = nn.LayerNorm(d_model)
            self.mlp_s = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lead_shape = x.shape[:-3]
            D = x.shape[-1]

            xf = x.reshape(-1, N_FAMS, D)
            xf_n = self.norm_f1(xf)
            xf = xf + self.attn_f(xf_n, xf_n, xf_n, need_weights=False)[0]
            xf = xf + self.mlp_f(self.norm_f2(xf))

            xs = xf.reshape(*lead_shape, N_STORES, N_FAMS, D).transpose(-3, -2).contiguous().reshape(-1, N_STORES, D)
            xs_n = self.norm_s1(xs)
            xs = xs + self.attn_s(xs_n, xs_n, xs_n, need_weights=False)[0]
            xs = xs + self.mlp_s(self.norm_s2(xs))

            return xs.reshape(*lead_shape, N_FAMS, N_STORES, D).transpose(-3, -2).contiguous()

    class CrossAxialSWANet(nn.Module):
        def __init__(self, past_dim: int = 27, fut_dim: int = 32, d_past: int = 128, d_fut: int = 96, dropout: float = 0.16):
            super().__init__()
            self.emb_store = nn.Embedding(N_STORES, 16)
            self.emb_cluster = nn.Embedding(17, 8)
            self.emb_city = nn.Embedding(22, 8)
            self.emb_state = nn.Embedding(16, 8)
            self.emb_type = nn.Embedding(5, 4)
            self.emb_fam = nn.Embedding(N_FAMS, 20)
            emb_dim = 64

            self.t_conv1 = nn.Conv1d(past_dim, d_past, kernel_size=3, padding=1)
            self.t_conv7 = nn.Conv1d(d_past, d_past, kernel_size=3, dilation=7, padding=7)
            self.t_norm = nn.LayerNorm(d_past)

            d_dow = 48
            self.week_weights = nn.Parameter(torch.linspace(0.5, 1.5, 8).view(1, 1, 1, 8, 1, 1))
            self.dow_proj = nn.Linear(d_past, d_dow)
            self.past_summary_proj = nn.Sequential(
                nn.Linear(d_past + d_dow + emb_dim, d_past),
                nn.LayerNorm(d_past),
                nn.GELU(),
            )

            self.past_axial = nn.ModuleList([AxialAttentionBlock(d_past, n_heads=4, dropout=dropout) for _ in range(2)])

            self.linear_norm_map = nn.Linear(PAST_LEN, HORIZON)
            self.anchor_gate = nn.Linear(emb_dim, 3)

            self.fut_proj = nn.Sequential(
                nn.Linear(fut_dim + d_past + d_dow + emb_dim, d_fut),
                nn.LayerNorm(d_fut),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.fut_pos = nn.Parameter(torch.randn(1, 1, 1, HORIZON, d_fut) * 0.02)
            self.fut_axial = AxialAttentionBlock(d_fut, n_heads=4, dropout=dropout)

            self.out_head = nn.Sequential(
                nn.Linear(d_fut + emb_dim, d_fut // 2),
                nn.LayerNorm(d_fut // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_fut // 2, 1),
            )

        def _build_entity_emb(self) -> torch.Tensor:
            s_e = torch.cat(
                [
                    self.emb_store(store_meta_t[:, 0]),
                    self.emb_cluster(store_meta_t[:, 1]),
                    self.emb_city(store_meta_t[:, 2]),
                    self.emb_state(store_meta_t[:, 3]),
                    self.emb_type(store_meta_t[:, 4]),
                ],
                dim=-1,
            )
            f_e = self.emb_fam(fam_idx_base)
            return torch.cat(
                [s_e.unsqueeze(1).expand(N_STORES, N_FAMS, -1), f_e.unsqueeze(0).expand(N_STORES, N_FAMS, -1)],
                dim=-1,
            )

        def forward(self, x_past, x_fut, p_sales_norm, revin_mean, revin_std, sdow_med8, sdow_mean8):
            B = x_past.shape[0]
            e_grid = self._build_entity_emb().unsqueeze(0).expand(B, -1, -1, -1)

            xp = x_past.reshape(B * N_SERIES, PAST_LEN, -1).transpose(1, 2)
            h = F.gelu(self.t_conv1(xp))
            h = h + F.gelu(self.t_conv7(h))
            h = self.t_norm(h.transpose(1, 2))

            h_weekly = h.view(B, N_STORES, N_FAMS, 8, 7, -1)
            w_soft = F.softmax(self.week_weights, dim=3)
            dow_lat = self.dow_proj((h_weekly * w_soft).sum(dim=3))

            h_last_week = h_weekly[:, :, :, -1, :, :].mean(dim=3)
            state = self.past_summary_proj(torch.cat([h_last_week, dow_lat.mean(dim=3), e_grid], dim=-1))

            for blk in self.past_axial:
                state = blk(state)

            lin_log = revin_mean + revin_std * self.linear_norm_map(p_sales_norm)
            aw = F.softmax(self.anchor_gate(e_grid), dim=-1)
            base_anchor = aw[..., 0:1] * sdow_med8 + aw[..., 1:2] * sdow_mean8 + aw[..., 2:3] * lin_log

            dow_h = dow_lat[:, :, :, dow_gather, :]
            state_h = state.unsqueeze(3).expand(-1, -1, -1, HORIZON, -1)
            e_h = e_grid.unsqueeze(3).expand(-1, -1, -1, HORIZON, -1)

            fut_tok = self.fut_proj(torch.cat([x_fut, state_h, dow_h, e_h], dim=-1)) + self.fut_pos
            fut_out = self.fut_axial(fut_tok.permute(0, 3, 1, 2, 4).contiguous()).permute(0, 2, 3, 1, 4).contiguous()

            res_norm = self.out_head(torch.cat([fut_out, e_h], dim=-1)).squeeze(-1)
            return base_anchor + revin_std * res_norm

    val_c = tensors["val_cutoff"]
    val_dead_mask = (sales_t[val_c - 20 : val_c + 1].sum(dim=0) == 0.0).cpu().numpy()

    with torch.no_grad():
        va_past, va_fut, va_psn, va_rm, va_rs, va_smed, va_smean, va_y = gather_snapshots(
            torch.tensor([val_c], device=device, dtype=torch.long), is_test=False
        )
        te_past, te_fut, te_psn, te_rm, te_rs, te_smed, te_smean = gather_snapshots(
            torch.tensor([tensors["test_cutoff"]], device=device, dtype=torch.long), is_test=True
        )

    train_cutoffs_arr = tensors["train_cutoffs"]
    if smoke_test:
        train_cutoffs_arr = train_cutoffs_arr[-6:]
        seeds = [42]
        n_epochs = 3
        batch_dates = 2
        warmup_epochs = 1
    else:
        seeds = [42, 2026, 777, 1234, 9999]
        n_epochs = 48
        batch_dates = 6
        warmup_epochs = 3

    train_cutoffs_t = torch.tensor(train_cutoffs_arr, device=device, dtype=torch.long)
    n_cutoffs = len(train_cutoffs_t)

    seed_preds_va = []
    seed_preds_te = []
    seed_val_raw = []
    seed_val_swa = []

    t0 = time.time()
    beta_ch = 0.30
    for seed in seeds:
        print(
            f"\n--- Training Seed {seed} ({n_epochs} epochs, {n_cutoffs} day snapshots, "
            f"batch_dates={batch_dates} = {batch_dates * N_SERIES:,} series/step) ---"
        )
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = CrossAxialSWANet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.35e-3, weight_decay=3.5e-4)

        def lr_lambda(ep: int) -> float:
            if ep < warmup_epochs:
                return 0.25 + 0.75 * (ep + 1) / warmup_epochs
            progress = (ep - warmup_epochs) / max(1, n_epochs - warmup_epochs)
            return max(0.015, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        top_k: list[tuple[float, dict]] = []  # keep top-3 checkpoints by val_mse
        patience = 12
        patience_cnt = 0
        best_val_mse = float("inf")

        for epoch in range(1, n_epochs + 1):
            model.train()
            perm = torch.randperm(n_cutoffs, device=device)
            tr_loss_sum = 0.0
            n_steps = 0

            for start in range(0, n_cutoffs, batch_dates):
                c_batch = train_cutoffs_t[perm[start : start + batch_dates]]
                bx_p, bx_f, bpsn, brm, brs, bsmed, bsmean, by = gather_snapshots(c_batch, is_test=False)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    pred = model(bx_p, bx_f, bpsn, brm, brs, bsmed, bsmean)
                    diff = pred - by
                    # Smooth C^inf Pseudo-Huber (Charbonnier) loss
                    loss = (torch.sqrt(diff * diff + beta_ch * beta_ch) - beta_ch).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss_sum += loss.item()
                n_steps += 1

            scheduler.step()
            tr_loss = tr_loss_sum / max(1, n_steps)

            model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    va_pred = torch.clamp(
                        model(va_past, va_fut, va_psn, va_rm, va_rs, va_smed, va_smean).float(), min=0.0
                    )
                va_mse = F.mse_loss(va_pred[0, :, :, :15], va_y[0, :, :, :15]).item()
                va_rmsle = np.sqrt(va_mse)

                va_pred_z = va_pred[0, :, :, :15].clone()
                va_pred_z[torch.tensor(val_dead_mask, device=device)] = 0.0
                va_zmask_rmsle = np.sqrt(F.mse_loss(va_pred_z, va_y[0, :, :, :15]).item())

            # Maintain Top-3 checkpoints per seed
            state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            top_k.append((va_mse, state_cpu))
            top_k.sort(key=lambda x: x[0])
            top_k = top_k[:3]

            if epoch % 5 == 0 or epoch == 1 or smoke_test:
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch:2d}/{n_epochs} [{elapsed:.0f}s] | Train Charb: {tr_loss:.5f} | "
                    f"Val Raw: {va_rmsle:.5f} | Val ZMask: {va_zmask_rmsle:.5f}",
                    flush=True,
                )

            if va_mse < best_val_mse:
                best_val_mse = va_mse
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"  [Early Stopping] Epoch {epoch} - Best Single Val Raw: {np.sqrt(best_val_mse):.5f}")
                    break

        # Top-3 Checkpoint Prediction Averaging within seed
        ckpt_va_list = []
        ckpt_te_list = []
        for _, st in top_k:
            model.load_state_dict(st)
            model.to(device).eval()
            with torch.no_grad():
                ckpt_va_list.append(
                    torch.clamp(model(va_past, va_fut, va_psn, va_rm, va_rs, va_smed, va_smean).float(), min=0.0)[0]
                    .reshape(N_SERIES, HORIZON)
                    .cpu()
                    .numpy()
                )
                ckpt_te_list.append(
                    torch.clamp(model(te_past, te_fut, te_psn, te_rm, te_rs, te_smed, te_smean).float(), min=0.0)[0]
                    .reshape(N_SERIES, HORIZON)
                    .cpu()
                    .numpy()
                )

        seed_va_swa = np.mean(ckpt_va_list, axis=0)
        seed_te_swa = np.mean(ckpt_te_list, axis=0)
        swa_rmsle = float(np.sqrt(np.mean((seed_va_swa[:, :15] - va_y[0].reshape(N_SERIES, HORIZON)[:, :15].cpu().numpy()) ** 2)))

        seed_val_raw.append(float(np.sqrt(best_val_mse)))
        seed_val_swa.append(swa_rmsle)
        print(f"  Seed {seed} Best Single Val Raw: {seed_val_raw[-1]:.5f} | Top-3 SWA Val Raw: {swa_rmsle:.5f}")

        seed_preds_va.append(seed_va_swa)
        seed_preds_te.append(seed_te_swa)

    y_va_np = va_y[0].reshape(N_SERIES, HORIZON)[:, :15].cpu().numpy()
    ens_va = np.mean(seed_preds_va, axis=0)[:, :15]
    ens_raw = float(np.sqrt(np.mean((ens_va - y_va_np) ** 2)))
    ens_va_z = ens_va.copy()
    ens_va_z[val_dead_mask.reshape(N_SERIES)] = 0.0
    ens_zmask = float(np.sqrt(np.mean((ens_va_z - y_va_np) ** 2)))

    print(f"\n[EVAL] 5-Seed Mean Single Best Val Raw:  {np.mean(seed_val_raw):.5f}")
    print(f"[EVAL] 5-Seed Mean Top-3 SWA Val Raw:    {np.mean(seed_val_swa):.5f}")
    print(f"[EVAL] 5-Seed ENSEMBLE Val Raw RMSLE:    {ens_raw:.5f}")
    print(f"[EVAL] 5-Seed ENSEMBLE Val ZMask RMSLE:  {ens_zmask:.5f}")

    avg_log_te = np.mean(seed_preds_te, axis=0)
    test_dates = pd.date_range("2017-08-16", "2017-08-31")
    df_pred_matrix = pd.DataFrame(avg_log_te.T, index=test_dates, columns=tensors["series_cols"])
    long_preds = df_pred_matrix.stack(["store_nbr", "family"]).reset_index()
    long_preds.columns = ["date", "store_nbr", "family", "log_sales"]
    long_preds["sales"] = np.clip(np.expm1(long_preds["log_sales"]), 0.0, None)

    test_df = tensors["test_df"]
    test_sub = pd.merge(
        test_df[["id", "date", "store_nbr", "family"]],
        long_preds,
        on=["date", "store_nbr", "family"],
        how="left",
    )
    sub = pd.DataFrame({"id": test_sub["id"], "sales": test_sub["sales"]})

    assert len(sub) == 28512, f"Expected 28512 rows, got {len(sub)}"
    assert int(sub["sales"].isna().sum()) == 0, "NaNs found in predictions!"

    if smoke_test:
        print("\n[SMOKE TEST PASSED]")
        return sub

    out_path = "003_store-sales/submission_exp063_colab_l4_cross_swa.csv"
    os.makedirs("003_store-sales", exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"\n[SAVE] Saved Test to: {out_path} ({len(sub)} rows, Mean={sub['sales'].mean():.2f})")

    # Also save exact Val predictions (1782 x 15 in log1p space) for local Family-adaptive OOF verification
    os.makedirs("003_store-sales/data", exist_ok=True)
    val_out_path = "003_store-sales/data/val_exp063_cross_swa.npy"
    np.save(val_out_path, ens_va.astype(np.float32))

    drive_dir = "/content/drive/MyDrive"
    if os.path.exists(drive_dir):
        sub.to_csv(os.path.join(drive_dir, "submission_exp063_colab_l4_cross_swa.csv"), index=False)
        np.save(os.path.join(drive_dir, "val_exp063_cross_swa.npy"), ens_va.astype(np.float32))

    with open(out_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    print(f"\n===BASE64_SUBMISSION_START===\n{b64}\n===BASE64_SUBMISSION_END===")

    buf = io.BytesIO()
    np.save(buf, ens_va.astype(np.float32))
    val_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    print(f"\n===BASE64_VAL_START===\n{val_b64}\n===BASE64_VAL_END===")
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp #063: Cross-Series Axial + Top-3 SWA + Pseudo-Huber (Colab L4).")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("  003 Store Sales: Exp #063 - Cross-Series Axial + Top-3 SWA (Colab L4)")
    print("=" * 80)

    if not args.smoke_test:
        start_heartbeat(20)
    data_dir = ensure_environment_and_data()
    tensors = build_tensor_store(data_dir)
    train_and_predict(tensors, smoke_test=args.smoke_test)
    print("\n[COMPLETE] Exp #063 finished successfully!")


if __name__ == "__main__":
    main()
