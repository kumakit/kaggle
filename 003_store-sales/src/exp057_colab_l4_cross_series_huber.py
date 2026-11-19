"""
003_store-sales: Exp #057 - Cross-Series Factorized Axial Attention + SmoothL1/Huber (Colab L4 GPU)
Kaggle Competition: Store Sales - Time Series Forecasting
Designed for Google Colab L4 GPU (VRAM 24GB) Ephemeral Execution

True 6th Orthogonal Pillar (Complementary to #036 GBDT1, #043 ResNet, #050 BiGRU-TCN, #052 Patch-DLinear, #054 GBDT2):
  - Full Cross-Series Spatiotemporal Snapshots: Every sample is an entire day snapshot (54 Stores x 33 Families = 1,782)
    processed simultaneously rather than independent flattened series.
  - Factorized 2D Axial Attention across BOTH Past State & Future 16-Day Horizons:
      * Axis 1 (Cross-Family within Store): Models basket halo & category cannibalization on promo/holiday shocks.
      * Axis 2 (Cross-Store within Family): Models city/regional/cluster demand waves across Ecuador.
  - Weekly-Folded 56d Lookback (8 weeks x 7 DOW phases): Exact weekday-aligned representation for horizon h.
  - Robust Orthogonal Loss & Anchor:
      * Trained with Huber/SmoothL1 loss (beta=0.45) to decorrelate outlier/holiday errors from all 5 MSE pillars.
      * Dual robust Same-DOW Median + Mean + RevIN Trend gated anchor.
  - Strictly validated on Val Raw RMSLE (2017-08-01..15) across 5 seeds [42, 2026, 777, 1234, 9999].
"""

from __future__ import annotations

import argparse
import base64
import glob
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
            print("[HEARTBEAT] L4 GPU Cross-Series Axial Attention active...", flush=True)

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
    """Precompute dense aligned (T, 54, 33) spatiotemporal tensors."""
    print("\n[DATA] Building aligned (T, 54, 33) cross-series tensors...")
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

    # Store transactions (normalized strictly on <= 2017-07-31) -> (T, 54, 33)
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

    # Holidays (National, Work Day, Jan 1, and City/State matched Local/Regional)
    stores_map = stores.set_index("store_nbr")
    active_hol = holidays[~holidays["transferred"]].copy()
    nat_dates = set(active_hol[(active_hol["locale"] == "National") & (active_hol["type"] != "Work Day")]["date"])
    work_dates = set(active_hol[active_hol["type"] == "Work Day"]["date"])
    is_nat_hol = np.array([1.0 if d in nat_dates else 0.0 for d in full_dates], dtype=np.float32)
    is_work_day = np.array([1.0 if d in work_dates else 0.0 for d in full_dates], dtype=np.float32)
    is_jan1 = np.array([1.0 if (d.month == 1 and d.day == 1) else 0.0 for d in full_dates], dtype=np.float32)

    reg_map = active_hol[active_hol["locale"] == "Regional"].groupby("locale_name")["date"].apply(set).to_dict()
    loc_map = active_hol[active_hol["locale"] == "Local"].groupby("locale_name")["date"].apply(set).to_dict()
    is_local_hol_store = np.zeros((n_dates, N_STORES), dtype=np.float32)
    for s_idx, s in enumerate(sorted_stores):
        matched_dates = reg_map.get(stores_map.loc[s, "state"], set()) | loc_map.get(stores_map.loc[s, "city"], set())
        for d in matched_dates:
            if d in date2idx:
                is_local_hol_store[date2idx[d], s_idx] = 1.0
    is_local_hol = np.repeat(is_local_hol_store[:, :, None], N_FAMS, axis=2)

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
            is_work_day,
            is_jan1,
            is_payday,
            dow_sin,
            dow_cos,
            dom_sin,
            dom_cos,
            mon_sin,
            mon_cos,
            eq_decay,
        ],
        axis=-1,
    ).astype(np.float32)  # (T, 13)

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
    )  # (54, 5)

    train_cutoffs = np.array(
        [date2idx[d] for d in pd.date_range("2015-05-01", "2017-07-15", freq="1D")],
        dtype=np.int64,
    )
    val_cutoff = date2idx[pd.Timestamp("2017-07-31")]
    test_cutoff = date2idx[pd.Timestamp("2017-08-15")]

    print(
        f"  - Tensors ready: T={n_dates}, Shape=({N_STORES}, {N_FAMS}) | "
        f"Train cutoff snapshots={len(train_cutoffs)} ({len(train_cutoffs) * N_SERIES:,} series-windows)"
    )

    return {
        "sales_log": sales_log,
        "promo_log": promo_log,
        "tx_norm": tx_norm,
        "is_local_hol": is_local_hol,
        "date_cov": date_cov,
        "store_meta": store_meta,
        "train_cutoffs": train_cutoffs,
        "val_cutoff": val_cutoff,
        "test_cutoff": test_cutoff,
        "series_cols": series_cols,
        "test_df": test,
    }


def train_and_predict(tensors: dict, smoke_test: bool = False) -> pd.DataFrame:
    """Train 5-Seed Cross-Series Axial Attention + Huber/SmoothL1 on GPU and infer Test."""
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

    sales_t = torch.tensor(tensors["sales_log"], device=device)      # (T, 54, 33)
    promo_t = torch.tensor(tensors["promo_log"], device=device)      # (T, 54, 33)
    tx_t = torch.tensor(tensors["tx_norm"], device=device)           # (T, 54, 33)
    loc_hol_t = torch.tensor(tensors["is_local_hol"], device=device) # (T, 54, 33)
    date_cov_t = torch.tensor(tensors["date_cov"], device=device)    # (T, 13)
    store_meta_t = torch.tensor(tensors["store_meta"], device=device)  # (54, 5)
    fam_idx_base = torch.arange(N_FAMS, device=device, dtype=torch.long)

    tr_end = tensors["val_cutoff"]
    s_mean = sales_t[:tr_end].mean()
    s_std = sales_t[:tr_end].std() + 1e-5
    p_mean = promo_t[:tr_end].mean()
    p_std = promo_t[:tr_end].std() + 1e-5

    past_offsets = torch.arange(-PAST_LEN + 1, 1, device=device)   # (56,)
    fut_offsets = torch.arange(1, HORIZON + 1, device=device)      # (16,)
    dow_gather = torch.tensor([(h - 1) % 7 for h in range(1, HORIZON + 1)], device=device, dtype=torch.long)
    h_frac_base = torch.arange(1, HORIZON + 1, device=device, dtype=torch.float32) / HORIZON

    def gather_snapshots(cutoffs_idx: torch.Tensor, is_test: bool = False):
        """Gather full-cross-section snapshots of shape (B, 54, 33, ...)."""
        B = len(cutoffs_idx)
        t_past = cutoffs_idx.unsqueeze(1) + past_offsets.unsqueeze(0)  # (B, 56)
        t_fut = cutoffs_idx.unsqueeze(1) + fut_offsets.unsqueeze(0)    # (B, 16)

        # Permute (B, T_win, 54, 33) -> (B, 54, 33, T_win)
        p_sales = sales_t[t_past].permute(0, 2, 3, 1).contiguous()
        p_promo = promo_t[t_past].permute(0, 2, 3, 1).contiguous()
        p_tx = tx_t[t_past].permute(0, 2, 3, 1).contiguous()
        p_lhol = loc_hol_t[t_past].permute(0, 2, 3, 1).contiguous()
        p_dcov = date_cov_t[t_past][:, None, None, :, :].expand(B, N_STORES, N_FAMS, PAST_LEN, 13)

        # Window RevIN statistics per (B, 54, 33, 1)
        revin_mean = p_sales.mean(dim=-1, keepdim=True)
        revin_std = torch.clamp(p_sales.std(dim=-1, keepdim=True), min=0.25)
        p_sales_norm = (p_sales - revin_mean) / revin_std

        # Cross-sectional relative signals within Store (over 33 fams) and within Family (over 54 stores)
        p_sales_store_rel = p_sales - p_sales.mean(dim=2, keepdim=True)
        p_sales_fam_rel = p_sales - p_sales.mean(dim=1, keepdim=True)
        p_promo_store_mean = (p_promo.mean(dim=2, keepdim=True) - p_mean) / p_std
        p_promo_fam_mean = (p_promo.mean(dim=1, keepdim=True) - p_mean) / p_std
        p_promo_store_b = p_promo_store_mean.expand(-1, -1, N_FAMS, -1)
        p_promo_fam_b = p_promo_fam_mean.expand(-1, N_STORES, -1, -1)

        # Exact weekly fold (B, 54, 33, 8, 7) -> Same-DOW Median & Mean Anchors
        p_weekly = p_sales.view(B, N_STORES, N_FAMS, 8, 7)
        sdow_med8 = p_weekly.median(dim=-2).values[..., dow_gather]   # (B, 54, 33, 16)
        sdow_mean8 = p_weekly.mean(dim=-2)[..., dow_gather]           # (B, 54, 33, 16)

        f_promo = promo_t[t_fut].permute(0, 2, 3, 1).contiguous()
        f_lhol = loc_hol_t[t_fut].permute(0, 2, 3, 1).contiguous()
        f_yoy1 = sales_t[t_fut - 364].permute(0, 2, 3, 1).contiguous()
        f_yoy2 = sales_t[t_fut - 728].permute(0, 2, 3, 1).contiguous()
        f_dcov = date_cov_t[t_fut][:, None, None, :, :].expand(B, N_STORES, N_FAMS, HORIZON, 13)

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
                (p_sales > 0).float().unsqueeze(-1),
                p_dcov,  # 13 cols -> 24 total
            ],
            dim=-1,
        )  # (B, 54, 33, 56, 24)

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
                f_lhol.unsqueeze(-1),
                h_frac,
                f_dcov,  # 13 cols -> 24 total
            ],
            dim=-1,
        )  # (B, 54, 33, 16, 24)

        if is_test:
            return x_past, x_fut, p_sales_norm, revin_mean, revin_std, sdow_med8, sdow_mean8
        y_fut = sales_t[t_fut].permute(0, 2, 3, 1).contiguous()       # (B, 54, 33, 16)
        return x_past, x_fut, p_sales_norm, revin_mean, revin_std, sdow_med8, sdow_mean8, y_fut

    class AxialAttentionBlock(nn.Module):
        """Factorized Cross-Family (intra-store) + Cross-Store (intra-family) Transformer block."""
        def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.15):
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
            # x: (N_leading, 54, 33, D)
            lead_shape = x.shape[:-3]
            D = x.shape[-1]

            # 1. Cross-Family within each Store
            xf = x.reshape(-1, N_FAMS, D)
            xf_n = self.norm_f1(xf)
            xf = xf + self.attn_f(xf_n, xf_n, xf_n, need_weights=False)[0]
            xf = xf + self.mlp_f(self.norm_f2(xf))

            # 2. Cross-Store within each Family
            xs = xf.reshape(*lead_shape, N_STORES, N_FAMS, D).transpose(-3, -2).contiguous().reshape(-1, N_STORES, D)
            xs_n = self.norm_s1(xs)
            xs = xs + self.attn_s(xs_n, xs_n, xs_n, need_weights=False)[0]
            xs = xs + self.mlp_s(self.norm_s2(xs))

            return xs.reshape(*lead_shape, N_FAMS, N_STORES, D).transpose(-3, -2).contiguous()

    class CrossSeriesAxialNet(nn.Module):
        def __init__(self, past_dim: int = 24, fut_dim: int = 24, d_past: int = 128, d_fut: int = 96, dropout: float = 0.15):
            super().__init__()
            # Rich hierarchical entity embeddings for 54 stores x 33 families
            self.emb_store = nn.Embedding(N_STORES, 16)
            self.emb_cluster = nn.Embedding(17, 8)
            self.emb_city = nn.Embedding(22, 8)
            self.emb_state = nn.Embedding(16, 8)
            self.emb_type = nn.Embedding(5, 4)   # sum store_emb = 44
            self.emb_fam = nn.Embedding(N_FAMS, 20)
            emb_dim = 64

            # Multi-scale Temporal Conv over 56d past
            self.t_conv1 = nn.Conv1d(past_dim, d_past, kernel_size=3, padding=1)
            self.t_conv7 = nn.Conv1d(d_past, d_past, kernel_size=3, dilation=7, padding=7)
            self.t_norm = nn.LayerNorm(d_past)

            # Collapse 8 weeks -> Same-DOW weekday latent (7, d_dow) & Global summary (d_past)
            d_dow = 48
            self.week_weights = nn.Parameter(torch.linspace(0.5, 1.5, 8).view(1, 1, 1, 8, 1, 1))
            self.dow_proj = nn.Linear(d_past, d_dow)
            self.past_summary_proj = nn.Sequential(
                nn.Linear(d_past + d_dow + emb_dim, d_past),
                nn.LayerNorm(d_past),
                nn.GELU(),
            )

            # 2 Factorized Past Axial Attention Blocks across (54 stores x 33 families)
            self.past_axial = nn.ModuleList([AxialAttentionBlock(d_past, n_heads=4, dropout=dropout) for _ in range(2)])

            # Robust Base Anchor mixing Same-DOW Median, Same-DOW Mean, and Linear RevIN Projection
            self.linear_norm_map = nn.Linear(PAST_LEN, HORIZON)
            self.anchor_gate = nn.Linear(emb_dim, 3)

            # Future Horizon Projection + 1 Horizon-Wise Cross-Series Axial Block
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
            )  # (54, 44)
            f_e = self.emb_fam(fam_idx_base)  # (33, 20)
            s_grid = s_e.unsqueeze(1).expand(N_STORES, N_FAMS, -1)
            f_grid = f_e.unsqueeze(0).expand(N_STORES, N_FAMS, -1)
            return torch.cat([s_grid, f_grid], dim=-1)  # (54, 33, 64)

        def forward(self, x_past, x_fut, p_sales_norm, revin_mean, revin_std, sdow_med8, sdow_mean8):
            B = x_past.shape[0]
            e_grid = self._build_entity_emb().unsqueeze(0).expand(B, -1, -1, -1)  # (B, 54, 33, 64)

            # 1. Past Multi-Scale Temporal Conv over 56d
            xp = x_past.reshape(B * N_SERIES, PAST_LEN, -1).transpose(1, 2)  # (N_seq, 24, 56)
            h = F.gelu(self.t_conv1(xp))
            h = h + F.gelu(self.t_conv7(h))
            h = self.t_norm(h.transpose(1, 2))  # (N_seq, 56, d_past)

            # Reshape to (B, 54, 33, 8 weeks, 7 dow, d_past) -> recency-weighted weekday latent
            h_weekly = h.view(B, N_STORES, N_FAMS, 8, 7, -1)
            w_soft = F.softmax(self.week_weights, dim=3)
            dow_lat = self.dow_proj((h_weekly * w_soft).sum(dim=3))  # (B, 54, 33, 7, d_dow)

            h_last_week = h_weekly[:, :, :, -1, :, :].mean(dim=3)     # (B, 54, 33, d_past)
            dow_pool = dow_lat.mean(dim=3)                            # (B, 54, 33, d_dow)
            state = self.past_summary_proj(torch.cat([h_last_week, dow_pool, e_grid], dim=-1))

            # 2. Factorized Past Axial Cross-Series Attention
            for blk in self.past_axial:
                state = blk(state)  # (B, 54, 33, d_past)

            # 3. Robust 3-Way Anchor (Same-DOW Median, Same-DOW Mean, RevIN Linear)
            lin_log = revin_mean + revin_std * self.linear_norm_map(p_sales_norm)
            anchor_w = F.softmax(self.anchor_gate(e_grid), dim=-1)  # (B, 54, 33, 3)
            base_anchor = (
                anchor_w[..., 0:1] * sdow_med8
                + anchor_w[..., 1:2] * sdow_mean8
                + anchor_w[..., 2:3] * lin_log
            )

            # 4. Future Horizon Representation + Horizon-Wise Axial Cross-Series Attention
            dow_h = dow_lat[:, :, :, dow_gather, :]  # (B, 54, 33, 16, d_dow)
            state_h = state.unsqueeze(3).expand(-1, -1, -1, HORIZON, -1)
            e_h = e_grid.unsqueeze(3).expand(-1, -1, -1, HORIZON, -1)

            fut_in = torch.cat([x_fut, state_h, dow_h, e_h], dim=-1)
            fut_tok = self.fut_proj(fut_in) + self.fut_pos  # (B, 54, 33, 16, d_fut)

            # Transpose to (B, 16, 54, 33, d_fut) so AxialAttentionBlock runs cross-series per horizon
            fut_by_h = self.fut_axial(fut_tok.permute(0, 3, 1, 2, 4).contiguous())
            fut_out = fut_by_h.permute(0, 2, 3, 1, 4).contiguous()  # (B, 54, 33, 16, d_fut)

            res_norm = self.out_head(torch.cat([fut_out, e_h], dim=-1)).squeeze(-1)
            return base_anchor + revin_std * res_norm

    val_c = tensors["val_cutoff"]
    val_dead_mask = (sales_t[val_c - 20 : val_c + 1].sum(dim=0) == 0.0).cpu().numpy()  # (54, 33)

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
        n_epochs = 2
        batch_dates = 2
    else:
        seeds = [42, 2026, 777, 1234, 9999]
        n_epochs = 48
        batch_dates = 6

    train_cutoffs_t = torch.tensor(train_cutoffs_arr, device=device, dtype=torch.long)
    n_cutoffs = len(train_cutoffs_t)

    seed_preds_va = []
    seed_preds_te = []
    seed_val_raw = []
    seed_val_zmask = []

    t0 = time.time()
    for seed in seeds:
        print(
            f"\n--- Training Seed {seed} ({n_epochs} epochs, {n_cutoffs} day snapshots, "
            f"batch_dates={batch_dates} = {batch_dates * N_SERIES:,} series/step) ---"
        )
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = CrossSeriesAxialNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=2e-5)

        best_val_mse = float("inf")
        best_state = None
        best_zmask = float("inf")
        patience = 12
        patience_cnt = 0

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
                    # Robust SmoothL1/Huber loss (beta=0.45): quadratic near core, outlier-resistant -> orthogonal to MSE
                    loss = F.smooth_l1_loss(pred, by, beta=0.45)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss_sum += loss.item()
                n_steps += 1

            scheduler.step()
            tr_huber = tr_loss_sum / max(1, n_steps)

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

            if epoch % 5 == 0 or epoch == 1 or smoke_test:
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch:2d}/{n_epochs} [{elapsed:.0f}s] | Train SmoothL1: {tr_huber:.5f} | "
                    f"Val Raw: {va_rmsle:.5f} | Val ZMask: {va_zmask_rmsle:.5f}",
                    flush=True,
                )

            if va_mse < best_val_mse:
                best_val_mse = va_mse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_zmask = va_zmask_rmsle
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"  [Early Stopping] Epoch {epoch} - Best Val Raw: {np.sqrt(best_val_mse):.5f}")
                    break

        seed_val_raw.append(np.sqrt(best_val_mse))
        seed_val_zmask.append(best_zmask)
        print(f"  Seed {seed} Best Val Raw: {seed_val_raw[-1]:.5f} | ZMask: {seed_val_zmask[-1]:.5f}")

        model.load_state_dict(best_state)
        model.to(device).eval()
        with torch.no_grad():
            va_out = torch.clamp(
                model(va_past, va_fut, va_psn, va_rm, va_rs, va_smed, va_smean).float(), min=0.0
            )[0].reshape(N_SERIES, HORIZON).cpu().numpy()
            te_out = torch.clamp(
                model(te_past, te_fut, te_psn, te_rm, te_rs, te_smed, te_smean).float(), min=0.0
            )[0].reshape(N_SERIES, HORIZON).cpu().numpy()
            seed_preds_va.append(va_out)
            seed_preds_te.append(te_out)

    y_va_np = va_y[0].reshape(N_SERIES, HORIZON)[:, :15].cpu().numpy()
    ens_va = np.mean(seed_preds_va, axis=0)[:, :15]
    ens_raw = np.sqrt(np.mean((ens_va - y_va_np) ** 2))
    ens_va_z = ens_va.copy()
    ens_va_z[val_dead_mask.reshape(N_SERIES)] = 0.0
    ens_zmask = np.sqrt(np.mean((ens_va_z - y_va_np) ** 2))

    print(f"\n[EVAL] 5-Seed Mean Individual Val Raw:   {np.mean(seed_val_raw):.5f} (ZMask: {np.mean(seed_val_zmask):.5f})")
    print(f"[EVAL] 5-Seed ENSEMBLE Val Raw RMSLE:   {ens_raw:.5f}")
    print(f"[EVAL] 5-Seed ENSEMBLE Val ZMask RMSLE: {ens_zmask:.5f}")

    avg_log_te = np.mean(seed_preds_te, axis=0)  # (1782, 16)
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

    out_path = "003_store-sales/submission_exp057_colab_l4_cross_series_huber.csv"
    os.makedirs("003_store-sales", exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"\n[SAVE] Saved to: {out_path} ({len(sub)} rows, Mean={sub['sales'].mean():.2f})")

    drive_dir = "/content/drive/MyDrive"
    if os.path.exists(drive_dir):
        drive_path = os.path.join(drive_dir, "submission_exp057_colab_l4_cross_series_huber.csv")
        sub.to_csv(drive_path, index=False)
        print(f"  - Synchronized to Google Drive: {drive_path}")

    with open(out_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    print(f"\n===BASE64_SUBMISSION_START===\n{b64}\n===BASE64_SUBMISSION_END===")
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp #057: Cross-Series Axial Attention + SmoothL1/Huber (Colab L4).")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("  003 Store Sales: Exp #057 - Cross-Series Axial Attention + Huber (Colab L4)")
    print("=" * 80)

    if not args.smoke_test:
        start_heartbeat(20)
    data_dir = ensure_environment_and_data()
    tensors = build_tensor_store(data_dir)
    train_and_predict(tensors, smoke_test=args.smoke_test)
    print("\n[COMPLETE] Exp #057 finished successfully!")


if __name__ == "__main__":
    main()
