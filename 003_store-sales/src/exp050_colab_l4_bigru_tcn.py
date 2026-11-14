"""
003_store-sales: Exp #050 - Direct Multi-Horizon Temporal CNN + BiGRU + Cross-Attention (Seq2Horizon)
Kaggle Competition: Store Sales - Time Series Forecasting
Designed for Google Colab L4 GPU (VRAM 24GB) Ephemeral Execution

Orthogonal Architecture to Both Exp #036 (GBDT) & Exp #043 (Flat MLP ResNet):
  - Temporal Sequence Modeling: Multi-scale Dilated 1D-Conv (k=3, d=1 & d=7 weekly) + 2-layer BiGRU
    encoding a 60-day trailing sequence per (store, family) series.
  - Exogenous Rich Covariates missing in Exp #043:
    * Daily store traffic (`transactions.csv` log1p)
    * Crude oil macro signal (`oil.csv` normalized + 14d delta)
    * Granular calendar holidays (`holidays_events.csv`: National, Regional, Local, Work Day)
  - Future Horizon Cross-Attention & BiGRU Decoder (16 target days):
    * Each future day h attends across 60 past encoder states (capturing weekly/payday periodicity)
    * Residual anchor over trailing Same-Day-of-Week 4-week average (DLinear/N-BEATS style)
    * Entity Embeddings injected via FiLM modulation + skip concatenation
  - GPU-resident on-the-fly sequence slicing with daily cutoffs (`freq="1D"`) and bfloat16 AMP
  - 5-Seed Ensemble [42, 2026, 777, 1234, 9999] with Google Drive + Base64 stdout sync
"""

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


def start_heartbeat(interval_sec: int = 15) -> None:
    """Keep WebSocket connection alive by periodic stdout flushing."""
    def beat():
        while True:
            time.sleep(interval_sec)
            print("[HEARTBEAT] L4 GPU Seq2Horizon training active...", flush=True)

    threading.Thread(target=beat, daemon=True).start()


def ensure_environment_and_data() -> str:
    """Verify Kaggle credentials and locate or download competition CSV files."""
    print("[INIT] Verifying packages and datasets...")
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    username = os.environ.get("KAGGLE_USERNAME", "nobuyukikumagai")
    key = os.environ.get("KAGGLE_KEY", "757b1a8d9ce71f3b129fbae7cc34b1b2")
    if not os.path.exists(kaggle_json_path):
        with open(kaggle_json_path, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json_path, 0o600)

    data_dir = "data"
    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        candidates = ["003_store-sales/data", "/content/data", "/content/kaggle/003_store-sales/data"]
        for c in candidates:
            if os.path.exists(os.path.join(c, "train.csv")):
                data_dir = c
                break

    if not os.path.exists(os.path.join(data_dir, "train.csv")):
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


def build_tensor_store(data_dir: str):
    """Precompute dense aligned date x series tensors for O(1) GPU window slicing."""
    print("\n[DATA] Building aligned multi-source spatiotemporal tensors...")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["date"])
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=["date"])
    stores = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    oil = pd.read_csv(os.path.join(data_dir, "oil.csv"), parse_dates=["date"])
    holidays = pd.read_csv(os.path.join(data_dir, "holidays_events.csv"), parse_dates=["date"])
    transactions = pd.read_csv(os.path.join(data_dir, "transactions.csv"), parse_dates=["date"])

    full_dates = pd.date_range("2013-01-01", "2017-08-31", freq="1D")
    n_dates = len(full_dates)
    date2idx = {d: i for i, d in enumerate(full_dates)}

    # 1. Sales (log1p) & Promotion (log1p) -> shape (T, 1782)
    piv_sales = train.pivot(index="date", columns=["store_nbr", "family"], values="sales").reindex(full_dates).fillna(0.0)
    series_cols = piv_sales.columns  # MultiIndex [(store_nbr, family)] -> 1782
    n_series = len(series_cols)

    piv_promo = (
        pd.concat([
            train.pivot(index="date", columns=["store_nbr", "family"], values="onpromotion"),
            test.pivot(index="date", columns=["store_nbr", "family"], values="onpromotion"),
        ])
        .reindex(full_dates)
        .fillna(0.0)
    )
    sales_log = np.log1p(np.clip(piv_sales.values, 0.0, None)).astype(np.float32)  # (T, 1782)
    promo_log = np.log1p(np.clip(piv_promo.values, 0.0, None)).astype(np.float32)  # (T, 1782)

    # 2. Store transactions (log1p, mapped to series) -> shape (T, 1782)
    piv_tx = transactions.pivot(index="date", columns="store_nbr", values="transactions").reindex(full_dates).fillna(0.0)
    # Normalize per store using train stats (<= 2017-07-31)
    tx_raw = np.log1p(np.clip(piv_tx.values, 0.0, None)).astype(np.float32)  # (T, 54)
    tx_mean = tx_raw[: date2idx[pd.Timestamp("2017-07-31")] + 1].mean()
    tx_std = tx_raw[: date2idx[pd.Timestamp("2017-07-31")] + 1].std() + 1e-5
    tx_norm_store = (tx_raw - tx_mean) / tx_std
    store_col_idx = np.array([s - 1 for s, _ in series_cols], dtype=np.int64)
    tx_norm_series = tx_norm_store[:, store_col_idx]  # (T, 1782)

    # 3. Oil macro series (normalized + 14d momentum) -> shape (T, 2)
    oil_s = oil.set_index("date")["dcoilwtico"].reindex(full_dates).interpolate().ffill().bfill()
    oil_vals = oil_s.values.astype(np.float32)
    oil_tr_end = date2idx[pd.Timestamp("2017-07-31")] + 1
    oil_norm = (oil_vals - oil_vals[:oil_tr_end].mean()) / (oil_vals[:oil_tr_end].std() + 1e-5)
    oil_diff14 = np.zeros_like(oil_norm)
    oil_diff14[14:] = oil_norm[14:] - oil_norm[:-14]

    # 4. Granular Holidays -> National (T,), Work Day (T,), Local/Regional (T, 1782)
    stores_map = stores.set_index("store_nbr")
    active_hol = holidays[~holidays["transferred"]].copy()
    nat_dates = set(active_hol[(active_hol["locale"] == "National") & (active_hol["type"] != "Work Day")]["date"])
    work_dates = set(active_hol[active_hol["type"] == "Work Day"]["date"])

    is_nat_hol = np.array([1.0 if d in nat_dates else 0.0 for d in full_dates], dtype=np.float32)
    is_work_day = np.array([1.0 if d in work_dates else 0.0 for d in full_dates], dtype=np.float32)

    # New Year's Day (virtually all supermarkets closed Jan 1)
    is_jan1 = np.array([1.0 if (d.month == 1 and d.day == 1) else 0.0 for d in full_dates], dtype=np.float32)

    # Local/Regional matched strictly to each (store_nbr, family)'s city/state
    reg_map = active_hol[active_hol["locale"] == "Regional"].groupby("locale_name")["date"].apply(set).to_dict()
    loc_map = active_hol[active_hol["locale"] == "Local"].groupby("locale_name")["date"].apply(set).to_dict()
    is_local_hol = np.zeros((n_dates, n_series), dtype=np.float32)
    for j, (s, _) in enumerate(series_cols):
        city = stores_map.loc[s, "city"]
        state = stores_map.loc[s, "state"]
        matched_dates = reg_map.get(state, set()) | loc_map.get(city, set())
        for d in matched_dates:
            if d in date2idx:
                is_local_hol[date2idx[d], j] = 1.0

    # 5. Deterministic Calendar / Quincena / Earthquake features -> shape (T, ...)
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

    # Static categorical IDs -> shape (1782, 6)
    cities = sorted(stores["city"].unique())
    states = sorted(stores["state"].unique())
    types = sorted(stores["type"].unique())
    families = sorted(train["family"].unique())
    city2idx = {c: i for i, c in enumerate(cities)}
    state2idx = {s: i for i, s in enumerate(states)}
    type2idx = {t: i for i, t in enumerate(types)}
    fam2idx = {f: i for i, f in enumerate(families)}

    static_cats = np.array(
        [
            [
                s - 1,
                fam2idx[f],
                stores_map.loc[s, "cluster"] - 1,
                city2idx[stores_map.loc[s, "city"]],
                state2idx[stores_map.loc[s, "state"]],
                type2idx[stores_map.loc[s, "type"]],
            ]
            for s, f in series_cols
        ],
        dtype=np.int64,
    )

    # Daily grid tensor of date covariates -> (T, 11)
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

    # Index splits (no validation target leakage: T + 16 <= 2017-07-31 -> T <= 2017-07-15)
    train_cutoffs = np.array([date2idx[d] for d in pd.date_range("2015-05-01", "2017-07-15", freq="1D")], dtype=np.int64)
    val_cutoff = date2idx[pd.Timestamp("2017-07-31")]
    test_cutoff = date2idx[pd.Timestamp("2017-08-15")]

    print(
        f"  - Tensors built: T={n_dates}, S={n_series} | "
        f"Train cutoffs={len(train_cutoffs)} ({len(train_cutoffs) * n_series:,} windows)"
    )

    return {
        "sales_log": sales_log,
        "promo_log": promo_log,
        "tx_norm": tx_norm_series,
        "is_local_hol": is_local_hol,
        "date_cov": date_cov,
        "static_cats": static_cats,
        "train_cutoffs": train_cutoffs,
        "val_cutoff": val_cutoff,
        "test_cutoff": test_cutoff,
        "series_cols": series_cols,
        "test_df": test,
    }


def train_and_predict(tensors: dict, smoke_test: bool = False) -> pd.DataFrame:
    """Train 5-Seed Seq2Horizon TCN+BiGRU+CrossAttention on GPU and infer Test."""
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

    PAST_LEN = 60
    HORIZON = 16

    # Move compact read-only tensors directly into GPU memory (~120 MB total!)
    sales_t = torch.tensor(tensors["sales_log"], device=device)        # (T, S)
    promo_t = torch.tensor(tensors["promo_log"], device=device)        # (T, S)
    tx_t = torch.tensor(tensors["tx_norm"], device=device)             # (T, S)
    loc_hol_t = torch.tensor(tensors["is_local_hol"], device=device)   # (T, S)
    date_cov_t = torch.tensor(tensors["date_cov"], device=device)      # (T, 13)
    cats_t = torch.tensor(tensors["static_cats"], device=device)       # (S, 6)

    n_series = sales_t.shape[1]

    # Precompute past sales stats across whole train range for clean global scaling
    s_mean = sales_t[: tensors["val_cutoff"]].mean()
    s_std = sales_t[: tensors["val_cutoff"]].std() + 1e-5
    p_mean = promo_t[: tensors["val_cutoff"]].mean()
    p_std = promo_t[: tensors["val_cutoff"]].std() + 1e-5

    def gather_batch(cutoffs_idx: torch.Tensor, series_idx: torch.Tensor, is_test: bool = False):
        """Slice (B, 60, C_past), (B, 16, C_fut), (B, 6), anchor (B, 16), and y (B, 16) on GPU."""
        # Past offsets: -59 .. 0 relative to cutoff T
        past_offsets = torch.arange(-PAST_LEN + 1, 1, device=device)  # (60,)
        fut_offsets = torch.arange(1, HORIZON + 1, device=device)     # (16,)

        t_past = cutoffs_idx.unsqueeze(1) + past_offsets.unsqueeze(0)  # (B, 60)
        t_fut = cutoffs_idx.unsqueeze(1) + fut_offsets.unsqueeze(0)    # (B, 16)
        s_past = series_idx.unsqueeze(1).expand(-1, PAST_LEN)          # (B, 60)
        s_fut = series_idx.unsqueeze(1).expand(-1, HORIZON)            # (B, 16)

        p_sales = sales_t[t_past, s_past]                              # (B, 60)
        p_promo = promo_t[t_past, s_past]                              # (B, 60)
        p_tx = tx_t[t_past, s_past]                                    # (B, 60)
        p_lhol = loc_hol_t[t_past, s_past]                             # (B, 60)
        p_dcov = date_cov_t[t_past]                                    # (B, 60, 13)

        # Anchor: trailing 4-week same-day-of-week average in log1p space (indices match h+1 mod 7)
        # Since t_past ends at T and t_fut starts at T+1:
        #   t_fut[h] - 7 = T + 1 + h - 7 -> offset in past_offsets (-59..0):
        #   -7 -> idx 53, -14 -> idx 46, ..., so index in 0..59 for lag k weeks before day h (h=0..15):
        #   t+1+h - 7*w -> offset = (1+h) - 7*w -> idx = PAST_LEN - 1 + (1+h) - 7*w
        # For h in 0..15:
        #   w=1..4 gives lags -7,-14,-21,-28 for days 0..6, and -14,-21,-28,-35 for days 7..13, etc.
        # Clean exact same-DOW indices relative to cutoff T (same weekday modulo 7 as fut_offsets):
        anchor_list = []
        for w in (7, 14, 21, 28):
            # shift fut_offsets back by multiples of 7 until <= 0 (always <= 0 past)
            lag_off = ((fut_offsets - 1) % 7 + 1) - w  # e.g. h=0 (T+1) -> +1-7=-6, h=7 -> +1-7=-6, etc.
            # Actually exact same weekday within last 4 occurrences prior to T:
            # For offset 1..7 -> -6..0; for offset 8..14 -> +1..+7 not in past so use 1..7 - w
            lag_idx = PAST_LEN - 1 + lag_off
            anchor_list.append(p_sales[:, lag_idx])
        same_dow_mean = torch.stack(anchor_list, dim=0).mean(dim=0)    # (B, 16)

        # Past level & centered series (exposes shape independently of scale)
        past28_mean = p_sales[:, -28:].mean(dim=1, keepdim=True)       # (B, 1)
        p_sales_centered = p_sales - past28_mean
        p_promo_mean = p_promo.mean(dim=1, keepdim=True)

        x_past = torch.cat(
            [
                ((p_sales - s_mean) / s_std).unsqueeze(-1),
                p_sales_centered.unsqueeze(-1),
                ((p_promo - p_mean) / p_std).unsqueeze(-1),
                (p_promo - p_promo_mean).unsqueeze(-1),
                p_tx.unsqueeze(-1),
                p_lhol.unsqueeze(-1),
                (p_sales > 0).float().unsqueeze(-1),
                p_dcov,  # 13 cols -> total C_past = 20
            ],
            dim=-1,
        )

        # Future known covariates (16 steps)
        f_promo = promo_t[t_fut, s_fut]                                # (B, 16)
        f_lhol = loc_hol_t[t_fut, s_fut]                               # (B, 16)
        f_yoy1 = sales_t[t_fut - 364, s_fut]                           # (B, 16) exact 52-week same weekday
        f_yoy2 = sales_t[t_fut - 371, s_fut]                           # (B, 16) 53-week same weekday
        f_dcov = date_cov_t[t_fut]                                     # (B, 16, 13)
        h_frac = (torch.arange(1, HORIZON + 1, device=device, dtype=torch.float32) / HORIZON).view(1, HORIZON, 1).expand(len(cutoffs_idx), -1, -1)

        # Exclude transactions index? date_cov has no transactions (oil + calendar + holidays only, all known!)
        x_fut = torch.cat(
            [
                ((f_promo - p_mean) / p_std).unsqueeze(-1),
                (f_promo - p_promo_mean).unsqueeze(-1),
                ((f_yoy1 - s_mean) / s_std).unsqueeze(-1),
                (f_yoy1 - past28_mean).unsqueeze(-1),
                ((f_yoy2 - s_mean) / s_std).unsqueeze(-1),
                same_dow_mean.unsqueeze(-1),
                (same_dow_mean - past28_mean).unsqueeze(-1),
                f_lhol.unsqueeze(-1),
                h_frac,
                f_dcov,  # 13 cols -> total C_fut = 22
            ],
            dim=-1,
        )

        x_cat = cats_t[series_idx]                                     # (B, 6)
        if is_test:
            return x_past, x_fut, x_cat, same_dow_mean, past28_mean
        y_fut = sales_t[t_fut, s_fut]                                  # (B, 16)
        return x_past, x_fut, x_cat, same_dow_mean, past28_mean, y_fut

    class Seq2HorizonTCNBiGRU(nn.Module):
        def __init__(self, past_dim: int = 20, fut_dim: int = 22, d_model: int = 192, dropout: float = 0.15):
            super().__init__()
            self.emb_store = nn.Embedding(54, 16)
            self.emb_fam = nn.Embedding(33, 16)
            self.emb_cluster = nn.Embedding(17, 8)
            self.emb_city = nn.Embedding(22, 8)
            self.emb_state = nn.Embedding(16, 8)
            self.emb_type = nn.Embedding(5, 4)
            emb_dim = 60

            # FiLM conditioning from entity embeddings onto hidden representations
            self.film_gamma = nn.Linear(emb_dim, d_model)
            self.film_beta = nn.Linear(emb_dim, d_model)

            # Past Encoder: projection + multi-scale dilated 1D-Conv + 2-layer BiGRU
            self.past_proj = nn.Sequential(
                nn.Linear(past_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.conv_d1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, dilation=1)
            self.conv_d7 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=7, dilation=7)
            self.conv_norm = nn.LayerNorm(d_model)
            self.enc_gru = nn.GRU(
                d_model, d_model // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout
            )
            self.enc_norm = nn.LayerNorm(d_model)

            # Future Horizon Processor + Cross-Attention over Past Sequence + Future BiGRU
            self.fut_proj = nn.Sequential(
                nn.Linear(fut_dim + emb_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
            self.attn_norm = nn.LayerNorm(d_model)
            self.dec_gru = nn.GRU(
                d_model * 2, d_model // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout
            )
            self.dec_norm = nn.LayerNorm(d_model)

            # Anchor gate + Residual Prediction Head
            self.anchor_weights = nn.Sequential(
                nn.Linear(emb_dim + d_model, 32),
                nn.GELU(),
                nn.Linear(32, 2),  # weight on [same_dow_mean, past28_mean]
            )
            self.res_head = nn.Sequential(
                nn.Linear(d_model + emb_dim, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x_past, x_fut, x_cat, same_dow_mean, past28_mean):
            B, H, _ = x_fut.shape
            e = torch.cat(
                [
                    self.emb_store(x_cat[:, 0]),
                    self.emb_fam(x_cat[:, 1]),
                    self.emb_cluster(x_cat[:, 2]),
                    self.emb_city(x_cat[:, 3]),
                    self.emb_state(x_cat[:, 4]),
                    self.emb_type(x_cat[:, 5]),
                ],
                dim=-1,
            )  # (B, 60)

            gamma = 1.0 + 0.1 * torch.tanh(self.film_gamma(e)).unsqueeze(1)
            beta = self.film_beta(e).unsqueeze(1)

            # Past TCN + BiGRU
            hp = self.past_proj(x_past) * gamma + beta                     # (B, 60, D)
            hc = hp.transpose(1, 2)                                        # (B, D, 60)
            conv_out = F.gelu(self.conv_d1(hc) + self.conv_d7(hc)).transpose(1, 2)
            hp = self.conv_norm(hp + conv_out)
            enc_mem, _ = self.enc_gru(hp)                                  # (B, 60, D)
            enc_mem = self.enc_norm(hp + enc_mem)

            # Future Projection + Cross-Attention to Past Memory
            e_rep = e.unsqueeze(1).expand(-1, H, -1)                       # (B, 16, 60)
            hf = self.fut_proj(torch.cat([x_fut, e_rep], dim=-1))          # (B, 16, D)
            attn_out, _ = self.cross_attn(hf, enc_mem, enc_mem)            # (B, 16, D)
            hf_ctx = self.attn_norm(hf + attn_out)

            # Future Horizon BiGRU (captures payday transitions & consecutive promo effects)
            dec_in = torch.cat([hf_ctx, enc_mem[:, -1:, :].expand(-1, H, -1)], dim=-1)
            dec_out, _ = self.dec_gru(dec_in)                              # (B, 16, D)
            dec_out = self.dec_norm(hf_ctx + dec_out)

            # Convex anchor + residual
            w = F.softmax(self.anchor_weights(torch.cat([dec_out, e_rep], dim=-1)), dim=-1)  # (B, 16, 2)
            base_anchor = w[..., 0] * same_dow_mean + w[..., 1] * past28_mean                # (B, 16)
            delta = self.res_head(torch.cat([dec_out, e_rep], dim=-1)).squeeze(-1)           # (B, 16)
            return base_anchor + delta

    # Precompute validation zero-mask (series dead for 21+ days before val_cutoff -> compare raw & masked val)
    val_c = tensors["val_cutoff"]
    val_dead_mask = (sales_t[val_c - 20 : val_c + 1].sum(dim=0) == 0.0).cpu().numpy()  # (1782,)
    all_series_idx = torch.arange(n_series, device=device, dtype=torch.long)

    # Validation and Test inputs
    with torch.no_grad():
        val_cutoff_vec = torch.full((n_series,), val_c, device=device, dtype=torch.long)
        va_past, va_fut, va_cat, va_sdow, va_p28, va_y = gather_batch(val_cutoff_vec, all_series_idx, is_test=False)

        test_cutoff_vec = torch.full((n_series,), tensors["test_cutoff"], device=device, dtype=torch.long)
        te_past, te_fut, te_cat, te_sdow, te_p28 = gather_batch(test_cutoff_vec, all_series_idx, is_test=True)

    train_cutoffs_arr = tensors["train_cutoffs"]
    if smoke_test:
        train_cutoffs_arr = train_cutoffs_arr[-10:]
        seeds = [42]
        n_epochs = 2
        batch_size = 1024
    else:
        seeds = [42, 2026, 777, 1234, 9999]
        n_epochs = 25
        batch_size = 2048

    # Flat grid of (cutoff_idx, series_idx) pairs -> keep on CPU/GPU cleanly (~1.44M pairs)
    grid_c = np.repeat(train_cutoffs_arr, n_series)
    grid_s = np.tile(np.arange(n_series, dtype=np.int64), len(train_cutoffs_arr))
    n_train = len(grid_c)
    grid_c_t = torch.tensor(grid_c, device=device, dtype=torch.long)
    grid_s_t = torch.tensor(grid_s, device=device, dtype=torch.long)

    seed_preds_va = []
    seed_preds_te = []
    seed_val_raw = []
    seed_val_zmask = []

    t0 = time.time()
    for seed in seeds:
        print(f"\n--- Training Seed {seed} ({n_epochs} epochs, {n_train:,} samples/epoch) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = Seq2HorizonTCNBiGRU().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=2e-5)

        best_val_loss = float("inf")
        best_state = None
        patience = 7
        patience_cnt = 0

        for epoch in range(1, n_epochs + 1):
            model.train()
            perm = torch.randperm(n_train, device=device)
            tr_loss_sum = 0.0
            n_steps = 0

            for start in range(0, n_train - batch_size + 1, batch_size):
                idx = perm[start : start + batch_size]
                bx_p, bx_f, bx_c, b_sdow, b_p28, by = gather_batch(grid_c_t[idx], grid_s_t[idx], is_test=False)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    pred = model(bx_p, bx_f, bx_c, b_sdow, b_p28)
                    loss = F.mse_loss(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss_sum += loss.item()
                n_steps += 1

            scheduler.step()
            tr_rmse = np.sqrt(tr_loss_sum / max(1, n_steps))

            model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    va_pred = torch.clamp(model(va_past, va_fut, va_cat, va_sdow, va_p28).float(), min=0.0)
                va_loss = F.mse_loss(va_pred[:, :15], va_y[:, :15]).item()
                va_rmsle = np.sqrt(va_loss)

                va_pred_z = va_pred[:, :15].clone()
                va_pred_z[torch.tensor(val_dead_mask, device=device)] = 0.0
                va_zmask_rmsle = np.sqrt(F.mse_loss(va_pred_z, va_y[:, :15]).item())

            if epoch % 5 == 0 or epoch == 1 or smoke_test:
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch:2d}/{n_epochs} [{elapsed:.0f}s] | Train RMSLE: {tr_rmse:.4f} | "
                    f"Val Raw: {va_rmsle:.5f} | Val ZMask: {va_zmask_rmsle:.5f}",
                    flush=True,
                )

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_zmask = va_zmask_rmsle
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"  [Early Stopping] Epoch {epoch} - Best Val Raw: {np.sqrt(best_val_loss):.5f}")
                    break

        seed_val_raw.append(np.sqrt(best_val_loss))
        seed_val_zmask.append(best_zmask)
        print(f"  Seed {seed} Best Val Raw: {seed_val_raw[-1]:.5f} | ZMask: {seed_val_zmask[-1]:.5f}")

        model.load_state_dict(best_state)
        model.to(device).eval()
        with torch.no_grad():
            seed_preds_va.append(torch.clamp(model(va_past, va_fut, va_cat, va_sdow, va_p28).float(), min=0.0).cpu().numpy())
            seed_preds_te.append(torch.clamp(model(te_past, te_fut, te_cat, te_sdow, te_p28).float(), min=0.0).cpu().numpy())

    # 5-Seed Ensemble Evaluation
    y_va_np = va_y[:, :15].cpu().numpy()
    ens_va = np.mean(seed_preds_va, axis=0)[:, :15]
    ens_raw = np.sqrt(np.mean((ens_va - y_va_np) ** 2))
    ens_va_z = ens_va.copy()
    ens_va_z[val_dead_mask] = 0.0
    ens_zmask = np.sqrt(np.mean((ens_va_z - y_va_np) ** 2))

    print(f"\n[EVAL] 5-Seed Mean Individual Val Raw:   {np.mean(seed_val_raw):.5f} (ZMask: {np.mean(seed_val_zmask):.5f})")
    print(f"[EVAL] 5-Seed ENSEMBLE Val Raw RMSLE:   {ens_raw:.5f}")
    print(f"[EVAL] 5-Seed ENSEMBLE Val ZMask RMSLE: {ens_zmask:.5f}")

    # Reconstruct test submission DataFrame
    avg_log_te = np.mean(seed_preds_te, axis=0)  # (1782, 16)
    test_dates = pd.date_range("2017-08-16", "2017-08-31")
    df_pred_matrix = pd.DataFrame(avg_log_te.T, index=test_dates, columns=tensors["series_cols"])
    long_preds = df_pred_matrix.stack(["store_nbr", "family"]).reset_index()
    long_preds.columns = ["date", "store_nbr", "family", "log_sales"]
    long_preds["sales"] = np.clip(np.expm1(long_preds["log_sales"]), 0.0, None)

    test_df = tensors["test_df"]
    test_sub = pd.merge(
        test_df[["id", "date", "store_nbr", "family"]], long_preds, on=["date", "store_nbr", "family"], how="left"
    )
    sub = pd.DataFrame({"id": test_sub["id"], "sales": test_sub["sales"]})

    assert len(sub) == 28512, f"Expected 28512 rows, got {len(sub)}"
    assert sub["sales"].isna().sum() == 0, "NaNs found in predictions!"

    if smoke_test:
        print("\n[SMOKE TEST PASSED] All shapes, losses, and submission merge verified cleanly!")
        return sub

    out_path = "003_store-sales/submission_exp050_colab_l4_bigru_tcn.csv"
    os.makedirs("003_store-sales", exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"\n[SAVE] Saved to: {out_path} ({len(sub)} rows, Mean={sub['sales'].mean():.2f})")

    drive_dir = "/content/drive/MyDrive"
    if os.path.exists(drive_dir):
        drive_path = os.path.join(drive_dir, "submission_exp050_colab_l4_bigru_tcn.csv")
        sub.to_csv(drive_path, index=False)
        print(f"  - Synchronized to Google Drive: {drive_path}")

    with open(out_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    print(f"\n===BASE64_SUBMISSION_START===\n{b64}\n===BASE64_SUBMISSION_END===")
    return sub


def main():
    parser = argparse.ArgumentParser(description="Exp #050: Colab L4 GPU Seq2Horizon TCN+BiGRU.")
    parser.add_argument("--smoke-test", action="store_true", help="Fast CPU/GPU sanity test on small subset")
    args = parser.parse_args()

    print("=" * 80)
    print("  003 Store Sales: Exp #050 - Seq2Horizon TCN + BiGRU + CrossAttn (Colab L4 GPU)")
    print("=" * 80)

    if not args.smoke_test:
        start_heartbeat(15)
    data_dir = ensure_environment_and_data()
    tensors = build_tensor_store(data_dir)
    train_and_predict(tensors, smoke_test=args.smoke_test)
    print("\n[COMPLETE] Exp #050 finished successfully!")


if __name__ == "__main__":
    main()
