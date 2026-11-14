"""
003_store-sales: Exp #052 - RevIN + Seasonal-Trend DLinear + Temporal Transformer (Colab L4 GPU)
Kaggle Competition: Store Sales - Time Series Forecasting
Designed for Google Colab L4 GPU (VRAM 24GB) Ephemeral Execution

Orthogonal 4th Architecture (Complementary to GBDT #036, Flat ResNet #043, TCN-BiGRU #050):
  - 90-Day Lookback Window (quarterly & 12-week cycles, vs 30d in #043 and 60d in #050)
  - Dual-Year YoY Anchors (-364d 1Y AND -728d 2Y same-weekday signals missing in prior NNs)
  - Series Seasonal-Trend Decomposition (15-day AvgPool trend + seasonal remainder) & RevIN
  - Dual Branch Prediction:
      1. DLinear-Style Direct Multi-Horizon Linear Decomposition Branch (Trend + Seasonal + Future Covs)
      2. Patch/Token Transformer Self-Attention + Future Horizon Query Decoder (pure Tensor Core parallel, no RNNs)
  - bfloat16 AMP + batch_size=4096 fast L4 Tensor Core execution across 5 seeds [42, 2026, 777, 1234, 9999]
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


def start_heartbeat(interval_sec: int = 20) -> None:
    """Keep WebSocket connection alive via periodic stdout flush."""
    def beat():
        while True:
            time.sleep(interval_sec)
            print("[HEARTBEAT] L4 GPU Transformer+DLinear active...", flush=True)

    threading.Thread(target=beat, daemon=True).start()


def ensure_environment_and_data() -> str:
    """Verify Kaggle auth and dataset CSV location."""
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
    """Precompute dense aligned spatiotemporal tensors."""
    print("\n[DATA] Building aligned 90d+2Y-YoY spatiotemporal tensors...")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["date"])
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=["date"])
    stores = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    oil = pd.read_csv(os.path.join(data_dir, "oil.csv"), parse_dates=["date"])
    holidays = pd.read_csv(os.path.join(data_dir, "holidays_events.csv"), parse_dates=["date"])
    transactions = pd.read_csv(os.path.join(data_dir, "transactions.csv"), parse_dates=["date"])

    full_dates = pd.date_range("2013-01-01", "2017-08-31", freq="1D")
    n_dates = len(full_dates)
    date2idx = {d: i for i, d in enumerate(full_dates)}

    piv_sales = train.pivot(index="date", columns=["store_nbr", "family"], values="sales").reindex(full_dates).fillna(0.0)
    series_cols = piv_sales.columns
    n_series = len(series_cols)

    piv_promo = (
        pd.concat([
            train.pivot(index="date", columns=["store_nbr", "family"], values="onpromotion"),
            test.pivot(index="date", columns=["store_nbr", "family"], values="onpromotion"),
        ])
        .reindex(full_dates)
        .fillna(0.0)
    )
    sales_log = np.log1p(np.clip(piv_sales.values, 0.0, None)).astype(np.float32)
    promo_log = np.log1p(np.clip(piv_promo.values, 0.0, None)).astype(np.float32)

    piv_tx = transactions.pivot(index="date", columns="store_nbr", values="transactions").reindex(full_dates).fillna(0.0)
    tx_raw = np.log1p(np.clip(piv_tx.values, 0.0, None)).astype(np.float32)
    tr_end = date2idx[pd.Timestamp("2017-07-31")] + 1
    tx_norm_store = (tx_raw - tx_raw[:tr_end].mean()) / (tx_raw[:tr_end].std() + 1e-5)
    store_col_idx = np.array([s - 1 for s, _ in series_cols], dtype=np.int64)
    tx_norm_series = tx_norm_store[:, store_col_idx]

    oil_s = oil.set_index("date")["dcoilwtico"].reindex(full_dates).interpolate().ffill().bfill()
    oil_vals = oil_s.values.astype(np.float32)
    oil_norm = (oil_vals - oil_vals[:tr_end].mean()) / (oil_vals[:tr_end].std() + 1e-5)
    oil_diff14 = np.zeros_like(oil_norm)
    oil_diff14[14:] = oil_norm[14:] - oil_norm[:-14]

    stores_map = stores.set_index("store_nbr")
    active_hol = holidays[~holidays["transferred"]].copy()
    nat_dates = set(active_hol[(active_hol["locale"] == "National") & (active_hol["type"] != "Work Day")]["date"])
    work_dates = set(active_hol[active_hol["type"] == "Work Day"]["date"])
    is_nat_hol = np.array([1.0 if d in nat_dates else 0.0 for d in full_dates], dtype=np.float32)
    is_work_day = np.array([1.0 if d in work_dates else 0.0 for d in full_dates], dtype=np.float32)
    is_jan1 = np.array([1.0 if (d.month == 1 and d.day == 1) else 0.0 for d in full_dates], dtype=np.float32)

    reg_map = active_hol[active_hol["locale"] == "Regional"].groupby("locale_name")["date"].apply(set).to_dict()
    loc_map = active_hol[active_hol["locale"] == "Local"].groupby("locale_name")["date"].apply(set).to_dict()
    is_local_hol = np.zeros((n_dates, n_series), dtype=np.float32)
    for j, (s, _) in enumerate(series_cols):
        matched_dates = reg_map.get(stores_map.loc[s, "state"], set()) | loc_map.get(stores_map.loc[s, "city"], set())
        for d in matched_dates:
            if d in date2idx:
                is_local_hol[date2idx[d], j] = 1.0

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
    ).astype(np.float32)

    train_cutoffs = np.array([date2idx[d] for d in pd.date_range("2015-05-01", "2017-07-15", freq="1D")], dtype=np.int64)
    val_cutoff = date2idx[pd.Timestamp("2017-07-31")]
    test_cutoff = date2idx[pd.Timestamp("2017-08-15")]

    print(
        f"  - Tensors ready: T={n_dates}, S={n_series} | "
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
    """Train 5-Seed RevIN + DLinear + Patch-Transformer on GPU and infer Test."""
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

    PAST_LEN = 90
    HORIZON = 16

    sales_t = torch.tensor(tensors["sales_log"], device=device)
    promo_t = torch.tensor(tensors["promo_log"], device=device)
    tx_t = torch.tensor(tensors["tx_norm"], device=device)
    loc_hol_t = torch.tensor(tensors["is_local_hol"], device=device)
    date_cov_t = torch.tensor(tensors["date_cov"], device=device)
    cats_t = torch.tensor(tensors["static_cats"], device=device)
    n_series = sales_t.shape[1]

    s_mean = sales_t[: tensors["val_cutoff"]].mean()
    s_std = sales_t[: tensors["val_cutoff"]].std() + 1e-5
    p_mean = promo_t[: tensors["val_cutoff"]].mean()
    p_std = promo_t[: tensors["val_cutoff"]].std() + 1e-5

    past_offsets = torch.arange(-PAST_LEN + 1, 1, device=device)
    fut_offsets = torch.arange(1, HORIZON + 1, device=device)
    h_frac_base = torch.arange(1, HORIZON + 1, device=device, dtype=torch.float32) / HORIZON
    sdow_indices = torch.stack(
        [PAST_LEN - 1 + ((fut_offsets - 1) % 7 + 1) - w for w in (7, 14, 21, 28, 35, 42, 49, 56)],
        dim=0,
    )  # (8 weeks, 16 horizons)

    def gather_batch(cutoffs_idx: torch.Tensor, series_idx: torch.Tensor, is_test: bool = False):
        B = len(cutoffs_idx)
        t_past = cutoffs_idx.unsqueeze(1) + past_offsets.unsqueeze(0)  # (B, 90)
        t_fut = cutoffs_idx.unsqueeze(1) + fut_offsets.unsqueeze(0)    # (B, 16)
        s_past = series_idx.unsqueeze(1).expand(-1, PAST_LEN)          # (B, 90)
        s_fut = series_idx.unsqueeze(1).expand(-1, HORIZON)            # (B, 16)

        p_sales = sales_t[t_past, s_past]                              # (B, 90)
        p_promo = promo_t[t_past, s_past]                              # (B, 90)
        p_tx = tx_t[t_past, s_past]                                    # (B, 90)
        p_lhol = loc_hol_t[t_past, s_past]                             # (B, 90)
        p_dcov = date_cov_t[t_past]                                    # (B, 90, 13)

        # RevIN window stats (robust min std=0.25 avoids amplifying dead-series jitter)
        revin_mean = p_sales.mean(dim=1, keepdim=True)                 # (B, 1)
        revin_std = torch.clamp(p_sales.std(dim=1, keepdim=True), min=0.25)
        p_sales_norm = (p_sales - revin_mean) / revin_std              # (B, 90)

        # 8-week same-DOW average anchor
        sdow8 = p_sales[:, sdow_indices].mean(dim=1)                   # (B, 16)

        f_promo = promo_t[t_fut, s_fut]                                # (B, 16)
        f_lhol = loc_hol_t[t_fut, s_fut]                               # (B, 16)
        f_yoy1 = sales_t[t_fut - 364, s_fut]                           # (B, 16) 1-Year YoY
        f_yoy2 = sales_t[t_fut - 728, s_fut]                           # (B, 16) 2-Year YoY
        f_dcov = date_cov_t[t_fut]                                     # (B, 16, 13)
        p_promo_mean = p_promo.mean(dim=1, keepdim=True)
        h_frac = h_frac_base.view(1, HORIZON, 1).expand(B, -1, -1)

        x_past = torch.cat(
            [
                p_sales_norm.unsqueeze(-1),
                ((p_sales - s_mean) / s_std).unsqueeze(-1),
                ((p_promo - p_mean) / p_std).unsqueeze(-1),
                (p_promo - p_promo_mean).unsqueeze(-1),
                p_tx.unsqueeze(-1),
                p_lhol.unsqueeze(-1),
                (p_sales > 0).float().unsqueeze(-1),
                p_dcov,  # 13 cols -> 20 total
            ],
            dim=-1,
        )

        x_fut = torch.cat(
            [
                ((f_promo - p_mean) / p_std).unsqueeze(-1),
                (f_promo - p_promo_mean).unsqueeze(-1),
                ((f_yoy1 - revin_mean) / revin_std).unsqueeze(-1),
                ((f_yoy2 - revin_mean) / revin_std).unsqueeze(-1),
                ((f_yoy1 - s_mean) / s_std).unsqueeze(-1),
                ((sdow8 - revin_mean) / revin_std).unsqueeze(-1),
                f_lhol.unsqueeze(-1),
                h_frac,
                f_dcov,  # 13 cols -> 21 total
            ],
            dim=-1,
        )

        x_cat = cats_t[series_idx]
        if is_test:
            return x_past, x_fut, x_cat, p_sales_norm, revin_mean, revin_std, sdow8
        y_fut = sales_t[t_fut, s_fut]
        return x_past, x_fut, x_cat, p_sales_norm, revin_mean, revin_std, sdow8, y_fut

    class SeriesDecompDLinearTransformer(nn.Module):
        def __init__(self, past_dim: int = 20, fut_dim: int = 21, d_model: int = 192, n_heads: int = 6, dropout: float = 0.15):
            super().__init__()
            self.emb_store = nn.Embedding(54, 16)
            self.emb_fam = nn.Embedding(33, 16)
            self.emb_cluster = nn.Embedding(17, 8)
            self.emb_city = nn.Embedding(22, 8)
            self.emb_state = nn.Embedding(16, 8)
            self.emb_type = nn.Embedding(5, 4)
            emb_dim = 60

            # Branch 1: DLinear Moving-Average Trend + Seasonal Direct Multi-Horizon Linear Maps (in RevIN space)
            self.trend_linear = nn.Linear(90, 16)
            self.seasonal_linear = nn.Linear(90, 16)
            self.entity_bias = nn.Linear(emb_dim, 16)

            # Branch 2: Stride-6 Patching of 90d Past -> 15 Weekly-ish Tokens + Transformer Self-Attention
            # Input past (B, 90, C_past) -> Conv1d kernel=6, stride=6 gives (B, 15, d_model) tokens!
            self.patch_embed = nn.Conv1d(past_dim, d_model, kernel_size=6, stride=6)
            self.past_pos = nn.Parameter(torch.randn(1, 15, d_model) * 0.02)
            self.fut_proj = nn.Sequential(
                nn.Linear(fut_dim + emb_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.fut_pos = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 3,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

            # Horizon residual output head (modulated by RevIN std back to log1p sales)
            self.out_head = nn.Sequential(
                nn.Linear(d_model + emb_dim, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            self.anchor_gate = nn.Linear(emb_dim, 2)  # blend between revin-dlinear vs 8-week same-DOW anchor

        def forward(self, x_past, x_fut, x_cat, p_sales_norm, revin_mean, revin_std, sdow8):
            B = x_past.shape[0]
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

            # 1. Series Moving-Average Decomposition (15-day kernel, replicate pad)
            padded = F.pad(p_sales_norm.unsqueeze(1), (7, 7), mode="replicate")
            trend = F.avg_pool1d(padded, kernel_size=15, stride=1).squeeze(1)  # (B, 90)
            seasonal = p_sales_norm - trend
            dlinear_norm = self.trend_linear(trend) + self.seasonal_linear(seasonal) + 0.1 * self.entity_bias(e)
            dlinear_log = revin_mean + revin_std * dlinear_norm  # (B, 16)

            # Convex base anchor between DLinear and 8-week Same-DOW Anchor
            w = F.softmax(self.anchor_gate(e), dim=-1)  # (B, 2)
            base_anchor = w[:, 0:1] * dlinear_log + w[:, 1:2] * sdow8

            # 2. Patch-Transformer Encoder over [15 Past Patches + 16 Future Horizon Tokens]
            past_tok = self.patch_embed(x_past.transpose(1, 2)).transpose(1, 2) + self.past_pos  # (B, 15, D)
            e_rep = e.unsqueeze(1).expand(-1, 16, -1)
            fut_tok = self.fut_proj(torch.cat([x_fut, e_rep], dim=-1)) + self.fut_pos            # (B, 16, D)

            all_tok = torch.cat([past_tok, fut_tok], dim=1)                                      # (B, 31, D)
            enc_tok = self.transformer(all_tok)
            fut_enc = enc_tok[:, 15:, :]                                                         # (B, 16, D)

            res_norm = self.out_head(torch.cat([fut_enc, e_rep], dim=-1)).squeeze(-1)            # (B, 16)
            return base_anchor + revin_std * res_norm

    val_c = tensors["val_cutoff"]
    val_dead_mask = (sales_t[val_c - 20 : val_c + 1].sum(dim=0) == 0.0).cpu().numpy()
    all_series_idx = torch.arange(n_series, device=device, dtype=torch.long)

    with torch.no_grad():
        va_inputs = gather_batch(torch.full((n_series,), val_c, device=device, dtype=torch.long), all_series_idx, is_test=False)
        va_past, va_fut, va_cat, va_psn, va_rm, va_rs, va_sdow, va_y = va_inputs

        te_inputs = gather_batch(torch.full((n_series,), tensors["test_cutoff"], device=device, dtype=torch.long), all_series_idx, is_test=True)
        te_past, te_fut, te_cat, te_psn, te_rm, te_rs, te_sdow = te_inputs

    train_cutoffs_arr = tensors["train_cutoffs"]
    if smoke_test:
        train_cutoffs_arr = train_cutoffs_arr[-10:]
        seeds = [42]
        n_epochs = 2
        batch_size = 1024
    else:
        seeds = [42, 2026, 777, 1234, 9999]
        n_epochs = 28
        batch_size = 4096

    grid_c_t = torch.tensor(np.repeat(train_cutoffs_arr, n_series), device=device, dtype=torch.long)
    grid_s_t = torch.tensor(np.tile(np.arange(n_series, dtype=np.int64), len(train_cutoffs_arr)), device=device, dtype=torch.long)
    n_train = len(grid_c_t)

    seed_preds_va = []
    seed_preds_te = []
    seed_val_raw = []
    seed_val_zmask = []

    t0 = time.time()
    for seed in seeds:
        print(f"\n--- Training Seed {seed} ({n_epochs} epochs, {n_train:,} samples/epoch) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = SeriesDecompDLinearTransformer().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=2e-5)

        best_val_loss = float("inf")
        best_state = None
        best_zmask = float("inf")
        patience = 8
        patience_cnt = 0

        for epoch in range(1, n_epochs + 1):
            model.train()
            perm = torch.randperm(n_train, device=device)
            tr_loss_sum = 0.0
            n_steps = 0

            for start in range(0, n_train - batch_size + 1, batch_size):
                idx = perm[start : start + batch_size]
                bx_p, bx_f, bx_c, bpsn, brm, brs, bsdow, by = gather_batch(grid_c_t[idx], grid_s_t[idx], is_test=False)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    pred = model(bx_p, bx_f, bx_c, bpsn, brm, brs, bsdow)
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
                    va_pred = torch.clamp(model(va_past, va_fut, va_cat, va_psn, va_rm, va_rs, va_sdow).float(), min=0.0)
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
            seed_preds_va.append(torch.clamp(model(va_past, va_fut, va_cat, va_psn, va_rm, va_rs, va_sdow).float(), min=0.0).cpu().numpy())
            seed_preds_te.append(torch.clamp(model(te_past, te_fut, te_cat, te_psn, te_rm, te_rs, te_sdow).float(), min=0.0).cpu().numpy())

    y_va_np = va_y[:, :15].cpu().numpy()
    ens_va = np.mean(seed_preds_va, axis=0)[:, :15]
    ens_raw = np.sqrt(np.mean((ens_va - y_va_np) ** 2))
    ens_va_z = ens_va.copy()
    ens_va_z[val_dead_mask] = 0.0
    ens_zmask = np.sqrt(np.mean((ens_va_z - y_va_np) ** 2))

    print(f"\n[EVAL] 5-Seed Mean Individual Val Raw:   {np.mean(seed_val_raw):.5f} (ZMask: {np.mean(seed_val_zmask):.5f})")
    print(f"[EVAL] 5-Seed ENSEMBLE Val Raw RMSLE:   {ens_raw:.5f}")
    print(f"[EVAL] 5-Seed ENSEMBLE Val ZMask RMSLE: {ens_zmask:.5f}")

    avg_log_te = np.mean(seed_preds_te, axis=0)
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
        print("\n[SMOKE TEST PASSED]")
        return sub

    out_path = "003_store-sales/submission_exp052_colab_l4_patch_dlinear.csv"
    os.makedirs("003_store-sales", exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"\n[SAVE] Saved to: {out_path} ({len(sub)} rows, Mean={sub['sales'].mean():.2f})")

    drive_dir = "/content/drive/MyDrive"
    if os.path.exists(drive_dir):
        drive_path = os.path.join(drive_dir, "submission_exp052_colab_l4_patch_dlinear.csv")
        sub.to_csv(drive_path, index=False)
        print(f"  - Synchronized to Google Drive: {drive_path}")

    with open(out_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    print(f"\n===BASE64_SUBMISSION_START===\n{b64}\n===BASE64_SUBMISSION_END===")
    return sub


def main():
    parser = argparse.ArgumentParser(description="Exp #052: Colab L4 GPU RevIN + DLinear + Patch-Transformer.")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("  003 Store Sales: Exp #052 - RevIN + DLinear + Patch-Transformer (Colab L4 GPU)")
    print("=" * 80)

    if not args.smoke_test:
        start_heartbeat(20)
    data_dir = ensure_environment_and_data()
    tensors = build_tensor_store(data_dir)
    train_and_predict(tensors, smoke_test=args.smoke_test)
    print("\n[COMPLETE] Exp #052 finished successfully!")


if __name__ == "__main__":
    main()
