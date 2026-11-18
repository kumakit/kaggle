"""
005_name-that-llm: Exp #012 - Colab L4 GPU Two-Stage Open Pool + 5-Fold DeBERTa-v3-large Multi-Layer
Competition: Name That LLM
Target Metric: Balanced Accuracy (18 Classes)
Key Design:
  1. Architecture Diversity: microsoft/deberta-v3-large (435M params, SentencePiece + Disentangled Attention)
     provides strong architectural diversity alongside Qwen3 (Causal LLM) and ModernBERT (RoPE Encoder).
  2. Multi-Layer Style+Semantic Mean Pooling across layers [6, 24] with LayerNorm before concatenation.
  3. Fidelity-filtered Open Pool (fidelity_ratio >= 0.45, kimi-k3 high, cap=6,000/cls -> ~99k rows).
  4. L4 VRAM Guard: bfloat16 + gradient_checkpointing(use_reentrant=False) + BucketBatchSampler.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import subprocess
import json
import time
import math
import base64
import random
import zipfile
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
MODEL_NAME = "microsoft/deberta-v3-large"
DATASET_SLUG = "nobuyukikumagai/name-that-llm-exp012-deberta-large"
DATASET_TITLE = "Exp012 DeBERTa-v3-large OpenPool"
assert 6 <= len(DATASET_TITLE) <= 50, f"Invalid title length: {len(DATASET_TITLE)}"
MAX_LEN = 416
CAP_PER_LABEL = 6000
LAYER_INDICES = [6, 24]

EPOCHS_S1 = 2
LR_S1_BACKBONE = 2.0e-5
LR_S1_HEAD = 6.0e-5

EPOCHS_S2 = 5
LR_S2_BACKBONE = 1.0e-5
LR_S2_HEAD = 3.0e-5


def seed_everything(seed=SEED):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_environment():
    print("[INIT] Setting up environment and packages...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio", "torchao"], check=False)
    packages = ["transformers>=4.48.0", "sentencepiece", "protobuf", "accelerate", "kaggle"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U"] + packages, check=True)

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(kaggle_json):
        for drive_cand in [
            "/content/drive/MyDrive/.kaggle/kaggle.json",
            "/content/drive/MyDrive/kaggle.json",
        ]:
            if os.path.exists(drive_cand):
                import shutil
                shutil.copy2(drive_cand, kaggle_json)
                os.chmod(kaggle_json, 0o600)
                print(f"  - Copied kaggle credentials from {drive_cand}", flush=True)
                break
    if not os.path.exists(kaggle_json):
        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")
        if not (username and key):
            try:
                from google.colab import userdata
                username = username or userdata.get("KAGGLE_USERNAME")
                key = key or userdata.get("KAGGLE_KEY")
            except Exception:
                pass
        if not (username and key):
            raise RuntimeError(
                "Missing Kaggle credentials. Configure ~/.kaggle/kaggle.json, "
                "Google Drive kaggle.json, Colab Secrets, or KAGGLE_USERNAME/KAGGLE_KEY env vars."
            )
        with open(kaggle_json, "w") as f:
            json.dump({"username": username.strip(), "key": key.strip()}, f)
        os.chmod(kaggle_json, 0o600)
        print(f"  - Configured kaggle credentials for {username}", flush=True)


def download_data():
    data_dir = "/content/data"
    pool_dir = "/content/open_pool"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(pool_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        print("[DATA] Downloading name-that-llm competition dataset...", flush=True)
        subprocess.run(["kaggle", "competitions", "download", "-c", "name-that-llm", "-p", data_dir], check=True)
        zip_path = os.path.join(data_dir, "name-that-llm.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(data_dir)
            os.remove(zip_path)

    pool_parquet = os.path.join(pool_dir, "ntl_open_pool.parquet")
    fidelity_csv = os.path.join(pool_dir, "fidelity_by_source.csv")

    for fname in ["ntl_open_pool.parquet", "fidelity_by_source.csv"]:
        fpath = os.path.join(pool_dir, fname)
        if not os.path.exists(fpath):
            print(f"[DATA] Downloading Open Pool file: {fname}...", flush=True)
            subprocess.run(
                [
                    "kaggle", "datasets", "download",
                    "crtproduction/name-that-llm-open-pool",
                    "-f", fname,
                    "-p", pool_dir,
                ],
                check=True,
            )
            for zname in os.listdir(pool_dir):
                if zname.endswith(".zip"):
                    zpath = os.path.join(pool_dir, zname)
                    with zipfile.ZipFile(zpath, "r") as z:
                        z.extractall(pool_dir)
                    os.remove(zpath)

    assert os.path.exists(train_path), f"Missing {train_path}"
    assert os.path.exists(pool_parquet), f"Missing {pool_parquet}"
    assert os.path.exists(fidelity_csv), f"Missing {fidelity_csv}"
    return data_dir, pool_parquet, fidelity_csv


def get_cosine_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps):
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def curate_open_pool(pool_path, fidelity_csv_path, classes, cap_per_label=CAP_PER_LABEL):
    print(f"[POOL] Loading {pool_path} & {fidelity_csv_path}...", flush=True)
    cols = [
        "id", "text", "model", "source_dataset", "prompt_canon",
        "match_weight", "gen_reasoning", "flag_near_train_prompt", "flag_train_prompt_regen"
    ]
    pool = pd.read_parquet(pool_path, columns=cols)
    pool = pool.dropna(subset=["text", "model"]).copy()
    pool = pool[pool["model"].isin(classes)]
    if "flag_near_train_prompt" in pool.columns:
        pool = pool[~pool["flag_near_train_prompt"].fillna(False)]
    if "flag_train_prompt_regen" in pool.columns:
        pool = pool[~pool["flag_train_prompt_regen"].fillna(False)]

    pool = pool[pool["source_dataset"] != "prvsiyan/prvsiyans-extended-name-that-llm-dataset"]
    pool = pool[~((pool["model"] == "qwen3.8-max") & (pool["gen_reasoning"] == "low"))]

    is_kimi = pool["model"] == "kimi-k3"
    pool = pool[(~is_kimi) | (pool["gen_reasoning"] == "high")]

    fid = pd.read_csv(fidelity_csv_path)[["model", "source_dataset", "fidelity_ratio"]]
    pool = pool.merge(fid, on=["model", "source_dataset"], how="left")
    keep_mask = (pool["model"] == "kimi-k3") | (pool["fidelity_ratio"].fillna(1.0) >= 0.45)
    dropped_low_fid = int((~keep_mask).sum())
    pool = pool[keep_mask].copy()
    print(f"  - Dropped {dropped_low_fid:,} low-fidelity (model, source) rows (fidelity < 0.45)", flush=True)

    pool["match_weight"] = pd.to_numeric(pool["match_weight"], errors="coerce").fillna(1.0).clip(lower=1e-4)

    sampled_parts = []
    for cls in classes:
        sub = pool[pool["model"] == cls]
        if len(sub) > cap_per_label:
            weights = sub["match_weight"].values
            weights = weights / weights.sum()
            sub = sub.sample(n=cap_per_label, replace=False, weights=weights, random_state=SEED)
        sampled_parts.append(sub)

    curated = pd.concat(sampled_parts, axis=0).reset_index(drop=True)
    curated["group_key"] = np.where(
        curated["prompt_canon"].fillna("").astype(str).str.len() > 0,
        curated["prompt_canon"].astype(str),
        curated["id"].astype(str),
    )
    vc = curated["model"].value_counts()
    print(f"  - Curated Stage 1 rows: {len(curated):,} (min={vc.min():,}, max={vc.max():,})", flush=True)
    return curated


def build_tokenized_dataset(texts, labels, tokenizer, max_len=MAX_LEN):
    from torch.utils.data import Dataset

    enc = tokenizer(
        [str(t) for t in texts],
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    lengths = [len(ids) for ids in enc["input_ids"]]

    class PreTokenizedDataset(Dataset):
        def __init__(self, input_ids, attention_mask, y=None):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.y = y

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            item = {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
            }
            if self.y is not None:
                item["labels"] = int(self.y[idx])
            return item

    return PreTokenizedDataset(enc["input_ids"], enc["attention_mask"], labels), lengths


class BucketBatchSampler:
    """Packs length-sorted sequences under a strict max-token budget to cap VRAM."""

    def __init__(self, lengths, max_tokens=6144, max_batch_size=32, seed=SEED):
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.seed = seed
        self.epoch = 0
        self._batches_len = len(self._build_batches(seed))

    def _build_batches(self, seed_val):
        rng = np.random.default_rng(seed_val)
        noisy = self.lengths.astype(np.float32) + rng.uniform(-12.0, 12.0, size=len(self.lengths))
        order = np.argsort(noisy, kind="mergesort")
        batches = []
        cur = []
        cur_max_len = 0
        for idx in order:
            l = int(self.lengths[idx])
            new_max = max(cur_max_len, l)
            if cur and ((len(cur) + 1) * new_max > self.max_tokens or len(cur) >= self.max_batch_size):
                batches.append(cur)
                cur = [int(idx)]
                cur_max_len = l
            else:
                cur.append(int(idx))
                cur_max_len = new_max
        if cur:
            batches.append(cur)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._build_batches(self.seed + self.epoch)
        self.epoch += 1
        for b in batches:
            yield b

    def __len__(self):
        return self._batches_len


def build_deberta_multilayer(model_name, config, n_classes, layer_indices, use_grad_ckpt=True):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel

    class DebertaMultiLayerClassifier(nn.Module):
        def __init__(self, backbone_model, h_dim, num_cls, layers):
            super().__init__()
            self.backbone = backbone_model
            self.layers = list(layers)
            self.h_dim = int(h_dim)
            self.head = nn.Sequential(
                nn.LayerNorm(self.h_dim * len(self.layers)),
                nn.Dropout(0.15),
                nn.Linear(self.h_dim * len(self.layers), num_cls),
            )

        def forward(self, input_ids, attention_mask):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hs = out.hidden_states
            mask_f = attention_mask.unsqueeze(-1).float()
            den = mask_f.sum(dim=1).clamp(min=1e-6)
            vecs = [
                F.layer_norm((hs[lyr].float() * mask_f).sum(dim=1) / den, (self.h_dim,))
                for lyr in self.layers
            ]
            return self.head(torch.cat(vecs, dim=-1))

    backbone = AutoModel.from_pretrained(model_name, config=config, torch_dtype=torch.float32)
    assert next(backbone.parameters()).dtype == torch.float32
    if use_grad_ckpt:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return DebertaMultiLayerClassifier(backbone, config.hidden_size, n_classes, layer_indices)


def evaluate_model(model, loader, device, use_bf16):
    import torch
    model.eval()
    probs_list = []
    targets_list = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits.float(), dim=-1)
            probs_list.append(probs.cpu().numpy())
            if "labels" in batch:
                targets_list.append(batch["labels"].cpu().numpy())
    probs_arr = np.concatenate(probs_list, axis=0)
    targets_arr = np.concatenate(targets_list, axis=0) if targets_list else None
    return probs_arr, targets_arr


def upload_private_dataset(export_dir, dataset_slug, message):
    meta = {
        "title": DATASET_TITLE,
        "id": dataset_slug,
        "licenses": [{"name": "CC-BY-SA-4.0"}],
    }
    meta_path = os.path.join(export_dir, "dataset-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[UPLOAD] Uploading {export_dir} to Kaggle Dataset {dataset_slug}...", flush=True)
    check = subprocess.run(["kaggle", "datasets", "status", dataset_slug], capture_output=True, text=True)
    if check.returncode == 0:
        cmd = ["kaggle", "datasets", "version", "-p", export_dir, "-m", message, "-r", "zip", "-t"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", export_dir, "-r", "zip", "-t"]

    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout, flush=True)
    if res.returncode != 0:
        print(f"[UPLOAD WARN] stderr: {res.stderr}", flush=True)
        if "create" in cmd:
            res2 = subprocess.run(
                ["kaggle", "datasets", "version", "-p", export_dir, "-m", message, "-r", "zip", "-t"],
                capture_output=True, text=True
            )
            print(res2.stdout, flush=True)
            if res2.returncode != 0:
                print(f"[UPLOAD ERR] stderr: {res2.stderr}", flush=True)


def train_and_evaluate():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import transformers
    from transformers import AutoTokenizer, AutoConfig, DataCollatorWithPadding
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import balanced_accuracy_score

    seed_everything(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Master weights stay fp32 (preventing Half AdamW eps=1e-8 NaN), while autocast runs fast bf16 on L4 Tensor Cores
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else 0.0

    if vram_gb >= 36.0:
        max_tokens_s1, max_bs_s1 = 8192, 48
        max_tokens_s2, max_bs_s2 = 4096, 24
    else:
        max_tokens_s1, max_bs_s1 = 6144, 32
        max_tokens_s2, max_bs_s2 = 3072, 16

    print(
        f"[DEVICE] {device} | GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} "
        f"({vram_gb:.1f}GB) | bf16={use_bf16} | layers={LAYER_INDICES} | transformers={transformers.__version__}",
        flush=True,
    )

    data_dir, pool_parquet, fidelity_csv = download_data()
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    classes = sorted(train_df["model"].unique().tolist())
    n_classes = len(classes)
    label2idx = {cls: i for i, cls in enumerate(classes)}
    idx2label = {i: cls for i, cls in enumerate(classes)}
    train_df["label"] = train_df["model"].map(label2idx)

    print(f"[MODEL] Loading tokenizer & config: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.update({"attention_probs_dropout_prob": 0.0, "hidden_dropout_prob": 0.05})
    assert max(LAYER_INDICES) <= config.num_hidden_layers

    # =========================================================
    # STAGE 1: Curated Open Pool Pre-Training (DeBERTa-v3-large)
    # =========================================================
    pool_df = curate_open_pool(pool_parquet, fidelity_csv, classes, cap_per_label=CAP_PER_LABEL)
    pool_df["label"] = pool_df["model"].map(label2idx)

    s1_splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    s1_tr_idx, s1_val_idx = next(s1_splitter.split(pool_df, pool_df["label"].values, groups=pool_df["group_key"].values))
    pool_tr = pool_df.iloc[s1_tr_idx].reset_index(drop=True)
    pool_val = pool_df.iloc[s1_val_idx].reset_index(drop=True)
    print(f"[STAGE 1] Train={len(pool_tr):,}, Pool-Val={len(pool_val):,}", flush=True)

    counts = np.bincount(pool_tr["label"].values, minlength=n_classes).astype(np.float32)
    cls_weights = (np.mean(counts) / np.maximum(counts, 1.0)) ** 0.75
    cls_weights = cls_weights / np.mean(cls_weights)
    cls_weights_tensor = torch.tensor(cls_weights, dtype=torch.float32, device=device)

    t0 = time.time()
    s1_tr_ds, s1_tr_lens = build_tokenized_dataset(pool_tr["text"].values, pool_tr["label"].values, tokenizer, MAX_LEN)
    s1_val_ds, _ = build_tokenized_dataset(pool_val["text"].values, pool_val["label"].values, tokenizer, MAX_LEN)
    print(f"  - Pre-tokenized in {time.time() - t0:.1f}s (mean len={np.mean(s1_tr_lens):.1f} toks)", flush=True)

    s1_sampler = BucketBatchSampler(s1_tr_lens, max_tokens=max_tokens_s1, max_batch_size=max_bs_s1, seed=SEED)
    s1_tr_loader = DataLoader(s1_tr_ds, batch_sampler=s1_sampler, collate_fn=collator, num_workers=2, pin_memory=True)
    s1_val_loader = DataLoader(s1_val_ds, batch_size=48, shuffle=False, collate_fn=collator, num_workers=2, pin_memory=True)

    model = build_deberta_multilayer(MODEL_NAME, config, n_classes, LAYER_INDICES, use_grad_ckpt=True).to(device)

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    opt_params = [
        {"params": [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], "lr": LR_S1_BACKBONE, "weight_decay": 0.01},
        {"params": [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], "lr": LR_S1_BACKBONE, "weight_decay": 0.0},
        {"params": list(model.head.parameters()), "lr": LR_S1_HEAD, "weight_decay": 0.01},
    ]
    optimizer = torch.optim.AdamW(opt_params)
    total_steps_s1 = len(s1_tr_loader) * EPOCHS_S1
    scheduler = get_cosine_warmup_scheduler(optimizer, int(total_steps_s1 * 0.06), total_steps_s1)
    criterion_s1 = nn.CrossEntropyLoss(weight=cls_weights_tensor, label_smoothing=0.05)

    stage1_ckpt_path = "/content/stage1_best_deberta_large.pt"
    best_s1_score = -1.0

    for epoch in range(EPOCHS_S1):
        t_ep = time.time()
        model.train()
        running_loss = 0.0
        seen_samples = 0
        n_steps = 0

        for step, batch in enumerate(s1_tr_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion_s1(logits.float(), labels)
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"Non-finite Stage 1 loss at epoch {epoch + 1} step {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            seen_samples += input_ids.size(0)
            n_steps += 1

            if (step + 1) == 50 or (step + 1) % 400 == 0:
                elapsed = time.time() - t_ep
                rate = seen_samples / max(elapsed, 1e-3)
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                print(
                    f"    [S1 E{epoch + 1}] step {step + 1} ({seen_samples:,} samples) "
                    f"| loss={running_loss / n_steps:.4f} | {rate:.1f} samples/s | max_vram={mem_gb:.2f}GB",
                    flush=True,
                )

        train_loss = running_loss / max(n_steps, 1)
        val_probs, val_targets = evaluate_model(model, s1_val_loader, device, use_bf16)
        val_bal_acc = balanced_accuracy_score(val_targets, np.argmax(val_probs, axis=1))
        print(
            f"  [S1 Epoch {epoch + 1}/{EPOCHS_S1}] ({time.time() - t_ep:.1f}s) "
            f"Loss: {train_loss:.4f} | Pool-Val BalAcc: {val_bal_acc:.5f}",
            flush=True,
        )
        if val_bal_acc > best_s1_score:
            best_s1_score = val_bal_acc
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, stage1_ckpt_path)

    del model, optimizer, scheduler, s1_tr_loader, s1_val_loader, s1_tr_ds, s1_val_ds
    torch.cuda.empty_cache()

    # =========================================================
    # STAGE 2: Zero-Shot Diagnostic + 5-Fold Fine-Tuning
    # =========================================================
    print("\n[STAGE 2] Pre-tokenizing competition train.csv & test.csv...", flush=True)
    train_ds_all, _ = build_tokenized_dataset(train_df["text"].values, train_df["label"].values, tokenizer, MAX_LEN)
    train_loader_all = DataLoader(train_ds_all, batch_size=48, shuffle=False, collate_fn=collator)

    stage1_state = torch.load(stage1_ckpt_path, map_location="cpu")

    diag_model = build_deberta_multilayer(MODEL_NAME, config, n_classes, LAYER_INDICES, use_grad_ckpt=False)
    diag_model.load_state_dict(stage1_state)
    diag_model.to(device)
    zs_probs, zs_targets = evaluate_model(diag_model, train_loader_all, device, use_bf16)
    zs_score = balanced_accuracy_score(zs_targets, np.argmax(zs_probs, axis=1))
    print(f"[STAGE 1 DIAGNOSTIC] Zero-Shot train.csv Balanced Accuracy: {zs_score:.5f}", flush=True)
    del diag_model
    torch.cuda.empty_cache()

    export_dir = "/content/export_deberta_large"
    os.makedirs(export_dir, exist_ok=True)
    tokenizer.save_pretrained(export_dir)
    config.save_pretrained(export_dir)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros((len(train_df), n_classes), dtype=np.float32)
    fold_scores = []

    y = train_df["label"].values
    groups = train_df["prompt_id"].values
    criterion_s2 = nn.CrossEntropyLoss(label_smoothing=0.02)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df, y, groups=groups)):
        print(f"\n{'='*20} Stage 2 Fold {fold + 1}/5 {'='*20}", flush=True)
        assert len(set(train_df.iloc[train_idx]["prompt_id"]).intersection(set(train_df.iloc[val_idx]["prompt_id"]))) == 0

        tr_sub = train_df.iloc[train_idx]
        va_sub = train_df.iloc[val_idx]

        tr_ds, tr_lens = build_tokenized_dataset(tr_sub["text"].values, tr_sub["label"].values, tokenizer, MAX_LEN)
        va_ds, _ = build_tokenized_dataset(va_sub["text"].values, va_sub["label"].values, tokenizer, MAX_LEN)
        tr_sampler = BucketBatchSampler(tr_lens, max_tokens=max_tokens_s2, max_batch_size=max_bs_s2, seed=SEED + fold * 100)
        tr_loader = DataLoader(tr_ds, batch_sampler=tr_sampler, collate_fn=collator, num_workers=2, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=48, shuffle=False, collate_fn=collator, num_workers=2, pin_memory=True)

        fold_model = build_deberta_multilayer(MODEL_NAME, config, n_classes, LAYER_INDICES, use_grad_ckpt=True)
        fold_model.load_state_dict(stage1_state)
        fold_model.to(device)

        opt_params_s2 = [
            {"params": [p for n, p in fold_model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], "lr": LR_S2_BACKBONE, "weight_decay": 0.02},
            {"params": [p for n, p in fold_model.backbone.named_parameters() if any(nd in n for nd in no_decay)], "lr": LR_S2_BACKBONE, "weight_decay": 0.0},
            {"params": list(fold_model.head.parameters()), "lr": LR_S2_HEAD, "weight_decay": 0.02},
        ]
        optimizer = torch.optim.AdamW(opt_params_s2)
        total_steps_s2 = len(tr_loader) * EPOCHS_S2
        scheduler = get_cosine_warmup_scheduler(optimizer, int(total_steps_s2 * 0.1), total_steps_s2)

        best_fold_score = -1.0
        best_fold_probs = None
        fold_ckpt_path = os.path.join(export_dir, f"fold_{fold}.pt")

        for epoch in range(EPOCHS_S2):
            t_ep = time.time()
            fold_model.train()
            tr_loss = 0.0
            for batch in tr_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    logits = fold_model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion_s2(logits.float(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fold_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                tr_loss += loss.item()

            tr_loss /= max(len(tr_loader), 1)
            va_probs, va_targets = evaluate_model(fold_model, va_loader, device, use_bf16)
            va_score = balanced_accuracy_score(va_targets, np.argmax(va_probs, axis=1))
            print(f"  Epoch {epoch + 1}/{EPOCHS_S2} ({time.time() - t_ep:.1f}s) - Loss: {tr_loss:.4f} - Val BalAcc: {va_score:.5f}", flush=True)

            if va_score > best_fold_score:
                best_fold_score = va_score
                best_fold_probs = va_probs
                torch.save({k: v.bfloat16().cpu() for k, v in fold_model.state_dict().items()}, fold_ckpt_path)

        oof_probs[val_idx] = best_fold_probs
        fold_scores.append(best_fold_score)
        print(f"Fold {fold + 1} Best BalAcc: {best_fold_score:.5f}", flush=True)

        del fold_model, optimizer, scheduler, tr_loader, va_loader
        torch.cuda.empty_cache()

    overall_preds = np.argmax(oof_probs, axis=1)
    overall_score = balanced_accuracy_score(y, overall_preds)
    mean_fold = float(np.mean(fold_scores))

    print("\n" + "=" * 60, flush=True)
    print("Exp #012 Two-Stage OpenPool+train DeBERTa-v3-large MultiLayer", flush=True)
    print(f"Stage 1 Zero-Shot train.csv BalAcc : {zs_score:.5f}", flush=True)
    print(f"Stage 2 5-Fold OOF Balanced Acc    : {overall_score:.5f}", flush=True)
    print(f"Fold Scores                        : {[round(float(s), 5) for s in fold_scores]}", flush=True)
    print(f"Mean Fold (+/- std)                : {mean_fold:.5f} +/- {np.std(fold_scores):.5f}", flush=True)
    print("=" * 60, flush=True)

    meta_info = {
        "model_name": MODEL_NAME,
        "hidden_size": int(config.hidden_size),
        "layer_indices": LAYER_INDICES,
        "max_len": MAX_LEN,
        "classes": classes,
        "weight_dtype": "bfloat16",
        "stage1_zeroshot_bal_acc": float(zs_score),
        "stage2_oof_bal_acc": float(overall_score),
        "fold_scores": [float(s) for s in fold_scores],
    }
    with open(os.path.join(export_dir, "classes.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    oof_df = pd.DataFrame(oof_probs, columns=classes)
    oof_df["id"] = train_df["id"]
    oof_df["true_model"] = train_df["model"]
    oof_df["pred_model"] = [idx2label[p] for p in overall_preds]
    oof_csv_path = "/content/oof_exp012_deberta_large.csv"
    oof_df.to_csv(oof_csv_path, index=False)

    with open(oof_csv_path, "rb") as f:
        b64_oof = base64.b64encode(f.read()).decode("utf-8")
    print(f"\n===BASE64_OOF_START===\n{b64_oof}\n===BASE64_OOF_END===", flush=True)

    upload_private_dataset(
        export_dir,
        DATASET_SLUG,
        f"Exp012 OpenPool+5Fold DeBERTa-v3-large (OOF {overall_score:.5f})",
    )


if __name__ == "__main__":
    ensure_environment()
    train_and_evaluate()
