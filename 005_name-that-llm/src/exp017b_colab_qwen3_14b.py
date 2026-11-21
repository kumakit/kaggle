"""
005_name-that-llm: Exp #017b - Colab A100/L4 Qwen3-14B-Base (Truncated 27L) LoRA/QLoRA + 8-Layer Readout
Upgrades over Exp #013 (Qwen3-8B-Base 0.59570) & exp015_colab_a100_qwen3_14b.py:
  1. Model scale-up: Qwen/Qwen3-14B-Base (hidden=5120) truncated to KEEP_LAYERS=27 (~10B params).
  2. 8-Layer Adaptive Readout: LAYER_INDICES = [3, 6, 9, 12, 16, 20, 24, 27].
  3. Hybrid precision: full bfloat16 LoRA on A100 (40GB+) OR auto 4-bit NF4 QLoRA on L4 (24GB).
  4. Full resumability (--resume):逐次保存 (Stage 1 + each Fold 0..4) to Kaggle Private Dataset.
  5. Strict credential hygiene (no hardcoded keys) & --smoke-test support.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc
import sys
import subprocess
import argparse
import json
import time
import math
import shutil
import base64
import random
import threading
import zipfile
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
MODEL_NAME = "Qwen/Qwen3-14B-Base"
KEEP_LAYERS = 27
LAYER_INDICES = [3, 6, 9, 12, 16, 20, 24, 27]
MAX_LEN = 384

DATASET_SLUG = "nobuyukikumagai/name-that-llm-exp017b-qwen3-14b-lora"
DATASET_TITLE = "Exp017b Qwen3-14B-27L LoRA"
assert 6 <= len(DATASET_TITLE) <= 50, f"Invalid Kaggle dataset title length: {len(DATASET_TITLE)}"

LR_S1_LORA = 8.0e-5
LR_S1_HEAD = 2.0e-4

EPOCHS_S2 = 5
LR_S2_LORA = 2.5e-5
LR_S2_HEAD = 8.0e-5


def seed_everything(seed=SEED):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_environment():
    print("[INIT] Setting up environment and packages...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio", "torchao"], check=False)
    packages = ["transformers>=4.51.0", "peft>=0.14.0", "accelerate", "bitsandbytes>=0.45.0", "kaggle"]
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


def _extract_all_zips(folder):
    for zname in os.listdir(folder):
        if zname.endswith(".zip"):
            zpath = os.path.join(folder, zname)
            with zipfile.ZipFile(zpath, "r") as z:
                z.extractall(folder)
            os.remove(zpath)


def download_data():
    data_dir = "/content/data"
    pool_dir = "/content/open_pool"
    extra_dir = "/content/extra_pools"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(pool_dir, exist_ok=True)
    os.makedirs(extra_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        print("[DATA] Downloading name-that-llm competition dataset...", flush=True)
        subprocess.run(["kaggle", "competitions", "download", "-c", "name-that-llm", "-p", data_dir], check=True)
        _extract_all_zips(data_dir)

    for fname in ["ntl_open_pool.parquet", "fidelity_by_source.csv"]:
        fpath = os.path.join(pool_dir, fname)
        if not os.path.exists(fpath):
            print(f"[DATA] Downloading Open Pool file: {fname}...", flush=True)
            subprocess.run(
                ["kaggle", "datasets", "download", "crtproduction/name-that-llm-open-pool", "-f", fname, "-p", pool_dir],
                check=True,
            )
            _extract_all_zips(pool_dir)

    extra_datasets = [
        ("prvsiyan/ntl-public-in-style-pool-v1", "pool_v1.parquet"),
        ("seyitkaangunes/ntl-extra-api-passages", "ntl_extra_passages_claude.csv"),
        ("zbigniewkirchner/ntl-extra-gpt-passages", "ntl_gpt_passages.csv"),
    ]
    for slug, check_file in extra_datasets:
        if not os.path.exists(os.path.join(extra_dir, check_file)):
            print(f"[DATA] Downloading extra pool dataset: {slug}...", flush=True)
            subprocess.run(["kaggle", "datasets", "download", slug, "-p", extra_dir, "--unzip"], check=True)

    return data_dir, os.path.join(pool_dir, "ntl_open_pool.parquet"), os.path.join(pool_dir, "fidelity_by_source.csv"), extra_dir


def get_cosine_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps):
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def curate_all_pools(pool_path, fidelity_csv_path, extra_dir, train_df, classes, cap_per_label):
    print("[POOL] Loading Open Pool + In-Style Pool + Extra Passages...", flush=True)
    train_texts = set(train_df["text"].fillna("").astype(str).str.strip().tolist())

    cols = [
        "id", "text", "model", "source_dataset", "prompt_canon",
        "match_weight", "style_score", "gen_reasoning",
        "flag_near_train_prompt", "flag_train_prompt_regen", "flag_kb_proxy_after_ruling",
    ]
    pool = pd.read_parquet(pool_path, columns=cols).dropna(subset=["text", "model"]).copy()
    pool = pool[pool["model"].isin(classes)]
    for flg in ["flag_near_train_prompt", "flag_train_prompt_regen", "flag_kb_proxy_after_ruling"]:
        if flg in pool.columns:
            pool = pool[~pool[flg].fillna(False)]
    pool = pool[pool["source_dataset"] != "prvsiyan/prvsiyans-extended-name-that-llm-dataset"]
    pool = pool[~((pool["model"] == "qwen3.8-max") & (pool["gen_reasoning"] == "low"))]
    pool = pool[(pool["model"] != "kimi-k3") | (pool["gen_reasoning"] == "high")]

    fid = pd.read_csv(fidelity_csv_path)[["model", "source_dataset", "fidelity_ratio"]]
    pool = pool.merge(fid, on=["model", "source_dataset"], how="left")
    pool = pool[(pool["model"] == "kimi-k3") | (pool["fidelity_ratio"].fillna(1.0) >= 0.45)].copy()

    mw = pd.to_numeric(pool["match_weight"], errors="coerce").fillna(1.0).clip(lower=0.05)
    ss = pd.to_numeric(pool["style_score"], errors="coerce").fillna(0.2).clip(lower=0.02)
    pool["sample_w"] = mw * (0.15 + ss)
    pool["group_key"] = np.where(
        pool["prompt_canon"].fillna("").astype(str).str.len() > 0,
        "op_" + pool["prompt_canon"].astype(str),
        "op_" + pool["id"].astype(str),
    )
    pool["tier"] = 1
    op_clean = pool[["text", "model", "group_key", "sample_w", "tier"]].copy()

    extra_parts = []
    pv_candidates = [os.path.join(extra_dir, "pool_v1.parquet"), os.path.join(os.path.dirname(pool_path), "../prvsiyan/pool_v1.parquet")]
    pv_path = next((p for p in pv_candidates if os.path.exists(p)), None)
    if pv_path:
        pv = pd.read_parquet(pv_path, columns=["prompt_id", "model", "text"]).dropna(subset=["text", "model"])
        pv = pv[pv["model"].isin(classes)].copy()
        pv["group_key"] = "pv_" + pv["prompt_id"].astype(str)
        pv["sample_w"] = 1.0
        pv["tier"] = 0
        extra_parts.append(pv[["text", "model", "group_key", "sample_w", "tier"]])

    cl_path = os.path.join(extra_dir, "ntl_extra_passages_claude.csv")
    if os.path.exists(cl_path):
        cl = pd.read_csv(cl_path)[["prompt_id", "model", "text"]].dropna(subset=["text", "model"])
        cl = cl[cl["model"].isin(classes)].copy()
        cl["group_key"] = "cl_" + cl["prompt_id"].astype(str)
        cl["sample_w"] = 1.0
        cl["tier"] = 0
        extra_parts.append(cl[["text", "model", "group_key", "sample_w", "tier"]])

    gp_path = os.path.join(extra_dir, "ntl_gpt_passages.csv")
    if os.path.exists(gp_path):
        gp = pd.read_csv(gp_path)[["prompt_id", "model", "text"]].dropna(subset=["text", "model"])
        gp = gp[gp["model"].isin(classes)].copy()
        gp["group_key"] = "gp_" + gp["prompt_id"].astype(str)
        gp["sample_w"] = 1.0
        gp["tier"] = 0
        extra_parts.append(gp[["text", "model", "group_key", "sample_w", "tier"]])

    instyle_df = pd.concat(extra_parts, ignore_index=True) if extra_parts else op_clean.iloc[:0].copy()
    combined = pd.concat([instyle_df, op_clean], ignore_index=True)
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 20]
    combined = combined[~combined["text"].isin(train_texts)]
    combined = combined.drop_duplicates(subset=["model", "text"]).reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    sampled_parts = []
    for cls in classes:
        sub_t0 = combined[(combined["model"] == cls) & (combined["tier"] == 0)]
        sub_t1 = combined[(combined["model"] == cls) & (combined["tier"] == 1)]
        t0_cap = min(cap_per_label, 2500)
        if len(sub_t0) > t0_cap:
            sub_t0 = sub_t0.sample(n=t0_cap, replace=False, random_state=SEED)
        rem = max(0, cap_per_label - len(sub_t0))
        if rem == 0:
            sub_t1 = sub_t1.iloc[:0]
        elif len(sub_t1) > rem:
            w = np.clip(sub_t1["sample_w"].to_numpy(dtype=np.float64), 1e-4, None)
            u = rng.uniform(1e-9, 1.0, size=len(w))
            keys = np.log(w) - np.log(-np.log(u))
            top_idx = np.argpartition(-keys, rem - 1)[:rem]
            sub_t1 = sub_t1.iloc[top_idx]
        sampled_parts.append(pd.concat([sub_t0, sub_t1], ignore_index=True))

    curated = pd.concat(sampled_parts, axis=0).reset_index(drop=True)
    vc = curated["model"].value_counts()
    t0_cnt = int((curated["tier"] == 0).sum())
    print(f"  - Curated Stage 1 rows: {len(curated):,} (tier0_instyle={t0_cnt:,}, min_cls={vc.min():,}, max_cls={vc.max():,})", flush=True)
    return curated


def build_tokenized_dataset(texts, labels, tokenizer, max_len=MAX_LEN):
    from torch.utils.data import Dataset

    enc = tokenizer([str(t) for t in texts], truncation=True, max_length=max_len, padding=False)
    lengths = [len(ids) for ids in enc["input_ids"]]

    class PreTokenizedDataset(Dataset):
        def __init__(self, input_ids, attention_mask, y=None):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.y = y

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            item = {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx]}
            if self.y is not None:
                item["labels"] = int(self.y[idx])
            return item

    return PreTokenizedDataset(enc["input_ids"], enc["attention_mask"], labels), lengths


class BucketBatchSampler:
    def __init__(self, lengths, max_tokens=3072, max_batch_size=24, seed=SEED):
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
        batches, cur, cur_max = [], [], 0
        for idx in order:
            l = int(self.lengths[idx])
            new_max = max(cur_max, l)
            if cur and ((len(cur) + 1) * new_max > self.max_tokens or len(cur) >= self.max_batch_size):
                batches.append(cur)
                cur, cur_max = [int(idx)], l
            else:
                cur.append(int(idx))
                cur_max = new_max
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


def build_multilayer_head(hidden_size, n_classes, num_layers):
    import torch
    import torch.nn as nn

    class MultiLayerAuthorshipHead(nn.Module):
        def __init__(self, h_dim, num_cls, n_lyr):
            super().__init__()
            self.h_dim = h_dim
            self.n_lyr = n_lyr
            self.layer_logits_last = nn.Parameter(torch.zeros(n_lyr))
            self.layer_logits_mean = nn.Parameter(torch.zeros(n_lyr))
            self.classifier = nn.Sequential(
                nn.LayerNorm(h_dim * 2),
                nn.Dropout(0.15),
                nn.Linear(h_dim * 2, num_cls),
            )

        def forward(self, last_stack, mean_stack):
            w_last = torch.softmax(self.layer_logits_last, dim=0).view(1, self.n_lyr, 1)
            w_mean = torch.softmax(self.layer_logits_mean, dim=0).view(1, self.n_lyr, 1)
            last_vec = (last_stack * w_last).sum(dim=1)
            mean_vec = (mean_stack * w_mean).sum(dim=1)
            return self.classifier(torch.cat([last_vec, mean_vec], dim=-1))

    return MultiLayerAuthorshipHead(hidden_size, n_classes, num_layers)


def forward_logits(backbone, head, input_ids, attention_mask, hidden_size, layer_indices=LAYER_INDICES):
    import torch
    import torch.nn.functional as F

    out = backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    bsz = input_ids.size(0)
    bsz_idx = torch.arange(bsz, device=input_ids.device)
    last_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0)
    mask_den = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
    mask_dtype = None

    last_list, mean_list = [], []
    for lyr in layer_indices:
        h = hs[lyr]
        if mask_dtype is None:
            mask_dtype = attention_mask.unsqueeze(-1).to(dtype=h.dtype if h.dtype == torch.bfloat16 else torch.float32)
        last_list.append(F.layer_norm(h[bsz_idx, last_idx].float(), (hidden_size,)))
        if h.dtype == torch.bfloat16:
            pooled = (h * mask_dtype).sum(dim=1).float() / mask_den
        else:
            pooled = (h.float() * mask_dtype).sum(dim=1) / mask_den
        mean_list.append(F.layer_norm(pooled, (hidden_size,)))

    return head(torch.stack(last_list, dim=1), torch.stack(mean_list, dim=1))


def evaluate_model(backbone, head, loader, device, use_bf16, hidden_size, max_batches=None):
    import torch
    backbone.eval()
    head.eval()
    probs_list, targets_list = [], []
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = forward_logits(backbone, head, input_ids, attention_mask, hidden_size)
            probs_list.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
            if "labels" in batch:
                targets_list.append(batch["labels"].cpu().numpy())
    return np.concatenate(probs_list, axis=0), (np.concatenate(targets_list, axis=0) if targets_list else None)


def upload_private_dataset(export_dir, dataset_slug, message):
    safe_msg = message[:48]
    assert 6 <= len(safe_msg) <= 50
    meta = {
        "title": DATASET_TITLE,
        "id": dataset_slug,
        "licenses": [{"name": "CC-BY-SA-4.0"}],
    }
    with open(os.path.join(export_dir, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[UPLOAD] Uploading {export_dir} to Kaggle Dataset {dataset_slug}...", flush=True)
    check = subprocess.run(["kaggle", "datasets", "status", dataset_slug], capture_output=True, text=True)
    if check.returncode == 0:
        cmd = ["kaggle", "datasets", "version", "-p", export_dir, "-m", safe_msg, "-r", "zip", "-t"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", export_dir, "-r", "zip", "-t"]

    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout, flush=True)
    if res.returncode != 0:
        print(f"[UPLOAD WARN] stderr: {res.stderr}", flush=True)
        if "create" in cmd:
            res2 = subprocess.run(
                ["kaggle", "datasets", "version", "-p", export_dir, "-m", safe_msg, "-r", "zip", "-t"],
                capture_output=True, text=True,
            )
            print(res2.stdout, flush=True)
            if res2.returncode != 0:
                print(f"[UPLOAD ERR] stderr: {res2.stderr}", flush=True)


def _start_heartbeat(interval_s=180):
    stop_event = threading.Event()
    t0 = time.time()

    def _beat():
        while not stop_event.wait(interval_s):
            print(f"[HEARTBEAT] elapsed={time.time() - t0:.0f}s", flush=True)

    th = threading.Thread(target=_beat, daemon=True)
    th.start()
    return stop_event


def train_and_evaluate(smoke_test=False, cap_per_label_arg=2400, epochs_s1_arg=1, force_bf16=False, resume=False):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import transformers
    import peft
    from transformers import AutoTokenizer, AutoModel, AutoConfig, DataCollatorWithPadding, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import balanced_accuracy_score

    seed_everything(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    hb_stop = _start_heartbeat(180)
    try:
        _run_train_and_evaluate(
            smoke_test,
            cap_per_label_arg,
            epochs_s1_arg,
            force_bf16,
            resume,
            torch,
            nn,
            DataLoader,
            transformers,
            peft,
            AutoTokenizer,
            AutoModel,
            AutoConfig,
            DataCollatorWithPadding,
            BitsAndBytesConfig,
            LoraConfig,
            TaskType,
            get_peft_model,
            get_peft_model_state_dict,
            set_peft_model_state_dict,
            prepare_model_for_kbit_training,
            StratifiedGroupKFold,
            balanced_accuracy_score,
        )
    finally:
        hb_stop.set()


def _run_train_and_evaluate(
    smoke_test,
    cap_per_label_arg,
    epochs_s1_arg,
    force_bf16,
    resume,
    torch,
    nn,
    DataLoader,
    transformers,
    peft,
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
    StratifiedGroupKFold,
    balanced_accuracy_score,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else 0.0
    use_4bit = (not force_bf16) and torch.cuda.is_available() and (vram_gb < 32.0)

    cap_per_label = 200 if smoke_test else cap_per_label_arg
    if use_4bit:
        max_tokens_s1, max_bs_s1, grad_accum_s1 = 2560, 16, 3
        max_tokens_s2, max_bs_s2, grad_accum_s2 = 2048, 12, 3
    else:
        max_tokens_s1, max_bs_s1, grad_accum_s1 = 3072, 24, 2
        max_tokens_s2, max_bs_s2, grad_accum_s2 = 2560, 16, 2

    print(
        f"[DEVICE] {device} | GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} "
        f"({vram_gb:.1f}GB) | bf16={use_bf16} | qlora_4bit={use_4bit} | smoke={smoke_test} | resume={resume} "
        f"| keep_layers={KEEP_LAYERS} | layers={LAYER_INDICES} | cap={cap_per_label} | ep_s1={epochs_s1_arg} "
        f"| transformers={transformers.__version__} | peft={peft.__version__}",
        flush=True,
    )

    data_dir, pool_parquet, fidelity_csv, extra_dir = download_data()
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    classes = sorted(train_df["model"].unique().tolist())
    n_classes = len(classes)
    label2idx = {cls: i for i, cls in enumerate(classes)}
    idx2label = {i: cls for i, cls in enumerate(classes)}
    train_df["label"] = train_df["model"].map(label2idx)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.pad_token_id = tokenizer.pad_token_id
    config.use_cache = False
    print(f"[MODEL] Truncating {MODEL_NAME}: {config.num_hidden_layers} -> {KEEP_LAYERS} layers (hidden={config.hidden_size})", flush=True)
    config.num_hidden_layers = KEEP_LAYERS
    if getattr(config, "layer_types", None) is not None:
        config.layer_types = config.layer_types[:KEEP_LAYERS]
    hidden_size = config.hidden_size
    assert max(LAYER_INDICES) <= config.num_hidden_layers

    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
        base_model = AutoModel.from_pretrained(
            MODEL_NAME,
            config=config,
            quantization_config=bnb_cfg,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        base_model = AutoModel.from_pretrained(
            MODEL_NAME,
            config=config,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
            attn_implementation="sdpa",
        )

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16,
        lora_alpha=32,
        lora_dropout=0.08,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    backbone = get_peft_model(base_model, lora_cfg)
    if not use_4bit:
        backbone.enable_input_require_grads()
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for p in backbone.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        backbone.to(device)
    backbone.print_trainable_parameters()

    head = build_multilayer_head(hidden_size, n_classes, len(LAYER_INDICES)).to(device)
    trainable_params = [p for p in backbone.parameters() if p.requires_grad] + list(head.parameters())
    export_dir = "/content/export_exp017b_qwen3_14b_lora"
    os.makedirs(export_dir, exist_ok=True)
    tokenizer.save_pretrained(export_dir)
    s1_ckpt_dir = os.path.join(export_dir, "stage1_best")

    if resume and not smoke_test:
        print(f"[RESUME] Downloading checkpoints from {DATASET_SLUG}...", flush=True)
        dl_dir = "/content/resume_exp017b"
        os.makedirs(dl_dir, exist_ok=True)
        subprocess.run(["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", dl_dir, "--unzip"], check=True)
        s1_loaded = os.path.join(dl_dir, "stage1_best")
        best_s1_lora = torch.load(os.path.join(s1_loaded, "lora_s1.pt"), map_location="cpu")
        best_s1_head = torch.load(os.path.join(s1_loaded, "head_s1.pt"), map_location="cpu")
        with open(os.path.join(s1_loaded, "stage1_meta.json")) as f:
            zs_score = float(json.load(f)["zs_score"])
        os.makedirs(s1_ckpt_dir, exist_ok=True)
        shutil.copy2(os.path.join(s1_loaded, "lora_s1.pt"), os.path.join(s1_ckpt_dir, "lora_s1.pt"))
        shutil.copy2(os.path.join(s1_loaded, "head_s1.pt"), os.path.join(s1_ckpt_dir, "head_s1.pt"))
        shutil.copy2(os.path.join(s1_loaded, "stage1_meta.json"), os.path.join(s1_ckpt_dir, "stage1_meta.json"))
        for fk in range(5):
            src_f = os.path.join(dl_dir, f"fold_{fk}")
            if os.path.isdir(src_f):
                shutil.copytree(src_f, os.path.join(export_dir, f"fold_{fk}"), dirs_exist_ok=True)
        print(f"[RESUME] Loaded Stage 1 checkpoint (ZS BalAcc={zs_score:.5f})", flush=True)
    else:
        pool_df = curate_all_pools(pool_parquet, fidelity_csv, extra_dir, train_df, classes, cap_per_label=cap_per_label)
        pool_df["label"] = pool_df["model"].map(label2idx)
        if smoke_test:
            pool_df = pool_df.sample(n=min(240, len(pool_df)), random_state=SEED).reset_index(drop=True)

        s1_splitter = StratifiedGroupKFold(n_splits=10 if smoke_test else 20, shuffle=True, random_state=SEED)
        s1_tr_idx, s1_val_idx = next(s1_splitter.split(pool_df, pool_df["label"].values, groups=pool_df["group_key"].values))
        pool_tr = pool_df.iloc[s1_tr_idx].reset_index(drop=True)
        pool_val = pool_df.iloc[s1_val_idx].reset_index(drop=True)
        print(f"[STAGE 1] Train={len(pool_tr):,}, Pool-Val={len(pool_val):,}", flush=True)

        counts = np.bincount(pool_tr["label"].values, minlength=n_classes).astype(np.float32)
        cls_weights = (np.mean(counts) / np.maximum(counts, 1.0)) ** 0.75
        cls_weights = cls_weights / np.mean(cls_weights)
        cls_weights_tensor = torch.tensor(cls_weights, dtype=torch.float32, device=device)

        s1_tr_ds, s1_tr_lens = build_tokenized_dataset(pool_tr["text"].values, pool_tr["label"].values, tokenizer, MAX_LEN)
        s1_val_ds, _ = build_tokenized_dataset(pool_val["text"].values, pool_val["label"].values, tokenizer, MAX_LEN)
        s1_sampler = BucketBatchSampler(s1_tr_lens, max_tokens=max_tokens_s1, max_batch_size=max_bs_s1, seed=SEED)
        s1_tr_loader = DataLoader(s1_tr_ds, batch_sampler=s1_sampler, collate_fn=collator, num_workers=2, pin_memory=True)
        s1_val_loader = DataLoader(s1_val_ds, batch_size=16 if use_4bit else 24, shuffle=False, collate_fn=collator, num_workers=2, pin_memory=True)

        n_epochs_s1 = 1 if smoke_test else epochs_s1_arg
        opt_s1 = torch.optim.AdamW(
            [
                {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": LR_S1_LORA, "weight_decay": 0.01},
                {"params": list(head.parameters()), "lr": LR_S1_HEAD, "weight_decay": 0.01},
            ]
        )
        opt_steps_s1 = int(math.ceil(len(s1_tr_loader) / grad_accum_s1) * n_epochs_s1 * 1.02)
        sched_s1 = get_cosine_warmup_scheduler(opt_s1, int(opt_steps_s1 * 0.05), max(1, opt_steps_s1))
        criterion_s1 = nn.CrossEntropyLoss(weight=cls_weights_tensor, label_smoothing=0.05)

        best_s1_score, best_s1_lora, best_s1_head = -1.0, None, None

        for epoch in range(n_epochs_s1):
            t_ep = time.time()
            backbone.train()
            head.train()
            running_loss, seen_samples, n_steps = 0.0, 0, 0
            opt_s1.zero_grad(set_to_none=True)

            for step, batch in enumerate(s1_tr_loader):
                if smoke_test and step >= 6:
                    break
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    logits = forward_logits(backbone, head, input_ids, attention_mask, hidden_size)
                loss = criterion_s1(logits.float(), labels) / grad_accum_s1
                loss.backward()
                running_loss += loss.item() * grad_accum_s1
                seen_samples += input_ids.size(0)
                n_steps += 1
                if (step + 1) % grad_accum_s1 == 0 or (step + 1) == len(s1_tr_loader):
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    opt_s1.step()
                    sched_s1.step()
                    opt_s1.zero_grad(set_to_none=True)
                if (step + 1) % (3 if smoke_test else 250) == 0:
                    elapsed = time.time() - t_ep
                    print(
                        f"    [S1 E{epoch + 1}] step {step + 1}/{len(s1_tr_loader)} ({seen_samples:,} samples) "
                        f"| loss={running_loss / n_steps:.4f} | {seen_samples / max(elapsed, 1e-3):.1f}/s "
                        f"| max_vram={torch.cuda.max_memory_allocated() / 1e9:.2f}GB",
                        flush=True,
                    )

            val_probs, val_targets = evaluate_model(backbone, head, s1_val_loader, device, use_bf16, hidden_size, max_batches=2 if smoke_test else None)
            val_bal_acc = balanced_accuracy_score(val_targets, np.argmax(val_probs, axis=1))
            w_last = torch.softmax(head.layer_logits_last.detach(), dim=0).cpu().numpy().round(3).tolist()
            w_mean = torch.softmax(head.layer_logits_mean.detach(), dim=0).cpu().numpy().round(3).tolist()
            print(
                f"  [S1 Epoch {epoch + 1}/{n_epochs_s1}] ({time.time() - t_ep:.1f}s) "
                f"Loss: {running_loss / max(n_steps, 1):.4f} | Pool-Val BalAcc: {val_bal_acc:.5f} | w_last={w_last} | w_mean={w_mean}",
                flush=True,
            )
            if val_bal_acc > best_s1_score:
                best_s1_score = val_bal_acc
                best_s1_lora = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(backbone).items()}
                best_s1_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

        del opt_s1, sched_s1, s1_tr_loader, s1_val_loader, s1_tr_ds, s1_val_ds
        gc.collect()
        torch.cuda.empty_cache()

        if smoke_test:
            backbone.train()
            head.train()
            sub_df = train_df.head(48).copy()
            tr_ds, tr_lens = build_tokenized_dataset(sub_df["text"].values, sub_df["label"].values, tokenizer, MAX_LEN)
            tr_sampler = BucketBatchSampler(tr_lens, max_tokens=max_tokens_s2, max_batch_size=max_bs_s2, seed=SEED)
            tr_loader = DataLoader(tr_ds, batch_sampler=tr_sampler, collate_fn=collator)
            opt_t = torch.optim.AdamW(trainable_params, lr=LR_S2_LORA)
            for batch in tr_loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    logits = forward_logits(backbone, head, batch["input_ids"].to(device), batch["attention_mask"].to(device), hidden_size)
                loss = nn.CrossEntropyLoss()(logits.float(), batch["labels"].to(device))
                loss.backward()
                opt_t.step()
                opt_t.zero_grad(set_to_none=True)
                break
            del opt_t, tr_loader, tr_ds
            gc.collect()
            torch.cuda.empty_cache()
            test_dir = "/content/smoke_exp017b"
            os.makedirs(test_dir, exist_ok=True)
            backbone.save_pretrained(test_dir)
            torch.save(head.state_dict(), os.path.join(test_dir, "classifier_head.pt"))
            res_k = subprocess.run(["kaggle", "competitions", "submission-limits", "-c", "name-that-llm"], capture_output=True, text=True)
            print(f"[SMOKE PASS] max_vram={torch.cuda.max_memory_allocated() / 1e9:.2f}GB | saved={os.listdir(test_dir)} | kaggle_auth_rc={res_k.returncode}", flush=True)
            return

        train_ds_all, _ = build_tokenized_dataset(train_df["text"].values, train_df["label"].values, tokenizer, MAX_LEN)
        train_loader_all = DataLoader(train_ds_all, batch_size=16 if use_4bit else 24, shuffle=False, collate_fn=collator)

        set_peft_model_state_dict(backbone, {k: v.to(device) for k, v in best_s1_lora.items()})
        head.load_state_dict({k: v.to(device) for k, v in best_s1_head.items()})
        zs_probs, zs_targets = evaluate_model(backbone, head, train_loader_all, device, use_bf16, hidden_size)
        zs_score = balanced_accuracy_score(zs_targets, np.argmax(zs_probs, axis=1))
        print(f"[STAGE 1 DIAGNOSTIC] Zero-Shot train.csv Balanced Accuracy: {zs_score:.5f}", flush=True)
        del train_ds_all, train_loader_all
        gc.collect()

        os.makedirs(s1_ckpt_dir, exist_ok=True)
        torch.save(best_s1_lora, os.path.join(s1_ckpt_dir, "lora_s1.pt"))
        torch.save(best_s1_head, os.path.join(s1_ckpt_dir, "head_s1.pt"))
        with open(os.path.join(s1_ckpt_dir, "stage1_meta.json"), "w") as f:
            json.dump({"zs_score": float(zs_score), "pool_val_score": float(best_s1_score)}, f, indent=2)
        upload_private_dataset(export_dir, DATASET_SLUG, f"Exp017b Qwen3-14B-27L S1 ZS {zs_score:.5f}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros((len(train_df), n_classes), dtype=np.float32)
    fold_scores = []

    y = train_df["label"].values
    groups = train_df["prompt_id"].values
    criterion_s2 = nn.CrossEntropyLoss(label_smoothing=0.02)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df, y, groups=groups)):
        fold_dir = os.path.join(export_dir, f"fold_{fold}")
        vp_path = os.path.join(fold_dir, "val_probs.npy")
        vs_path = os.path.join(fold_dir, "val_score.json")
        hd_path = os.path.join(fold_dir, "classifier_head.pt")
        ad_path = os.path.join(fold_dir, "adapter_model.safetensors")
        if resume and os.path.exists(vp_path) and os.path.exists(vs_path) and os.path.exists(hd_path) and os.path.exists(ad_path):
            cached_vp = np.load(vp_path)
            if cached_vp.shape == (len(val_idx), n_classes):
                with open(vs_path) as f:
                    cached_score = float(json.load(f)["best_fold_score"])
                oof_probs[val_idx] = cached_vp
                fold_scores.append(cached_score)
                print(f"\n[RESUME] Skipping completed Fold {fold + 1}/5 (BalAcc={cached_score:.5f})", flush=True)
                continue

        print(f"\n{'='*20} Stage 2 Fold {fold + 1}/5 {'='*20}", flush=True)
        assert len(set(train_df.iloc[train_idx]["prompt_id"]).intersection(set(train_df.iloc[val_idx]["prompt_id"]))) == 0

        set_peft_model_state_dict(backbone, {k: v.to(device) for k, v in best_s1_lora.items()})
        head.load_state_dict({k: v.to(device) for k, v in best_s1_head.items()})

        tr_sub, va_sub = train_df.iloc[train_idx], train_df.iloc[val_idx]
        tr_ds, tr_lens = build_tokenized_dataset(tr_sub["text"].values, tr_sub["label"].values, tokenizer, MAX_LEN)
        va_ds, _ = build_tokenized_dataset(va_sub["text"].values, va_sub["label"].values, tokenizer, MAX_LEN)
        tr_sampler = BucketBatchSampler(tr_lens, max_tokens=max_tokens_s2, max_batch_size=max_bs_s2, seed=SEED + fold * 100)
        tr_loader = DataLoader(tr_ds, batch_sampler=tr_sampler, collate_fn=collator, num_workers=2, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=16 if use_4bit else 24, shuffle=False, collate_fn=collator, num_workers=2, pin_memory=True)

        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": LR_S2_LORA, "weight_decay": 0.03},
                {"params": list(head.parameters()), "lr": LR_S2_HEAD, "weight_decay": 0.03},
            ]
        )
        opt_steps_s2 = int(math.ceil(len(tr_loader) / grad_accum_s2) * EPOCHS_S2 * 1.02)
        scheduler = get_cosine_warmup_scheduler(optimizer, int(opt_steps_s2 * 0.1), opt_steps_s2)

        best_fold_score, best_fold_probs, best_fold_lora, best_fold_head = -1.0, None, None, None

        for epoch in range(EPOCHS_S2):
            t_ep = time.time()
            backbone.train()
            head.train()
            tr_loss, n_steps = 0.0, 0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(tr_loader):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    logits = forward_logits(backbone, head, input_ids, attention_mask, hidden_size)
                loss = criterion_s2(logits.float(), labels) / grad_accum_s2
                loss.backward()
                tr_loss += loss.item() * grad_accum_s2
                n_steps += 1
                if (step + 1) % grad_accum_s2 == 0 or (step + 1) == len(tr_loader):
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            va_probs, va_targets = evaluate_model(backbone, head, va_loader, device, use_bf16, hidden_size)
            va_score = balanced_accuracy_score(va_targets, np.argmax(va_probs, axis=1))
            print(f"  Epoch {epoch + 1}/{EPOCHS_S2} ({time.time() - t_ep:.1f}s) - Loss: {tr_loss / max(n_steps, 1):.4f} - Val BalAcc: {va_score:.5f}", flush=True)

            if va_score > best_fold_score:
                best_fold_score = va_score
                best_fold_probs = va_probs
                best_fold_lora = {k: v.detach().float().cpu().clone() for k, v in get_peft_model_state_dict(backbone).items()}
                best_fold_head = {k: v.detach().float().cpu().clone() for k, v in head.state_dict().items()}

        oof_probs[val_idx] = best_fold_probs
        fold_scores.append(best_fold_score)
        print(f"Fold {fold + 1} Best BalAcc: {best_fold_score:.5f}", flush=True)

        os.makedirs(fold_dir, exist_ok=True)
        set_peft_model_state_dict(backbone, {k: v.to(device) for k, v in best_fold_lora.items()})
        backbone.save_pretrained(fold_dir)
        torch.save(best_fold_head, os.path.join(fold_dir, "classifier_head.pt"))
        np.save(os.path.join(fold_dir, "val_probs.npy"), best_fold_probs)
        with open(os.path.join(fold_dir, "val_score.json"), "w") as f:
            json.dump({"best_fold_score": float(best_fold_score)}, f)
        upload_private_dataset(export_dir, DATASET_SLUG, f"Exp017b Qwen3-14B F{fold + 1}/5 {best_fold_score:.4f}")

        del optimizer, scheduler, tr_loader, va_loader
        gc.collect()
        torch.cuda.empty_cache()

    overall_preds = np.argmax(oof_probs, axis=1)
    overall_score = balanced_accuracy_score(y, overall_preds)
    mean_fold = float(np.mean(fold_scores))

    print("\n" + "=" * 60, flush=True)
    print("Exp #017b Two-Stage Qwen3-14B-Base (27L) 8-Layer LoRA", flush=True)
    print(f"Stage 1 Zero-Shot train.csv BalAcc : {zs_score:.5f}", flush=True)
    print(f"Stage 2 5-Fold OOF Balanced Acc    : {overall_score:.5f}", flush=True)
    print(f"Fold Scores                        : {[round(float(s), 5) for s in fold_scores]}", flush=True)
    print(f"Mean Fold (+/- std)                : {mean_fold:.5f} +/- {np.std(fold_scores):.5f}", flush=True)
    print("=" * 60, flush=True)

    meta_info = {
        "model_name": MODEL_NAME,
        "hidden_size": int(hidden_size),
        "keep_layers": int(KEEP_LAYERS),
        "max_len": MAX_LEN,
        "layer_indices": LAYER_INDICES,
        "pooling": "multilayer_normed_weighted_last_mean",
        "classes": classes,
        "lora_r": 16,
        "lora_alpha": 32,
        "qlora_4bit": bool(use_4bit),
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
    oof_csv_path = "/content/oof_exp017b_qwen3_14b_lora.csv"
    oof_df.to_csv(oof_csv_path, index=False)
    oof_df.to_csv(os.path.join(export_dir, "oof_exp017b_qwen3_14b_lora.csv"), index=False)

    if os.path.isdir("/content/drive/MyDrive"):
        drv_dir = "/content/drive/MyDrive/kaggle_exp017b_qwen3_14b_lora"
        shutil.copytree(export_dir, drv_dir, dirs_exist_ok=True)
        shutil.copy2(oof_csv_path, os.path.join(drv_dir, "oof_exp017b_qwen3_14b_lora.csv"))
        print(f"[DRIVE] Backed up to {drv_dir}", flush=True)

    upload_private_dataset(
        export_dir,
        DATASET_SLUG,
        f"Exp017b Qwen3-14B-27L OOF {overall_score:.5f}",
    )

    with open(oof_csv_path, "rb") as f:
        b64_oof = base64.b64encode(f.read()).decode("utf-8")
    print(f"\n===BASE64_OOF_START===\n{b64_oof}\n===BASE64_OOF_END===", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--cap-per-label", type=int, default=2400)
    parser.add_argument("--epochs-s1", type=int, default=1)
    parser.add_argument("--force-bf16", action="store_true")
    parser.add_argument("--resume", "--resume-s1", dest="resume", action="store_true")
    args = parser.parse_args()
    ensure_environment()
    train_and_evaluate(
        smoke_test=args.smoke_test,
        cap_per_label_arg=args.cap_per_label,
        epochs_s1_arg=args.epochs_s1,
        force_bf16=args.force_bf16,
        resume=args.resume,
    )
