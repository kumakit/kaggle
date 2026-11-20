"""
005_name-that-llm: Exp #016 - Colab L4 Feature Extraction for Two-Way Gaussian Head & Embedding Clustering
Extracts unified train.csv (2,699 rows) fp16 hidden-state / embedding matrices using EXISTING weights:
  1. E_qwen_emb8b.npy   (2699, 4096)  - Qwen/Qwen3-Embedding-8B L2-normalized EOS embeddings (clustering thr=0.70)
  2. Z_llama_s1.npy     (2699, 32768) - Exp #015 Llama-3.1-8B (24L) Stage-1 LoRA (unseen train.csv, 8-layer mean pool)
  3. Z_llama_oof.npy    (2699, 32768) - Exp #015 Llama-3.1-8B (24L) Fold-OOF LoRA (strict val_idx per fold)
  4. Z_qwen8b_base.npy  (2699, 24576) - Qwen/Qwen3-8B-Base raw (6-layer mean pool)
  5. Z_qwen8b_oof.npy   (2699, 24576) - Exp #013 Qwen3-8B Fold-OOF LoRA (strict val_idx per fold)
Uploads all assets to Private Dataset nobuyukikumagai/name-that-llm-exp016-gauss-assets.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc
import sys
import subprocess
import argparse
import json
import time
import shutil
import warnings
import zipfile
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
DATASET_SLUG = "nobuyukikumagai/name-that-llm-exp016-gauss-assets"
DATASET_TITLE = "Exp016 Gauss Head and Embedding Assets"
assert 6 <= len(DATASET_TITLE) <= 50, f"Invalid title len: {len(DATASET_TITLE)}"

EXP013_SLUG = "nobuyukikumagai/name-that-llm-exp013-qwen3-8b-lora"
EXP015_SLUG = "nobuyukikumagai/name-that-llm-exp015-llama3-8b-lora"


def ensure_environment():
    print("[INIT] Setting up packages & Kaggle auth...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio", "torchao"], check=False)
    packages = ["transformers>=4.51.0", "peft>=0.14.0", "accelerate", "safetensors", "kaggle"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U"] + packages, check=True)

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
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
            raise RuntimeError("Missing Kaggle credentials (env KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json).")
        with open(kaggle_json, "w") as f:
            json.dump({"username": username.strip(), "key": key.strip()}, f)
        os.chmod(kaggle_json, 0o600)
        print(f"  - Configured kaggle credentials for {username}", flush=True)


def extract_zips(folder):
    for fn in os.listdir(folder):
        if fn.endswith(".zip"):
            zp = os.path.join(folder, fn)
            with zipfile.ZipFile(zp, "r") as z:
                z.extractall(folder)
            os.remove(zp)


def download_inputs():
    comp_dir = "/content/comp_data"
    exp13_dir = "/content/exp013_lora"
    exp15_dir = "/content/exp015_lora"
    os.makedirs(comp_dir, exist_ok=True)
    if not os.path.exists(os.path.join(comp_dir, "train.csv")):
        subprocess.run(["kaggle", "competitions", "download", "-c", "name-that-llm", "-p", comp_dir], check=True)
        extract_zips(comp_dir)
    if not os.path.exists(os.path.join(exp13_dir, "classes.json")):
        subprocess.run(["kaggle", "datasets", "download", "-d", EXP013_SLUG, "-p", exp13_dir, "--unzip"], check=True)
    if not os.path.exists(os.path.join(exp15_dir, "classes.json")):
        subprocess.run(["kaggle", "datasets", "download", "-d", EXP015_SLUG, "-p", exp15_dir, "--unzip"], check=True)
    return comp_dir, exp13_dir, exp15_dir


def resolve_module_name(prefix, name_to_mod):
    cands = [prefix]
    for sp in ("base_model.model.model.", "base_model.model.", "base_model.", "model."):
        if prefix.startswith(sp):
            cands.append(prefix[len(sp):])
    for c in cands:
        if c in name_to_mod:
            return c
    for mname in name_to_mod:
        if prefix.endswith("." + mname) or prefix.endswith(mname):
            return mname
    return None


def extract_qwen_embedding(texts, device, smoke_test=False):
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_id = "Qwen/Qwen3-Embedding-8B"
    print(f"[1/3] Extracting {model_id} EOS embeddings ({len(texts)} rows)...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 151643
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos

    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16, attn_implementation="sdpa").to(device).eval()
    hid = int(model.config.hidden_size)
    enc = tok(list(texts), truncation=True, max_length=511)["input_ids"]
    ids = [x if (x and x[-1] == eos) else (x + [eos]) for x in enc]
    order = np.argsort([len(x) for x in ids], kind="stable")

    emb = np.zeros((len(texts), hid), dtype=np.float16)
    tok_budget = 4096 if not smoke_test else 2048
    done = 0
    with torch.inference_mode():
        while done < len(order):
            end = done + 1
            while end < len(order) and (end - done + 1) * len(ids[order[end]]) <= tok_budget and (end - done) < 16:
                end += 1
            idx = order[done:end]
            width = max(len(ids[i]) for i in idx)
            inp = torch.full((len(idx), width), pad, dtype=torch.long)
            att = torch.zeros((len(idx), width), dtype=torch.long)
            for r, i in enumerate(idx):
                inp[r, :len(ids[i])] = torch.tensor(ids[i], dtype=torch.long)
                att[r, :len(ids[i])] = 1
            hidden = model(input_ids=inp.to(device), attention_mask=att.to(device), use_cache=False).last_hidden_state
            vec = hidden[torch.arange(len(idx), device=device), (att.sum(1) - 1).to(device)].float()
            emb[idx] = torch.nn.functional.normalize(vec, dim=-1).cpu().numpy().astype(np.float16)
            done = end

    print(f"  - E_qwen_emb8b done shape={emb.shape} in {time.time() - t0:.1f}s | max_vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return emb


def _make_tokenized_batches(texts, tok, max_len, batch_size=12):
    enc = tok(list(texts), truncation=True, max_length=max_len, padding=False)["input_ids"]
    order = np.argsort([len(x) for x in enc], kind="stable")
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    return enc, batches


def _featurize_multilayer(model, enc, pad_id, batches, layers, hidden_size, device):
    import torch
    import torch.nn.functional as F

    feats = np.zeros((len(enc), hidden_size * len(layers)), dtype=np.float16)
    with torch.inference_mode():
        for b in batches:
            width = max(len(enc[i]) for i in b)
            inp = torch.full((len(b), width), pad_id, dtype=torch.long)
            att = torch.zeros((len(b), width), dtype=torch.long)
            for r, i in enumerate(b):
                inp[r, :len(enc[i])] = torch.tensor(enc[i], dtype=torch.long)
                att[r, :len(enc[i])] = 1
            hs = model(input_ids=inp.to(device), attention_mask=att.to(device), output_hidden_states=True, use_cache=False).hidden_states
            mask = att.to(device).unsqueeze(-1).float()
            den = mask.sum(dim=1).clamp(min=1e-6)
            chunks = []
            for lyr in layers:
                mean_vec = (hs[lyr].float() * mask).sum(dim=1) / den
                chunks.append(F.layer_norm(mean_vec, (hidden_size,)))
            feats[b] = torch.cat(chunks, dim=-1).cpu().numpy().astype(np.float16)
    return feats


def extract_llama_assets(texts, folds, exp15_dir, device):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from safetensors.torch import load_file as load_safetensors

    t0 = time.time()
    with open(os.path.join(exp15_dir, "classes.json")) as f:
        meta = json.load(f)
    model_id = meta["model_name"]
    keep_layers = int(meta["keep_layers"])
    layers = list(meta["layer_indices"])
    max_len = int(meta["max_len"])
    hid = int(meta["hidden_size"])
    print(f"[2/3] Extracting {model_id} ({keep_layers}L) Stage-1 & Fold-OOF features...", flush=True)

    tok = AutoTokenizer.from_pretrained(exp15_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(model_id)
    cfg.pad_token_id = tok.pad_token_id
    cfg.num_hidden_layers = keep_layers
    cfg.use_cache = False

    base_model = AutoModel.from_pretrained(model_id, config=cfg, torch_dtype=torch.float16, attn_implementation="sdpa").to(device).eval()
    name_to_mod = dict(base_model.named_modules())

    with open(os.path.join(exp15_dir, "fold_0", "adapter_config.json")) as f:
        acfg = json.load(f)
    scaling = float(acfg["lora_alpha"]) / float(acfg["r"])

    # Build target mapping once using fold_0 safetensors
    sf0 = load_safetensors(os.path.join(exp15_dir, "fold_0", "adapter_model.safetensors"))
    target_map = {}
    for k in sf0:
        if k.endswith(".lora_A.weight"):
            prefix = k[:-len(".lora_A.weight")]
            mname = resolve_module_name(prefix, name_to_mod)
            assert mname is not None, f"Unresolved Llama LoRA key: {k}"
            target_map[prefix] = mname
    del sf0

    def apply_delta(sd, sign=+1.0):
        for prefix, mname in target_map.items():
            # stage1 lora_s1.pt keys may include .default.weight
            ka = prefix + ".lora_A.default.weight" if (prefix + ".lora_A.default.weight") in sd else prefix + ".lora_A.weight"
            kb = prefix + ".lora_B.default.weight" if (prefix + ".lora_B.default.weight") in sd else prefix + ".lora_B.weight"
            la = sd[ka].to(device, dtype=torch.float32)
            lb = sd[kb].to(device, dtype=torch.float32)
            mod = name_to_mod[mname]
            delta = (sign * scaling * (lb @ la)).to(mod.weight.dtype)
            mod.weight.data.add_(delta)

    # 1) Stage-1 unseen train.csv features (unified coordinate space)
    s1_sd = torch.load(os.path.join(exp15_dir, "stage1_best", "lora_s1.pt"), map_location="cpu", weights_only=True)
    apply_delta(s1_sd, +1.0)
    enc_all, batches_all = _make_tokenized_batches(texts, tok, max_len, batch_size=12)
    z_s1 = _featurize_multilayer(base_model, enc_all, tok.pad_token_id, batches_all, layers, hid, device)
    apply_delta(s1_sd, -1.0)
    del s1_sd

    # 2) Fold-OOF features (val_idx only per fold)
    z_oof = np.zeros_like(z_s1)
    for k, (_, va_idx) in enumerate(folds):
        if len(va_idx) == 0:
            continue
        f_sd = load_safetensors(os.path.join(exp15_dir, f"fold_{k}", "adapter_model.safetensors"))
        apply_delta(f_sd, +1.0)
        enc_va, batches_va = _make_tokenized_batches(texts[va_idx], tok, max_len, batch_size=12)
        z_oof[va_idx] = _featurize_multilayer(base_model, enc_va, tok.pad_token_id, batches_va, layers, hid, device)
        apply_delta(f_sd, -1.0)
        del f_sd

    print(f"  - Llama S1/OOF done shape={z_s1.shape} in {time.time() - t0:.1f}s | max_vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    del base_model, name_to_mod
    gc.collect()
    torch.cuda.empty_cache()
    return z_s1, z_oof


def extract_qwen8b_assets(texts, folds, exp13_dir, device):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from safetensors.torch import load_file as load_safetensors

    t0 = time.time()
    with open(os.path.join(exp13_dir, "classes.json")) as f:
        meta = json.load(f)
    model_id = meta["model_name"]
    layers = list(meta["layer_indices"])
    max_len = int(meta["max_len"])
    hid = int(meta["hidden_size"])
    print(f"[3/3] Extracting {model_id} Raw Base & Fold-OOF features...", flush=True)

    tok = AutoTokenizer.from_pretrained(exp13_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(model_id)
    cfg.pad_token_id = tok.pad_token_id
    cfg.use_cache = False

    base_model = AutoModel.from_pretrained(model_id, config=cfg, torch_dtype=torch.float16, attn_implementation="sdpa").to(device).eval()
    name_to_mod = dict(base_model.named_modules())

    with open(os.path.join(exp13_dir, "fold_0", "adapter_config.json")) as f:
        acfg = json.load(f)
    scaling = float(acfg["lora_alpha"]) / float(acfg["r"])
    sf0 = load_safetensors(os.path.join(exp13_dir, "fold_0", "adapter_model.safetensors"))
    target_map = {}
    for k in sf0:
        if k.endswith(".lora_A.weight"):
            prefix = k[:-len(".lora_A.weight")]
            mname = resolve_module_name(prefix, name_to_mod)
            assert mname is not None, f"Unresolved Qwen8B LoRA key: {k}"
            target_map[prefix] = mname
    del sf0

    # 1) Unmodified base model (zero leakage, unified coordinate space)
    enc_all, batches_all = _make_tokenized_batches(texts, tok, max_len, batch_size=12)
    z_base = _featurize_multilayer(base_model, enc_all, tok.pad_token_id, batches_all, layers, hid, device)

    # 2) Fold-OOF features (val_idx only per fold)
    z_oof = np.zeros_like(z_base)
    for k, (_, va_idx) in enumerate(folds):
        if len(va_idx) == 0:
            continue
        f_sd = load_safetensors(os.path.join(exp13_dir, f"fold_{k}", "adapter_model.safetensors"))
        for prefix, mname in target_map.items():
            la = f_sd[prefix + ".lora_A.weight"].to(device, dtype=torch.float32)
            lb = f_sd[prefix + ".lora_B.weight"].to(device, dtype=torch.float32)
            mod = name_to_mod[mname]
            mod.weight.data.add_((scaling * (lb @ la)).to(mod.weight.dtype))
        enc_va, batches_va = _make_tokenized_batches(texts[va_idx], tok, max_len, batch_size=12)
        z_oof[va_idx] = _featurize_multilayer(base_model, enc_va, tok.pad_token_id, batches_va, layers, hid, device)
        for prefix, mname in target_map.items():
            la = f_sd[prefix + ".lora_A.weight"].to(device, dtype=torch.float32)
            lb = f_sd[prefix + ".lora_B.weight"].to(device, dtype=torch.float32)
            mod = name_to_mod[mname]
            mod.weight.data.sub_((scaling * (lb @ la)).to(mod.weight.dtype))
        del f_sd

    print(f"  - Qwen8B Base/OOF done shape={z_base.shape} in {time.time() - t0:.1f}s | max_vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    del base_model, name_to_mod
    gc.collect()
    torch.cuda.empty_cache()
    return z_base, z_oof


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
    check = subprocess.run(["kaggle", "datasets", "status", dataset_slug], capture_output=True, text=True)
    cmd = ["kaggle", "datasets", "version" if check.returncode == 0 else "create", "-p", export_dir]
    if check.returncode == 0:
        cmd += ["-m", safe_msg]
    cmd += ["-r", "zip", "-t"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout, flush=True)
    if res.returncode != 0:
        print(f"[UPLOAD WARN] {res.stderr}", flush=True)


def run(smoke_test=False):
    import torch
    from sklearn.model_selection import StratifiedGroupKFold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comp_dir, exp13_dir, exp15_dir = download_inputs()
    train_df = pd.read_csv(os.path.join(comp_dir, "train.csv"))
    train_df["text"] = train_df["text"].fillna("").astype(str)
    classes = sorted(train_df["model"].unique().tolist())
    c2i = {c: i for i, c in enumerate(classes)}
    y = train_df["model"].map(c2i).values
    groups = train_df["prompt_id"].values

    if smoke_test:
        train_df = train_df.head(32).copy()
        texts = train_df["text"].values
        folds = [(np.arange(24), np.arange(24, 32))] + [(np.arange(0), np.arange(0)) for _ in range(4)]
    else:
        texts = train_df["text"].values
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        folds = list(sgkf.split(train_df, y, groups=groups))

    e_emb = extract_qwen_embedding(texts, device, smoke_test=smoke_test)
    z_ll_s1, z_ll_oof = extract_llama_assets(texts, folds, exp15_dir, device)
    z_q8_base, z_q8_oof = extract_qwen8b_assets(texts, folds, exp13_dir, device)

    assert all(np.isfinite(x).all() for x in (e_emb, z_ll_s1, z_ll_oof, z_q8_base, z_q8_oof))
    if smoke_test:
        res_k = subprocess.run(["kaggle", "competitions", "submission-limits", "-c", "name-that-llm"], capture_output=True, text=True)
        print(f"[SMOKE PASS] shapes={(e_emb.shape, z_ll_s1.shape, z_q8_base.shape)} | max_vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB | kaggle_auth_rc={res_k.returncode}", flush=True)
        return

    out_dir = "/content/export_exp016_gauss_assets"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "E_qwen_emb8b.npy"), e_emb)
    np.save(os.path.join(out_dir, "Z_llama_s1.npy"), z_ll_s1)
    np.save(os.path.join(out_dir, "Z_llama_oof.npy"), z_ll_oof)
    np.save(os.path.join(out_dir, "Z_qwen8b_base.npy"), z_q8_base)
    np.save(os.path.join(out_dir, "Z_qwen8b_oof.npy"), z_q8_oof)
    upload_private_dataset(out_dir, DATASET_SLUG, "Exp016 Gauss & Qwen-Emb train assets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    ensure_environment()
    run(smoke_test=args.smoke_test)
