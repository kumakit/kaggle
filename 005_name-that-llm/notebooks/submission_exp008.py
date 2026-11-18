"""
005_name-that-llm: Exp #008 Inference-Only 4-Model Multi-Architecture Ensemble Submission
Log-Prob Geometric Ensemble (OOF Balanced Accuracy: 0.64312 | Nested-CV: 0.63831):
  - 25% Exp #007: OpenPool (fidelity >= 0.45) + 5-Fold Qwen3-4B-Base LoRA (Single OOF: 0.61460)
  - 40% Exp #008: OpenPool (fidelity >= 0.45) + 5-Fold ModernBERT-large (Single OOF: 0.58790)
  - 30% Exp #006: OpenPool + 5-Fold Qwen3-1.7B-Base LoRA (Single OOF: 0.57606)
  -  5% Exp #005: OpenPool + 5-Fold ModernBERT-base (Single OOF: 0.51675)
Optimizations:
  - Length-sorted batching with exact original-order restoration (~1.8-2x faster on hidden test)
  - Pure PyTorch + Safetensors LoRA merging (zero PEFT dependency at runtime)
  - Sequential VRAM cleanup across all 4 model ensembles
"""

import os
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file as load_safetensors
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

MAX_LEN = 416
W_QWEN3_4B = 0.25
W_MB_LARGE = 0.40
W_QWEN3_17B = 0.30
W_MB_BASE = 0.05
EPS = 1e-8


def find_file(filename, path_hint=None):
    if os.path.exists("/kaggle/input"):
        matches = []
        for root, _, files in os.walk("/kaggle/input"):
            if filename in files:
                full = os.path.join(root, filename)
                if path_hint is None or path_hint.lower() in full.lower():
                    print(f"[FOUND] {filename}: {full}")
                    return full
                matches.append(full)
        if path_hint is None and matches:
            print(f"[FOUND] {filename}: {matches[0]}")
            return matches[0]
    for cand in [os.path.join("005_name-that-llm", "data", filename), os.path.join("data", filename), filename]:
        if os.path.exists(cand):
            print(f"[FOUND] {filename} locally: {cand}")
            return cand
    raise FileNotFoundError(f"Could not locate {filename} (hint={path_hint})")


def find_qwen_base_dir(expected_hidden_size):
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "config.json" in files and (
                "model.safetensors" in files or "model.safetensors.index.json" in files
            ):
                try:
                    with open(os.path.join(root, "config.json"), "r") as f:
                        cfg = json.load(f)
                    if cfg.get("hidden_size") == expected_hidden_size and "qwen" in str(cfg.get("model_type", "")).lower():
                        print(f"[FOUND] Qwen3 Base Model dir (hidden={expected_hidden_size}): {root}")
                        return root
                except Exception:
                    pass
    raise FileNotFoundError(f"Could not locate Qwen3 base model weights (hidden_size={expected_hidden_size}) in /kaggle/input")


def find_qwen_lora_dir(expected_hidden_size, path_hint):
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "classes.json" in files:
                try:
                    with open(os.path.join(root, "classes.json"), "r") as f:
                        meta = json.load(f)
                    if int(meta.get("hidden_size", -1)) == expected_hidden_size:
                        print(f"[FOUND] Qwen3 LoRA dir (hidden={expected_hidden_size}, hint={path_hint}): {root}")
                        return root
                except Exception:
                    pass
    raise FileNotFoundError(f"Could not locate Qwen3 LoRA weights (hidden={expected_hidden_size}, hint={path_hint})")


def find_bert_dir(expected_hidden_size, path_hint):
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "fold_0.pt" in files and "config.json" in files:
                try:
                    with open(os.path.join(root, "config.json"), "r") as f:
                        cfg = json.load(f)
                    if cfg.get("hidden_size") == expected_hidden_size and "modernbert" in str(cfg.get("model_type", "")).lower():
                        print(f"[FOUND] ModernBERT dir (hidden={expected_hidden_size}, hint={path_hint}): {root}")
                        return root
                except Exception:
                    pass
    raise FileNotFoundError(f"Could not locate ModernBERT weights (hidden={expected_hidden_size}, hint={path_hint})")


class PreTokenizedDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def make_sorted_loader(test_texts, tokenizer, max_len, batch_size):
    enc = tokenizer(
        [str(t) for t in test_texts],
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    lengths = np.asarray([len(x) for x in enc["input_ids"]], dtype=np.int32)
    order = np.argsort(lengths, kind="mergesort")
    inv_order = np.argsort(order, kind="mergesort")

    sorted_ids = [enc["input_ids"][i] for i in order]
    sorted_mask = [enc["attention_mask"][i] for i in order]
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(
        PreTokenizedDataset(sorted_ids, sorted_mask),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    return loader, inv_order


def make_classifier_head(hidden_size, n_classes):
    return nn.Sequential(
        nn.LayerNorm(hidden_size * 2),
        nn.Dropout(0.1),
        nn.Linear(hidden_size * 2, n_classes),
    )


def resolve_module_name(lora_prefix, name_to_mod):
    cands = [lora_prefix]
    for strip_prefix in ["base_model.model.model.", "base_model.model.", "base_model.", "model."]:
        if lora_prefix.startswith(strip_prefix):
            cands.append(lora_prefix[len(strip_prefix):])
    for c in cands:
        if c in name_to_mod:
            return c
    for mname in name_to_mod:
        if lora_prefix.endswith("." + mname) or lora_prefix.endswith(mname):
            return mname
    return None


def predict_qwen3_ensemble(test_texts, classes, qwen_base_dir, qwen_lora_dir, device, amp_dtype, use_amp, batch_size=24, tag="QWEN3"):
    t0 = time.time()
    with open(os.path.join(qwen_lora_dir, "classes.json"), "r") as f:
        meta = json.load(f)
    assert meta["classes"] == classes, f"{tag} class list mismatch!"
    hidden_size = int(meta["hidden_size"])
    max_len = int(meta.get("max_len", MAX_LEN))

    tokenizer = AutoTokenizer.from_pretrained(qwen_lora_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    test_loader, inv_order = make_sorted_loader(test_texts, tokenizer, max_len, batch_size)

    config = AutoConfig.from_pretrained(qwen_base_dir)
    config.pad_token_id = tokenizer.pad_token_id
    model_dtype = torch.bfloat16 if (use_amp and amp_dtype == torch.bfloat16) else (torch.float16 if use_amp else torch.float32)

    print(f"[{tag}] Loading base model ({model_dtype}, hidden={hidden_size}) from {qwen_base_dir}...", flush=True)
    base_model = AutoModel.from_pretrained(qwen_base_dir, config=config, torch_dtype=model_dtype)
    base_model.to(device)
    base_model.eval()

    name_to_mod = dict(base_model.named_modules())

    with open(os.path.join(qwen_lora_dir, "fold_0", "adapter_config.json"), "r") as f:
        acfg = json.load(f)
    scaling = float(acfg["lora_alpha"]) / float(acfg["r"])

    sample_lora = load_safetensors(os.path.join(qwen_lora_dir, "fold_0", "adapter_model.safetensors"))
    target_map = {}
    for k in sample_lora:
        if k.endswith(".lora_A.weight"):
            prefix = k[:-len(".lora_A.weight")]
            mname = resolve_module_name(prefix, name_to_mod)
            assert mname is not None, f"Could not resolve LoRA module for key: {k}"
            target_map[prefix] = mname
    del sample_lora

    print(f"[{tag}] Snapshotting {len(target_map)} base linear weights on CPU...", flush=True)
    orig_weights = {mname: name_to_mod[mname].weight.data.cpu().clone() for mname in target_map.values()}

    head = make_classifier_head(hidden_size, len(classes)).to(device)
    head.eval()

    probs_qwen3 = np.zeros((len(test_texts), len(classes)), dtype=np.float64)

    for fold in range(5):
        fold_dir = os.path.join(qwen_lora_dir, f"fold_{fold}")
        lora_sd = load_safetensors(os.path.join(fold_dir, "adapter_model.safetensors"))

        for prefix, mname in target_map.items():
            lora_A = lora_sd[prefix + ".lora_A.weight"].to(device, dtype=torch.float32)
            lora_B = lora_sd[prefix + ".lora_B.weight"].to(device, dtype=torch.float32)
            w_orig = orig_weights[mname].to(device, dtype=torch.float32)
            w_merged = w_orig + scaling * (lora_B @ lora_A)
            name_to_mod[mname].weight.data.copy_(w_merged.to(model_dtype))
        del lora_sd

        head_sd = torch.load(os.path.join(fold_dir, "classifier_head.pt"), map_location=device)
        head.load_state_dict(head_sd)
        del head_sd

        fold_probs = []
        with torch.inference_mode():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    out = base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    hidden = out.last_hidden_state
                    bsz = hidden.size(0)
                    last_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0)
                    last_tok = hidden[torch.arange(bsz, device=device), last_idx].float()
                    mask_f = attention_mask.unsqueeze(-1).float()
                    mean_tok = (hidden.float() * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
                    pooled = torch.cat([last_tok, mean_tok], dim=-1)
                    logits = head(pooled)
                probs = torch.softmax(logits.float(), dim=-1)
                fold_probs.append(probs.cpu().numpy())

        probs_qwen3 += np.concatenate(fold_probs, axis=0)[inv_order] / 5.0
        print(f"  - {tag} Fold {fold + 1}/5 inference done ({len(target_map)} LoRA layers merged).", flush=True)

    del base_model, orig_weights, head, test_loader, name_to_mod, target_map
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[{tag}] Complete in {time.time() - t0:.1f}s", flush=True)
    return probs_qwen3


def predict_modernbert_ensemble(test_texts, classes, bert_dir, device, amp_dtype, use_amp, batch_size=48, tag="BERT"):
    t0 = time.time()
    with open(os.path.join(bert_dir, "classes.json"), "r") as f:
        meta = json.load(f)
    assert meta["classes"] == classes, f"{tag} class list mismatch!"

    tokenizer = AutoTokenizer.from_pretrained(bert_dir)
    config = AutoConfig.from_pretrained(bert_dir)
    test_loader, inv_order = make_sorted_loader(test_texts, tokenizer, MAX_LEN, batch_size)

    model = AutoModelForSequenceClassification.from_config(config)
    model.to(device)
    model.eval()

    probs_bert = np.zeros((len(test_texts), len(classes)), dtype=np.float64)
    for fold in range(5):
        state = {k: v.float() for k, v in torch.load(os.path.join(bert_dir, f"fold_{fold}.pt"), map_location="cpu").items()}
        model.load_state_dict(state)
        del state
        fold_probs = []
        with torch.inference_mode():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits.float(), dim=-1)
                fold_probs.append(probs.cpu().numpy())
        probs_bert += np.concatenate(fold_probs, axis=0)[inv_order] / 5.0
        print(f"  - {tag} Fold {fold + 1}/5 inference done.", flush=True)

    del model, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[{tag}] Complete in {time.time() - t0:.1f}s", flush=True)
    return probs_bert


def main():
    t0 = time.time()
    print("=== Name That LLM: Exp #008 4-Model Log-Prob Inference (Qwen3-4B + ModernBERT-large + Qwen3-1.7B + ModernBERT-base) ===", flush=True)

    train = pd.read_csv(find_file("train.csv"))
    test = pd.read_csv(find_file("test.csv"))
    classes = sorted(train["model"].unique().tolist())
    idx2label = {i: c for i, c in enumerate(classes)}

    qwen4b_base_dir = find_qwen_base_dir(2560)
    qwen4b_lora_dir = find_qwen_lora_dir(2560, path_hint="exp007")
    qwen17b_base_dir = find_qwen_base_dir(2048)
    qwen17b_lora_dir = find_qwen_lora_dir(2048, path_hint="exp006")
    mb_large_dir = find_bert_dir(1024, path_hint="exp008")
    mb_base_dir = find_bert_dir(768, path_hint="exp005")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[DEVICE] {device} | AMP={use_amp} ({amp_dtype})", flush=True)

    test_texts = test["text"].values
    probs_4b = predict_qwen3_ensemble(
        test_texts, classes, qwen4b_base_dir, qwen4b_lora_dir, device, amp_dtype, use_amp,
        batch_size=24, tag="QWEN3-4B",
    )
    probs_mb_large = predict_modernbert_ensemble(
        test_texts, classes, mb_large_dir, device, amp_dtype, use_amp,
        batch_size=48, tag="ModernBERT-large",
    )
    probs_17b = predict_qwen3_ensemble(
        test_texts, classes, qwen17b_base_dir, qwen17b_lora_dir, device, amp_dtype, use_amp,
        batch_size=32, tag="QWEN3-1.7B",
    )
    probs_mb_base = predict_modernbert_ensemble(
        test_texts, classes, mb_base_dir, device, amp_dtype, use_amp,
        batch_size=64, tag="ModernBERT-base",
    )

    log_blend = (
        W_QWEN3_4B * np.log(probs_4b + EPS)
        + W_MB_LARGE * np.log(probs_mb_large + EPS)
        + W_QWEN3_17B * np.log(probs_17b + EPS)
        + W_MB_BASE * np.log(probs_mb_base + EPS)
    )
    preds = [idx2label[p] for p in np.argmax(log_blend, axis=1)]

    sub = pd.DataFrame({"id": test["id"], "model": preds})
    sub.to_parquet("submission.parquet", index=False)
    print(f"[DONE] Saved submission.parquet shape={sub.shape} in {time.time() - t0:.1f}s", flush=True)
    print(sub.head(), flush=True)


if __name__ == "__main__":
    main()
