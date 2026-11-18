"""
005_name-that-llm: Exp #010 Phase 1 - Extract GLM-5.3-Flash & GLM-5.3 Likelihood Features on train.csv
Runs offline on Kaggle GPU T4 x2 using prebuilt ntl-segrun and GGUF segment mirrors.
MAX_TEXT_TOKENS=320 preserves exact L160 scores in s[:160] as well as L320 scores!
"""
import base64
import glob
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
MAX_TEXT_TOKENS = 320
SEQ_BATCH = 8
STATE_DTYPE = "auto"
SESSION_LIMIT_S = 11.2 * 3600
PREFIX = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|><|assistant|><think></think>"
NTL_SEGRUN_SHA256 = "9c39e3afc77c64c1930f438fef50f297a6a8a649252ab4af9f618804be84cc8c"

CODE_PAYLOAD_B64 = "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"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def find_train_csv():
    cands = [p for p in glob.glob("/kaggle/input/**/train.csv", recursive=True) if "name-that-llm" in p]
    if not cands:
        cands = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    assert cands, "train.csv not found under /kaggle/input"
    return cands[0]

def setup_ntl_segrun():
    cands = sorted(glob.glob("/kaggle/input/**/ntl-bin/ntl-segrun", recursive=True))
    ok = [
        p for p in cands
        if sha256_file(p) == NTL_SEGRUN_SHA256 and Path(p).with_name("lib").joinpath("libcublasLt.so.13").exists()
    ]
    assert ok, f"Valid ntl-segrun not found among {cands}"
    src_bin = Path(ok[0])
    bin_dir = Path("/kaggle/temp/v1-bin")
    bin_dir.mkdir(parents=True, exist_ok=True)
    dst_bin = bin_dir / "ntl-segrun"
    shutil.copy(src_bin, dst_bin)
    os.chmod(dst_bin, 0o755)
    lib_dir = src_bin.with_name("lib")
    env = dict(os.environ, LD_LIBRARY_PATH=f"{lib_dir}:" + os.environ.get("LD_LIBRARY_PATH", ""))
    return bin_dir, env

def link_segments(pattern, expected_count, name):
    seen = {}
    for p in sorted(glob.glob("/kaggle/input/**/" + pattern, recursive=True)):
        seen.setdefault(os.path.basename(p), p)
    assert len(seen) == expected_count, f"{name}: found {len(seen)} segments, expected {expected_count}"
    out_dir = Path(f"/kaggle/temp/v1-segs-{name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for base, p in seen.items():
        dst = out_dir / base
        if not dst.exists():
            dst.symlink_to(p)
    return out_dir

def export_sliced_features(scores_npz, train_df, out_csv, tok_cap):
    sys.path.insert(0, "/kaggle/temp/v1-code")
    import features as feat_mod
    z = np.load(scores_npz, allow_pickle=True)
    ids = z["ids"].astype(str)
    scores_all = z["scores"]
    offsets = z["offsets"]
    text_map = dict(zip(train_df["id"].astype(str), train_df["text"].fillna("").astype(str)))
    seq_list = [scores_all[offsets[i]:offsets[i + 1]][:tok_cap] for i in range(len(ids))]
    texts = [text_map[i] for i in ids]
    feats = feat_mod.text_features(seq_list, texts)
    feats.insert(0, "id", ids)
    feats.to_csv(out_csv, index=False)
    print(f"[EXPORT] Wrote {len(feats)} rows (cap={tok_cap}) -> {out_csv}", flush=True)

def main():
    os.system("nvidia-smi")
    train_csv = find_train_csv()
    train_df = pd.read_csv(train_csv)
    print(f"[DATA] train_csv={train_csv} rows={len(train_df)}", flush=True)

    code_dir = Path("/kaggle/temp/v1-code")
    code_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_PAYLOAD_B64)), mode="r:gz") as tf:
        tf.extractall(code_dir)

    bin_dir, v1_env = setup_ntl_segrun()
    flash_dir = link_segments("flash_iq3xxs-seg*-of-06-*.gguf", 6, "flash")
    glm53_dir = link_segments("glm53_iq3xxs-seg*-of-19-*.gguf", 19, "glm53")
    print("[SETUP] Binary + CUDA13 lib + 6 Flash + 19 GLM-5.3 segments ready.", flush=True)

    out_dir = Path("/kaggle/working/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path("/kaggle/temp/v1-work")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Flash already completed & saved locally; extract full train GLM-5.3 features now (~2.9h)
    for tag, seg_dir in [("glm53", glm53_dir)]:
        left_s = SESSION_LIMIT_S - (time.time() - T0)
        print(f"\n[RUN {tag.upper()}] Starting full train scoring ({len(train_df)} rows, max_tok={MAX_TEXT_TOKENS}, left={left_s:.0f}s)...", flush=True)
        t_start = time.time()
        cmd = [
            sys.executable,
            str(code_dir / "run_pipeline.py"),
            "--bin", str(bin_dir),
            "--segs", str(seg_dir / "*.gguf"),
            "--csv", train_csv,
            "--out", str(out_dir),
            "--tag", tag,
            "--work", str(work_dir),
            "--seq-batch", str(SEQ_BATCH),
            "--state-dtype", STATE_DTYPE,
            "--max-text-tokens", str(MAX_TEXT_TOKENS),
            "--deadline-s", f"{left_s:.0f}",
            "--prefix", PREFIX,
        ]
        res = subprocess.run(cmd, env=v1_env)
        assert res.returncode == 0, f"{tag} pipeline failed with exit code {res.returncode}"
        dt = time.time() - t_start
        print(f"[RUN {tag.upper()}] Finished in {dt/3600:.2f}h ({dt:.0f}s)", flush=True)

        # Print per-model sanity check (glm-5.3 / glm-5.3-flash should score highest log-prob!)
        f_df = pd.read_csv(out_dir / f"{tag}_features.csv").astype({"id": str})
        m_df = f_df.merge(train_df[["id", "model"]].astype({"id": str}), on="id")
        print(f"\n[SUMMARY {tag.upper()} L320]")
        print(m_df.groupby("model")[["lp_mean", "ent_mean", "top1", "fdg"]].mean().sort_values("lp_mean", ascending=False).round(4).to_string())

        export_sliced_features(
            out_dir / f"{tag}_scores.npz",
            train_df,
            out_dir / f"{tag}_features_L160.csv",
            160,
        )

    print(f"\n[DONE] Total elapsed: {(time.time() - T0)/3600:.2f}h", flush=True)

if __name__ == "__main__":
    main()
