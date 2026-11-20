"""
Two-Way Prompt-Centered Gaussian Covariance Head & Joint Inference (Gauss Head).
Fits within-prompt residual covariance (Se) and cross-prompt topic covariance (Sp)
with Ledoit-Wolf shrinkage, transforms embeddings into generalized-eigenvector space,
and computes exact marginals (size <= n_exact) or repulsive EM marginals for clusters.
"""

import itertools
import numpy as np

K_CLASSES = 18
DEFAULT_CFG = {
    "red": "wpca",
    "d": 128,
    "se_shrink": "lw",
    "sp_shrink": "lw",
    "lam_floor": 1e-4,
    "n_exact": 3,
    "inj": True,
    "iters": 5,
    "repel": True,
    "sp_scale": 1.0,
}

_MASK_CACHE = {}


def _lse(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    if keepdims:
        return out
    return np.squeeze(out, axis=axis) if axis is not None else float(out.item())


def log_softmax(logits):
    return logits - _lse(logits, axis=1, keepdims=True)


def _lw_shrinkage(x):
    from sklearn.covariance import ledoit_wolf_shrinkage
    return float(ledoit_wolf_shrinkage(x, assume_centered=True))


def _shrunk(cov, alpha):
    d = cov.shape[0]
    return (1.0 - alpha) * cov + alpha * (np.trace(cov) / d) * np.eye(d)


def _two_way(z, y, pinv, n_groups, iters=4):
    p_mat = np.zeros((n_groups, z.shape[1]), dtype=np.float64)
    cnt_p = np.bincount(pinv, minlength=n_groups).astype(np.float64)
    cls = np.unique(y)
    mu = np.zeros((K_CLASSES, z.shape[1]), dtype=np.float64)
    for _ in range(iters):
        res = z - p_mat[pinv]
        for c in cls:
            mu[c] = res[y == c].mean(0)
        res = z - mu[y]
        p_mat.fill(0.0)
        np.add.at(p_mat, pinv, res)
        p_mat /= cnt_p[:, None]
    return mu, p_mat, cnt_p


def fit(x, y, prompts, cfg=None):
    c = dict(DEFAULT_CFG)
    c.update(cfg or {})
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n, d_raw = x.shape
    pu, pinv = np.unique(np.asarray(prompts), return_inverse=True)
    g = len(pu)
    mx = x.mean(0)
    sx = x.std(0)
    sx = np.where(sx > 1e-8, sx, 1.0)
    xs = (x - mx) / sx

    if c["red"] == "none":
        v_proj = np.eye(d_raw)
    else:
        if c["red"] == "wpca":
            pm = np.zeros((g, d_raw), dtype=np.float64)
            np.add.at(pm, pinv, xs)
            pm /= np.bincount(pinv)[:, None]
            r_mat = xs - pm[pinv]
        elif c["red"] == "pca":
            r_mat = xs
        else:
            raise ValueError(f"Unknown red={c['red']}")
        d = int(min(c["d"], n - g - 1, d_raw))
        if n < d_raw:
            w, u = np.linalg.eigh(r_mat @ r_mat.T)
            order = np.argsort(w)[::-1][:d]
            w, u = np.clip(w[order], 1e-12, None), u[:, order]
            v_proj = (r_mat.T @ u) / np.sqrt(w)
        else:
            _, _, vt = np.linalg.svd(r_mat, full_matrices=False)
            v_proj = vt[:d].T

    z = xs @ v_proj
    d = z.shape[1]
    mu, p_mat, ng = _two_way(z, y, pinv, g)
    e_res = z - mu[y] - p_mat[pinv]
    dof = max(n - g - len(np.unique(y)) + 1, 1)
    se_raw = e_res.T @ e_res / dof
    ae = _lw_shrinkage(e_res) if c["se_shrink"] == "lw" else float(c["se_shrink"])
    se = _shrunk(se_raw, ae)

    pc = p_mat - p_mat.mean(0)
    sp_raw = pc.T @ pc / max(g - 1, 1) - np.mean(1.0 / ng) * se_raw
    ap = _lw_shrinkage(pc) if c["sp_shrink"] == "lw" else float(c["sp_shrink"])
    tr = max(np.trace(sp_raw), 1e-8 * np.trace(se))
    sp = ((1.0 - ap) * sp_raw + ap * (tr / d) * np.eye(d)) * c["sp_scale"]

    chol = np.linalg.cholesky(se)
    chol_inv = np.linalg.inv(chol)
    m_mat = chol_inv @ sp @ chol_inv.T
    m_mat = 0.5 * (m_mat + m_mat.T)
    lam, q_mat = np.linalg.eigh(m_mat)
    lam = np.clip(lam, c["lam_floor"], None)
    a_mat = q_mat.T @ chol_inv
    w_proj = v_proj @ a_mat.T
    return {
        "mx": mx,
        "sx": sx,
        "W": w_proj,
        "m": mu @ a_mat.T,
        "lam": lam,
        "T1": 1.0,
        "Tg": 1.0,
        "info": {"d": int(d), "a_e": round(ae, 4), "a_p": round(ap, 4)},
        "cfg": c,
    }


def transform(x, prm):
    return ((np.asarray(x, dtype=np.float64) - prm["mx"]) / prm["sx"]) @ prm["W"]


def _inj_mask(n, c_cls):
    if (n, c_cls) not in _MASK_CACHE:
        grids = np.meshgrid(*[np.arange(c_cls)] * n, indexing="ij")
        bad = np.zeros([c_cls] * n, dtype=bool)
        for i, j in itertools.combinations(range(n), 2):
            bad |= grids[i] == grids[j]
        _MASK_CACHE[(n, c_cls)] = bad
    return _MASK_CACHE[(n, c_cls)]


def singleton_ll(u, m, lam):
    r = u[:, None, :] - m[None]
    return -0.5 * (r ** 2 / (1.0 + lam)).sum(-1)


def joint_terms(u, m, lam):
    n = len(u)
    gam = lam / (1.0 + n * lam)
    r = u[:, None, :] - m[None]
    a = -0.5 * (r ** 2 * (1.0 - gam)).sum(-1)
    g_mat = r * np.sqrt(gam)
    b = np.einsum("icd,jed->ijce", g_mat, g_mat)
    return a, b


def joint_table(a, b):
    n, c_cls = a.shape
    q_tab = np.zeros([c_cls] * n, dtype=np.float64)
    for i in range(n):
        sh = [1] * n
        sh[i] = c_cls
        q_tab = q_tab + a[i].reshape(sh)
    for i, j in itertools.combinations(range(n), 2):
        sh = [1] * n
        sh[i] = c_cls
        sh[j] = c_cls
        q_tab = q_tab + b[i, j].reshape(sh)
    return q_tab


def exact_marginals(q_tab, temp=1.0, inj=True):
    n = q_tab.ndim
    c_cls = q_tab.shape[0]
    qt = q_tab / temp
    if inj and n >= 2:
        qt = np.where(_inj_mask(n, c_cls), -np.inf, qt)
    lz = _lse(qt)
    out = np.empty((n, c_cls), dtype=np.float64)
    for i in range(n):
        ax = tuple(k for k in range(n) if k != i)
        out[i] = (_lse(qt, axis=ax) if ax else qt) - lz
    return out


def em_marginals(u, m, lam, temp=1.0, iters=5, repel=False):
    n = len(u)
    v = lam / (1.0 + (n - 1) * lam)
    logits = singleton_ll(u, m, lam) / temp
    q = np.exp(log_softmax(logits))
    for _ in range(iters):
        s = u - q @ m
        s_sum = s.sum(0)
        pm = (s_sum[None] - s) * v
        r = u[:, None, :] - pm[:, None, :] - m[None]
        logits = -0.5 * (r ** 2 / (1.0 + v)).sum(-1) / temp
        if repel:
            lo = np.log(np.clip(1.0 - q, 1e-6, None))
            logits = logits + (lo.sum(0)[None] - lo)
        q = np.exp(log_softmax(logits))
    return log_softmax(logits)


def group_logpost(u, prm, t1=None, tg=None, cfg=None):
    c = dict(prm.get("cfg", DEFAULT_CFG))
    c.update(cfg or {})
    n = len(u)
    lg = np.full_like(prm["lam"], 1e6) if c.get("diff") else prm["lam"]
    t1 = prm["T1"] if t1 is None else t1
    tg = prm["Tg"] if tg is None else tg
    if n == 1:
        return log_softmax(singleton_ll(u, prm["m"], prm["lam"]) / t1)
    if n <= c["n_exact"]:
        a, b = joint_terms(u, prm["m"], lg)
        return exact_marginals(joint_table(a, b), tg, c["inj"])
    return em_marginals(u, prm["m"], lg, tg, c["iters"], c["repel"])


def scores(x_pool, lab, prm, t1=None, tg=None, cfg=None, u=None):
    u = transform(x_pool, prm) if u is None else u
    lab = np.asarray(lab)
    n = len(u)
    out = np.empty((n, prm["m"].shape[0]), dtype=np.float64)
    sizes = np.bincount(lab)
    s1 = sizes[lab] == 1
    t1 = prm["T1"] if t1 is None else t1
    if s1.any():
        out[s1] = log_softmax(singleton_ll(u[s1], prm["m"], prm["lam"]) / t1)
    for g in np.where(sizes >= 2)[0]:
        ii = np.where(lab == g)[0]
        out[ii] = group_logpost(u[ii], prm, t1, tg, cfg)
    return out
