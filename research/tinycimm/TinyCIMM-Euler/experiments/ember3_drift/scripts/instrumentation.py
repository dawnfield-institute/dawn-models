"""
Ember III drift instrumentation — external, side-effect-free measurement.

Everything here deliberately bypasses the substrate's forward() (which mutates
micro_memory/math_memory and advances complexity_factor — port-audit findings
7-8) by computing h = relu(x W^T + b), y = h V^T + c directly under no_grad.

The drift metric is linear CKA over a fixed probe set: dimension-agnostic, so
it is well-defined across grow/prune events that change the hidden width
(port-audit findings 5-6 make neuron-indexed comparison impossible by design).
"""

import copy
import io
from contextlib import redirect_stdout

import numpy as np
import torch


# ---------------------------------------------------------------- data stream

def sieve_primes(limit):
    """Primes up to limit via numpy sieve."""
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i:: i] = False
    return np.flatnonzero(sieve)


def prime_gaps(limit=3_200_000):
    """Gap sequence g_i = p_{i+1} - p_i, float32."""
    primes = sieve_primes(limit)
    return np.diff(primes).astype(np.float32)


def make_windows(gaps, indices, window):
    """X[i] = gaps[idx:idx+window], y[i] = gaps[idx+window]. indices are window
    start positions; caller guarantees idx+window < len(gaps)."""
    X = np.stack([gaps[i: i + window] for i in indices])
    y = np.array([[gaps[i + window]] for i in indices], dtype=np.float32)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ----------------------------------------------------- side-effect-free model IO

def _act(model, z):
    """Use the model's activation if it defines one (repair-ladder subclass),
    else the substrate's ReLU."""
    fn = getattr(model, "_act", None)
    return fn(z) if callable(fn) else torch.relu(z)


@torch.no_grad()
def hidden_reps(model, X):
    """Hidden representation on probe inputs, no side effects. (n, hidden_dim)."""
    h = _act(model, X.to(model.device) @ model.W.T + model.b)
    return h.detach().cpu().double().numpy()


@torch.no_grad()
def clean_predict(model, X):
    """Leak-free prediction: no higher_order_transform, no state mutation."""
    h = _act(model, X.to(model.device) @ model.W.T + model.b)
    y = h @ model.V.T + model.c
    return y.detach().cpu()


# ------------------------------------------------------------------ drift (CKA)

def linear_cka(H1, H2):
    """Linear CKA between two representation matrices (n_probes, dim1/dim2).
    Dimension-agnostic; returns a value in [0, 1] (1 = identical geometry)."""
    H1 = H1 - H1.mean(axis=0, keepdims=True)
    H2 = H2 - H2.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(H1.T @ H2, ord="fro") ** 2
    n1 = np.linalg.norm(H1.T @ H1, ord="fro")
    n2 = np.linalg.norm(H2.T @ H2, ord="fro")
    denom = n1 * n2
    if denom < 1e-12:
        return float("nan")
    return float(cross / denom)


def rep_stats(H):
    """Representation-health diagnostics for a probe-set hidden matrix
    (n_probes, hidden_dim). dead_frac = fraction of units with zero activation
    on EVERY probe (ReLU-dead for the probe distribution); const_frac = fraction
    of units with (near-)zero variance across probes (dead or saturated-constant
    — these contribute nothing to the representation's geometry)."""
    dead = (H <= 0).all(axis=0).mean()
    const = (H.std(axis=0) < 1e-9).mean()
    return {"dead_frac": float(dead), "const_frac": float(const),
            "h_std": float(H.std()), "h_absmax": float(np.abs(H).max())}


# ------------------------------------------------------------ plasticity probe

def safe_deepcopy(model):
    """Deepcopy the model despite the substrate storing non-leaf tensors from
    the last forward (last_h has a grad_fn, which torch refuses to deepcopy)."""
    saved = {}
    for attr in ("last_h", "last_x", "last_prediction"):
        v = getattr(model, attr, None)
        if torch.is_tensor(v):
            saved[attr] = v
            setattr(model, attr, None)
    try:
        return copy.deepcopy(model)
    finally:
        for k, v in saved.items():
            setattr(model, k, v)


def plasticity_probe(model, seg_X, seg_y, adapt_steps=200, edge=50):
    """Can the system still learn NEW structure right now?

    Deepcopies the model (parameters + optimizer + controller travel together, so
    the clone adapts under the same regime as its arm), adapts it on a fresh
    never-seen segment one token at a time, and reports the relative reduction in
    leak-free prequential MAE from the first `edge` tokens to the last `edge`.
    The clone is discarded; the measured model is never touched.
    """
    clone = safe_deepcopy(model)
    errs = []
    sink = io.StringIO()
    for i in range(min(adapt_steps, len(seg_X))):
        x = seg_X[i: i + 1].to(clone.device)
        y = seg_y[i: i + 1].to(clone.device)
        err = float(torch.abs(clean_predict(clone, x) - y.cpu()).mean())
        errs.append(err)
        # Mirror the live loop: arms whose controller consumes the error stream
        # keep observing during the probe (Arm C's observe() consumes surrogate).
        if hasattr(clone.structure_controller, "observe"):
            clone.structure_controller.observe(err)
        with redirect_stdout(sink):
            clone.online_adaptation_step(x, y)
        sink.truncate(0), sink.seek(0)
    first = float(np.mean(errs[:edge]))
    last = float(np.mean(errs[-edge:]))
    del clone
    return {
        "mae_first": first,
        "mae_last": last,
        "plasticity": (first - last) / (first + 1e-9),
    }
