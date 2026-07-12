#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""#311 TropNNC — Laguerre-cell-aware structured reduction of a FROZEN level-set witness trunk.

THE LEVER (the tropical-skeleton rate lever). The #284 deep-math chapters established that at low
softmax temperature the witness is piecewise-linear (tropical / max-plus) and its argmax partition
is a Laguerre power diagram; ``d_seg`` sees ONLY that partition, so any weight change that leaves
every Laguerre cell boundary unchanged is a byte cut at STRUCTURALLY zero ``Δd_seg``. This module is
the algorithmic core (per-layer tropical-dominance ranking + mean-compensated structured neuron
prune) that RANKS candidate reductions; the ACCEPT authority is EMPIRICAL — the full byte-close
decode + SegNet argmax equality on all 600 pairs (see ``tools/witness_apply_pass.py`` stage #311).

HONEST CAVEAT (why the certificate only RANKS, never ACCEPTS). The v752 checkpoint deploys with
``hosc`` activation ``tanh(beta*sin(omega*u))`` and a softmax temperature ``tau`` that is NOT in the
low-``tau`` (max-plus) limit — measured on ``levelset_v752_baseline`` it is ``beta=1.0``,
``omega=1.0``, ``tau=1.0`` (fully soft), and the trunk is DENSE (probe: min per-unit activation std
~0.026, every ``out_sdf`` column norm > 0.8, ZERO dead units). So the closed-form "boundary
unchanged" argument is only APPROXIMATE here; the tropical influence score is a heuristic ranker and
the n600 decode-argmax equality is the sole authority. A reduction is admissible iff it yields
EXACTLY ``Δd_seg == 0`` (near-zero is REJECT — this lever's whole value is exactness).

THE INFLATE CONSTRAINT (why width reduction is uniform, not per-layer). The byte-close inflate
forward (``levelset_rgb_forward_numpy``) reshapes ``film`` to ``(n_hidden, 2, hidden_dim)`` — it
assumes a UNIFORM hidden width across all layers. So a structured width reduction to ``w = H - k``
must drop ``k`` units at EVERY output layer (each layer keeps its own least-influential set of size
``w``); the reduced npz carries ``__cfg_hidden_dim = w`` and byte-closes unchanged. Output layers:
``L0`` = ``in_proj`` output (consumed by ``hidden.0`` columns); ``L1..L_{nH-1}`` = ``hidden.*``
outputs (consumed by the next hidden layer's columns); ``L_nH`` = final hidden output (consumed by
the linear ``out_sdf`` + ``out_tex`` heads). Dropping unit ``j`` at layer ``Lm``:
  * remove the ROW that PRODUCES it (in_proj / hidden.{m-1}) + its bias + its film gamma/beta;
  * MEAN-COMPENSATE its downstream contribution: ``consumer.bias += E[h_Lm_j] * consumer[:, j]``
    (the best constant fold — exact for the linear heads L_nH; a first-order compensation for the
    nonlinear hidden consumers), then remove the consumer COLUMN.
The mean ``E[h_Lm_j]`` is estimated over the probe/eval render (the deploy distribution).

TropNNC LINEAGE: the published TropNNC (Misiakos et al., tropical-geometry structured NN
compression) ranks neurons by their contribution to the layer's max-plus (zonotope) geometry and
prunes/merges the dominated ones. On a ReLU net that contribution is exact; on this near-tropical
``hosc`` witness we use the data-driven surrogate ``std(activation) * ||downstream weight||`` (the
zero-mean fluctuation a neuron injects into the argmax decision) as the dominance rank, and let the
uint8+argmax tolerance of the REAL decode decide admissibility.

AUTHORITY: every derived byte number here is ``[macOS-CPU advisory] NON-PROMOTABLE``. The pointer
moves only through a byte-closed ``upstream/evaluate.py`` n600 exact row. This module is MEANS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# canonical surfaces (reuse — never re-derive the codec/forward math). The reduced witness renders
# through the canonical ``levelset_rgb_forward_numpy`` (the byte-close/inflate forward) unchanged;
# this module only needs the archive quantizer for the byte accounting.
from tac.boundary_math.lever_b_levelset_generator import quantize_levelset_blob

# the weight params the trunk-reduction touches (learned; pose_carrier/__cfg_* are pass-through)
_TRUNK_WEIGHT_KEYS = ("in_proj", "hidden", "out_sdf", "out_tex")


def _act_hosc(u: np.ndarray, beta: float, omega: float) -> np.ndarray:
    """hosc activation ``tanh(beta*sin(omega*u))`` — bit-mirror of the canonical ``_act`` fused form
    (omega==1/beta==1 skip the identity multiplies). Returns float64 (probe-domain, not the fp32
    deploy cast — we only use it for statistics here, never for the shipped .raw)."""
    t = np.sin(u) if omega == 1.0 else np.sin(omega * u)
    if beta != 1.0:
        t = beta * t
    return np.tanh(t)


@dataclass
class WitnessCheckpoint:
    """A loaded level-set witness: float64 weight params + the verbatim ``__cfg_*``/aux arrays."""

    params: dict[str, np.ndarray]  # learned weights (in_proj/film/hidden.*/out_sdf/out_tex/palette/code)
    aux: dict[str, np.ndarray]  # everything else verbatim (pose_carrier.*, __cfg_*, __bank_*, ...)
    cfg: dict[str, object]  # decoded scalars (n_hidden, hidden_dim, activation, softmax_temp, ...)

    @property
    def n_hidden(self) -> int:
        return int(self.cfg["n_hidden"])

    @property
    def hidden_dim(self) -> int:
        return int(self.cfg["hidden_dim"])


def load_witness(npz_path: str | Path) -> WitnessCheckpoint:
    """Load a level-set witness npz into a :class:`WitnessCheckpoint` (float64 weights)."""
    z = np.load(npz_path)
    params: dict[str, np.ndarray] = {}
    aux: dict[str, np.ndarray] = {}
    for k in z.files:
        a = z[k]
        if k.startswith("__") or k.startswith("pose_carrier"):
            aux[k] = a
        else:
            params[k] = np.asarray(a, np.float64)
    for req in ("in_proj.weight", "in_proj.bias", "film.weight", "film.bias",
                "out_sdf.weight", "out_sdf.bias", "out_tex.weight", "out_tex.bias",
                "palette", "code"):
        if req not in params:
            raise ValueError(f"witness npz {npz_path} lacks required learned param {req!r}")
    n_hidden = int(aux.get("__cfg_n_hidden", np.int64(
        sum(1 for k in params if k.startswith("hidden.") and k.endswith(".weight")))))
    hidden_dim = int(aux.get("__cfg_hidden_dim", np.int64(params["in_proj.weight"].shape[0])))
    cfg = {
        "n_hidden": n_hidden,
        "hidden_dim": hidden_dim,
        "n_classes": int(params["out_sdf.weight"].shape[0]),
        "in_feat": int(params["in_proj.weight"].shape[1]),
        "mod_dim": int(params["code"].shape[1]),
        "n_pairs": int(params["code"].shape[0] // 2),
        "activation": str(aux.get("__cfg_activation", np.array("hosc")).item()
                          if hasattr(aux.get("__cfg_activation", "hosc"), "item")
                          else aux.get("__cfg_activation", "hosc")),
        "softmax_temp": float(aux.get("__cfg_softmax_temp", np.float64(1.0))),
        "hosc_beta": float(aux.get("__cfg_hosc_beta", np.float64(1.0))),
        "hosc_omega": float(aux.get("__cfg_hosc_omega", np.float64(1.0))),
        "wire_w0": float(aux.get("__cfg_wire_w0", np.float64(20.0))),
        "wire_s0": float(aux.get("__cfg_wire_s0", np.float64(10.0))),
        "chroma": bool(int(aux.get("__cfg_chroma", np.int64(1)))),
    }
    return WitnessCheckpoint(params=params, aux=aux, cfg=cfg)


def forward_all_layers(
    ck: WitnessCheckpoint, feats: np.ndarray, code_row: np.ndarray
) -> list[np.ndarray]:
    """Forward pass returning the activation stack ``[L0, L1, ..., L_nH]`` (each ``(Npix, width)``).

    ``L0`` = ``act(feats @ in_proj.W.T + b)``; ``L{p+1}`` = ``act((h @ hidden.p.W.T + b)*scale +
    shift)``. Bit-mirrors :func:`levelset_rgb_forward_numpy` up to the heads (hosc only path; the
    v752 witness has no film_pl/concat_pl). Used ONLY for statistics (float64)."""
    p = ck.params
    nH, hd = ck.n_hidden, ck.hidden_dim
    beta, omega = ck.cfg["hosc_beta"], ck.cfg["hosc_omega"]
    feats = np.asarray(feats, np.float64)
    code_row = np.asarray(code_row, np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        h = _act_hosc(feats @ p["in_proj.weight"].T + p["in_proj.bias"], beta, omega)
        stack = [h]
        film = (code_row @ p["film.weight"].T + p["film.bias"]).reshape(nH, 2, hd)
        for li in range(nH):
            pre = (h @ p[f"hidden.{li}.weight"].T + p[f"hidden.{li}.bias"]) * (1.0 + film[li, 0]) + film[li, 1]
            h = _act_hosc(pre, beta, omega)
            stack.append(h)
    return stack


@dataclass
class LayerStats:
    """Per-output-layer unit statistics used to rank tropical dominance.

    ``mean[m]`` / ``std[m]`` are ``(width,)`` over the probe pixels; ``influence[m]`` is
    ``std * ||downstream weight column||`` (the zero-mean argmax-perturbation a unit injects)."""

    mean: list[np.ndarray]
    std: list[np.ndarray]
    influence: list[np.ndarray]
    n_probe_px: int


def probe_layer_stats(
    ck: WitnessCheckpoint, feats: np.ndarray, pair_frame_indices: list[int]
) -> LayerStats:
    """Render the activation stack over ``pair_frame_indices`` (columns of ``code``) and compute
    per-output-layer mean/std + downstream-weighted tropical influence.

    ``feats`` is the (fixed) curvelet bank output for the render grid; ``pair_frame_indices`` are
    indices into ``code`` (``2*pair + frame``). d_seg rides frame1 (``2*pair+1``)."""
    p = ck.params
    nH = ck.n_hidden
    acc_sum: list[np.ndarray] | None = None
    acc_sq: list[np.ndarray] | None = None
    n_px = 0
    for ci in pair_frame_indices:
        stack = forward_all_layers(ck, feats, p["code"][ci])
        if acc_sum is None:
            acc_sum = [np.zeros(s.shape[1]) for s in stack]
            acc_sq = [np.zeros(s.shape[1]) for s in stack]
        for m, s in enumerate(stack):
            acc_sum[m] += s.sum(axis=0)
            acc_sq[m] += (s * s).sum(axis=0)
        n_px += stack[0].shape[0]
    assert acc_sum is not None and acc_sq is not None
    means, stds, infl = [], [], []
    for m in range(nH + 1):
        mean = acc_sum[m] / n_px
        var = np.maximum(acc_sq[m] / n_px - mean * mean, 0.0)
        std = np.sqrt(var)
        # downstream consumer column norm for output layer Lm
        if m == 0:
            dn = np.linalg.norm(p["hidden.0.weight"], axis=0)  # cols index L0 units
        elif m < nH:
            dn = np.linalg.norm(p[f"hidden.{m}.weight"], axis=0)  # cols index L{m} units
        else:  # m == nH: final hidden output feeds the two linear heads
            dn = np.linalg.norm(p["out_sdf.weight"], axis=0) + np.linalg.norm(p["out_tex.weight"], axis=0)
        means.append(mean)
        stds.append(std)
        infl.append(std * dn)
    return LayerStats(mean=means, std=stds, influence=infl, n_probe_px=n_px)


def select_kept_sets(stats: LayerStats, n_hidden: int, k: int) -> dict[int, np.ndarray]:
    """Drop the ``k`` least-influential units at EACH output layer (uniform width ``H-k``).

    Returns ``{output_layer_index: kept_unit_indices (sorted ascending)}`` for layers ``0..n_hidden``.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    kept: dict[int, np.ndarray] = {}
    for m in range(n_hidden + 1):
        infl = stats.influence[m]
        width = infl.shape[0]
        if k == 0:
            kept[m] = np.arange(width)
            continue
        if k >= width:
            raise ValueError(f"k={k} >= layer {m} width {width}: cannot drop all units")
        drop = np.argsort(infl, kind="stable")[:k]  # smallest influence first
        keep = np.setdiff1d(np.arange(width), drop, assume_unique=False)
        kept[m] = np.sort(keep)
    return kept


def apply_reduction(
    ck: WitnessCheckpoint, kept: dict[int, np.ndarray], stats: LayerStats
) -> tuple[dict[str, np.ndarray], int]:
    """Apply a uniform-width structured reduction with mean-compensation.

    ``kept[m]`` = surviving unit indices at output layer ``Lm`` (all same size ``w``). ``stats.mean``
    provides ``E[h_Lm]`` for the bias fold. Returns ``(reduced_params_float64, new_width)``.
    """
    p = ck.params
    nH = ck.n_hidden
    widths = {m: kept[m].shape[0] for m in kept}
    w = widths[0]
    if any(v != w for v in widths.values()):
        raise ValueError(f"non-uniform kept widths {widths}: inflate needs uniform hidden_dim")
    r: dict[str, np.ndarray] = {k: v.copy() for k, v in p.items()}

    def _dropped(m: int) -> np.ndarray:
        return np.setdiff1d(np.arange(stats.mean[m].shape[0]), kept[m], assume_unique=True)

    # --- L0: in_proj rows + hidden.0 input columns (fold into hidden.0.bias) ---
    d0 = _dropped(0)
    if d0.size:
        r["hidden.0.bias"] = r["hidden.0.bias"] + (p["hidden.0.weight"][:, d0] * stats.mean[0][d0]).sum(axis=1)
    r["in_proj.weight"] = p["in_proj.weight"][kept[0], :]
    r["in_proj.bias"] = p["in_proj.bias"][kept[0]]
    r["hidden.0.weight"] = r["hidden.0.weight"][:, kept[0]]

    # --- L1..L_{nH-1}: hidden.{p-1} rows + hidden.p input cols (fold into hidden.p.bias) ---
    for m in range(1, nH):
        dm = _dropped(m)
        # rows that PRODUCE Lm live in hidden.{m-1}
        r[f"hidden.{m-1}.weight"] = r[f"hidden.{m-1}.weight"][kept[m], :]
        r[f"hidden.{m-1}.bias"] = r[f"hidden.{m-1}.bias"][kept[m]]
        # columns that CONSUME Lm live in hidden.{m}
        if dm.size:
            r[f"hidden.{m}.bias"] = r[f"hidden.{m}.bias"] + (
                p[f"hidden.{m}.weight"][:, dm] * stats.mean[m][dm]).sum(axis=1)
        r[f"hidden.{m}.weight"] = r[f"hidden.{m}.weight"][:, kept[m]]

    # --- L_nH (final hidden output): hidden.{nH-1} rows + out_sdf/out_tex cols (fold into head bias) ---
    dN = _dropped(nH)
    r[f"hidden.{nH-1}.weight"] = r[f"hidden.{nH-1}.weight"][kept[nH], :]
    r[f"hidden.{nH-1}.bias"] = r[f"hidden.{nH-1}.bias"][kept[nH]]
    if dN.size:
        r["out_sdf.bias"] = r["out_sdf.bias"] + (p["out_sdf.weight"][:, dN] * stats.mean[nH][dN]).sum(axis=1)
        r["out_tex.bias"] = r["out_tex.bias"] + (p["out_tex.weight"][:, dN] * stats.mean[nH][dN]).sum(axis=1)
    r["out_sdf.weight"] = r["out_sdf.weight"][:, kept[nH]]
    r["out_tex.weight"] = r["out_tex.weight"][:, kept[nH]]

    # --- film: (nH*2*hd, 32) -> (nH*2*w, 32). film[li] modulates hidden.li pre-act = output L{li+1}. ---
    hd = ck.hidden_dim
    fw = p["film.weight"].reshape(nH, 2, hd, -1)
    fb = p["film.bias"].reshape(nH, 2, hd)
    new_fw = np.stack([fw[li][:, kept[li + 1], :] for li in range(nH)], axis=0).reshape(nH * 2 * w, -1)
    new_fb = np.stack([fb[li][:, kept[li + 1]] for li in range(nH)], axis=0).reshape(nH * 2 * w)
    r["film.weight"] = new_fw
    r["film.bias"] = new_fb
    return r, w


def write_reduced_npz(
    ck: WitnessCheckpoint, reduced_params: dict[str, np.ndarray], new_width: int, out_path: str | Path
) -> Path:
    """Write a reduced witness npz that ``levelset_byte_close_and_eval.py`` consumes verbatim.

    All ``__cfg_*``/``__bank_*``/``pose_carrier.*`` arrays are preserved EXACTLY except
    ``__cfg_hidden_dim`` which is overridden to ``new_width``. Weights are cast to float32 (the npz
    storage dtype of the original checkpoint)."""
    out_path = Path(out_path)
    out: dict[str, np.ndarray] = {}
    for k, v in reduced_params.items():
        out[k] = np.asarray(v, np.float32)
    for k, v in ck.aux.items():
        out[k] = np.asarray(v)
    out["__cfg_hidden_dim"] = np.asarray(new_width, dtype=np.int64)
    np.savez(out_path, **out)
    return out_path


def trunk_blob_bytes(params: dict[str, np.ndarray]) -> int:
    """MEASURED int8+brotli trunk+code blob bytes (the counted rate term for the learned trunk).

    Uses the canonical :func:`quantize_levelset_blob` over the learned weights + ``code`` (the
    pose_carrier/lane sections are unchanged by a trunk reduction, so this delta == the archive
    delta up to the fixed sections). Exact byte count (noise floor 0)."""
    learned = {k: np.asarray(v, np.float32) for k, v in params.items()
               if any(k == f"{g}.weight" or k == f"{g}.bias" or k.startswith("hidden.")
                      for g in _TRUNK_WEIGHT_KEYS) or k in ("film.weight", "film.bias", "palette", "code")}
    rep = quantize_levelset_blob(learned)
    return int(rep["total_quantized_blob_bytes"])


@dataclass
class ReductionPlan:
    """A ranked reduction candidate: k units/layer dropped, the kept sets, and its byte accounting."""

    k: int
    new_width: int
    kept: dict[int, np.ndarray]
    baseline_blob_bytes: int
    reduced_blob_bytes: int
    dropped_params: int
    influence_dropped: list[float] = field(default_factory=list)

    @property
    def bytes_saved(self) -> int:
        return self.baseline_blob_bytes - self.reduced_blob_bytes

    def to_json_dict(self) -> dict:
        return {
            "k_per_layer": self.k,
            "new_width": self.new_width,
            "baseline_blob_bytes": self.baseline_blob_bytes,
            "reduced_blob_bytes": self.reduced_blob_bytes,
            "trunk_blob_bytes_saved": self.bytes_saved,
            "dropped_params": self.dropped_params,
            "max_influence_dropped": max(self.influence_dropped) if self.influence_dropped else 0.0,
            "kept_widths": {int(m): int(v.shape[0]) for m, v in self.kept.items()},
        }


def build_reduction_plan(ck: WitnessCheckpoint, stats: LayerStats, k: int) -> tuple[ReductionPlan, dict[str, np.ndarray]]:
    """Rank + apply a k-per-layer reduction; return the plan (byte accounting) + reduced params."""
    kept = select_kept_sets(stats, ck.n_hidden, k)
    reduced, w = apply_reduction(ck, kept, stats)
    base_bytes = trunk_blob_bytes(ck.params)
    red_bytes = trunk_blob_bytes(reduced)
    n_before = sum(int(np.prod(v.shape)) for v in ck.params.values())
    n_after = sum(int(np.prod(v.shape)) for v in reduced.values())
    infl_dropped: list[float] = []
    for m in range(ck.n_hidden + 1):
        drop = np.setdiff1d(np.arange(stats.influence[m].shape[0]), kept[m], assume_unique=True)
        infl_dropped.extend(float(x) for x in stats.influence[m][drop])
    plan = ReductionPlan(
        k=k, new_width=w, kept=kept, baseline_blob_bytes=base_bytes,
        reduced_blob_bytes=red_bytes, dropped_params=n_before - n_after,
        influence_dropped=infl_dropped)
    return plan, reduced
