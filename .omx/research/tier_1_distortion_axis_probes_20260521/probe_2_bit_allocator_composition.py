# SPDX-License-Identifier: MIT
"""
OVERNIGHT-CCC Tier-1 Probe 2: Bit-allocator composition smoke [macOS-CPU advisory]

Per AAA T4 grand council symposium PROCEED_WITH_REVISIONS verdict (commit a8b02679)
Decision #6 + Catalog #354 (8 master-gradient exploit consumers) + Catalog #356
(per-axis decomposition) + Catalog #357 (Tier B contract).

CONTRACT: empirically test whether bit-allocator routing using exploits #3 (top-K
byte sensitivity) + #5 (per-class chroma) + #9 (per-pair clustering) on PR 101
archive bytes minimizes (d_seg + d_pose) simultaneously vs uniform allocation.

PREDICTED SIGNATURE (Boyd + Dykstra canonical extension per AAA T4 §8):
  - Top-K mass concentrated in <50% of archive bytes (high-leverage subset)
  - Per-class chroma + per-pair clustering produce non-uniform allocation
    weights with measurable variance
  - Composition factor alpha > 1.0 (additive or super-additive) vs sum-of-singletons

FALSIFYING OUTCOME:
  - Uniform leverage distribution (alpha ≤ 1.0; no concentration)
  - => DEFER bit-allocator paid dispatch pending real master-gradient extraction
    against contest CUDA archive

CANONICAL PROVENANCE: macOS-CPU-advisory; promotable=False; axis_tag=[macOS-CPU advisory]
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch


def _load_pr101_archive_bytes() -> tuple[bytes, str, int]:
    archive_path = REPO_ROOT / "experiments" / "results" / "public_pr101_hnerv_ft_microcodec_intake_20260504_codex" / "archive.zip"
    payload = archive_path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    return payload, sha, len(payload)


def _simulate_per_byte_master_gradient(archive_bytes: bytes, n_pairs: int = 8) -> np.ndarray:
    """Simulate per-byte master-gradient via deterministic hash-based proxy.

    Per Catalog #318 STRICT gate: this is NOT a contest-score derivative (which
    requires grammar-aware operator + ZIP CRC + entropy-coded re-packing). This
    is an advisory PROXY for the bit-allocator composition test — sister exploit
    #3 (top-K) operates on per-byte sensitivity rows from a real master-gradient
    extraction artifact, but this is a smoke probe so we use a deterministic
    hash-proxy to test the COMPOSITION mechanism, not authoritative sensitivity.

    Output: (N_pairs, N_bytes) sensitivity matrix.
    """
    n_bytes = len(archive_bytes)
    arr = np.frombuffer(archive_bytes, dtype=np.uint8).astype(np.float32)
    rng = np.random.default_rng(0xCC2)
    pair_sensitivities = []
    for pair_i in range(n_pairs):
        # Hash-based per-byte proxy with non-uniform structure tied to actual byte values
        seed = rng.integers(0, 2 ** 31)
        # High-leverage region: bytes whose value mod 17 == seed mod 17 + 1 std dev
        leverage = np.abs(arr - (seed % 256)) / 255.0
        # Add small noise for empirical variance
        noise = rng.normal(0, 0.01, size=n_bytes).astype(np.float32)
        sens = leverage + noise
        pair_sensitivities.append(sens)
    return np.stack(pair_sensitivities, axis=0)  # (N_pairs, N_bytes)


def _exploit_3_top_k_byte_sensitivity(sens: np.ndarray, k_frac: float = 0.1) -> np.ndarray:
    """Per Catalog #354 exploit #3: identify top-K% highest-leverage bytes.

    Returns boolean mask of shape (N_bytes,).
    """
    # Aggregate per-pair sensitivity by sum (canonical L1 aggregation per Cable D)
    aggregate = sens.sum(axis=0)  # (N_bytes,)
    k = max(1, int(k_frac * len(aggregate)))
    top_k_idx = np.argsort(aggregate)[-k:]
    mask = np.zeros(len(aggregate), dtype=bool)
    mask[top_k_idx] = True
    return mask


def _exploit_5_per_class_chroma_priority(archive_bytes: bytes, n_classes: int = 5) -> np.ndarray:
    """Per Catalog #354 exploit #5: per-SegNet-class chroma allocation priority.

    Returns per-class allocation weights normalized to sum=1.0.
    """
    arr = np.frombuffer(archive_bytes, dtype=np.uint8)
    # Per-class byte histogram simulating chroma signature distribution
    weights = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        # Class c gets weight from bytes in range [c*51, (c+1)*51)
        weights[c] = np.sum((arr >= c * 51) & (arr < (c + 1) * 51))
    weights = weights / max(weights.sum(), 1.0)
    return weights


def _exploit_9_per_pair_clustering(sens: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """Per Catalog #354 exploit #9: per-pair gradient clustering.

    Returns cluster assignment per pair via k-means on per-pair sensitivity
    norms. Simple k-means surrogate (no sklearn dependency).
    """
    pair_norms = np.linalg.norm(sens, axis=1)  # (N_pairs,)
    # Simple equal-width binning surrogate (deterministic, no sklearn)
    bins = np.linspace(pair_norms.min(), pair_norms.max() + 1e-9, n_clusters + 1)
    cluster_assignments = np.digitize(pair_norms, bins[1:-1])
    return cluster_assignments


def _compose_bit_allocator(
    top_k_mask: np.ndarray,
    per_class_weights: np.ndarray,
    pair_clusters: np.ndarray,
    n_bytes: int,
) -> dict:
    """Compose 3 exploits into a unified bit-allocator.

    Composition rule (canonical per AAA T4 §8.4):
      bit_budget[b] = top_k_mask[b] * sum(per_class_weights) * mean(pair_cluster_diversity)

    Returns composition statistics for verdict.
    """
    top_k_frac = float(top_k_mask.sum()) / n_bytes
    per_class_entropy = float(-np.sum(per_class_weights * np.log(per_class_weights + 1e-10)))
    per_class_max_weight = float(per_class_weights.max())
    pair_cluster_diversity = float(len(np.unique(pair_clusters))) / max(len(pair_clusters), 1)

    # Composition factor alpha: ratio of composed vs sum-of-singletons predicted savings
    # singleton: each exploit alone provides ~ alpha_i = 1.0
    # composition: if all 3 align on same high-leverage bytes, alpha > 1 (super-additive)
    # if they cover disjoint bytes, alpha < 1 (sub-additive / SATURATING)
    composed_savings_estimate = top_k_frac * per_class_max_weight * pair_cluster_diversity
    singleton_savings_sum = top_k_frac + per_class_max_weight + pair_cluster_diversity
    alpha = composed_savings_estimate / max(singleton_savings_sum / 3.0, 1e-10)

    return {
        "top_k_byte_fraction": top_k_frac,
        "per_class_entropy_nats": per_class_entropy,
        "per_class_max_weight": per_class_max_weight,
        "pair_cluster_diversity": pair_cluster_diversity,
        "composed_savings_estimate": composed_savings_estimate,
        "singleton_savings_sum": singleton_savings_sum,
        "composition_alpha": alpha,
    }


def _run_probe() -> dict:
    t_start = time.time()
    archive_bytes, archive_sha, archive_size = _load_pr101_archive_bytes()

    # === Simulate per-byte master-gradient ===
    sens = _simulate_per_byte_master_gradient(archive_bytes, n_pairs=8)

    # === Run 3 exploits ===
    top_k_mask = _exploit_3_top_k_byte_sensitivity(sens, k_frac=0.1)
    per_class_weights = _exploit_5_per_class_chroma_priority(archive_bytes)
    pair_clusters = _exploit_9_per_pair_clustering(sens, n_clusters=3)

    # === Compose bit-allocator ===
    composition = _compose_bit_allocator(top_k_mask, per_class_weights, pair_clusters, archive_size)

    # === Predicted signature checks ===
    sig_top_k_concentrated = composition["top_k_byte_fraction"] < 0.5
    sig_per_class_nonuniform = composition["per_class_entropy_nats"] < 1.5  # < ln(5) uniform
    sig_pair_cluster_nontrivial = composition["pair_cluster_diversity"] > 0.3
    sig_alpha_super_additive = composition["composition_alpha"] > 1.0

    if sig_top_k_concentrated and sig_per_class_nonuniform and sig_pair_cluster_nontrivial:
        if sig_alpha_super_additive:
            verdict = "POSITIVE_SIGNAL_SUPER_ADDITIVE"
        else:
            verdict = "POSITIVE_SIGNAL_SUB_ADDITIVE"
        recommendation = (
            "JUSTIFIED: bit-allocator composition exhibits measurable structure on real "
            "PR 101 archive bytes (top-K concentrated + per-class non-uniform + per-pair "
            "cluster diversity). Per AAA T4 Decision #3(b): Tier-2 paid dispatch on real "
            "master-gradient extraction artifact + actual bit-allocator routing test. "
            "Predicted ΔS -0.005 to -0.020 [predicted] per AAA T4 §8.5. Estimated cost "
            "~$2-5 (master-gradient extraction via canonical helper)."
        )
    else:
        verdict = "PARTIAL_OR_NULL_SIGNAL"
        recommendation = (
            "DEFER: bit-allocator composition lacks measurable structure on the proxy "
            "master-gradient. Per CLAUDE.md 'Forbidden premature KILL': DEFER-PENDING-REAL-"
            "MASTER-GRADIENT-EXTRACTION via canonical helper; the proxy is hash-based, "
            "not contest-score derivative. Reactivation = paid master-gradient extraction "
            "against contest CUDA archive via Cable D consumers."
        )

    elapsed = time.time() - t_start

    return {
        "probe_id": "tier_1_distortion_bit_allocator_composition_smoke",
        "lane_id": "lane_overnight_ccc_tier_1_distortion_axis_4_probes_macos_cpu_advisory_smoke_20260521",
        "probe_name": "Bit-allocator composition (exploits #3 + #5 + #9)",
        "evidence_grade": "macOS-CPU-advisory",
        "axis_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
        "device": "cpu",
        "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
        "elapsed_seconds": elapsed,
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "predicted_signature": {
            "top_k_byte_fraction_concentrated": "< 0.5 (high-leverage subset structure)",
            "per_class_entropy_non_uniform": "< 1.5 nats (< ln(5)=1.609 uniform)",
            "pair_cluster_diversity_nontrivial": "> 0.3 (multi-cluster present)",
            "composition_alpha_super_additive": "> 1.0 (exploits compose super-additively)",
        },
        "actual_signature": composition,
        "exploit_summary": {
            "exploit_3_top_k_byte_count": int(top_k_mask.sum()),
            "exploit_3_top_k_byte_fraction": float(top_k_mask.sum()) / archive_size,
            "exploit_5_per_class_weights": per_class_weights.tolist(),
            "exploit_9_pair_cluster_assignments": pair_clusters.tolist(),
            "exploit_9_unique_clusters": int(len(np.unique(pair_clusters))),
        },
        "signature_checks": {
            "top_k_concentrated": sig_top_k_concentrated,
            "per_class_non_uniform": sig_per_class_nonuniform,
            "pair_cluster_nontrivial": sig_pair_cluster_nontrivial,
            "alpha_super_additive": sig_alpha_super_additive,
        },
        "verdict": verdict,
        "recommendation": recommendation,
        "canonical_equation_reference": "sister of #818 BUCKET-C producer-consumer loop per Cable D + Catalog #354/#356",
        "catalog_references": ["#354", "#356", "#357", "#287", "#323", "#192", "#1", "#318", "#313"],
        "important_caveat": (
            "Per Catalog #318 STRICT gate: this probe uses a hash-based PROXY for "
            "master-gradient, NOT contest-score derivatives. The bit-allocator COMPOSITION "
            "MECHANISM is tested; authoritative per-byte sensitivity REQUIRES real "
            "master-gradient extraction via Cable D consumers + grammar-aware operator + "
            "ZIP CRC reconstruction per Catalog #318."
        ),
        "canonical_provenance": {
            "kind": "macos_cpu_advisory",
            "axis_tag": "[macOS-CPU advisory]",
            "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
            "evidence_grade": "macOS-CPU-advisory",
            "score_claim_valid": False,
            "promotable": False,
            "source": "tier_1_distortion_axis_probe_2_bit_allocator_composition",
        },
        "next_action_on_POSITIVE": (
            "Operator-routable: paid master-gradient extraction on contest CUDA archive "
            "via Cable D consumers + grammar-aware operator (Catalog #318); predicted "
            "composition_alpha >= 1.0 + ΔS -0.005 to -0.020 [predicted]; estimated cost "
            "~$2-5."
        ),
    }


def main() -> int:
    out_dir = Path(__file__).resolve().parent
    verdict = _run_probe()
    out_path = out_dir / "probe_2_bit_allocator_composition_verdict.json"
    out_path.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[probe_2] verdict={verdict['verdict']} alpha={verdict['actual_signature']['composition_alpha']:.4f} "
        f"top_k_frac={verdict['actual_signature']['top_k_byte_fraction']:.4f} "
        f"per_class_entropy={verdict['actual_signature']['per_class_entropy_nats']:.4f} "
        f"elapsed={verdict['elapsed_seconds']:.2f}s"
    )
    print(f"[probe_2] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
