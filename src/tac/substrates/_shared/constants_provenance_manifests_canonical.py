# SPDX-License-Identifier: MIT
"""Canonical seed constants-provenance manifests (the operator-required hi_nerv seed).

Source: operator hardening packet 2026-06-09. Seeds the constants-provenance
manifest for ``hi_nerv`` with the REAL current constant values (read from
``src/tac/substrates/hi_nerv/architecture.py`` + ``launch_manifest.py`` +
``score_aware_loss.py`` + the shared MLX harness on 2026-06-09 — NOT guessed):

  * ``sin_frequency = 30.0`` (architecture.py:121) — the worked ARBITRARY symptom.
  * ``ema_decay = 0.997`` / ``grad_clip_max_norm = 1.0`` (launch_manifest.py).
  * ``latent_dim_coarse/mid/fine = 16/20/24`` (architecture.py:105,108,111).
  * ``decoder_channels = (48,40,32,24,20,16,12)`` (architecture.py:119).
  * ``mid/fine_injection_block_index = 2/4`` (architecture.py:125,128).
  * ``segnet/pose_distillation_weight = 0.0`` (the Mistake-B objective-starvation
    default in the shared MLX harness — bundle.py:540 / dual_ascent.py:463,473).
  * ``research_total_epochs = 3000`` / ``warmup_epochs = 10`` (launch_manifest.py).
  * ``latent_init_std = 0.02`` (architecture.py:404-410).

Provenance classification per CLAUDE.md cargo-cult audit (Catalog #303) applied at
the constant level. The headline finding: ``sin_frequency=30.0`` is ARBITRARY +
score_relevant + blocking at L2, with a real ``replacement_path`` (the v2 scorer
spectral-sensitivity atlas) so the gate SURFACES it as "must be resolved before
L2" rather than blocking immediately — until the carrier declares L2, the manifest
records the debt without halting work. ``distill_weight=0.0`` is the canonical
Mistake-B: an ARBITRARY score-relevant default that should be MEASURED/LEARNED.

These are WARN-surface seeds: the manifest is emitted un-verified so the Catalog
#385 gate can surface the ARBITRARY-at-declared-maturity findings; ``hi_nerv``
currently declares L1 (mechanism-present per the 2026-06-09 audits), so the L2
blockers are recorded but NOT yet firing — they fire structurally the moment the
carrier claims L2 without resolving them.

Sister of ``tac.substrates._shared.vehicle_fidelity_manifests_canonical`` (the
name-laundering seed manifests).
"""

from __future__ import annotations

from .constants_provenance_manifest import (
    ConstantProvenance,
    ConstantsProvenanceManifest,
)

__all__ = ["CANONICAL_CONSTANTS_MANIFESTS", "HI_NERV_CONSTANTS_MANIFEST"]


# The carrier declares L1 today (mechanism-present per the 2026-06-09 Vehicle OS
# audits); the L2 blockers are recorded but do not fire until it claims L2.
_HI_NERV_DECLARED_MATURITY = "L1"


HI_NERV_CONSTANTS_MANIFEST = ConstantsProvenanceManifest(
    vehicle_id="hi_nerv",
    declared_maturity_level=_HI_NERV_DECLARED_MATURITY,
    constants=(
        # --- the worked ARBITRARY symptom (the whole reason for this packet) ---
        ConstantProvenance(
            constant_name="sin_frequency",
            value=30.0,
            provenance="ARBITRARY",
            score_relevant=True,
            stability_critical=False,
            owner="src/tac/substrates/hi_nerv/architecture.py:121 (HinervConfig.sin_frequency)",
            replacement_path=(
                "tools/measure_scorer_spectral_sensitivity.py v2 (scorer transfer "
                "function) -> the MEASURED peak siren_w_equivalent; or a per-scale "
                "Nyquist DERIVED cap; or a LEARNED per-channel omega"
            ),
            blocking_maturity_level="L2",
            notes=(
                "The single global SIREN w is a 0-parameter stand-in for a whole "
                "spectral geometry. Empirical w=30 alias trap (F1 arm A flat at "
                "21.73 dB). MUST become DERIVED/MEASURED/LEARNED before L2."
            ),
        ),
        # --- the Mistake-B objective-starvation defaults (score-relevant ARBITRARY) ---
        ConstantProvenance(
            constant_name="segnet_distillation_weight",
            value=0.0,
            provenance="ARBITRARY",
            score_relevant=True,
            stability_critical=False,
            owner=(
                "src/tac/substrates/_shared/mlx_score_aware/dual_ascent.py:463 "
                "(shared harness default)"
            ),
            replacement_path=(
                "Mistake-B fix: set an explicit NONZERO SegNet objective weight "
                "tuned against d_seg (Catalog #384 sister gate); 0.0 silently "
                "trains recon-MSE-only"
            ),
            blocking_maturity_level="L2",
            notes="The canonical OBJECTIVE-STARVATION default (Catalog #384 anchor).",
        ),
        ConstantProvenance(
            constant_name="pose_distillation_weight",
            value=0.0,
            provenance="ARBITRARY",
            score_relevant=True,
            stability_critical=False,
            owner=(
                "src/tac/substrates/_shared/mlx_score_aware/dual_ascent.py:473 "
                "(shared harness default)"
            ),
            replacement_path=(
                "Mistake-B fix: explicit NONZERO PoseNet objective weight tuned "
                "against d_pose (Catalog #384 sister gate)"
            ),
            blocking_maturity_level="L2",
            notes="Sister of segnet_distillation_weight; pose-axis objective starvation.",
        ),
        # --- stability-critical training constants (convention; need provenance) ---
        ConstantProvenance(
            constant_name="ema_decay",
            value=0.997,
            provenance="ARBITRARY",
            score_relevant=False,
            stability_critical=True,
            owner="src/tac/substrates/hi_nerv/launch_manifest.py:571 (ema_decay default)",
            replacement_path=(
                "EMA decay tau relates to the effective averaging window N ~ 1/(1-tau); "
                "DERIVE tau from the stage epoch budget (window = a target fraction of "
                "stage length), or sweep MEASURED against final d_seg/d_pose"
            ),
            blocking_maturity_level="L2",
            notes=(
                "Quantizr canonical 0.997 (CLAUDE.md 'EMA -- NON-NEGOTIABLE'); "
                "inherited convention. Stability-critical (wrong decay freezes or "
                "destabilizes the inference shadow). Has a derivation path -> not "
                "blocking, but flagged for provenance."
            ),
        ),
        ConstantProvenance(
            constant_name="grad_clip_max_norm",
            value=1.0,
            provenance="ARBITRARY",
            score_relevant=False,
            stability_critical=True,
            owner="src/tac/substrates/hi_nerv/launch_manifest.py:834 (grad_clip_max_norm)",
            replacement_path=(
                "DERIVE/MEASURE from the observed gradient-norm distribution during "
                "warmup (clip at a high percentile), rather than the convention 1.0"
            ),
            blocking_maturity_level="L2",
            notes="Wave N+11 canonical max_norm=1.0; convention inherited from PR95.",
        ),
        # --- latent dims (score-relevant via bytes + capacity) ---
        ConstantProvenance(
            constant_name="latent_dim_coarse",
            value=16,
            provenance="ARBITRARY",
            score_relevant=True,
            stability_critical=False,
            owner="src/tac/substrates/hi_nerv/architecture.py:105",
            replacement_path=(
                "DERIVE from a rate-distortion sweep (the per-pair latent bytes vs "
                "d_seg/d_pose marginal), or an MDL/information-bottleneck bound on the "
                "per-pair code"
            ),
            blocking_maturity_level="L2",
            notes="16/20/24 coarse/mid/fine inherited from the PR95-family taper.",
        ),
        ConstantProvenance(
            constant_name="latent_dim_mid",
            value=20,
            provenance="ARBITRARY",
            score_relevant=True,
            owner="src/tac/substrates/hi_nerv/architecture.py:108",
            replacement_path="rate-distortion sweep / MDL bound (sister of latent_dim_coarse)",
            blocking_maturity_level="L2",
        ),
        ConstantProvenance(
            constant_name="latent_dim_fine",
            value=24,
            provenance="ARBITRARY",
            score_relevant=True,
            owner="src/tac/substrates/hi_nerv/architecture.py:111",
            replacement_path="rate-distortion sweep / MDL bound (sister of latent_dim_coarse)",
            blocking_maturity_level="L2",
        ),
        # --- channel taper (score-relevant: param count -> bytes + capacity) ---
        ConstantProvenance(
            constant_name="decoder_channels",
            value=(48, 40, 32, 24, 20, 16, 12),
            provenance="ARBITRARY",
            score_relevant=True,
            stability_critical=False,
            owner="src/tac/substrates/hi_nerv/architecture.py:119 (the 229K taper)",
            replacement_path=(
                "DERIVE the taper from a per-scale Nyquist capacity argument (each "
                "upsample block resolves only so many cycles), or a LEARNED width "
                "search; currently the hand-set 228,903-param taper"
            ),
            blocking_maturity_level="L2",
            notes="The canonical 229K-param taper; inherited from PR95.",
        ),
        # --- injection block indices (score-relevant placement of detail) ---
        ConstantProvenance(
            constant_name="mid_injection_block_index",
            value=2,
            provenance="ARBITRARY",
            score_relevant=True,
            owner="src/tac/substrates/hi_nerv/architecture.py:125",
            replacement_path=(
                "DERIVE from the per-scale Nyquist schedule (inject mid/fine latents "
                "at the block whose resolution matches the latent's spatial frequency)"
            ),
            blocking_maturity_level="L2",
        ),
        ConstantProvenance(
            constant_name="fine_injection_block_index",
            value=4,
            provenance="ARBITRARY",
            score_relevant=True,
            owner="src/tac/substrates/hi_nerv/architecture.py:128",
            replacement_path="per-scale Nyquist schedule (sister of mid_injection_block_index)",
            blocking_maturity_level="L2",
        ),
        # --- stage durations (score-relevant via convergence quality) ---
        ConstantProvenance(
            constant_name="research_total_epochs",
            value=3000,
            provenance="ARBITRARY",
            score_relevant=True,
            stability_critical=False,
            owner="src/tac/substrates/hi_nerv/launch_manifest.py:828",
            replacement_path=(
                "MEASURE the convergence knee (epochs at which d_seg/d_pose stop "
                "improving) and DERIVE the stage budget from it, rather than the "
                "PR95-inherited 8-stage 29,650-epoch convention scaled down"
            ),
            blocking_maturity_level="L2",
            notes="PR95 8-stage curriculum totals 29,650; the research default is 3000.",
        ),
        # --- a DERIVED constant (contrast: this one is NOT arbitrary) ---
        ConstantProvenance(
            constant_name="latent_init_std",
            value=0.02,
            provenance="DERIVED",
            score_relevant=False,
            stability_critical=True,
            owner="src/tac/substrates/hi_nerv/architecture.py:404-410",
            replacement_path="",
            blocking_maturity_level="L2",
            notes=(
                "0.02 is the standard small-init std for per-pair latents (a "
                "near-zero start so the decoder learns the mean first); a sane "
                "initialization convention, classified DERIVED (small-init "
                "principle) rather than score-tuned. Contrast example: shows the "
                "manifest is not 'everything is ARBITRARY'."
            ),
        ),
        # --- a harmless engineering constant (the guardrail: must NOT block) ---
        ConstantProvenance(
            constant_name="checkpoint_cadence_epochs",
            value=250,
            provenance="ARBITRARY",
            score_relevant=False,
            stability_critical=False,
            owner="src/tac/substrates/hi_nerv/launch_manifest.py:572",
            replacement_path="",
            blocking_maturity_level="L2",
            notes=(
                "How often to checkpoint. ARBITRARY but score_relevant=False AND "
                "stability_critical=False -> EXEMPT from the gate (the guardrail: "
                "do not bureaucratize harmless engineering constants)."
            ),
        ),
    ),
    summary=(
        "hi_nerv constants provenance: sin_frequency=30 + distill_weights=0.0 are the "
        "headline ARBITRARY score-relevant constants (both carry replacement_paths so "
        "they record debt without halting L1 work); latent dims / taper / injection "
        "indices / stage epochs are ARBITRARY-with-replacement; ema_decay + grad_clip "
        "are ARBITRARY stability-critical; latent_init_std is DERIVED; "
        "checkpoint_cadence is the guardrail exemption (harmless engineering)."
    ),
    source_memos=(
        ".omx/research/principled_frequency_basis_synthesis_20260609.md",
        "docs/vehicle_operating_system.md",
    ),
)


CANONICAL_CONSTANTS_MANIFESTS: tuple[ConstantsProvenanceManifest, ...] = (
    HI_NERV_CONSTANTS_MANIFEST,
)
