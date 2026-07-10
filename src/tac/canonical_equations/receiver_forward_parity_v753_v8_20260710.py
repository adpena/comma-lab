# SPDX-License-Identifier: MIT
"""Canonical equation: byte-close RECEIVER-forward parity for v7.5.3 / v8 weight groups (#417 fix half).

MEASURED 2026-07-10 (tools/tests/test_receiver_bijection_v753_v8_parity.py). The #417 two-landing's
FIX half taught the byte-close receiver (the ``_INFLATE_PY`` numpy forward in
``tools/levelset_byte_close_and_eval.py``) to CONSUME the three previously counted-but-INERT weight
groups: v7.5.3 ``tex_trunk.*`` (#395 band Gabor texture trunk) + ``out_tex_h.*`` (#395 A2 widened
texture head) and v8 ``decoupled_head.*`` (#398 B1 per-class decoupled partition fields). Before the
fix those groups paid rate yet rendered the shared-head CONTROL through R -> a scored A/B would be a
FAKE lever verdict (NO-FAKE #8). The fix is REAL iff the receiver numpy forward MATCHES the trainer
MLX submodule; this equation registers that MEASURED parity.

THE LAW (numpy-fp32-is-the-bit-identical-authority non-negotiable, applied to the receiver). The
receiver forward is the numpy-fp64 authority; each new group's receiver mirror reproduces the trainer
MLX (fp32) submodule to FLOAT32 LEVEL (not bit-exact -- the numpy reference runs fp64 then the MLX
module runs fp32, per the module docstrings). Isolated group forwards (receiver mirror vs the exact
``make_texture_trunk_mlx`` / ``make_decoupled_field_head_mlx``): relmax ~7e-4. End-to-end compose
(through softmax->palette->sigmoid*255, which AMPLIFIES the fp32 divergence of boundary-adjacent
logits): relmax <=3.2e-3 (< ~1 uint8/255 -> benign for argmax-d_seg / MSE-d_pose). The regenerated
FREE Gabor bank equals the trainer's excluded ``tex_trunk.bank_B`` to relmax < 1e-6 (rule-118 table).
A SHARED-HEAD witness (none of the three groups) is BYTE-IDENTICAL byte-close (blob sha
693a76912d9d... unchanged vs HEAD) -- the new branches are provably inert (default-off guarantee).

means != ends: a receiver-parity cert is a code-correctness fact, never a score. It UNBLOCKS future
v7.5.3/v8 scored rows (they no longer render the fake control) but MOVES nothing itself. Pointer
contest-CPU 0.19108282 UNMOVED. Axis tags: all anchors [macOS-MLX research-signal] /
[macOS-CPU advisory], score_claim=false, promotable=false.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

RECEIVER_FORWARD_PARITY_V753_V8_EQUATION_ID = "receiver_forward_parity_v753_v8_v1"

_MLX_SIGNAL = "[macOS-MLX research-signal]"
_ADVISORY = "[macOS-CPU advisory]"

_TEST = "tools/tests/test_receiver_bijection_v753_v8_parity.py"
_RECEIVER = "tools/levelset_byte_close_and_eval.py"

# MEASURED relmax (fp64 receiver vs fp32 MLX submodule), 2026-07-10.
TEX_TRUNK_ISO_RELMAX = 7.0e-4          # receiver _tex_trunk_forward vs make_texture_trunk_mlx
DECOUPLED_ISO_RELMAX = 7.5e-4          # receiver _decoupled_phi vs make_decoupled_field_head_mlx
OUT_TEX_H_COMPOSE_RELMAX = 1.4e-3      # end-to-end _outputs_from_h0 rgb vs MLX compose
TEX_TRUNK_COMPOSE_RELMAX = 1.2e-3
DECOUPLED_COMPOSE_RELMAX = 3.2e-3      # sigmoid*255 amplification (< ~1 uint8/255)
GABOR_BANK_REGEN_RELMAX = 1e-6         # regenerated free bank vs trainer bank_B
SHARED_HEAD_BLOB_BYTE_IDENTICAL = True  # blob sha unchanged vs HEAD on a real witness
FLOAT32_PARITY_CEILING = 8e-3          # the compose-level tolerance (documented, benign)


def build_receiver_forward_parity_v753_v8_v1() -> CanonicalEquation:
    """Build the receiver-forward parity equation. All anchors MEASURED
    (VERIFIED_VIA_EMPIRICAL_ANCHOR) by the parity test; the group-forward match IS the proof the
    #417 fix is REAL (consumed, not inert)."""

    anchor_texture = EmpiricalAnchor(
        anchor_id="receiver_tex_trunk_out_tex_h_numpy_mlx_parity_20260710",
        measurement_utc="2026-07-10T20:00:00Z",
        inputs={
            "groups": "tex_trunk.* (#395 band Gabor trunk) + out_tex_h.* (#395 A2 widened head)",
            "receiver": "_tex_trunk_forward / _outputs_from_h0 out_tex_h branch (numpy fp64)",
            "trainer": "make_texture_trunk_mlx / nn.relu(out_tex_h)->out_tex (mlx fp32)",
            "coverage": "synthetic 6x8 grid, K=5, random soft/h; annulus_power in {0.0, 1.5}",
        },
        predicted_output={"relmax": "<= float32 level (~1e-3)", "inert": False},
        empirical_output={
            "tex_trunk_isolated_relmax": TEX_TRUNK_ISO_RELMAX,
            "tex_trunk_compose_relmax": TEX_TRUNK_COMPOSE_RELMAX,
            "out_tex_h_compose_relmax": OUT_TEX_H_COMPOSE_RELMAX,
            "gabor_bank_regen_vs_bank_B_relmax": GABOR_BANK_REGEN_RELMAX,
            "verdict": "CONSUMED (render CHANGES vs shared control -> a live lever, not inert)",
        },
        residual=TEX_TRUNK_COMPOSE_RELMAX,
        source_artifact=_TEST,
        measurement_method="numpy_receiver_vs_mlx_submodule_relmax_parity",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_TEST,
            reactivation_criteria="re-run the parity test if the trainer texture_trunk/out_tex_h "
                                  "forward changes (#395); the receiver mirror must be re-matched",
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_decoupled = EmpiricalAnchor(
        anchor_id="receiver_decoupled_head_numpy_mlx_parity_20260710",
        measurement_utc="2026-07-10T20:00:00Z",
        inputs={
            "group": "decoupled_head.* (v8 B1 per-class decoupled partition fields, #398)",
            "receiver": "_decoupled_phi (numpy fp64, block-diagonal einsums, relu)",
            "trainer": "make_decoupled_field_head_mlx.phi_single (mlx fp32)",
            "coverage": "synthetic P=40, in_feat=12, mod=7, K=5, H=8, L=2",
        },
        predicted_output={"relmax": "<= float32 level (~1e-3)", "inert": False},
        empirical_output={
            "decoupled_isolated_relmax": DECOUPLED_ISO_RELMAX,
            "decoupled_compose_relmax": DECOUPLED_COMPOSE_RELMAX,
            "verdict": "CONSUMED (partition phi replaced -> render CHANGES vs shared control)",
        },
        residual=DECOUPLED_COMPOSE_RELMAX,
        source_artifact=_TEST,
        measurement_method="numpy_receiver_vs_mlx_submodule_relmax_parity",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_TEST,
            reactivation_criteria="re-run if the v8 decoupled_field forward changes (#398); v8 byte-"
                                  "close residual coder (1b/2) is a SEPARATE future anchor",
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_byte_identity = EmpiricalAnchor(
        anchor_id="receiver_shared_head_byte_close_byte_identical_20260710",
        measurement_utc="2026-07-10T20:05:00Z",
        inputs={
            "witness": "real shared-head levelset_witness_ema_mlx.npz (no v7.5.3/v8 groups)",
            "comparison": "build_levelset_blob HEAD vs #417-edited tool, same npz",
        },
        predicted_output={"byte_identical": True},
        empirical_output={
            "blob_sha256_prefix": "693a76912d9d5fa9",
            "blob_bytes": 70918, "manifest_bytes": 1889,
            "byte_identical": SHARED_HEAD_BLOB_BYTE_IDENTICAL,
            "verdict": "the new branches + base_order/_B filter + conditional manifest key are INERT "
                       "for the shared head (default-off guarantee preserved)",
        },
        residual=0.0,
        source_artifact=_RECEIVER,
        measurement_method="head_vs_workingtree_blob_sha_equality",
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_RECEIVER,
            reactivation_criteria="any future receiver edit must re-prove shared-head byte-identity",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=RECEIVER_FORWARD_PARITY_V753_V8_EQUATION_ID,
        name="Byte-close receiver-forward parity for v7.5.3/v8 groups (#417 consumed-not-inert)",
        one_line_summary=(
            "byte-close receiver now CONSUMES tex_trunk/out_tex_h/decoupled_head, matching trainer "
            "MLX to float32 level (iso relmax ~7e-4, compose <=3.2e-3); shared-head byte-IDENTICAL"
        ),
        latex_form=(
            r"\max_{x}\left|\mathrm{recv}_{\mathrm{np64}}(g)(x) - \mathrm{train}_{\mathrm{mlx32}}"
            r"(g)(x)\right| \big/ \max|\cdot| \le \varepsilon_{\mathrm{fp32}} \approx 8\times10^{-3}"
            r",\quad g \in \{\mathrm{tex\_trunk}, \mathrm{out\_tex\_h}, \mathrm{decoupled\_head}\}"
        ),
        python_callable_module_path="tools.levelset_byte_close_and_eval:build_levelset_blob",
        domain_of_validity={
            "groups": "v7.5.3 tex_trunk.* + out_tex_h.* (#395), v8 decoupled_head.* (#398 B1 "
                      "composition forward; the v8 residual coder 1b/2 is NOT yet covered)",
            "parity_level": "float32 (fp64 receiver vs fp32 MLX) -- NOT bit-exact; compose amplifies "
                            "via sigmoid*255 (< ~1 uint8/255)",
            "shared_head": "BYTE-IDENTICAL (default-off inert)",
            "measurement_axis": [_MLX_SIGNAL, _ADVISORY],
        },
        units_in={"group": "param_keys", "coords": "grid", "code": "per_pair_latent"},
        units_out={"relmax": "dimensionless", "byte_identical": "boolean"},
        empirical_anchors=(anchor_texture, anchor_decoupled, anchor_byte_identity),
        predicted_vs_empirical_residual={
            "tex_trunk_iso": TEX_TRUNK_ISO_RELMAX,
            "decoupled_iso": DECOUPLED_ISO_RELMAX,
            "decoupled_compose": DECOUPLED_COMPOSE_RELMAX,
            "shared_head_byte_identity": 0.0,
        },
        last_calibration_utc="2026-07-10T20:05:00Z",
        next_recalibration_trigger="when_3+_new_empirical_anchors_in_domain",
        canonical_consumers=(
            "tools.levelset_byte_close_and_eval",   # the receiver whose parity this certifies
            "tools.levelset_receiver_bijection_gate",  # the fail-closed sister gate (#417 self-protect)
        ),
        canonical_producers=(
            "tools.tests.test_receiver_bijection_v753_v8_parity",  # the parity test that measures it
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_RECEIVER,
            reactivation_criteria="extend anchors when the v8 residual coder byte-close lands, or when "
                                  "tex_trunk/decoupled forwards change; re-match the receiver mirror",
            measurement_axis=_MLX_SIGNAL,
            hardware_substrate="m5_max_cpu",
        ),
    )


def populate_receiver_forward_parity_v753_v8_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (latest-row-wins query semantics)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_receiver_forward_parity_v753_v8_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="receiver_consumption_bijection_fix_417_20260710",
    )
    return eq


__all__ = [
    "RECEIVER_FORWARD_PARITY_V753_V8_EQUATION_ID",
    "build_receiver_forward_parity_v753_v8_v1",
    "populate_receiver_forward_parity_v753_v8_equation",
]
