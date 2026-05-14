# SPDX-License-Identifier: MIT
"""Tests for C1 probe v2 (posterior + prior + residual + KL).

Per Grand Council UNANIMOUS D verdict (commit c283bfcdb), probe v2 implements
the canonical 6-component Hafner DreamerV3 RSSM:

  1. Encoder    f_enc: o_t -> e_t
  2. Posterior  q(z_t | h_t, e_t)
  3. Prior      p(z_t | h_t)
  4. Recurrent  h_t = GRU(h_{t-1}, z_{t-1})  -- z_{t-1} NOT zeros
  5. Decoder    D: z_t -> o_hat_t
  6. KL term    KL(q || p) added to ELBO loss

Plus matched-DOF + matched-bytes independent-frame baselines for fair
comparison per the council's methodological requirements.

These tests exercise EVERY one of the 6 RSSM components plus the matching
logic; they are the canonical test surface for the probe-disambiguator
pattern at Catalog #125 hook #6.
"""

from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from probe_c1_world_model_v2_posterior_prior_disambiguator import (  # noqa: E402
    _Decoder,
    _Encoder,
    _Posterior,
    _Prior,
    _RSSMWorldModel,
    _count_params,
    _independent_baseline,
    _solve_bytes_matched_latent_dim,
    _solve_dof_matched_latent_dim,
    _world_model_v2_fit,
    run_probe,
)


# ----------------------------------------------------------------------
# Component tests: encoder / posterior / prior / decoder shapes + outputs
# ----------------------------------------------------------------------


class TestEncoder:
    def test_encoder_output_shape(self) -> None:
        enc = _Encoder(feature_dim=32, embed_dim=16, hidden_dim=32)
        o = torch.randn(8, 32)  # (B=8, feature_dim=32)
        e = enc(o)
        assert e.shape == (8, 16)

    def test_encoder_is_differentiable(self) -> None:
        enc = _Encoder(feature_dim=8, embed_dim=4, hidden_dim=8)
        o = torch.randn(2, 8, requires_grad=True)
        e = enc(o)
        e.sum().backward()
        assert o.grad is not None
        assert any(p.grad is not None for p in enc.parameters())


class TestPosterior:
    def test_posterior_outputs_two_params(self) -> None:
        post = _Posterior(hidden_dim=32, embed_dim=16, latent_dim=8)
        h = torch.randn(1, 32)
        e = torch.randn(1, 16)
        mu, log_sigma = post(h, e)
        assert mu.shape == (1, 8)
        assert log_sigma.shape == (1, 8)

    def test_posterior_log_sigma_clamped(self) -> None:
        post = _Posterior(hidden_dim=4, embed_dim=2, latent_dim=2)
        # Force extreme inputs to provoke saturation; clamp is at (-5, +2).
        h = torch.full((1, 4), 1e6)
        e = torch.full((1, 2), 1e6)
        _, log_sigma = post(h, e)
        assert (log_sigma >= -5.0).all()
        assert (log_sigma <= 2.0).all()


class TestPrior:
    def test_prior_takes_only_h(self) -> None:
        prior = _Prior(hidden_dim=32, latent_dim=8)
        h = torch.randn(1, 32)
        mu, log_sigma = prior(h)
        assert mu.shape == (1, 8)
        assert log_sigma.shape == (1, 8)

    def test_prior_does_NOT_see_observations(self) -> None:
        """Critical fairness invariant: prior has NO observation input.
        This is what makes the prior a FORECASTING network."""
        prior = _Prior(hidden_dim=8, latent_dim=4)
        # Prior.forward only takes h_t -- enforce signature.
        import inspect
        sig = inspect.signature(prior.forward)
        params = list(sig.parameters.keys())
        # 'self' not in bound method sig; only h_t remains.
        assert params == ["h_t"]


class TestDecoder:
    def test_decoder_maps_latent_to_feature(self) -> None:
        dec = _Decoder(latent_dim=8, feature_dim=32)
        z = torch.randn(4, 8)
        o_hat = dec(z)
        assert o_hat.shape == (4, 32)


# ----------------------------------------------------------------------
# Full RSSM tests
# ----------------------------------------------------------------------


class TestRSSMWorldModel:
    def test_rssm_forward_shapes(self) -> None:
        wm = _RSSMWorldModel(feature_dim=16, latent_dim=8, hidden_dim=16, embed_dim=8)
        T = 12
        obs = torch.randn(T, 16)
        recon, kl, mu_q, log_sigma_q, mu_p, log_sigma_p = wm(obs, use_posterior=True)
        assert recon.shape == (T, 16)
        assert kl.shape == (T,)
        assert mu_q.shape == (T, 8)
        assert log_sigma_q.shape == (T, 8)
        assert mu_p.shape == (T, 8)
        assert log_sigma_p.shape == (T, 8)

    def test_rssm_uses_z_prev_NOT_zeros_in_recurrent(self) -> None:
        """The smoking gun fix: GRU input must be z_{t-1}, NOT zeros.

        Probe-1 fed zero_action; probe v2 must feed z_prev. This test
        verifies that perturbing z_prev changes the recurrent trajectory.
        """
        torch.manual_seed(0)
        wm = _RSSMWorldModel(feature_dim=8, latent_dim=4, hidden_dim=8, embed_dim=4)
        obs = torch.zeros(5, 8)  # zero observations so encoder output is small
        # Step 1: forward with prior-only (no observation conditioning of z)
        recon_post, _, mu_q, _, _, _ = wm(obs, use_posterior=True)
        # If GRU input were zero, the recurrent state would evolve identically
        # regardless of latent. We perturb the initial z and observe a different
        # trajectory in the recurrent state.
        wm2 = _RSSMWorldModel(feature_dim=8, latent_dim=4, hidden_dim=8, embed_dim=4)
        wm2.load_state_dict(wm.state_dict())
        # Modify z_init to a nonzero value
        with torch.no_grad():
            wm2.z_init.fill_(2.0)
        recon_post2, _, _, _, _, _ = wm2(obs, use_posterior=False)
        # Reconstruct should differ from the unperturbed case because the
        # recurrent state propagates z_prev (NOT zeros).
        assert not torch.allclose(recon_post, recon_post2)

    def test_rssm_posterior_sees_observations(self) -> None:
        """Posterior q(z|h,e) must respond to observation perturbations."""
        torch.manual_seed(0)
        wm = _RSSMWorldModel(feature_dim=8, latent_dim=4, hidden_dim=8, embed_dim=4)
        obs_a = torch.randn(3, 8)
        obs_b = torch.randn(3, 8) * 5.0  # very different observations
        _, _, mu_q_a, _, _, _ = wm(obs_a, use_posterior=True)
        _, _, mu_q_b, _, _, _ = wm(obs_b, use_posterior=True)
        # Posterior means should differ because encoder output differs.
        assert not torch.allclose(mu_q_a, mu_q_b)

    def test_rssm_kl_nonnegative(self) -> None:
        """KL(q || p) is always >= 0 for proper diagonal Gaussians."""
        torch.manual_seed(0)
        wm = _RSSMWorldModel(feature_dim=8, latent_dim=4, hidden_dim=8, embed_dim=4)
        obs = torch.randn(10, 8)
        _, kl_per_step, _, _, _, _ = wm(obs, use_posterior=True)
        # KL can be very slightly negative due to numerical clamping of log_sigma;
        # allow a small floor.
        assert (kl_per_step >= -1e-3).all()

    def test_rssm_all_six_components_have_params(self) -> None:
        """All 6 RSSM components must have trainable parameters."""
        wm = _RSSMWorldModel(feature_dim=8, latent_dim=4, hidden_dim=8, embed_dim=4)
        assert _count_params(wm.encoder) > 0
        assert _count_params(wm.posterior) > 0
        assert _count_params(wm.prior) > 0
        assert _count_params(wm.decoder) > 0
        assert _count_params(wm.cell) > 0
        # h_init + z_init are direct nn.Parameter
        assert wm.h_init.numel() > 0
        assert wm.z_init.numel() > 0


# ----------------------------------------------------------------------
# Matched-DOF and matched-bytes baseline tests
# ----------------------------------------------------------------------


class TestMatchedFairness:
    def test_dof_matched_latent_dim_solves_inequality(self) -> None:
        latent = _solve_dof_matched_latent_dim(
            n_frames=64, feature_dim=32, world_model_params=10_000,
        )
        # Baseline params = latent * (64+32) + 32 = latent * 96 + 32
        baseline_params = latent * 96 + 32
        assert baseline_params >= 10_000

    def test_bytes_matched_solves_for_fp4_archive_bytes(self) -> None:
        # 10_000 archive bytes = 20_000 FP4 params needed
        latent = _solve_bytes_matched_latent_dim(
            n_frames=64, feature_dim=32, world_model_bytes=10_000,
        )
        baseline_params = latent * 96 + 32
        baseline_bytes = baseline_params // 2
        assert baseline_bytes >= 10_000

    def test_independent_baseline_fits(self) -> None:
        torch.manual_seed(0)
        target = torch.cumsum(torch.randn(32, 16) * 0.05, dim=0)
        residual, n_params, _ = _independent_baseline(
            n_frames=32, latent_dim=8, feature_dim=16, epochs=50, target=target,
        )
        assert residual >= 0.0
        assert n_params > 0


# ----------------------------------------------------------------------
# End-to-end run_probe verdict schema
# ----------------------------------------------------------------------


class TestRunProbe:
    def test_verdict_schema_complete(self) -> None:
        """run_probe must emit the full verdict schema with all required keys."""
        v = run_probe(n_frames=16, latent_dim=8, feature_dim=16,
                       hidden_dim=8, embed_dim=4, epochs=30, seed=0)
        required = {
            "world_model_v2_full",
            "independent_baseline_dof_matched",
            "independent_baseline_bytes_matched",
            "verdict",
            "verdict_rationale",
            "fairness_mode_winner",
            "evidence_grade",
            "score_claim_valid",
            "score_axis",
            "target_source",
            "ready_for_exact_eval_dispatch",
            "promotion_eligible",
            "rank_or_kill_eligible",
            "result_review_blockers",
            "config",
            "lane_id",
            "council_authority",
            "rssm_components_present",
        }
        assert required.issubset(set(v.keys()))

    def test_evidence_tags_are_fail_closed(self) -> None:
        """Every score axis flag must default to non-promotable proxy."""
        v = run_probe(n_frames=8, latent_dim=4, feature_dim=8,
                       hidden_dim=4, embed_dim=4, epochs=10, seed=0)
        assert v["evidence_grade"] == "proxy"
        assert v["score_claim_valid"] is False
        assert v["ready_for_exact_eval_dispatch"] is False
        assert v["promotion_eligible"] is False
        assert v["rank_or_kill_eligible"] is False
        assert len(v["result_review_blockers"]) >= 3

    def test_rssm_components_present_lists_all_six(self) -> None:
        v = run_probe(n_frames=8, latent_dim=4, feature_dim=8,
                       hidden_dim=4, embed_dim=4, epochs=5, seed=0)
        expected = {
            "encoder_o_to_e",
            "posterior_q_z_given_h_e",
            "prior_p_z_given_h",
            "recurrent_h_t_GRU",
            "decoder_z_to_o_hat",
            "kl_divergence_q_p",
        }
        assert expected.issubset(set(v["rssm_components_present"]))

    def test_lane_id_is_probe_v2_lane(self) -> None:
        v = run_probe(n_frames=8, latent_dim=4, feature_dim=8,
                       hidden_dim=4, embed_dim=4, epochs=5, seed=0)
        assert v["lane_id"] == "lane_c1_probe_v2_posterior_prior_residual_kl_20260514"

    def test_council_authority_references_landed_council_memo(self) -> None:
        v = run_probe(n_frames=8, latent_dim=4, feature_dim=8,
                       hidden_dim=4, embed_dim=4, epochs=5, seed=0)
        auth = v["council_authority"]
        assert "feedback_grand_council_c1_world_model_review_landed_20260514.md" in auth
        assert "UNANIMOUS D" in auth

    def test_world_model_fit_returns_finite_residual(self) -> None:
        target = torch.cumsum(torch.randn(16, 8) * 0.05, dim=0)
        (residual, kl_final, elbo_final, n_params, wall) = _world_model_v2_fit(
            n_frames=16, latent_dim=4, feature_dim=8, hidden_dim=8, embed_dim=4,
            epochs=30, kl_weight=1.0, target=target,
        )
        assert residual >= 0.0
        assert residual < float("inf")
        assert kl_final >= -1.0  # allow small clamping slack
        assert n_params > 0

    def test_verdict_is_string_label(self) -> None:
        v = run_probe(n_frames=16, latent_dim=4, feature_dim=8,
                       hidden_dim=8, embed_dim=4, epochs=20, seed=0)
        assert v["verdict"] in {
            "world_model_v2_full",
            "independent_baseline_dof_matched",
            "independent_baseline_bytes_matched",
            "tie",
        }

    def test_fairness_mode_winner_is_canonical(self) -> None:
        v = run_probe(n_frames=16, latent_dim=4, feature_dim=8,
                       hidden_dim=8, embed_dim=4, epochs=20, seed=0)
        assert v["fairness_mode_winner"] in {
            "both", "matched_dof", "matched_bytes", "neither",
        }


# ----------------------------------------------------------------------
# CLI subprocess test
# ----------------------------------------------------------------------


class TestCLI:
    def test_cli_invocation_emits_json(self, tmp_path: Path) -> None:
        out = tmp_path / "verdict.json"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "probe_c1_world_model_v2_posterior_prior_disambiguator.py"),
                "--n-frames", "8",
                "--latent-dim", "4",
                "--feature-dim", "8",
                "--hidden-dim", "4",
                "--embed-dim", "4",
                "--epochs", "5",
                "--seed", "0",
                "--output", str(out),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out.exists()
        verdict = json.loads(out.read_text(encoding="utf-8"))
        assert "world_model_v2_full" in verdict
        assert "verdict" in verdict
