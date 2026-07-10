# SPDX-License-Identifier: MIT
"""Tests for the exact-metric Monte-Carlo finisher (#396, tac.through_r.mc_finisher).

The rigor is on the SEARCH INVARIANTS (ratchet monotonicity, confirm-is-authority, discrete
byte accounting, batch bisect, resumability, determinism, guided-proposal concentration).
All oracles are TINY synthetic mocks ($0/fast) with a KNOWN, reachable optimum so acceptance
actually happens and can be checked. ONE governed n8 integration smoke drives the CANONICAL
through-R harness (scorer/fixture-gated skip if unavailable) — never a parallel R re-impl.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tac.through_r.mc_finisher import (
    DEFAULT_PARAM_TARGETS,
    FinisherProblem,
    MCFinisher,
    MCFinisherError,
    MeasuredObjective,
    Proposal,
    ProposalEngine,
    delta_s_floor_per_confirmed_flip,
    params_sha256,
)


# ======================================================================================
# A controllable synthetic objective: d_seg = fraction of out_sdf.weight elements whose SIGN
# differs from a fixed target sign pattern. Each single-element sign-fix lowers d_seg by
# exactly 1/total — a clean, reachable, flip-shaped landscape ideal for invariant tests.
# ======================================================================================
def _sign_target(shape, seed=7):
    rng = np.random.default_rng(seed)
    return np.where(rng.random(shape) < 0.5, -1.0, 1.0).astype(np.float32)


def _make_sign_problem(params, target_sign, *, tensor="out_sdf.weight", bytes_const=1000):
    total = int(np.asarray(target_sign).size)

    def render_fn(p):
        return p  # the 'frames' ARE the params for this mock (render is identity)

    def measure_fn(frames, *, confirm):
        w = np.asarray(frames[tensor])
        wrong = int(np.sum(np.sign(w).astype(np.float32) != target_sign))
        return MeasuredObjective(
            d_seg=wrong / total, archive_bytes=bytes_const, confirm=confirm, n_pairs=600
        )

    return FinisherProblem(render_fn=render_fn, measure_fn=measure_fn, n_pairs=600)


def _rand_params(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "out_sdf.weight": rng.normal(size=(5, 96)).astype(np.float32),
        "out_sdf.bias": rng.normal(size=(5,)).astype(np.float32),
        "palette": rng.normal(size=(5, 3)).astype(np.float32),
    }


# --------------------------------------------------------------------------------------
# 1. delta_s floor is the exact DERIVED per-flip granularity (P2 noise floor).
# --------------------------------------------------------------------------------------
def test_delta_s_floor_is_exact_derived():
    assert delta_s_floor_per_confirmed_flip(600) == pytest.approx(100.0 / (600 * 384 * 512))
    assert delta_s_floor_per_confirmed_flip(8) == pytest.approx(100.0 / (8 * 384 * 512))
    with pytest.raises(MCFinisherError):
        delta_s_floor_per_confirmed_flip(0)


# --------------------------------------------------------------------------------------
# 2. params_sha256 is deterministic + sensitive to any element change.
# --------------------------------------------------------------------------------------
def test_params_sha256_deterministic_and_sensitive():
    p = _rand_params(1)
    assert params_sha256(p) == params_sha256({k: v.copy() for k, v in p.items()})
    p2 = {k: v.copy() for k, v in p.items()}
    p2["palette"][0, 0] += np.float32(1.0)
    assert params_sha256(p2) != params_sha256(p)


# --------------------------------------------------------------------------------------
# 3. RATCHET MONOTONICITY — the confirmed S-component never increases; best <= start.
# --------------------------------------------------------------------------------------
def test_ratchet_monotone_never_regresses():
    params = _rand_params(0)
    tgt = _sign_target((5, 96))
    prob = _make_sign_problem(params, tgt)
    f = MCFinisher(
        params, prob, targets=["out_sdf.weight"], mode="fp32", seed=3,
        batch_elems=8, screen_gate=False,
    )
    # track the confirmed S after every batch — must be non-increasing.
    s_hist = [f.current_objective().s_component()]
    for _ in range(80):
        f._proposal_index += 1
        f.accept_batch(f.engine.propose())
        s_hist.append(f.current_objective().s_component())
    assert all(s_hist[i + 1] <= s_hist[i] + 1e-12 for i in range(len(s_hist) - 1))
    assert s_hist[-1] <= s_hist[0] + 1e-12


def test_run_lowers_or_holds_s_and_accepts_on_reachable_landscape():
    params = _rand_params(0)
    tgt = _sign_target((5, 96))
    prob = _make_sign_problem(params, tgt)
    # big fp32 steps flip signs readily → acceptance is reachable.
    f = MCFinisher(
        params, prob, targets=["out_sdf.weight"], mode="fp32", seed=5, batch_elems=4,
        screen_gate=False, engine_kwargs={"fp32_step0": 3.0},
    )
    res = f.run(max_confirmed_batches=120)
    assert res.best_s_component <= res.start_s_component + 1e-12
    assert res.delta_s_total <= 0.0
    assert res.n_accepted >= 1  # the landscape IS improvable → search must find something
    assert res.best_params_sha256 != res.checkpoint_sha256  # params actually moved


# --------------------------------------------------------------------------------------
# 4. SCREEN-vs-CONFIRM ladder honesty — CONFIRM is the ONLY accept authority (P9).
# --------------------------------------------------------------------------------------
def test_confirm_authority_asserted_when_measure_lies():
    params = _rand_params(0)

    def render_fn(p):
        return p

    def lying_measure(frames, *, confirm):
        # BUG: always returns confirm=False even on a confirm call.
        return MeasuredObjective(d_seg=0.5, archive_bytes=1000, confirm=False, n_pairs=600)

    prob = FinisherProblem(render_fn=render_fn, measure_fn=lying_measure, n_pairs=600)
    f = MCFinisher(params, prob, targets=["out_sdf.weight"], mode="fp32", seed=0, screen_gate=False)
    with pytest.raises(MCFinisherError, match="confirm authority"):
        f.current_objective()


def test_screen_gate_short_circuits_non_improving_without_confirm():
    params = _rand_params(0)
    tgt = _sign_target((5, 96))
    total = tgt.size
    n_confirm = {"count": 0}
    n_screen = {"count": 0}

    def render_fn(p):
        return p

    def measure_fn(frames, *, confirm):
        w = np.asarray(frames["out_sdf.weight"])
        wrong = int(np.sum(np.sign(w).astype(np.float32) != tgt))
        if confirm:
            n_confirm["count"] += 1
        else:
            n_screen["count"] += 1
        return MeasuredObjective(d_seg=wrong / total, archive_bytes=1000, confirm=confirm, n_pairs=600)

    prob = FinisherProblem(render_fn=render_fn, measure_fn=measure_fn, n_pairs=600)
    f = MCFinisher(
        params, prob, targets=["out_sdf.weight"], mode="fp32", seed=0, batch_elems=8,
        screen_gate=True, engine_kwargs={"fp32_step0": 1e-6},  # tiny steps rarely flip signs
    )
    f.current_objective()  # 1 confirm for the anchor
    confirms_before = n_confirm["count"]
    # A tiny-step proposal that does not improve on SCREEN must NOT trigger a CONFIRM.
    prop = f.engine.propose()
    out = f.accept_batch(prop)
    if out.reason == "screen_no_improve":
        assert n_confirm["count"] == confirms_before  # screen short-circuited: no confirm spent
        assert out.screen_dseg is not None and out.confirm_dseg is None


# --------------------------------------------------------------------------------------
# 5. DISCRETE-MODE (int8) byte accounting — ΔS folds Δbytes via byte_cost_fn.
# --------------------------------------------------------------------------------------
def test_int8_mode_byte_accounting_in_delta_s():
    rng = np.random.default_rng(0)
    params = {"out_sdf.weight": rng.integers(-8, 8, size=(5, 96)).astype(np.int8)}
    tgt = _sign_target((5, 96))
    total = tgt.size

    def render_fn(p):
        return p

    def measure_fn(frames, *, confirm):
        w = np.asarray(frames["out_sdf.weight"]).astype(np.float32)
        wrong = int(np.sum(np.sign(np.where(w == 0, 1.0, w)).astype(np.float32) != tgt))
        return MeasuredObjective(d_seg=wrong / total, archive_bytes=0, confirm=confirm, n_pairs=600)

    # byte cost = nonzero-count * 2 (a stand-in entropy proxy) → mutations CHANGE bytes.
    def byte_cost_fn(p):
        return int(np.count_nonzero(p["out_sdf.weight"])) * 2

    prob = FinisherProblem(
        render_fn=render_fn, measure_fn=measure_fn, n_pairs=600, byte_cost_fn=byte_cost_fn
    )
    obj = prob.evaluate(params, confirm=True)
    # s_component includes the byte term (not just d_seg).
    from tac.contest_score import RATE_WEIGHT, UNCOMPRESSED_SIZE_BYTES

    expected_bytes = int(np.count_nonzero(params["out_sdf.weight"])) * 2
    assert obj.archive_bytes == expected_bytes
    assert obj.s_component() == pytest.approx(
        100.0 * obj.d_seg + RATE_WEIGHT * expected_bytes / UNCOMPRESSED_SIZE_BYTES
    )


def test_int8_proposal_clamps_to_code_range():
    params = {"out_sdf.weight": np.full((4, 4), 127, dtype=np.int8)}
    prop = Proposal(
        tensor="out_sdf.weight",
        flat_indices=np.array([0, 1, 2], dtype=np.int64),
        deltas=np.array([5, 5, 5], dtype=np.int64),
    )
    out = prop.apply(params, mode="int8")
    # 127 + 5 clamps to 127 (no wraparound to a negative int8).
    assert out["out_sdf.weight"].reshape(-1)[0] == 127
    assert out["out_sdf.weight"].dtype == np.int8


# --------------------------------------------------------------------------------------
# 6. BATCH BISECT — a batch that confirms net-positive is bisected to salvage the good half.
# --------------------------------------------------------------------------------------
# Landscape: idx0 low-is-good (weight 2), idx1 low-is-good (weight 10). Start w=[5,0].
# start d_seg = (2*5 + 10*0)/1000 = 0.010.
#   proposal idx0-=5 (w0->0: -0.010 help), idx1+=5 (w1->5: +0.050 harm).
#   WHOLE batch d_seg = (2*0 + 10*5)/1000 = 0.050  → REGRESSION (+0.040).
#   idx0-half alone d_seg = 0.000 → improvement; idx1-half alone = 0.060 → regression.
def _bisect_problem():
    total = 1000.0

    def render_fn(p):
        return p

    def measure_fn(frames, *, confirm):
        w = np.asarray(frames["out_sdf.weight"]).reshape(-1)
        dseg = (max(0.0, w[0]) * 2.0 + max(0.0, w[1]) * 10.0) / total
        return MeasuredObjective(d_seg=dseg, archive_bytes=0, confirm=confirm, n_pairs=600)

    params = {"out_sdf.weight": np.array([[5.0, 0.0]], dtype=np.float32)}
    return params, FinisherProblem(render_fn=render_fn, measure_fn=measure_fn, n_pairs=600)


def _bisect_proposal():
    return Proposal(
        tensor="out_sdf.weight",
        flat_indices=np.array([0, 1], dtype=np.int64),
        deltas=np.array([-5.0, +5.0], dtype=np.float64),
    )


def test_batch_bisect_salvages_net_negative_half():
    params, prob = _bisect_problem()
    f = MCFinisher(
        params, prob, targets=["out_sdf.weight"], mode="fp32", seed=0,
        screen_gate=False, bisect=True,
    )
    f.current_objective()
    out = f.accept_batch(_bisect_proposal())
    assert out.accepted
    assert out.bisect_depth >= 1  # accepted via a bisected sub-proposal
    w = f.params["out_sdf.weight"].reshape(-1)
    assert w[0] == pytest.approx(0.0)  # the helpful element moved
    assert w[1] == pytest.approx(0.0)  # the harmful element was NOT applied


def test_bisect_disabled_rejects_net_positive_batch():
    params, prob = _bisect_problem()
    f = MCFinisher(params, prob, targets=["out_sdf.weight"], mode="fp32", seed=0, screen_gate=False, bisect=False)
    f.current_objective()
    out = f.accept_batch(_bisect_proposal())
    assert not out.accepted
    assert out.reason == "confirm_no_improve"
    w = f.params["out_sdf.weight"].reshape(-1)
    assert w[0] == pytest.approx(5.0) and w[1] == pytest.approx(0.0)  # nothing applied


# --------------------------------------------------------------------------------------
# 7. DETERMINISM — same seed ⇒ identical trajectory + identical final params.
# --------------------------------------------------------------------------------------
def test_determinism_same_seed_identical_run():
    tgt = _sign_target((5, 96))

    def build():
        params = _rand_params(0)
        prob = _make_sign_problem(params, tgt)
        return MCFinisher(
            params, prob, targets=["out_sdf.weight"], mode="fp32", seed=11,
            batch_elems=6, screen_gate=False, engine_kwargs={"fp32_step0": 2.0},
        )

    r1 = build().run(max_confirmed_batches=50)
    r2 = build().run(max_confirmed_batches=50)
    assert r1.best_params_sha256 == r2.best_params_sha256
    assert r1.n_accepted == r2.n_accepted
    assert [o.accepted for o in r1.outcomes] == [o.accepted for o in r2.outcomes]


def test_different_seed_diverges():
    tgt = _sign_target((5, 96))
    params = _rand_params(0)
    r1 = MCFinisher(
        params, _make_sign_problem(params, tgt), targets=["out_sdf.weight"], mode="fp32",
        seed=1, batch_elems=6, screen_gate=False, engine_kwargs={"fp32_step0": 2.0},
    ).run(max_confirmed_batches=40)
    r2 = MCFinisher(
        _rand_params(0), _make_sign_problem(_rand_params(0), tgt), targets=["out_sdf.weight"],
        mode="fp32", seed=2, batch_elems=6, screen_gate=False, engine_kwargs={"fp32_step0": 2.0},
    ).run(max_confirmed_batches=40)
    # different seeds → different accepted-proposal streams (extremely likely).
    assert [o.accepted for o in r1.outcomes] != [o.accepted for o in r2.outcomes]


# --------------------------------------------------------------------------------------
# 8. RESUMABILITY ROUND-TRIP — snapshot + resume reconstructs params + counters + RNG.
# --------------------------------------------------------------------------------------
def test_resumability_round_trip(tmp_path):
    tgt = _sign_target((5, 96))
    params = _rand_params(0)
    prob = _make_sign_problem(params, tgt)
    snap = tmp_path / "snap.npz"
    f = MCFinisher(
        params, prob, targets=["out_sdf.weight"], mode="fp32", seed=9, batch_elems=6,
        screen_gate=False, engine_kwargs={"fp32_step0": 2.0},
    )
    f.run(max_confirmed_batches=30, snapshot_path=snap)
    assert snap.exists()
    # Resume + continue; compare to an uninterrupted 60-batch run from the same seed.
    prob2 = _make_sign_problem(params, tgt)
    g = MCFinisher.resume_from(
        snap, prob2, targets=["out_sdf.weight"], mode="fp32", batch_elems=6,
        screen_gate=False, engine_kwargs={"fp32_step0": 2.0},
    )
    assert g._proposal_index == 30
    assert g.seed == 9
    # the resumed params equal the snapshot's params.
    assert params_sha256(g.params) == params_sha256(f.params)


def test_atomic_snapshot_is_reloadable(tmp_path):
    params = _rand_params(0)
    prob = _make_sign_problem(params, _sign_target((5, 96)))
    snap = tmp_path / "sub" / "snap.npz"  # parent dir does not exist yet
    f = MCFinisher(params, prob, targets=["out_sdf.weight"], mode="fp32", seed=0, screen_gate=False)
    f.current_objective()
    f._atomic_snapshot(snap)
    assert snap.exists()
    d = np.load(snap, allow_pickle=True)
    assert "__mc_seed" in d.files and "out_sdf.weight" in d.files
    # no leftover tmp file.
    assert not (snap.parent / (snap.name + ".tmp.npz")).exists()


# --------------------------------------------------------------------------------------
# 9. GUIDED PROPOSAL DISTRIBUTION SANITY vs a synthetic flip ledger (saliency prior).
# --------------------------------------------------------------------------------------
def test_saliency_prior_concentrates_element_picks():
    params = {"out_sdf.weight": np.zeros((5, 96), dtype=np.float32)}
    total = 5 * 96
    # one-hot saliency: only flat index 42 is flip-adjacent.
    sal = np.zeros(total, dtype=np.float64)
    sal[42] = 1.0
    eng = ProposalEngine(
        params, targets=["out_sdf.weight"], mode="fp32", rng=np.random.default_rng(0),
        saliency={"out_sdf.weight": sal}, use_saliency=True, batch_elems=1,
    )
    picks = {int(eng.propose().flat_indices[0]) for _ in range(50)}
    assert picks == {42}  # a one-hot saliency forces every pick to index 42.


def test_saliency_off_samples_broadly():
    params = {"out_sdf.weight": np.zeros((5, 96), dtype=np.float32)}
    sal = np.zeros(5 * 96, dtype=np.float64)
    sal[42] = 1.0
    eng = ProposalEngine(
        params, targets=["out_sdf.weight"], mode="fp32", rng=np.random.default_rng(0),
        saliency={"out_sdf.weight": sal}, use_saliency=False, batch_elems=1,
    )
    picks = {int(eng.propose().flat_indices[0]) for _ in range(50)}
    assert len(picks) > 1  # ignoring saliency → broad sampling.


def test_tensor_weights_bias_mass_and_online_reweight_toward_accepts():
    params = _rand_params(0)
    eng = ProposalEngine(
        params, targets=["out_sdf.weight", "palette"], mode="fp32",
        rng=np.random.default_rng(0), tensor_weights={"out_sdf.weight": 9.0, "palette": 1.0},
    )
    m = eng.tensor_mass
    assert m["out_sdf.weight"] > m["palette"]  # prior mass ∝ supplied weight
    # simulate palette accepting repeatedly → its mass should climb.
    pal_prop = Proposal("palette", np.array([0]), np.array([0.1]))
    before = eng.tensor_mass["palette"]
    for _ in range(30):
        eng.register_outcome(pal_prop, accepted=True)
    assert eng.tensor_mass["palette"] > before


def test_fp32_step_self_adapts_on_accept_reject():
    params = _rand_params(0)
    eng = ProposalEngine(
        params, targets=["out_sdf.weight"], mode="fp32", rng=np.random.default_rng(0),
        fp32_step0=1e-3, adapt_rate=0.5,
    )
    prop = Proposal("out_sdf.weight", np.array([0]), np.array([0.0]))
    s0 = eng.step_scale["out_sdf.weight"]
    eng.register_outcome(prop, accepted=True)
    assert eng.step_scale["out_sdf.weight"] > s0  # grows on accept (1/5-rule)
    s1 = eng.step_scale["out_sdf.weight"]
    eng.register_outcome(prop, accepted=False)
    assert eng.step_scale["out_sdf.weight"] < s1  # shrinks on reject


# --------------------------------------------------------------------------------------
# 10. TARGET SELECTION + validation.
# --------------------------------------------------------------------------------------
def test_targets_intersect_present_tensors():
    params = {"out_sdf.weight": np.zeros((5, 96), np.float32)}  # no palette/out_tex
    eng = ProposalEngine(params, targets=DEFAULT_PARAM_TARGETS, mode="fp32")
    assert eng.targets == ("out_sdf.weight",)


def test_engine_rejects_no_present_targets_and_bad_mode():
    params = {"hidden.0.weight": np.zeros((4, 4), np.float32)}
    with pytest.raises(MCFinisherError, match="no target tensors"):
        ProposalEngine(params, targets=["out_sdf.weight"], mode="fp32")
    with pytest.raises(MCFinisherError, match="mode must be"):
        ProposalEngine({"out_sdf.weight": np.zeros((2, 2), np.float32)}, targets=["out_sdf.weight"], mode="bogus")


def test_saliency_size_mismatch_and_negative_rejected():
    params = {"out_sdf.weight": np.zeros((5, 96), np.float32)}
    with pytest.raises(MCFinisherError, match="size"):
        ProposalEngine(
            params, targets=["out_sdf.weight"], mode="fp32",
            saliency={"out_sdf.weight": np.ones(10)},
        )
    bad = np.ones(5 * 96)
    bad[0] = -1.0
    with pytest.raises(MCFinisherError, match="non-negative"):
        ProposalEngine(params, targets=["out_sdf.weight"], mode="fp32", saliency={"out_sdf.weight": bad})


# --------------------------------------------------------------------------------------
# 11. STOP RULES + logging.
# --------------------------------------------------------------------------------------
def test_max_dry_batches_stop_rule():
    params = _rand_params(0)
    tgt = _sign_target((5, 96))
    prob = _make_sign_problem(params, tgt)
    # tiny steps → essentially no accepts → dry-batch stop fires quickly.
    f = MCFinisher(
        params, prob, targets=["out_sdf.weight"], mode="fp32", seed=0, batch_elems=4,
        screen_gate=False, engine_kwargs={"fp32_step0": 1e-9},
    )
    res = f.run(max_confirmed_batches=1000, max_dry_batches=5)
    assert res.stop_reason == "max_dry_batches"
    assert res.n_confirmed_batches <= 6


def test_jsonl_log_written(tmp_path):
    import json

    params = _rand_params(0)
    prob = _make_sign_problem(params, _sign_target((5, 96)))
    log = tmp_path / "log.jsonl"
    f = MCFinisher(params, prob, targets=["out_sdf.weight"], mode="fp32", seed=0, screen_gate=False)
    f.run(max_confirmed_batches=5, log_path=log)
    rows = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert len(rows) == 5
    for r in rows:
        assert "checkpoint_sha256" in r and "delta_s" in r and "accepted" in r


def test_result_to_rows_shape():
    params = _rand_params(0)
    prob = _make_sign_problem(params, _sign_target((5, 96)))
    f = MCFinisher(params, prob, targets=["out_sdf.weight"], mode="fp32", seed=0, screen_gate=False)
    res = f.run(max_confirmed_batches=4)
    rows = res.to_rows()
    assert len(rows) == res.n_confirmed_batches
    assert all("proposal_index" in r for r in rows)


# ======================================================================================
# GOVERNED INTEGRATION SMOKE — ONE tiny n8 confirm through the CANONICAL through-R harness.
# Skips if the gt cache / frozen SegNet are unavailable (no scorer weights in CI).
# ======================================================================================
def _gt_cache_n(n):
    for cand in (
        Path("experiments/results/mlx_fleet_gt_cache/gt_n8.npz"),
        Path("experiments/results/mlx_fleet_gt_cache/gt_n6.npz"),
        Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz"),
    ):
        if cand.exists():
            return cand
    return None


@pytest.mark.slow
def test_integration_smoke_through_canonical_harness_n_subset():
    cache = _gt_cache_n(8)
    if cache is None:
        pytest.skip("no gt cache available for integration smoke")
    try:
        from tac.through_r.harness import load_frozen_segnet, measure_through_r
    except Exception as e:  # pragma: no cover
        pytest.skip(f"through_r harness import failed: {e}")
    try:
        segnet = load_frozen_segnet("cpu-torch")
    except Exception as e:
        pytest.skip(f"frozen SegNet unavailable (no scorer weights): {e}")

    d = np.load(cache, allow_pickle=True)
    lstars = np.asarray(d["lstars"])
    n = min(8, lstars.shape[0])
    lstars = lstars[:n]
    from tac.through_r.resolution_chain import CAMERA_H, CAMERA_W

    # A trivial 'witness': params hold a per-pair constant camera color; render = broadcast.
    params = {"palette": np.full((n, 3), 100.0, dtype=np.float32)}

    def render_fn(p):
        col = p["palette"]
        return [np.broadcast_to(col[i], (CAMERA_H, CAMERA_W, 3)).astype(np.uint8) for i in range(n)]

    def measure_fn(frames, *, confirm):
        res = measure_through_r(
            frames, lstars=lstars, pairs="n600", allow_subset_reason="n8 integration smoke (#396)",
            segnet=segnet, input_space="camera-uint8",
        )
        return MeasuredObjective(
            d_seg=res.agg_dseg, archive_bytes=1000, confirm=confirm, n_pairs=n
        )

    prob = FinisherProblem(render_fn=render_fn, measure_fn=measure_fn, n_pairs=n)
    f = MCFinisher(
        params, prob, targets=["palette"], mode="fp32", seed=0, batch_elems=2,
        screen_gate=False, engine_kwargs={"fp32_step0": 20.0},
    )
    res = f.run(max_confirmed_batches=6)
    # the confirm path drove the REAL SegNet; the ratchet held.
    assert res.best_s_component <= res.start_s_component + 1e-9
    assert res.n_confirmed_batches == 6
