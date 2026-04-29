"""Lane 8 multi-pass inflate optimizer — inner loop MVP tests.

Council Y3→G3 promotion (2026-04-29) prescribes 3 inner-loop tests:
    (a) cold-start: after one outer iter, archive bytes change vs input
    (b) plateau:    when scores plateau within tol for `patience` iters,
                    the loop stops at the right index
    (c) max-iters:  when score keeps improving, loop stops at max_iters

Tests use a tiny synthetic archive fixture (a few-byte renderer.bin +
masks.mkv + optimized_poses.pt) plus an injected ``score_fn`` and an
injected ``inner_step_fn`` so they run without GPU + without contest
auth eval. The production wiring of the inner step lives behind
``_default_inner_step``; these tests verify the OUTER LOOP shape, not
the per-step deltas (which the underlying ``optimize_poses_batch`` and
``learnable_class_targets.ema_update`` tests already cover).
"""
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable

import pytest

REPO = Path(__file__).resolve().parents[3]
MULTI_PASS_PATH = REPO / "experiments" / "multi_pass_inflate_optimizer.py"


def _load_module():
    """Import experiments/multi_pass_inflate_optimizer.py as a module.

    Mirrors the importlib pattern used by
    ``test_lane_m_v3_clean_train_inference_parity.py`` — keeps the
    experiments/ scripts importable without a setuptools entry.
    """
    for p in (REPO / "src",):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(
        "multi_pass_inflate_optimizer", MULTI_PASS_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass field-type lookup (which walks
    # `sys.modules.get(cls.__module__)`) finds the module while it's
    # still being initialised. Without this, `@dataclass` raises
    # `AttributeError: 'NoneType' object has no attribute '__dict__'`
    # under `from __future__ import annotations`.
    sys.modules["multi_pass_inflate_optimizer"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def synthetic_archive(tmp_path: Path) -> Path:
    """Build a tiny synthetic archive that satisfies the multi-pass loop.

    The contents are NOT a valid contest submission — they're just enough
    for ``_extract_archive`` and ``_repack_archive`` to round-trip + for
    the injected ``score_fn`` to compute ``current_archive.stat().st_size``.
    """
    mp = _load_module()
    contents = {
        "renderer.bin": b"FAKE_RENDERER_PAYLOAD" * 100,  # ~2KB
        "masks.mkv": b"FAKE_MASKS_PAYLOAD" * 50,         # ~0.9KB
        "optimized_poses.pt": b"FAKE_POSES_PAYLOAD" * 20,  # ~0.4KB
    }
    archive_path = tmp_path / "synthetic.zip"
    mp._repack_archive(contents, archive_path)
    return archive_path


def _make_score_fn(scores: list[float]) -> Callable:
    """Return a score_fn that emits the next score from `scores` each call.

    The seg/pose components are derived deterministically from the score
    so tests can assert on them too (seg = score/2, pose = score/2 — the
    multi-pass loop only consumes the third value for plateau).
    """
    iter_state = {"i": 0}

    def _score(archive_path: Path, device: str) -> tuple[float, float, float]:
        idx = iter_state["i"]
        iter_state["i"] = idx + 1
        if idx >= len(scores):
            # Repeat the final score forever — simulates a true plateau.
            s = scores[-1]
        else:
            s = scores[idx]
        return (s / 2.0, s / 2.0, s)

    return _score


def _make_inner_step_mutator(suffix_token: bytes) -> Callable:
    """Return an inner_step that appends `suffix_token + iter_idx` to poses.

    The mutation guarantees the archive bytes change every iter (so
    test (a) sees a delta) AND that re-runs are deterministic (the
    suffix is byte-stable for a given iter_idx).
    """
    def _step(
        contents: dict[str, bytes],
        cfg,
        ctx: dict[str, Any],
    ) -> dict[str, bytes]:
        new_contents = dict(contents)
        token = suffix_token + str(ctx["iter_idx"]).encode("ascii")
        new_contents["optimized_poses.pt"] = (
            contents["optimized_poses.pt"] + token
        )
        return new_contents

    return _step


# ─────────────────────────────────────────────────────────────────────────────
# Test (a): cold-start — after one inner-step iter, archive bytes differ
# ─────────────────────────────────────────────────────────────────────────────


def test_cold_start_archive_bytes_change_after_one_iter(
    synthetic_archive: Path, tmp_path: Path,
) -> None:
    """One outer iter should produce a re-packed archive with different bytes.

    The injected inner-step mutator appends a token to optimized_poses.pt;
    the deterministic re-pack therefore yields a different zip on disk.
    The test reads the zip BEFORE convergence (max_iters=2, patience=99
    so plateau never fires) and confirms iter-1 archive bytes != iter-0.
    """
    mp = _load_module()
    cfg = mp.MultiPassConfig(
        max_iters=2,
        patience=99,  # disable plateau early-stop
        tol=0.0005,
        output_archive=tmp_path / "out.zip",
    )

    # Constant score → no plateau triggered (patience=99 anyway). The
    # inner-step mutator is what we're verifying.
    score_fn = _make_score_fn([1.0, 1.0, 1.0])
    inner_step = _make_inner_step_mutator(b"_MUTATED_")

    initial_bytes = synthetic_archive.read_bytes()
    out, history = mp.run_multi_pass(
        synthetic_archive, cfg,
        device="cuda",  # never reached — we override score_fn
        score_fn=score_fn,
        inner_step_fn=inner_step,
    )

    # Two scoring calls: iter 0 (input archive) and iter 1 (re-packed).
    assert len(history) == 2, f"expected 2 iters, got {len(history)}"

    # Verify iter 1 archive (the re-packed one) lives on disk and has
    # different bytes from the input.
    work_root = cfg.output_archive.parent / f"{cfg.output_archive.stem}_work"
    iter_1_archive = work_root / "iter_1.zip"
    assert iter_1_archive.exists(), (
        f"expected re-packed archive at {iter_1_archive}; "
        f"work_root contents: {list(work_root.iterdir())}"
    )
    re_packed_bytes = iter_1_archive.read_bytes()
    assert re_packed_bytes != initial_bytes, (
        "Re-pack produced byte-identical output — inner-step mutation "
        "didn't propagate."
    )

    # Verify the optimized_poses.pt entry actually grew by the token bytes.
    with zipfile.ZipFile(iter_1_archive, "r") as zf:
        new_poses = zf.read("optimized_poses.pt")
    with zipfile.ZipFile(synthetic_archive, "r") as zf:
        old_poses = zf.read("optimized_poses.pt")
    assert new_poses.startswith(old_poses), (
        "Mutated poses should be old_poses + token suffix"
    )
    assert len(new_poses) > len(old_poses), (
        f"Mutated poses ({len(new_poses)}B) should exceed original "
        f"({len(old_poses)}B)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test (b): plateau — when scores plateau within tol for patience iters,
#                     loop stops at the right index
# ─────────────────────────────────────────────────────────────────────────────


def test_plateau_stops_loop_at_correct_index(
    synthetic_archive: Path, tmp_path: Path,
) -> None:
    """Scores [1.0, 0.5, 0.4999, 0.4999, 0.4999] should plateau at iter 3.

    Improvement from 0.5 → 0.4999 = 0.0001, which is BELOW tol=0.0005.
    With patience=3, the plateau check fires once 3 consecutive iters
    have failed to improve best by >= tol. Iter sequence:

        iter 0: score=1.0, history=[1.0],     best_old=N/A          (need patience+1)
        iter 1: score=0.5, history=[1.0,0.5], best_old=N/A
        iter 2: score=0.4999, history=[..0.4999], best_old=N/A     (still need 4 entries)
        iter 3: score=0.4999, history=[1.0,0.5,0.4999,0.4999]
                best_old = min(history[:-3]) = min([1.0]) = 1.0
                recent_best = min(0.5, 0.4999, 0.4999) = 0.4999
                gap = 0.5001 >> 0.0005 → NOT plateau, continue
        iter 4: same recent_best vs best_old=min([1.0,0.5])=0.5
                gap = 0.5 - 0.4999 = 0.0001 < 0.0005 → PLATEAU, stop

    So with max_iters=10, the loop should stop at iter 4 (5 entries,
    converged=True on the last).
    """
    mp = _load_module()
    cfg = mp.MultiPassConfig(
        max_iters=10,
        patience=3,
        tol=0.0005,
        output_archive=tmp_path / "out.zip",
    )

    scores = [1.0, 0.5, 0.4999, 0.4999, 0.4999, 0.4999, 0.4999]
    score_fn = _make_score_fn(scores)
    inner_step = _make_inner_step_mutator(b"_PL_")

    out, history = mp.run_multi_pass(
        synthetic_archive, cfg,
        device="cuda",
        score_fn=score_fn,
        inner_step_fn=inner_step,
    )

    # Expected: stops at iter 4 (5 history entries).
    assert len(history) == 5, (
        f"plateau should fire at iter 4 (5 entries); got {len(history)} "
        f"iters with scores {[r.score for r in history]}"
    )
    assert history[-1].converged is True, (
        f"final iter should be marked converged; got "
        f"converged={history[-1].converged}"
    )
    # Best score is the early 0.4999 (iter 2), which is what the final
    # output should mirror.
    best_iter = min(range(len(history)), key=lambda i: history[i].score)
    assert best_iter == 2, (
        f"best iter should be the first 0.4999 entry (iter 2); got "
        f"best_iter={best_iter} score={history[best_iter].score}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test (c): max-iters — when score keeps improving, loop stops at max_iters
# ─────────────────────────────────────────────────────────────────────────────


def test_max_iters_stops_when_score_keeps_improving(
    synthetic_archive: Path, tmp_path: Path,
) -> None:
    """Strictly-decreasing scores (no plateau) hit max_iters and stop.

    Each iter improves by 0.1 (well above tol=0.0005), so plateau
    NEVER fires. The loop should run exactly max_iters times and
    converged should be False.
    """
    mp = _load_module()
    cfg = mp.MultiPassConfig(
        max_iters=5,
        patience=3,
        tol=0.0005,
        output_archive=tmp_path / "out.zip",
    )

    # Strictly decreasing — improvement always >= tol → no plateau.
    scores = [2.0, 1.5, 1.0, 0.7, 0.5, 0.3]
    score_fn = _make_score_fn(scores)
    inner_step = _make_inner_step_mutator(b"_MX_")

    out, history = mp.run_multi_pass(
        synthetic_archive, cfg,
        device="cuda",
        score_fn=score_fn,
        inner_step_fn=inner_step,
    )

    assert len(history) == cfg.max_iters, (
        f"loop should run exactly max_iters={cfg.max_iters} times; "
        f"got {len(history)} iters with scores {[r.score for r in history]}"
    )
    assert history[-1].converged is False, (
        f"max_iters termination should NOT mark converged=True; got "
        f"converged={history[-1].converged}"
    )
    # Best score is the last (most-improved) iter.
    best_iter = min(range(len(history)), key=lambda i: history[i].score)
    assert best_iter == cfg.max_iters - 1, (
        f"best iter should be the last (most-improved); got "
        f"best_iter={best_iter}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: deterministic re-pack — same inputs produce byte-identical archives.
# This is the codex R5-r6 #5 requirement that the rate-term feedback loop
# depends on. Worth testing inline so the contract can never silently break.
# ─────────────────────────────────────────────────────────────────────────────


def test_repack_is_deterministic(tmp_path: Path) -> None:
    """Same contents dict → byte-identical zip across two re-packs."""
    mp = _load_module()
    contents = {
        "renderer.bin": b"R" * 1024,
        "masks.mkv": b"M" * 512,
        "optimized_poses.pt": b"P" * 256,
    }
    a = tmp_path / "a.zip"
    b = tmp_path / "b.zip"
    mp._repack_archive(contents, a)
    mp._repack_archive(contents, b)
    assert a.read_bytes() == b.read_bytes(), (
        "Two re-packs of identical contents must be byte-identical "
        "(codex R5-r6 #5; rate-term feedback requires bit-stable bytes)."
    )
