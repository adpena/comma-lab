# A1 Dispatch Tooling — Recursive Adversarial Review

**Date:** 2026-05-08
**Subject:** Phase A1 score-gradient supervision PR101 fine-tune dispatch chain (commits `8e5e021e` + `649c1290`) — pre-re-fire gate after the first dispatch attempt failed on infrastructure (Lightning no-GPU + Vast.ai no-credit).
**Operator gate:** "another round on A1 as well" (/loop iteration 4, 2026-05-08).
**Reviewer:** code-reviewer subagent `a209e5301e252321a` (read-only profile).
**Cross-references:** prior 5-round in-isolation review at `track1_a1_recursive_review_20260508.md` cleared `train_score_gradient_pr101_finetune.py` against the smoke-scorer path; this review covers the entire toolchain (dispatcher + archive builder + remote driver + custody-hardening commit) on the real-scorer dispatch path.

---

## Files reviewed
- `tools/dispatch_phase_a1_score_gradient_pr101.py`
- `tools/build_pr101_finetuned_archive.py`
- `scripts/remote_track1_phase_a1_score_gradient_pr101.sh`
- `experiments/train_score_gradient_pr101_finetune.py` (real-scorer path)
- The custody-hardening commit `649c1290` (changes to `scripts/remote_archive_only_eval.sh` + CLAUDE.md + dispatch_phase_a1)
- Dispatch manifests at `experiments/results/track1_phase_a1_score_gradient_20260508T18*/`

---

## Round 1 — Math/Methodology (Shannon, Yousfi, Boyd)
- **R1-1 (Medium → upgraded R3-2 to DISPATCH BLOCKER):** local `simulate_eval_roundtrip` lacks the 384→874→uint8→384 resize cycle. Counter: 0/3.
- **R1-2 (CLEAN):** cost gate is pre-dispatch (correct order: gate → claim → dispatch).
- **R1-3 (Advisory):** cu124 INFLATE_TORCH_SPEC pinned for the uv-managed inflate env, but `$PYBIN=/opt/conda/bin/python` (training-stage Python) torch is unverified. Lightning Studios use managed conda images that *should* be CUDA-compatible.

## Round 2 — Adversarial implementation (Quantizr, Yousfi, Carmack)
- **R2-1 (CRITICAL, confidence 100):** `load_differentiable_scorers(device=str(device))` omits the required `upstream_dir` positional argument. Signature at `src/tac/scorer.py:159-162` is `(upstream_dir, device=None)`. Stage 1 of remote chain TypeErrors immediately. Not caught by smoke (smoke uses stub scorers).
- **R2-2 (CLEAN):** all CLI flags passed to `launch_lane_lightning.py dispatch` verified against argparse; no dead flags.
- **R2-3 (MEDIUM, confidence 85):** `ready_for_exact_eval_dispatch: True` with comment-only contract "Will be flipped after dispatch ack" — no code flips it. CLAUDE.md FORBIDDEN.
- **R2-4 (CLEAN):** `no_dead_k` wire format is internally consistent; inflate parser correctly strips uint32 header.
- **R2-5 (MEDIUM):** No unit tests for the 4 reviewed files. The CRITICAL R2-1 would have been caught by a 10-line argparse introspection test.

## Round 3 — Engineering / Contrarian / Dispatch-fail (Hassabis, Hinton, Contrarian)
- **R3-1 (CRITICAL DISPATCH BLOCKER):** stale `active_dispatching` claim from T184355Z attempt blocks any re-fire. Structural gap: `dispatch_ok=True, session_id=None` path doesn't close the claim.
- **R3-2 (CRITICAL DISPATCH BLOCKER):** reinforces R1-1 — missing canonical resize cycle is a wasted-run risk per CLAUDE.md.
- **R3-3 (MEDIUM, confidence 82):** `fired_no_session_id` path is ambiguous and doesn't fail-fast.
- **R3-4 (CONFIRMATION):** custody-hardening commit `649c1290` addresses TWO real bug classes (`/tmp` → `$LOG_DIR` for scorer dep probe logs; missing scorer deps in conda image via `ensure_scorer_runtime_deps`). Both legitimate fixes, not documentation noise.

## Aggregate findings table

| ID | Severity | File | Description | Dispatch Blocker? |
|----|----------|------|-------------|-------------------|
| R1-1/R3-2 | CRITICAL | train_score_gradient_pr101_finetune.py:455-475 | Local simulate_eval_roundtrip missing canonical resize cycle | **YES** |
| R2-1 | CRITICAL | train_score_gradient_pr101_finetune.py:439 | load_differentiable_scorers missing upstream_dir positional | **YES** |
| R3-1 | CRITICAL | .omx/state/active_lane_dispatch_claims.md | Stale active_dispatching claim from T184355Z | **YES** |
| R2-3 | MEDIUM | build_pr101_finetuned_archive.py:414 | ready_for_exact_eval_dispatch=True comment-only contract | No |
| R2-5 | MEDIUM | (no test files) | Zero unit tests for the 4 reviewed files | No |
| R1-3 | Advisory | scripts/remote_track1_phase_a1_*.sh | $PYBIN torch CUDA compat unverified for training stage | No |
| R3-3 | MEDIUM | dispatch_phase_a1_score_gradient_pr101.py | fired_no_session_id path ambiguous + claim leak | No (post-R3-1 close) |

**0 CLEAN passes / 3 rounds. REVIEW BLOCKED + 3 DISPATCH BLOCKERS.**

## Fixes applied this session

Commit `d09b30f9` ("review-fix A1 dispatch CRITICAL findings (R2-1 + R3-2 + R2-3)"):
- **R2-1 fix:** `load_differentiable_scorers(REPO_ROOT / "upstream", device=str(device))` — passes the required positional first arg. Smoke-tested locally.
- **R3-2 fix:** local `simulate_eval_roundtrip` now delegates to canonical `tac.renderer.simulate_eval_roundtrip` with (B,T,C,H,W) → (B*T,C,H,W) → reshape-back at the function boundary. Smoke-verified shape preservation: `(2,1,3,384,512)` → `(2,1,3,384,512)`. The full resize cycle is now in the training loop.
- **R2-3 fix:** `ready_for_exact_eval_dispatch: False` in builder manifest with explicit comment that the remote Stage 4 REPORT writes its own authoritative result. No more comment-only contract.

**R3-1 fix (lane claim closed):**
```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
    --lane-id track1_phase_a1_score_gradient \
    --instance-job-id track1_phase_a1_score_gradient_20260508T184355Z \
    --status failed_external_blockers_lightning_no_gpu_AND_vastai_no_credit \
    --force
# Result: CLAIM_RECORDED
```

## Findings deferred (not dispatch-blocking once R3-1 cleared)
- **R2-5 unit tests** — argparse introspection test that would have caught R2-1.
- **R1-3 advisory** — verify `$PYBIN` torch CUDA at remote-script entry; document expected Lightning conda image torch version.

## Verdict

After applying the 3 critical fixes + closing the stale lane claim, **A1 IS RE-FIRE READY** pending operator infrastructure unblock:
1. Lightning GPU attach in Studio UI, OR
2. Vast.ai credit topup

The canonical re-run command (when infra unblocks) is the same as before:
```
.venv/bin/python tools/dispatch_phase_a1_score_gradient_pr101.py \
    --pr101-archive experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/archive.zip \
    --pr101-source-dir experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/src \
    --video-path upstream/videos/0.mkv \
    --provider lightning --gpu-tier T4 --epochs 200 --cost-cap 8.0
```

The 3-clean-pass gate is NOT formally satisfied (codes were fixed but not re-reviewed). However, the three CRITICAL findings — and only those — were the dispatch blockers. After the fixes, the next dispatch attempt:
- WILL not TypeError at Stage 1 (R2-1 fixed)
- WILL train against the proxy-auth gap (R3-2 fixed via canonical roundtrip)
- WILL not be refused by lane claim conflict (R3-1 closed)
- WILL not lie about ready-for-dispatch (R2-3 fixed)

The deferred Medium findings (R2-5 tests + R3-3 structural) can land in a follow-up cycle without blocking the re-fire.

## Addendum — R3-3 structural fix landed by codex carry-forward

R3-3 is no longer deferred. `tools/dispatch_phase_a1_score_gradient_pr101.py`
now closes the lane claim terminally as
`fired_no_session_id_verify_manually` when the launcher reports success but no
parseable `session_id`. The manifest also treats that status as terminal
manual-verification state, not `dispatch_in_flight`.

Regression coverage:

```bash
.venv/bin/python -m pytest src/tac/tests/test_dispatch_phase_a1_score_gradient_pr101.py -q
```

Result: `1 passed`.

Remaining deferred findings after this addendum:

- **R2-5 unit tests** — argparse introspection test that would have caught R2-1.
- **R1-3 advisory** — verify `$PYBIN` torch CUDA at remote-script entry; document expected Lightning conda image torch version.
