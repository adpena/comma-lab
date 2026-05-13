# Exact Authority + Modal Hardening - 2026-05-13

## Scope

Recursive adversarial hardening pass after the fresh-eyes bug-hunter review of
the 2026-05-13 frontier stack. This landing is engineering work, not a score
claim: it tightens the dispatch, harvest, archive, and dashboard authority
boundaries before further paid GPU dispatch.

## Implemented

- `tools/parallel_dispatch_top_k.py` no longer mints `[contest-CUDA]` from raw
  `score` fields. It only emits CUDA score authority through
  `parse_auth_eval_score_claim(required_score_axis="contest_cuda",
  require_component_recompute=True)` and only from a label-bound auth-eval file
  written by the current dispatch run.
- `tools/harvest_modal_calls.py` is read-only by default, requires `--execute`
  for Modal contact/mutation, rejects unsafe remote artifact paths, re-polls
  nonterminal generated-only summaries, and terminal-closes function timeouts.
- The active exact-readiness score frontier now tracks
  `hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513`
  at `0.20638030907530963`, replacing the stale PR106/R2 score floor.
- Substrate trainers for SIREN, SABOR, Ballé, Cool-Chic, VQ-VAE, and wavelet
  now build charged `archive.zip` packets with `0.bin` only. Runtime files stay
  in `submission_dir` and are evaluated through explicit `--inflate-sh` custody.
- `src/tac/substrates/_shared/inflate_runtime.py` rejects unsafe `file_list`
  output names before writing raw files.
- TD-LoRA now initializes as an exact no-op against the frozen base by using
  tropical residual branches `max(base, base + delta_i, ...)`; a centered
  log-sum-exp straight-through tie-gradient keeps adapter training live at the
  all-zero residual start.
- `tools/auth_eval_records.py` demotes persisted promotion/rank flags when
  blocker lists are present, blocking stale dashboard laundering.
- T10 and pretrained-driving-prior recipes are fail-closed / smoke-only until
  real training paths land; S2SBS no longer logs a smoke result as
  `[contest-CUDA]`.
- Modal PR101 LC-v2 smoke rc=13 was harvested and classified as
  `failed_worker_sentinel_not_mounted`; a terminal claim row was appended.

## Verification

- Focused operator authorization tests: 4 passed.
- Broader operator/sentinel harvest suite: 166 passed.
- Hardening suite covering harvest, dispatch score authority, auth records,
  inflate output containment, TD-LoRA, smoke guards, and archive contract:
  234+ selected tests passed before final all-lanes rerun.
- `tools/audit_exact_ready_queues.py` passed with the updated HLM1 score floor.
- `tools/claim_lane_dispatch.py summary --live-only` reported no active claims.
- `tools/all_lanes_preflight.py` initially failed only on the two intentionally
  untracked files from this landing; rerun after canonicalization is required
  before dispatch.

## Remaining High-EV Bug-Hunter Queue

- Make PacketIR exact closure v2 require positive runtime-consumption,
  full-frame parity, current-frontier manifest, and global archive/runtime/axis
  duplicate-dispatch keys.
- Split JSCC scorer-conditional proxy API from an archive-ready API that either
  reconstructs side state or charges it in bytes.
- Reconcile score-aware RGB range contracts at the common loss boundary.
- Reclassify IGLT claims unless the implementation is upgraded to the stated
  Riemannian Langevin SDE.
