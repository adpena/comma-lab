# DDM M7 relaxed-receiver byte-close realization — implementation spec

Date: 2026-07-23  
Lane: `ddm_m7_relaxed_receiver_realize_byteclose`  
Authority: delegated M7 prompt, SHA-256
`98c29589b60adec5a2b438a55821387d4ff89fe2e975a6d4c8ed1180dba304dd`  
Execution authority: local `$0` only; no Modal, GPU, remote, paid dispatch, or
frontier-pointer mutation. MAIN owns any Task #381 exact-eval dispatch.

## Outcome contract

Build one typed, deterministic, resumable local verifier for the exact archive:

`/Volumes/VertigoDataTier/pact/evidence/joint_optimum_575_xhigh_20260720/n600_r1/n600_r1/candidate_archive.zip`

Expected custody:

- bytes: `177169`
- SHA-256:
  `cb6cf0ba719a535bf8874b31675a4ec66a893423d320f1e4071a2012cd88a56f`
- ZIP grammar: exactly one `ZIP_STORED` member named `x`
- receiver grammar: the existing PR110-lineage `FP11 -> CTXR -> latent +
  sidecar -> FECa selector (+ optional DQS1)` runtime
- lattice: the parsed editable state is the receiver-native `(600, 28)`
  `uint8` latent table. The audited member already resulted from deterministic
  `+1/-1` integer-coordinate polish. This task performs no continuous solve and
  no solve-then-round substitution.

The verifier must:

1. Fail closed unless archive bytes/SHA/member/framing match the typed config.
2. Parse with `tac.click_polish.FrozenPacket`, then require all
   `verify_roundtrip()` byte-identity legs. This is the receiver-consumption and
   parse-back proof for the actual counted archive.
3. Instantiate `tac.click_polish.Renderer` from the exact submission runtime.
   Render `packet.Q0` through decoder, sidecar, optional DQS1, bicubic camera
   resize, PR98 offsets, clamp/round to uint8, and FECa selector. No proxy
   renderer is allowed.
4. Decode the actual upstream source with `AVVideoDataset` in evaluator order.
   For every canonical batch, form ground-truth SegNet argmax and PoseNet first
   six outputs with the frozen upstream models, score the realized candidate,
   and then discard frame tensors. All 600 pairs are mandatory.
5. Use `batch_pairs=16`, `num_threads=2`, `seed=1234`, CPU Torch only. Record
   exact package, host, microarchitecture, model hashes, upstream source hashes,
   runtime source hashes, source-video hash, and both canonical manifest hashes.
6. Atomically write and preserve one immutable checkpoint JSON per completed
   batch. Each checkpoint must contain the exact pair interval, per-pair
   `d_seg`/`d_pose`, candidate-frame SHA-256, GT-frame SHA-256, and the full
   config/archive/runtime/upstream identity needed to reject stale resume.
   Resume may skip scoring only for a contiguous prefix whose checkpoint
   identity and content hash revalidate. At most one in-flight batch is lost.
7. Aggregate only after pair IDs are exactly `0..599` once each. Emit an SSD
   result receipt first, then allow the repo findings/receipt to cite it.
8. Recompute

   `S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489`

   and compare against the canonical pointer `0.1910828242`, the strict fork
   threshold `0.19108`, and the arithmetic counterfactual formed from
   `d_seg=0.00015196`, `d_pose=0.00010184`, and the same 177169 bytes.
9. Decompose the counterfactual-to-realized gap into Seg, Pose, and rate terms.
   Rate must be identical. Record the realization-transfer ratios and the
   counterfactual caveat: those solve-distortion values belonged to a different
   high-byte exact-C1 object and were never properties of this 177169-byte
   receiver member.
10. Always label the row `[macOS-CPU frozen-scorer advisory]`,
    `score_claim=false`, `promotion_eligible=false`,
    `ready_for_exact_eval_dispatch=false`. If `S < 0.19108`, emit the routing
    label `BYTE-CLOSED_CANDIDATE_FOR_MODAL_EXACT_EVAL` while preserving the
    non-authoritative flags and explicitly requiring MAIN review/dispatch.

## Typed config

The CLI accepts only `--config <json>`. The config schema is
`ddm_m7_relaxed_receiver_realize_config.v1` and binds:

- candidate archive, runtime, upstream, SSD output directory
- expected archive bytes/SHA/member
- `n_pairs=600`, `batch_pairs=16`, `num_threads=2`, `seed=1234`
- pointer/threshold/reference size and arithmetic counterfactual components
- `device=cpu`, evidence axis, and all false authority fields

Unknown keys, wrong constants, local-disk bulk output, missing source/runtime
files, insufficient storage, or a dirty/stale resume identity must refuse.
The tool may expose an internal pure receipt verifier for tests, but no second
ad-hoc CLI mode.

## Canonical equation

Land
`src/tac/canonical_equations/ddm_m7_realization_transfer_20260723.py`
with pure functions for:

- contest score terms;
- solve/counterfactual-to-realized distortion ratios;
- additive Seg/Pose/rate score-gap decomposition;
- invariant check that term gaps sum to total score gap.

The equation ID is
`ddm_m7_solve_to_realized_transfer_receiver_closed_v1`. Its domain declaration
must say the ratios are an instance-level diagnostic between explicitly named
objects, not a universal transfer coefficient.

## Tests and verification

Tests must cover typed-config refusal, score arithmetic, gap closure, immutable
checkpoint hashing/identity, contiguous resume, duplicate/missing pair
refusal, ZIP/member/hash refusal, round-trip gate refusal, final routing label,
false authority flags, and a synthetic scorer-free batch path. The real n600
measurement is not a unit test.

Required local gates before measurement:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tools/tests/test_realize_ddm_m7_relaxed_receiver.py \
  src/tac/canonical_equations/tests/test_ddm_m7_realization_transfer_20260723.py
PYTHONPATH=src .venv/bin/python -m ruff check \
  tools/realize_ddm_m7_relaxed_receiver.py \
  src/tac/canonical_equations/ddm_m7_realization_transfer_20260723.py \
  tools/tests/test_realize_ddm_m7_relaxed_receiver.py \
  src/tac/canonical_equations/tests/test_ddm_m7_realization_transfer_20260723.py
```

The implementation is committed before the n600 run. The measurement receipt
records that clean implementation commit as `git_head` and separately records
the upstream content-manifest SHA-256, closing the two provenance fields absent
from the historical #575 dispatch.

## Triality and review

- DSL/config: the single typed JSON schema above; no invented trainer flag.
- DAG: companion append-only DDM M7 feed after measurement.
- Equations: the realization-transfer module above.
- Durable result: SSD batch receipts plus repo machine receipt and dated
  findings memo.
- Review: three bounded re-derivation passes over custody, math, and scope.
  This isolated branch is never self-promoting; MAIN landing review is
  mandatory.

