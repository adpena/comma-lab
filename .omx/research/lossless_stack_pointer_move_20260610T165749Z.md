# Lossless-stack pointer move — verdict: NO MOVE (the four moves are already the frontier; S12 inapplicable)

**LEAD — EXACT POINTER DELTA: the frontier pointer did NOT move. It stays at contest-CPU 0.19109982419209975
(sha `b46897267ded…`, 177,169 B). No stacked archive was built, no paired eval fired — because building one
would re-derive the EXISTING frontier byte-for-byte and the eval would re-confirm an already-paired result.**

**Date:** 2026-06-10
**Subagent:** `task64_lossless_stack_pointer_move` (Task #64)
**Lane:** read-only verification of `lane_pr110_payload_entropy_recode_20260610` (the current frontier)
**Mode:** RACE (`RACE_MODE_ACTIVE.flag` present). Evidence grade: `[macOS-CPU advisory]` / structural + exact-byte.
NO score claim, NO dispatch, `promotable=false`, $0 spend.
**Proof artifact:** `experiments/results/lossless_stack_20260610T165749Z/lossless_stack_parity_proof.json`

## Headline

The task asked to assemble R1⊕R2⊕R3⊕S12 (four orthogonal lossless rate moves) onto the recoded-R3 frontier and
exact-score the stack. **SEARCH-FIRST verification against the EXACT frontier bytes (sha `b46897267ded`,
verified-matches-pointer) proves the premise is already spent:**

| move | premise (planning memo) | reality on the EXACT frontier | addressable new bytes |
|---|---|---|---:|
| **R1** decoder entropy recode | −1,023 B, to-apply | **ALREADY IN THE BASE** — `byte_closure_proof.decoder_delta_bytes = −1023` (162,127 brotli → 161,104 CTXR range). The leapfrog folded it in. | **0** |
| **R2** latent AR+range recode | −317 B, to-apply | **ALREADY IN THE BASE** — `latent_delta_bytes = −317` (15,387 LZMA1 → 15,070 CTXR AR). Folded in. | **0** |
| **R3** selector/framing | the base | **IS THE BASE** — FECa selector 222 B + DQS1 tail 42 B kept verbatim; recoded-R3 IS the pointer. | **0** |
| **S12** resize-null preimage | −10–19.5% of coded frame bytes | **INAPPLICABLE** — the frontier stores NO frame pixels (procedural HNeRV: decoder + latents). S12 acts on stored uint8 camera-frame planes; there are none. | **0** |

**The recoded-R3 frontier IS R1⊕R2⊕R3 already stacked** (the leapfrog verdict did this and the canonical pointer
records it). The fourth move, S12, has **zero addressable bytes** on this procedural vehicle. So the "stacked
archive" would be byte-identical to the existing frontier (`5e781e8e…` member `x`, 177,169 B). **byte_delta = 0 →
S_new = S_frontier = 0.19109982. The pointer cannot move via this path.**

## The exact-byte verification (NO FAKE — independently confirmed, not just trusting the memo)

1. **Frontier sha matches the pointer.** `shasum -a 256 …/submission_dir/archive.zip = b46897267ded…` == the
   canonical pointer's `our_local_frontier_contest_cpu.archive_sha256`. 177,169 B, single ZIP member `x` (177,069 B,
   `ZIP_STORED`).
2. **R1 + R2 are provably already in the base.** `byte_closure_proof.json` (the leapfrog's own fail-closed proof):
   `decoder_delta_bytes=−1023`, `latent_delta_bytes=−317`, `lossless_proof.decoder_raw_byte_identical=true`,
   `latent_raw_byte_identical=true`, `sidecar_byte_identical=true`. The recoded archive (`b46897267ded`) is the
   PR#112-coder-applied version of the R3 source (`1ccae18d…`, 178,495 B). R1 and R2 are not pending moves — they
   are the −1,326 B that already separates the frontier from its R3 source.
3. **S12 is inapplicable — confirmed at the inflate runtime.** `submission_dir/inflate.py:677` is
   `decoded = decoder(latents[i:j])` — the 1,200 frames are **generated** by `HNeRVDecoder(latents)` at inflate,
   rounded to uint8, and written. Member `x` layout (from inflate.py's own docstring): `decoder weights (CTXR) |
   latents (CTXR AR) | sidecar 607 | FECa selector 222 | DQS1 tail 42`. There is **no stored camera-frame uint8
   plane** with a resize-null structure for S12 to fill. S12's own docstring scopes it to frame-storing vehicles
   (SNeRV render, frontier-compose, raw-frame archives). This matches `t1_s12_lossless_stack_verdict_20260610.md`'s
   independent finding ("S12 has no addressable bytes here").

## Lossless-parity proof (the gate) — vacuous by construction

No candidate archive was materialized (the stack reduces to the existing frontier), so the LOSSLESS-PARITY gate
has nothing new to run. The frontier's parity is already proven exactly at two levels by the leapfrog +
cuda-pairing verdicts:

| level | proof | result |
|---|---|---|
| decode parity (1,200 frames, 3.66 GB raw) | `decode_parity_proof.json` | recoded sha `dacf6b33…` == R3 sha `dacf6b33…` BYTE-IDENTICAL |
| score parity contest-CPU (600 samples) | Modal `fc-01KTRAYS68…` | d_seg 0.00055978 / d_pose 0.00002942 byte-identical to R3 |
| score parity contest-CUDA (600 samples) | Modal `fc-01KTRCQ6KY…` | sha-identical archive, lossless-axis-invariant to 8 decimals |

Per-move parity verdict (what the task asked for):

| move | d_seg identical? | d_pose identical? | lossless? |
|---|:--:|:--:|---|
| R1 (already applied) | YES | YES | YES (proven, in base) |
| R2 (already applied) | YES | YES | YES (proven, in base) |
| R3 (the base) | — | — | — (it is the base) |
| **S12** | **N/A — no bytes to change** | **N/A** | **INAPPLICABLE (no stored frame pixels)** |

S12 was NOT dropped because it failed a parity proof — it was dropped because it has **no addressable bytes** on a
procedural HNeRV archive. There is nothing to fill, recode, or measure.

## Exact score (recomputed from components, the rounded field is irrelevant — they're equal)

`S = 100·d_seg + sqrt(10·d_pose) + 25·B / 37,545,489`, with the frontier's exact (already-measured, lossless)
components `d_seg=0.00055978`, `d_pose=2.942e-05`:

| | d_seg | d_pose | bytes | exact S |
|---|---|---|---:|---|
| frontier (pointer) | 0.00055978 | 2.942e-05 | 177,169 | **0.19109982419209975** |
| "stacked" (R1⊕R2⊕R3⊕S12) | 0.00055978 | 2.942e-05 | **177,169** | **0.19109982419209975** |
| **delta** | 0 | 0 | **0** | **0.0 — NO MOVE** |

Byte breakdown {R1 saved, R2 saved, S12 saved, new total}: R1 = −1,023 B (already banked in the base), R2 = −317 B
(already banked), S12 = **0 B (inapplicable)**, new total = **177,169 B = the frontier**. There is no fresh saving
to bank.

## Why no paired eval fired (the disciplined refusal)

Firing the prompt's confirming CPU+CUDA paired eval would dispatch the **exact same bytes** that are already
both-axis paired (CPU `fc-01KTRAYS68…` 0.19109982 + CUDA `fc-01KTRCQ6KY…` 0.22528084, both on sha
`b46897267ded`). It would spend ~$0.3–0.6 to re-confirm a confirmed frontier — exactly the waste the deferral
ledger §E E1 names ("re-confirming wastes ~$0.3 to learn nothing") and that MVP-first phasing forbids. **Refused.**

## Honesty tags (the DEFENSIVE BANK framing — unchanged, recorded)

- The recoded-R3 frontier is a **DEFENSIVE BANK**, not the innovative submission: R1/R2 are borrowed from PR #112
  (mattneel); it is a −2.6e-5 absorb-recode within contest reporting precision → it **fails the Innovation Gate as
  a SUBMISSION**.
- It is **submission-blocked** on two operator dispositions (deferral-ledger D1): the `constriction` import is not
  in the compliance allowlist, and the PR#112 attribution URL trips the no-network-string gate.
- It lowers (held) our LOCAL frontier; it is NOT the innovative submission. This verdict does not change that.

## Where the real remaining EV is (the honest reframe — feeds the planner)

The lossless RATE axis on this procedural HNeRV frontier is **exhausted** (decoder at 98.6% of iid Shannon; latents
at per-dim marginal floor, cross-pair MI = 0 — `t1_s12` + `frontier_latent_axis_waterfill` verdicts). The ONLY
lossless slivers left are LOW-EV, sub-contest-precision (T4 selector RLE −50–100 B; T9 decoder clustering −100–500 B;
T3 inflate-as-interpreter). **None is a pointer move.** The genuine remaining headroom is the **DISTORTION /
campaign axis** off the rate-saturated vertex (deferral-ledger §A): lever C (joint seg+pose frame1 carrier, the live
blocker on the confirmed lever B), AFSR-1 fresh-init smaller-arch + null-space-primary retrain, lever D contour
coder. S12 becomes a real lever there — **as a TRAINING CONSTRAINT** (synergy #3: train representation error INTO
the certified-invisible null space), the moment a frame-storing carrier exists. That is a NEEDS-CAMPAIGN build, not
a frozen-byte transform — out of this $0 ready-now task's scope.

## 6-hook wire-in (Catalog #125)

1. **sensitivity-map — ACTIVE:** the lossless rate axis (decoder + latents + selector) is confirmed at its floor;
   no fresh byte-saving direction on the frozen frontier. The aiming surface is the distortion/campaign axis.
2. **Pareto — ACTIVE:** the frozen-bytes lossless-rate vertex is saturated; R1⊕R2⊕R3 is the vertex, S12 is
   off-axis for this vehicle class. No off-vertex lossless move exists.
3. **bit-allocator — NEGATIVE:** S12 reduces no coded bytes here (no stored frame plane) → not a bit-allocator
   primitive on this archive.
4. **cathedral-autopilot — NEGATIVE:** do NOT queue a "lossless stack" materializer on the procedural HNeRV
   frontier; it re-derives the existing bytes. The leapfrog already captured the achievable lossless gain.
5. **continual-learning — ACTIVE:** reseeds the V3 judge that the R1/R2/R3 stack IS the current frontier (not four
   pending moves) and that S12 is class-scoped to frame-storing vehicles — closing the planning-memo's optimistic
   four-orthogonal-move framing against the EXACT frontier.
6. **probe-disambiguator — RESOLVED:** "does R1⊕R2⊕R3⊕S12 lower the exact pointer?" → NO. R1/R2/R3 are already
   stacked (the frontier); S12 has zero addressable bytes (procedural HNeRV, inflate.py:677 generates frames). No
   second interpretation survives the byte-closure proof + the inflate-runtime inspection.

## Provenance

- Frontier archive sha256 `b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e`, member `x`
  `5e781e8e…`, 177,169 B (== canonical pointer, verified). Both-axis paired (CPU 0.19109982 / CUDA 0.22528084).
- R1/R2 already-applied proof: `experiments/results/pr110_payload_entropy_recode_20260610/byte_closure_proof.json`
  (`decoder_delta_bytes=−1023`, `latent_delta_bytes=−317`).
- S12-inapplicable proof: `submission_dir/inflate.py:677` (`decoded = decoder(latents[i:j])`); independent
  confirmation in `t1_s12_lossless_stack_verdict_20260610.md`.
- This verdict's machine-readable artifact: `experiments/results/lossless_stack_20260610T165749Z/lossless_stack_parity_proof.json`.

**Cross-refs:** `leapfrog_pr112_absorb_recode_verdict_20260610.md` (R1+R2+R3 = the frontier) ·
`t1_s12_lossless_stack_verdict_20260610.md` (S12 INAPPLICABLE + latent floor) ·
`stacking_synergy_composition_plan_20260610.md` (the four-move premise, corrected here against the exact bytes) ·
`deferral_recovery_ledger_20260610T130200Z.md` §E E1 (the "do NOT re-confirm" guard) + §A (the live distortion-axis
levers) · `cuda_pairing_recoded_r3_verdict_20260610.md` (the both-axis pairing this would have re-confirmed).
