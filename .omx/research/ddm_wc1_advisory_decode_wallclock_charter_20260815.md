# CHARTER — ddm_wc1_advisory_decode_wallclock (2026-08-15)

**Operator steer (binding, 2026-08-15):** "Wall clock on decode side needs iteration and
optimization." The advisory-chain n600 eval is the measurement loop that gates every
admission decision; each eval costs 45–76 min, of which ~32 min is decode. This arm cuts
the ADVISORY decode wall-clock by composing existing, identity-proven pieces — it does NOT
touch the shipping packet.

## Objective

Cut the measured end-to-end advisory decode on the hv1 frontier base archive from
**1,905.44 s to ≤ 700 s (bar) / ≤ 450 s (stretch)** with EVERY identity gate PASS, and
make micro-edit candidate evals skip redundant decode work via a content-addressed cache.
All three levers land behind default-off flags on the advisory chain only; the shipping
`inflate.sh` / submission packet is OUT OF SCOPE (its decode budget is #835, a separate
adjudication).

## Measured starting state (provenance pins — verify each before building)

- **Decode profile receipt** (the f26p inflate report, base leg r2):
  `/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/launcher_r2/run.log` line 30.
  Total decode 1,905.44 s at cpu_num_threads=4 / interop=1 (torch 2.12.1 arm64):
  `neural_render_and_resize` **1,179.59 s (62%)** · `token_decode_or_checkpoint_load`
  **642.14 s (34%)** (token decode_runtime 566 s) · `frame0_selector_and_io` 79.68 s ·
  `archive_setup` 2.65 s. Raw output 3,662,409,600 B, sha
  `e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9` — **the bit-identity
  target for the hv1 base archive.**
- **hv1 frontier base archive** (the object the advisory chain decodes): sha
  `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`, 182,759 B.
- **Native token line, ALREADY ON MAIN:** f26q commit `8b8bb25e6f` (1.880632× token stage,
  383.354385 → 203.843359 s) and f26r commit `95fb2944f8` (final rung: token decode
  **147.005376584013 s** M5 4-thread, full-field identity receipts: token sha
  `9ba2e52b3096…`, quantized-logit sha `617e9fcf…`, CDF-input sha `ba0d529b…`, RC64 bit
  position 921964, forced-scalar twin PASS, deterministic 69,424 B binary rebuild).
  Sources: `experiments/ddm_f26q_f26_hpac_native.py` (ctypes binding),
  `experiments/ddm_f26r_hpac_hot_stage_final_rung.py` (rung driver),
  memo `.omx/research/ddm_f26r_hpac_hot_stage_final_rung_20260814.md`.
  CAVEAT: those receipts are on the MC36/F26 sealed archive (`f0ba4bb4…`, 186,269 B), the
  same F26/HPAC receiver family — applicability to the hv1 archive must be RE-PROVEN with
  fresh identity receipts on the hv1 bytes; never transferred by citation.
- **Advisory chain surfaces:** `experiments/ddm_f26p_f26_inflate_cpu.py` (the CPU-lift
  inflate driver that emitted the profile; has `.f26_decode_checkpoints` machinery,
  resume:false) · the canonical eval argv template in
  `experiments/ddm_mp2_advisory_queue.py` (contest_auth_eval.py → generation `inflate.sh`
  → generation runtime `f26_inflate.py`, sha `4718834f` = the pure-Python token path that
  burns the 642 s) · eval mirror `/Volumes/APDataStore/pact/upstream_eval_mirror_20260815`.
- **Frame-parallel precedent:** #592 INFLATE_WORKERS (banded trainer materializer) with
  parity receipts proving byte-identity under process-parallel decode.
- **Instrument law (et4, binding):** (code, weights, threads, batch) are part of the
  forward instrument. Any parallel/native variant must reproduce the retained baseline
  bytes exactly or it is a DIFFERENT instrument and its rows are inadmissible.

## The three levers (build in this order)

1. **Frame-parallel neural render (the 62%).** N worker processes, each running the SAME
   per-process thread/config instrument as the baseline (4 torch threads, interop 1);
   frames/pairs partitioned across workers; results assembled in canonical order.
   Byte-identity gate: full raw sha `e5539653…` reproduction on the hv1 base archive.
   If oversubscription breaks identity or throughput, derive worker count from free CPU/RAM
   at launch (derived-at-consumption; never a latched constant) and measure per-worker RSS.
   Expected: 1,179 s → ~200–300 s at 4–6 workers.
2. **Native token stage on the advisory chain (the 34%).** Wire the f26r native decoder in
   as an advisory-only, default-off path (env flag). Fresh identity receipts ON THE HV1
   ARCHIVE: decoded token field sha + quantized-logit sha + CDF-input sha + bit position +
   final raw sha, plus one forced-scalar twin run. Expected: 642 s → ~150–200 s.
3. **Content-addressed decode cache (the micro-edit multiplier).** Key = (archive section
   shas, decoder code sha, thread config). A candidate whose token/model sections match a
   cached decode reuses the cached token field (sha-verified) and only re-runs stages
   downstream of the changed sections. Parity gate: cached-vs-fresh identical on ONE real
   micro-edit candidate (e.g. an mp2-generation archive). Cache lives on the SSD tier with
   a manifest (path, bytes, sha256) per certify-or-block.

## OPTIMAL FORM

- **Reference form:** the measured chain itself — the f26p profile receipt
  (`/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/launcher_r2/run.log` line 30,
  raw sha `e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9`, archive sha
  `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`) and the f26r native
  line at its measured optimum (147.005376584013 s; commits `8b8bb25e6f` + `95fb2944f8`;
  token sha `9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52`; runtime
  `f26_inflate.py` sha `4718834f`). This arm composes those reference forms; it invents no
  new decoder math.
- **SCOPE reductions (legal):** n4/n32 prefix identity gates before any full n600 run;
  building levers behind flags before wiring defaults.
- **MECHANISM reductions (FORBIDDEN, would toy-bracket the arm):** approximate or
  reduced-resolution render, skipping the uint8/resize path, parity checked on subsets
  while claiming full-field identity, timing rows from a different thread/instrument
  config than the receipt claims.
- **Provenance pins for every reused component** are listed above (path + sha/commit).

## Gates, governance, discipline

- **Bit-identity is the admission gate for every lever.** A lever that cannot reproduce
  the retained baseline sha on the hv1 base archive does not ship, full stop. Per-frame
  sha manifests retained for every parity run; at least one full reference payload per
  archive retained on the SSD tier (ALWAYS KEEP THE PAYLOAD — never a scalars-only run);
  subsequent identical outputs may be certified byte-identical against the retained
  manifest instead of duplicating 3.66 GB payloads.
- **Every timing/parity receipt carries its instrument pin:** decoder code sha, archive
  sha, threads, interop, workers, host. Timing rows are `[M5-CPU scorer-free]` advisory —
  never score claims.
- **Full n600 runs are WATCHED detached launches** via
  `tools/launch_detached_process.py --arm-watchers` with liveness + quality configs AND
  the new `success_receipts` watcher family (tools/run_liveness_watcher.py — adopt it; a
  clean rc=0 exit must NOT alert). No hand-typed background launches; no fg sleeps >3 min.
- **Serialize behind the live advisory slot:** an mp2-queue eval may be running (check the
  attempt-dir `launcher/run.pid` liveness before firing any full-decode run); never
  co-run two 4-thread n600 decodes with a live Metal training fire without checking free
  RAM/CPU. Sweep AppleDouble `._*` from any mirror/runtime dir before launch and set
  PYTHONDONTWRITEBYTECODE=1 in launch envs (the pycache-contamination law).
- **Consumer wiring:** once parity is proven, thread the flags through the canonical
  advisory launch path (the mp2 queue argv template and the lh2 composer) so FUTURE
  admission evals consume the fast path by default — a lever without a recipient is
  orphan signal.
- **Commits** via `tools/subagent_commit_serializer.py` with post-edit
  `--expected-content-sha256`; .py files get 2 review_tracker passes; NEVER
  REVIEW_GATE_OVERRIDE on .py; no co-author trailer; `[no-triality] [p0-ledger-ok]` tags.
- upstream/ pinned snapshot READ-ONLY; the eval MIRROR is also never mutated (runtime
  copies live in generation dirs).
- Full research + original-derivation authority (the #984 clause) and full internal
  leverage (own code/docs/unwired = off-the-shelf) apply at this spawn site.

## Deliverables

1. The three levers as committed code behind default-off flags, each with its identity
   receipts (sha-pinned JSON) on the hv1 base archive.
2. ONE measured end-to-end advisory decode row on the hv1 base with all levers ON:
   total seconds + per-stage breakdown + instrument pin + final raw sha PASS.
3. Cache parity receipt on one real micro-edit candidate archive.
4. Memo `.omx/research/ddm_wc1_advisory_decode_wallclock_<date>.md` with: RECALL EVIDENCE ·
   measured table (before/after per stage) · verdict-scope labels · MEASURED vs DERIVED vs
   NOT-MEASURED boundaries · NEXT_IF_RESUMED · DEAD-ENDS. Lead with the finding.
5. Consumer wiring diff (mp2 queue / lh2 composer) gated on parity, or a typed blocker.

## What this arm must NOT do

- Touch the shipping packet, submission bundle, or `pq1` artifacts.
- Mutate upstream/, the eval mirror, live run dirs, or any retained custody.
- Claim any score row (decode wall-clock only; the scorer half of the eval is untouched).
- Transfer MC36-archive identity receipts to hv1 bytes by citation.
- Launch Metal/GPU work or paid dispatches.
