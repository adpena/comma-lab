# ck1 T4 refusal S=79.4 — mixed-lineage silent arithmetic desync (typed + cured)

`verdict_scope`: **INSTANCE** — archive `35c318d541d703708ab06c55473c200bb893491e24bea312e37be42f010677e3`
@ 177,182 B staged from the ck1 GENERATION runtime, contest-CUDA T4 n600
(call `fc-01M0BGYVF1RBYZKJPMA9KN7K9B`, ~$0.16, wall 456 s). The CLASS finding
(mixed-lineage assembly desyncs silently) is DERIVED from the module diff and the
receipt pair; the cured recompose is byte-identical on the archive side and
re-measured through the class-cure advisory below.

## 1. The row

d_seg **0.41580084** · d_pose **142.15991211** · S 79.40216174747616 — both scorers
destroyed. rc=0: the decode RAN to completion and produced garbage frames.
Receipt: `/Volumes/APDataStore/pact/ddm_ck1/t4_row/MODAL_REMOTE_RESULT.json` (persisted, P0).

## 2. Mechanism (EXACT, verified at source)

The compiled candidate paired **sz1's token bytes** with **ck1's runtime**:

- sz1's `runtime/free_corrector.py` imports `fx1_logistic_mixer_corrector` (+ `rr4_free_corrector`)
  — the fx1/rr4 probability chain the sz1 hpac section was ENCODED for.
- ck1's generation runtime forked BEFORE fx1/rr4: both modules ABSENT, its
  `free_corrector.py` diverges by 980 diff-lines. Same entry name, different
  probability model.
- Arithmetic decoding under a mismatched probability model never errors — it
  emits wrong symbols from the first divergent bin onward. Every downstream frame
  is garbage; both scorers pay.

Why nothing upstream caught it:
- **Parse-back is structure-level** (sections + shas round-trip) — it cannot see a
  probability-model mismatch.
- **All the scoring receipts in the admit arithmetic were measured on DIFFERENT
  objects**: the ck1 generation archive through ck1's runtime (advisory, authority
  solve) and sz1's archive through sz1's runtime. The COMPOSED candidate — sz1
  sections × ck1 runtime — was never decoded-and-scored anywhere before the paid row.
- NOT the rr2 DEVICE class (#1096) despite the surface resemblance: this desync is
  device-independent and would reproduce on CPU. The typing discriminator is
  "would CPU reproduce it" — here yes.

Bonus divergence found on the same diff: the RECEIVERS forked BOTH WAYS
(ck1 added SM3R mode-6; sz1 added SD1M depth rows). Neither tree alone decodes the
composed candidate.

## 3. Cures (landed this session)

1. **Recompose via `runtime_overlays`** (`experiments/ddm_sa3_rebase_sz1.py`):
   per-row list of (tree, relpath) files copied over the sz1 base runtime.
   ck1 row = sz1 runtime (fx1/rr4 chain intact) + ck1's
   `cpr1/ddm_mp2_semantic_receiver.py` (the only file that knows mode 6).
   Overlay refuses to ADD files blind (target must exist in base).
   r4 compile: parse-back PASS both variants, archive byte-identical
   (`35c318d5…` @ 177,182 — container assembly is deterministic).
2. **Pattern-based inflate.py re-pin, fail-closed** (same session, sister fix):
   the literal-substitution re-pin silently missed non-sz1 pins; now re-pins by
   assignment regex and refuses if no pin moved.
3. **CLASS CURE — composed-archive CPU advisory gate**: any candidate whose
   sections and runtime come from different lineages MUST be decoded AND scored
   end-to-end on local CPU (full n600 advisory) before a paid axis row. The
   keep01 chain got away without it because all its ingredients were
   same-lineage; ck1 is the first mixed-lineage compile and the first casualty.
   Fired as `/Volumes/APDataStore/pact/ddm_ck1/advisory_rebased/attempt_0001`.
   Structural (tool-enforced) version: follow-on — teach `stage_generation` (or
   the seal producer) to demand an advisory receipt keyed to the staged archive
   sha when `runtime_overlays`/`source_runtime` deviates from the base lineage.

## 4. Genus links

Sister of [[modal_dispatch_five_failure_classes_permanent_fixes_20260804]] (the
2-validators-disagree → env-coupled-digest law: here the two "validators" were
parse-back and the T4 scorer, and the missing shared object was a scored decode)
and of #1096 (rr2 device desync — same silent-arithmetic genus, different
trigger). The measured-object-vs-named-object law
([[measured_object_vs_named_object_20260816]]) applies exactly: the admit
arithmetic quoted receipts for objects that were not the fired object.

## 5. Chain state at memo time

Advisory attempt_0001 died at t=10s, rc=2 — NOT a decode failure: the #929
bare-`python` class through the LOCAL harness. The sz1-lineage `inflate.sh`
probes `python` (absent on this macOS host) → falls into its Brotli bootstrap
branch → `uv pip install --python python` → uv's "No interpreter found for
executable name `python`" → exit 2 under `set -e`, 0.0 s elapsed. Modal's T4
container has `python` on PATH, which is why the same script never failed
remotely. The V-series/keep01 advisories survived via
`/Volumes/APDataStore/pact/ddm_sa1/pyshim/python` (exec-wrapper → repo venv,
per `python-shim-must-be-exec-wrapper-never-symlink`) prepended to PATH in the
launch env — my attempt_0001 launch omitted that prefix. Cure = harness-env
fix ONLY (pyshim PATH + PYTHONDONTWRITEBYTECODE=1); the shipped tree stays
byte-identical. With the shim, the venv's Brotli 1.2.0 satisfies the probe and
the uv branch is skipped entirely.

Attempt_0002 relaunched with the corrected env (the proven keep01 template).
If d_seg lands ≈0.00043 band and pose residual ≈1.48e-4 band (ck1's own
AUTHORITY.json) → re-seal (new runtime digest; r3 seal STALE post-r4 overlay
re-stage) → T4 refire (~$0.16). If the advisory refuses on SCORE → the
authority solve's decoded-state identity assumption is next in the fault tree
(ck1's L_ck1 vs sz1's L_sz1 lattice).
