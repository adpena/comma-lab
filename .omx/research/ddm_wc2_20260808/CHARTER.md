# ddm_wc2 — HETEROGENEOUS COMPUTE (GPU+ANE+CPU same-run) + MEMORY-LEVERAGE design/build for the composed-vehicle receiver builds (operator 08-08 ×2 steers)

**Operator steers (binding):** (1) "Seems like we're not fully leveraging the memory on our
machine" — MEASURED: ARM-VEH peaked 1,552 MiB against a 45,000 MiB safe_run cap on a 128 GB
M5 Max (~1.2% of envelope; memory ceiling law m79 = 116 GiB, admission gate reclaimable-aware
m78). (2) "We have GPU, ANE, and CPU that we can all use and leverage and build custom kernels
for and use in the same run."

**Recall-first — the banked corpus (cite, extend, NEVER rebuild):** wc1 (26f6a5aa3d) built the
mx1 trainer's default-off levers + 5-variant bench (MAIN is firing it now — its measured rows
land in `.omx/research/ddm_wc1_20260807/wc1_bench_receipts.jsonl`; CONSUME them, the bench is
the baseline instrument). #477 precision-format × backend matrix · #482 ANE correction ladder
(decompose → precision-split → calibrated corrector → band-tile w/ donated SE → W8A8 →
composed) · #478 custom Metal conv suite (pointwise-GEMM + depthwise-fp16) · fm2/#980 verified
the CoreML/ANE on-device leg · #330 async subprocess verdict reclaim · #495 pose-verdict gate ·
#455/#456 cheap/distilled frozen-SegNet forwards · batch_saturation_throughput_20260623.json ·
MPS train/authority split law (train-gradient OK, NEVER score authority).

**Deliverables (research + design + IMPLEMENT to admission, default-off, on the LIVE mx1 vehicle):**
1. **Memory-for-compute config law:** derive + implement the n120/n600 batch/microbatch/cache
   plan against the 116 GiB ceiling — RAM-resident input/target caches (943 MB each today),
   full-batch vs chunked forward projections (measure via mem-probe path, no guessing),
   `--verdict-batch-size` scaling, and a derived (not hardcoded) microbatch law replacing the
   GPU-default-4 in `_derive_train_microbatch_pairs` (provenance-laddered).
2. **Same-run heterogeneous pipeline:** (a) CPU-torch verdict/facets CONCURRENT with Metal
   training (async subprocess per #330 pattern, killpg-reclaimed, custody-safe receipts) —
   verdict wall-clock leaves the Metal critical path; (b) ANE leg: frozen SegNet forward via
   CoreML (#482 ladder, fm2 wrappers) for VERDICT/FACETS duty — ARGMAX-PARITY-GATED vs the
   CPU-torch authority on real frames before ANY use, advisory-only labeling regardless
   (MPS/ANE never authority); (c) custom Metal kernel fit assessment for the w96 conv
   renderer fwd/bwd (#478 suite port cost vs measured MLX conv baseline — HONEST no-go if MLX
   convs are already near roofline; measure, don't presume).
3. **Bench extension:** add the new variants ({ane-verdict, concurrent-cpu-verdict, ram-cache,
   derived-microbatch-N}) to the wc1 bench harness so MAIN can fire ONE governed Metal batch
   for the whole matrix.
4. **n120/n600 receiver-build config recommendation** composing wc1's measured winners + the
   above, DSL/ticket-compatible, every constant provenance-laddered.

## OPTIMAL FORM
Build+design arm at reference form: real payloads/caches only (tq1c/gt caches, real ckpts),
no synthetic fixtures except kernel-parity unit vectors (declared TOY-BRACKET). Mechanism
reductions FORBIDDEN: ANE leg must run the REAL frozen SegNet (converted), parity measured on
REAL frames; memory projections via the existing mem-probe instrument, not arithmetic alone.
SCOPE reductions legal+declared (n32 first, parity subsets ≥ stratified n32). Provenance pins:
wc1 commit 26f6a5aa3d + bench receipts sha (consume post-fire) · fire_argv_final.json ·
m78/m79 memory laws · #477/#482/#478/#980 artifacts as recalled from their memos · repo HEAD.
**Boundaries:** CPU-only for the arm itself (OMP/MKL ≤4); NO Metal fires (emit bench
tickets/variants for MAIN); ANE/CoreML conversion + CPU-side parity checks allowed (they are
not Metal); NO scorer slot beyond CPU-torch parity subsets; $0.
**Discipline:** serializer + POST-EDIT --expected-content-sha256 per file; tags [no-triality]
[p0-ledger-ok]; NO Claude/AI attribution or Co-Authored-By; review_tracker ×2 per pact .py.
Recall-evidence section mandatory. Follow-on disposition table (no orphans). Findings:
`.omx/research/ddm_wc2_20260808/WC2_FINDINGS.md` + typed rows. If serializer hits sandbox
git-perms, write artifacts + say so.
