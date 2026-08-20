# CHARTER — ddm_f26p_runtime_cpu_lift (2026-08-14, operator runtime-lift grant)

OPERATOR GRANT (verbatim, binding): "we can monkey patch or port the inherited
runtime or lift it and then extend and enhance. We'd already done a lot of
porting work to port it to metal and MLX, which is most optimal if we have to
do long or expensive things." Memory:
runtime-lift-grant-port-inherited-decoders (08-14).

CONTEXT (recall, do not re-derive). The NEW effective_frontier is OUR MC36
Variant C row: S 0.1619344578804448 @ 186,269 B [contest-CUDA T4, n600],
archive f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de.
Its CPU leg FAILED structurally: `runtime/f26_inflate.py:106` raises
"F26 inflation requires a CUDA-capable GPU" (measured 3.2 s, rc=1, receipt at
experiments/results/modal_auth_eval_cpu/ddm_mc36_promotion_paired_modal_auth_20260814T182512Z_cpu/).
Per #998 the CPU axis on the PR130/PR135 lineage has NEVER been bought, and
device is a MEASURED score lever (PR102 CPU beat its own CUDA by 0.033; lc2
has a real CPU-vs-CUDA delta). The runtime ships in OUR archive — the CUDA
lock is inherited code, not a rule. T4 decode wall: 327.6 s. Contest CPU CI:
Linux x86_64, 4 cores / 16 GB, 30-min inflate budget.

## THE WORK (all local, scorer-light; NO Modal — MAIN fires all Modal)

1. **RECALL FIRST**: read `runtime_stage/` (the complete submission copy) at
   /Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/promotion_submission/runtime_stage/
   (archive.zip byte-exact candidate + inflate.py + inflate.sh + runtime/ +
   cpr1/). Read runtime/f26_inflate.py FULLY — every device-dependent site,
   not just :106. Recall the hb1/hb2 HPAC machinery (our trained-HPAC +
   round-trip fix work) + mlx_score_aware adapters + the fd135 decomposition
   memo for what each decode stage does.
2. **LEG A — CPU-unlock patch**: produce `f26_inflate_cpu.py` (or a device
   parameter in a LIFTED copy — never edit the sealed runtime_stage in place;
   copy to a work dir). Remove/parameterize the CUDA gate; route every
   `.cuda()` / device literal through one `device` argument. The decode must
   remain semantics-identical: same ops, same order, same dtypes where
   device-portable.
3. **LEG B — identity vs RETAINED custody ($0, decisive)**: run the CPU decode
   end-to-end locally on the REAL archive; sha256 the raw decoded output in
   the SAME canonical order the T4 worker used; compare to the retained T4
   aggregate sha `a41ca69d2288d3edd8f009b03404ef070661297a8f962a067e663ff26f7c0e8b`.
   MATCH ⇒ bit-identical decode (the CPU row's input to the scorer equals the
   CUDA row's; remaining CPU-vs-CUDA delta = scorer numerics, the exact #998
   question). MISMATCH ⇒ quantify (n frames divergent, max abs) — that is a
   finding, not a failure; report it typed.
4. **LEG C — budget measurement (honest instrument)**: measure full-decode
   wall-clock with `torch.set_num_threads(4)` + OMP/MKL caps = the contest
   core count. Label the number `[M5-CPU 4-thread LOWER BOUND on contest
   wall]` — M5 perf cores are faster than CI Xeon; do NOT claim the contest
   budget passes from the local number alone. Verdict ladder: <15 min local ⇒
   LIKELY-IN-BUDGET (MAIN buys a ~$0.10 Modal CPU dry run to confirm);
   15–30 min ⇒ MARGINAL (name the hot stage); >30 min ⇒ OVER — go to Leg D.
5. **LEG D — hot-stage profile + lowering plan (fires if C is MARGINAL/OVER)**:
   profile the decode (entropy/HPAC decode vs neural render vs resize vs I/O);
   for the hot stage, write the lowering plan ranked {CPU-torch vectorization ·
   numpy/BLAS rewrite · Rust lowering (full native grant, bit-parity gated per
   runtime-rs precedent)} with projected speedups DERIVED from op counts, not
   guessed. MLX/Metal is the LOCAL substrate for expensive encode/solve work,
   NOT the contest CPU row (MLX does not run on the contest host) — say so
   explicitly in the memo so nobody routes the contest row through MLX.
6. **LEG E — MLX/Metal asset inventory (the operator's pointer)**: table of
   our EXISTING ports (hb1/hb2 HPAC, mlx_score_aware adapters, Metal kernels)
   × which F26 decode/encode stages each maps to × what long/expensive
   lineage work each unlocks locally (js1-line solves, HPAC retraining,
   candidate screening) — named consumers, no orphan rows.

## OPTIMAL FORM

Family reference PINS: archive f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de
@186,269 B · runtime_stage runtime tree (CUDA-side expected sha
776849ba00fa0e942c84ec63643ef067324a021f139726afff80855cfb613db9) · retained
T4 raw-decode sha a41ca69d2288d3edd8f009b03404ef070661297a8f962a067e663ff26f7c0e8b ·
T4 decode wall 327.6 s · promotion memo
ddm_mc36_promotion_complete_s_verdict_20260814.md · CPU-failure receipt path
above · hb2 HPAC round-trip fix · #998 device-lever row. SCOPE reductions
legal (profile on a frame subset AFTER the full-decode identity run);
MECHANISM reductions = TOY-BRACKET (a synthetic-archive decode, an uncapped-
thread wall-clock quoted as the budget number, or skipping the sha identity
check can NOT produce a family verdict). Payload law DEF CON 1000: persist the
CPU raw decode (or its per-frame sha manifest at minimum) to the Vertigo tier
— never sha-and-discard without the manifest. Arms cannot run Modal or Metal;
MAIN fires the Modal CPU dry run + any exact row. Git-blocked ⇒ declare memo
SHA for MAIN handoff.

## OUTPUT

Work dir: /Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/.
Code: experiments/ddm_f26p_f26_cpu_lift.py (runner) + the lifted decode module.
Memo: .omx/research/ddm_f26p_runtime_cpu_lift_20260814.md — identity verdict ·
4-thread wall + honest label · budget-ladder verdict · hot-stage table (if
run) · MLX asset inventory · EXACT Modal-CPU dry-run fire order for MAIN
(experiments/modal_auth_eval_cpu.py::main form, submission_dir = the LIFTED
runtime staged as a NEW candidate dir with its own runtime-tree sha — a
runtime edit is a NEW object, full sha custody). Commit via
tools/subagent_commit_serializer.py (post-edit shas, `[no-triality]
[p0-ledger-ok]`, no co-author trailer; .py needs review_tracker mark-file).
End with NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS.
