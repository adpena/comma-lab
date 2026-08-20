# ddm_rc2 — the composed shipping object: clean port × rider, decode-proven, sealed

## MANDATE

`ddm_rc1` (`5047803cf0`) merged the rr8 native corrector with the rr5 CPR1 rider and proved
their **parse** semantics. `ddm_rv16` then narrowed that memo's title at source: the port leg's
bit-identity is INHERITED from rr6/rr8 measurements on a **different tree** (the instrumented
one), and the composed object has never executed. This arm pays exactly that debt.

Three items are owed before the sixteenth-move row can fire. All three are $0 and local.

1. **Clean-vs-instrumented byte-neutrality.** The rr8 T4 row measured
   `candidate_runtime_jg5_native_corrector_instrumented/` (36 files, emits per-stage timing).
   The shipping candidate must be the CLEAN tree `candidate_runtime_jg5_native_corrector/`.
   A seal proves content identity, not behaviour. Prove the clean tree decodes to the same
   bytes — do not infer it from the instrumented row.
2. **Composed-tree decode smoke through the REAL `inflate.sh`.** The rc1 proof ran the parse
   layer only. The port leg needs its native `f26_corrector_native.c` to COMPILE and run in the
   target env; the rider leg needs `restore_carrier_body` inside that same execution. Compare
   decoded output against the retained jg5 raws (`7246a4ff…` / `6bf8acf8…`).
3. **Seal ONE candidate** via `tools/make_candidate_seal.py` (dual-axis; content-only digest;
   NO hand-typed sha). Emit a sealed fire-order. **Do NOT fire** — MAIN owns dispatch.

Composed object: archive `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080`
@ 180,456 B × {clean port + rider} runtime. Conditional arithmetic gives
**S 0.14827847122030854** (−1.125302e-04, rate-only) IF composed semantic identity holds.
That number is a HYPOTHESIS until the composed decode runs — say so in every line that quotes it.

## THE 464 s QUESTION — do not transfer it

rv16 F1 narrowed the rr8 headline to "one instrumented instance passes." The composed clean
tree's inflate wall is **unmeasured**. If your smoke can time the decode at real n600 scope,
report the measured seconds. If it can only run a bounded subset, say which subset and do NOT
extrapolate to a wall — the [[cross-regime constant transfer]] genus has cost this campaign
three separate corrections this week (cd1's split, rr5's 183 B, rr7's 1.865×).

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. Do NOT touch `submissions/robust_current/jg5_sub015_runtime/`
  (seal-pinned custody, scanner-exempt).
- **Do NOT fire any Modal dispatch.** MAIN owns dispatch and single-flight.
- Custody on **APDataStore** (`/Volumes/APDataStore/pact/ddm_rc2/`) — Vertigo is full.
- ALWAYS KEEP THE PAYLOAD: every materialized artifact persisted with sha256 + byte count.
- `.py` edits: 2 genuine review-tracker passes; commit via `tools/commit_autosha.sh`.
- Detached launches ONLY via `tools/launch_detached_process.py`.

## OPTIMAL FORM

- **Family reference:** the canonical byte-close → decode-identity → seal chain at its landed
  form — `ddm_rr6` (native port decode identity, token bit-identity proven at n600 scope) and
  `ddm_rr5` C1/C2/C3 (arithmetic round-trip · carrier-body identity · real-receiver parse).
  Reference instance for the seal: `tools/make_candidate_seal.py` + the 36 executed controls
  landed at `361608c875`.
- **SCOPE reductions permitted, declared per row** (e.g. a bounded-pair decode smoke instead of
  n600 — state the pair count and that a wall figure does not follow).
  **MECHANISM reductions FORBIDDEN:** no parse-layer stand-in for a real `inflate.sh` run; no
  seal built on a hand-typed sha; no identity claim inherited from a different tree.
- **Provenance pins:** rc1 memo `.omx/research/ddm_rc1_composed_candidate_20260820.md`
  (`5047803cf0`) · rr8 verdict `.omx/research/ddm_rr8_t4_wallclock_verdict_20260820.md`
  (`fa6863305c`) · rr5 re-measure `.omx/research/ddm_rr5_jg5_rider_remeasure_20260820.md`
  (`edf4fc0608`) · rv16 findings `.omx/research/ddm_rv16_round3_finding_wave_20260820.md`
  (`052a5f3693`) · composed tree `/Volumes/APDataStore/pact/ddm_rc1/candidate_runtime_composed`
  (37 files) · clean port tree `candidate_runtime_jg5_native_corrector`.
- **PRIOR-LAW PREDICTION (derived, falsifiable):** the line-disjoint merge (`git merge-file`
  rc=0, byte arithmetic additive to the byte: 27,520 + 1,908 + 630 = 30,058) and the proven
  `carrier_blob` parse identity together predict the composed decode is **byte-identical to
  jg5's raws**. FALSIFIER: any decoded-byte difference, or a native compile failure in the
  target env, refutes the composition at INSTANCE scope and routes to per-leg bisection
  (run each leg's tree alone against the same raws to localize).

## DELIVERABLE

`.omx/research/ddm_rc2_composed_clean_decode_seal_20260820.md` — per-item rows
{item · what EXECUTED at what scope · measured result · verdict}, the composed seal JSON path
+ sha, a sealed fire-order for MAIN (dual-axis, single-flight), and any refutation localized
per leg. End with the own-vehicle frontier line.

---

## AMENDMENT 1 (MAIN, binding — 2026-08-20, post-spawn)

**The deliverable SPLITS into two independent objects. The original charter merged them; that was
MAIN's scoping error, not the arm's.**

**VERIFIED AT SOURCE:** `canonical_frontier_pointer.json` gives
`effective_frontier` = `our_local_frontier_contest_cuda` = **0.14839100138338618**, archive
`f3bce5d259a08183...`. The rr8 T4 receipt scored that **same 180,625 B archive** and recomputes
identical. ⚠ TRAP: that receipt's `final_score` FIELD reads `0.15` — the rounded 2-decimal DISPLAY
value (Catalog #877). Never cite the field; recompute S from components.

### OBJECT A — THE SHIP PATH (do FIRST, report separately)

`{jg5 archive f3bce5d2... 180,625 B, UNCHANGED}` x `{CLEAN port runtime
candidate_runtime_jg5_native_corrector}`

Its SCORE IS ALREADY CONFIRMED (0.14839100138338618, contest-CUDA T4 n600) and its decode wall is
MEASURED (464.559 s) — but both came from the INSTRUMENTED tree. The ONLY open question is whether
the CLEAN tree behaves identically. That is charter item 1 plus a real `inflate.sh` decode of the
clean tree against the retained jg5 raws. If clean-tree decode is byte-identical, Object A is a
shippable sub-0.15 candidate whose authority row ALREADY EXISTS — no new T4 fire is needed for the
score (MAIN may still elect one on the exact clean tree). **State that conclusion explicitly.**

### OBJECT B — THE UPGRADE (second)

`{rider archive df7fd266... 180,456 B}` x `{clean port + rider runtime}` — charter items 2+3 as
written. This is −169 B / −1.125e-4 S and **REQUIRES a fresh authority row** because the archive
bytes differ. Its S 0.14827847122030854 stays a HYPOTHESIS until that row lands.

Everything else in the charter stands unchanged: no Modal fire, APDataStore custody, keep the
payload, no mechanism reductions, per-row scope declarations. **If Object A's clean-tree decode
REFUTES byte-identity, that is the highest-value finding of the arm** — report it immediately and
do not proceed to Object B until it is localized (per-leg bisection as chartered).
