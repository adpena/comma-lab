# RELAY → ddm_mp2 (from MAIN, 2026-08-15): your admission baseline is MEASURED

The hv1-base advisory n600 row your charter's admission rule needs now EXISTS — do not re-run it.
- Receipt: /Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/contest_auth_eval.json
  (work dir retained at .../work_r2; launcher receipts in .../launcher_r2).
- Base on the mirror-CPU advisory chain (mirror upstream_eval_mirror_20260815, hv1 generation
  archive sha 80d9c8c6… @182,759 B): **d_seg 0.00042714 · d_pose 1.4747e-4 ·
  S_advisory 0.20280753928705508**. evidence_grade "auth-eval env mismatch advisory" — the
  DELTAS vs this row on the SAME chain are your decision quantities (admit < −3.5e-6 net,
  components recomputed).
- TWO launch-env laws proven on this chain today (bake into every advisory launch):
  (1) pre-launch sweep `/usr/bin/find <dirs+mirror> -name '._*' -delete` (ExFAT AppleDouble);
  (2) `PYTHONDONTWRITEBYTECODE=1` in the env AND sweep `__pycache__` from the mirror —
  a run otherwise writes bytecode INTO the mirror and the next run's authority hasher
  fail-closes (r1 of the base leg died exactly this way; r2 with the cure ran clean, ~45 min).

## RELAY 2 (operator steer 2026-08-15 "Remember quantum and regions and cells too")
Stage 2 (carrier rank/refit) MUST carry the quantization-toolbox arm: race per-cell/per-coefficient
ADAPTIVE quantization + sub-int16 depth of the 22,032 B basis+coeff pool AGAINST global rank
reduction (aware-solve in-loop, never round post-hoc; naive re-score first as the cheapest rung).
A rank verdict without the adaptive-quant arm is MECHANISM-scoped, not the family verdict.
Full amendment text: the OPERATOR AMENDMENT section of
.omx/research/charters/ddm_wd3_scorer_aware_width_distillation_20260815.md.

## RELAY 3 (operator steer 2026-08-15 "We can also do selective and surgical targeting")
Stage 1's FiLM-row candidates = surgical row-level edits: keep per-row attribution in receipts.
Stage 2's adaptive-quant waterfill = selective by construction: persist the SELECTION MAP (which
coefficients got which depth), not only aggregate bytes. Bounding laws: selection re-priced under
joint remeasure (never ranked once, #873); validation subsets strided/stratified never prefix
(m88/m96). Full clause: wd3 charter OPERATOR AMENDMENT clause 4.

## RELAY 4 (MAIN adjudication, 2026-08-15 ~20:3xZ — candidate 1 REFUSED, pose-dominated)
score_gated_selected_mixed_q3q4 attempt_0000 (rc=0, 4,550 s, receipt + work dir retained):
seg 0.00042828 · pose 0.00073123 · 181,936 B. Same-chain deltas vs base:
Δd_seg +1.14e-6 (ΔS +1.14e-4, near-neutral) · Δd_pose +5.8376e-4 = **4.96× base**
(ΔS +4.711e-2) · Δbytes −823 (ΔS −5.48e-4, exactly the mz2 projection).
**NET +4.668e-2 → REFUSED.** Mechanism = the recurring vehicle law (wd2, qs4, now q3/q4):
seg tolerates representation cuts; POSE reads near-photometric precision and pays 86× the
rate prize here. verdict_scope: instance (q3/q4 depth map on this pool).
CONSEQUENCE FOR STAGE 2 (binding): the adaptive per-cell waterfill's selection metric MUST
be pose-sensitivity-weighted (pose term in the per-coefficient gate, not bytes+seg alone);
a depth map selected without a pose channel repeats this refusal by construction. The
FiLM-row prune family (candidate 2 running) is a DIFFERENT mechanism (row sparsity vs
precision cut) — pose risk remains, same gate applies.

## RELAY 5 (operator doctrine 2026-08-15 "All negative signal is signal that directs us to
## what needs engineering and design and optimization attention and love" — the instruments)
The refusal trio (wd2 student · qs4 stale-Schur · q3/q4) all point at ONE engineering object:
a measured PER-COEFFICIENT POSE-SENSITIVITY FIELD over the carrier pool. Do not reinvent it —
assets that exist:
- ms6 receiver-support probe pattern (perturb at the actual quantum step THROUGH the real
  receiver/R/uint8/scorer; probe-then-aggregate, retained custody) — the proven mechanism for
  exactly this field, at row/tensor granularity.
- Strided/stratified subset law (m88/m96): a pose FD probe on a prefix subset is
  anti-conservative 2.5–4.2×; stride the pairs.
- pz4a DEAD-END (do not redo): sensitivity-allocated coarsening of the POSE coefficients
  themselves came out +2,232 B net (#1062). The live question is the SEMANTIC/FiLM pool's
  pose sensitivity through the RENDER — a different object.
- wd3 (parallel arm) is the training-time version of the same cure (pose term in-loop);
  your Stage-2 gate is the compress-time version. Same law, two surfaces.
Minimal Stage-2 compliance: per-row/per-tensor FD pose probe at quantization step size on a
strided pair subset → pose channel in the waterfill metric → joint remeasure on survivors.
