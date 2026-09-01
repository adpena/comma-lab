# ddm_jbp1_joint_batch_price — execute gen-3's QUEUED-BATCH-PRICE fire order on the NARROWED roster: exact joint G/M pricing of the B/H/W support + SFP1 changed-field proposals through the retained rxc1 instrument (task #1374; fire order in .omx/research/ddm_rxc1_gen3_gate1_verdict_20260901.md §NEXT_IF_RESUMED row 1)

## MANDATE

Gen-3 landed **GATE-1-PARTIAL, Branch 1** (memo ddm_rxc1_gen3_gate1_verdict_20260901.md):
exactness PROVEN (64/64 incremental comparisons byte-identical to full re-encode), economics
suffix-priced (best aggregate 450.593 s/proposal, splice closed 0/32) — so SCMDL pricing runs as
**bounded batched exact re-encodes**, never a per-proposal outer loop. Its fire order authorizes
one bounded batched pricing pass over the roster. RECONCILIATION (binding, post-dates gen-3's
spawn): ddm_dds1_ceiling_readjudication_20260901.md CLOSED xov1 candidates 1+2 (born expert ·
generator-conditioned peel chain) BY CEILING — GF1's own all-live ceiling ~613 B vs its 47,603 B
counted packet = 77.6× underwater; the M-only rider ceils at ~2.08 B. They do NOT enter this
pass; re-entry requires refuting that ceiling arithmetic at source, not silent re-inclusion.
The admitted roster: **(A) the 5,506-record directed B/H/W support** (xov1 candidate 3, rate-first
field edit) and **(B) SFP1's 3 ranked changed-field proposals + the byte-identical null control**.
Each is an executable change to the coded field X priced by ONE exact suffix re-encode through
the proven instrument (~450–718 s each; ≤5 re-encodes ≈ ≤1.2 h, DETACHED).

## SCOPE

1. **Identity gate first**: run the SFP1 null (empty) proposal through the instrument — the
   re-encoded archive must be BYTE-IDENTICAL to the base. Any deviation → STOP, typed blocker
   (the instrument's identity claim is INSTANCE-scoped to gen-3's 32 rows; this extends it to
   the pricing configuration before any candidate row is admitted).
2. **Candidate A — B/H/W support**: apply the 5,506 directed edits (records at
   `/Volumes/APDataStore/pact/ddm_xov1_crossover_pass/retained/cross_bhw/cross_parent_bhw.records`,
   custody per xov1 memo sha `ad093da3358996cb30700a2d0976af2436b10ea570eeada276580336b6ce6345`)
   to the coded field → ONE exact joint re-encode (G/M refit where the edit demands it — the
   jc1 joint leg, never an additive estimate) → exact archive bytes vs base.
3. **Candidates B1–B3 — SFP1 proposals**: same protocol per ranked proposal (memo
   ddm_sfp1_scmdl_field_proposal_prep_20260901.md sha
   `af70ab65c258b8700851bdf525ab8dc1c58b41cf34b374403e9c8e67ad48538b`; base X sha
   `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`, 117,964,800 B).
4. **Demand arithmetic per row**: exact Δbytes vs the 39,522.14 B cut demand (allowance
   87,403.86 B on the 126,926 B joint pool; affine archive ceiling 140,479.86 B — gen-3 §XOV1/JC1
   consequences). Typed verdict per candidate: {BYTE-WIN by X B / REFUSED +X B}, plus the honest
   fraction-of-demand line. RATE ROWS ONLY — no distortion claim of any kind; a byte-winning
   candidate owes fresh pose compensation + a matched n600 Seg/Pose gate before any authority
   request (gen-3's QUEUED-CONDITIONAL-AUTHORITY trigger: receiver-closed ≤140,479.86 B + fresh
   full-n600 component receipts).
5. **Route the outcome**: byte-winner(s) → typed fire order to MAIN (seal → advisory → authority
   chain); all-refused → the roster is EXHAUSTED-MEASURED — state plainly what remains live in
   #1374 (jc1 bounded-reset causal-state redesign per gen-3's QUEUED-STRUCTURAL-LOCALITY row).

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- The local SCORER LANE belongs to MAIN, always. Do NOT write who currently holds it into a
  charter: an occupancy claim goes stale the moment that holder exits, and the arm has no way
  to learn it did (the #1210 stale-precondition genus — MEASURED 2026-08-29, when
  `ddm_bz2_bornsmall_capacity_ceiling` correctly refused to claim a capacity ceiling because
  a charter told it a since-released lane was taken). If this arm's work needs a scorer run,
  emit a typed fire order naming its trigger and let MAIN fire it; landing an honest partial
  plus a fire order is the CORRECT outcome, never a failure.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_jbp1_joint_batch_price/`
  (per-candidate edited fields + exact streams + receipts, gen-3's row schema; preserve the
  mandatory 1 GiB AP reserve — free measured 52.4 GB at gen-3 close, re-verify AT START, #1210).
- OWNERSHIP: this arm inherits `experiments/ddm_rxc1_restartable_exact_coder.py` +
  `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/` (gen-3 exited rc=0). The 26+6
  sealed screen rows are read-only custody — extend, never recompute; sg2b + xov1 + sfp1 +
  dds1 stores READ-ONLY.
- DETACHED >30-MIN COMPUTE: any single compute step projected to exceed 30 minutes MUST
  launch outside the arm session with `nohup` + `disown`, a pidfile, crash-resumable stage
  checkpoints, and a durable done-receipt (script paths avoid claude/codex tokens — the
  fleet-reaper argv predicate). The arm MONITORS that process. In-session multi-hour loops
  FORBIDDEN.
- CLOSED-FORM-FIRST (operator 2026-08-31): the coder chain is deterministic exact math —
  price by REAL re-encode, never an entropy estimate (#1204/cm1); SCREEN bits only as
  refusal ceilings, never byte claims.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- ddm_sg2b_falsifier_verdict_20260901.md (receipt sha `ed592b68…a39b`): X-alone DISTORTION
  edits refused 3/3 at every dose — these rows are RATE-first; the charter makes NO distortion
  claim and forbids citing B/H/W labels as scorer evidence (wwc1 bd7c651b0a: label→realized
  transfer BROKEN at token-GT scope).
- ddm_rxc1_gen3_gate1_verdict_20260901.md DEAD-ENDS bind verbatim: no per-proposal outer loop ·
  splice closed on this instance · additive/entropy admission forbidden · exact-vs-exact
  correlation is vacuous.
- jt22/#1269: banked deltas are NOT additive without joint re-encode — every row here IS a
  joint re-encode; never sum candidate deltas.
- #1199: cross-parent agreement is a near-useless d_seg predictor (exponent 16.7) — selection
  heuristics stay heuristics.
- fcd2/fcd3 (fcd1 lineage): token-GT-selected edits failed REALIZED transfer twice — the reason
  rows are rate-only with the scorer gate deferred to MAIN.

## OPTIMAL FORM

- Family exemplar: gen-3's run is the reference form (memo ddm_rxc1_gen3_gate1_verdict_20260901.md;
  SCREEN.json sha `e6a400be9bb140bd220b4d5e77473a36dd692e2794cbd47f48a65b30d9309420`, MANIFEST.json
  sha `954ccada3094df4f22147d47225cab84238782291287ea51c52d1362d0a4ae2c`) — same instrument, same
  custody schema, same refusal discipline; this arm is its priced-candidate continuation.
  Reconciliation pin: ddm_dds1_ceiling_readjudication_20260901.md (roster narrowing authority).
- SCOPE reductions legal: none expected (≤5 re-encodes). MECHANISM reductions FORBIDDEN: real
  full suffix re-encodes per candidate, byte-exact archives, retained payloads; no sampling,
  no extrapolated costs.
- **PRIOR-LAW PREDICTION (falsifiable):** fcd1's win-win law (−5.7 bits/edit over 5,268
  GT-benefit edits ≈ −3.7 KB screened) predicts candidate A's exact joint re-encode shrinks the
  token subsystem by order −3–4 KB — roughly 10× SHORT of the 39,522.14 B demand. FALSIFIER:
  the exact row lands ≥ −10 KB (the joint refit finds structure the screen missed) → the
  win-win family re-opens at joint scope; count it plainly either way.

## DELIVERABLE

`.omx/research/ddm_jbp1_joint_batch_price_verdict_20260901.md` — typed rows: AP trigger check ·
null identity proof · per-candidate exact byte row {Δbytes, fraction-of-demand, verdict} ·
demand arithmetic · routing (fire order OR exhaustion + what stays live in #1374) · DEAD-ENDS +
denominator. Commit via the serializer (bundle-fallback on .git/objects denial, #1293). End with
the own-vehicle frontier line (S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600], afr1
sha cbb8d928…d405bf25 — UNMOVED unless a fire order lands).
