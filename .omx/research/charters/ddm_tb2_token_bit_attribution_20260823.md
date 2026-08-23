# ddm_tb2_token_bit_attribution — six arms measured the token stream in AGGREGATE; nobody asked where its 910,216 bits actually go, and the arithmetic coder already knows exactly (owning memo: ddm_ar1b_archive_residue_purchase_20260822.md)

## MANDATE

Operator 20260823: *"Audit all negative signal and continue with all when appropriate"* + the standing
full-authority directive to think divergently about frontier lowering.

`ddm_ar1b` mapped the dx2 archive with ZERO remainder: renderer 30,856 · carrier 22,010 · HPAC model
13,515 · compact residual 96 · framing 114 · **token stream 113,777 B**. The token stream is **63.1%
of the archive** and 2.68× the entire 42,382 B demand.

Six arms measured it — `ddm_to2` (ordering), `ddm_ad2` (addressing), `ddm_ef1` (entropy floor),
`ddm_cx3` (context ceiling), `ddm_sv2` (coder race), `ddm_rc4` (drop × carrier re-solve) — and every
one of them measured it **in aggregate**: a total, a floor, a delta. Not one asked the question the
arithmetic coder answers for free: **which symbols, at which positions, in which contexts, are
consuming the 910,216 bits?**

This is not a coder question. The coder axis is CLOSED (`ddm_r7`, #996: all four sections measured
vs their own memoryless bound). It is an ATTRIBUTION question, and it is the input every targeting
lever has been missing. `ddm_ad2` measured that reordering substitutes for a context model (+34.5%)
and `ddm_to2` measured the same law firing backwards (−196%) — both are aggregate symptoms of a
per-symbol structure that NO memo reports measuring (SEARCH SCOPE: all `.omx/research/ddm_*.md`
dated 2026-08-20..2026-08-23, read in full — state this scope in the deliverable).

`ddm_bl1` did per-position attribution for the **MODEL** bits and it was decisive: Lane is 0.59% of
area but 33.56% of model bits, top 1% of positions carry 96.32%, Gini 0.995. **The same instrument has
never been pointed at the token stream**, which is 8.4× larger than the model span.

## SCOPE

1. INSTRUMENT the live encoder to emit per-symbol bit cost — the arithmetic coder computes −log2 p
   per symbol by construction; this is a read, not a model. Persist the full per-symbol field
   (ALWAYS KEEP THE PAYLOAD: the field, not its summary statistics).
2. ATTRIBUTE the 910,216 bits along every axis the corpus already knows to matter: position ·
   spatial band · class (per the canonical comma10k order — Road 0 / Lane 1 / Undrivable 2 /
   Movable 3 / MyCar 4, and NEVER re-derive it by luma-sort) · pair index · context state · symbol
   value. Report concentration exactly as bl1 did: Gini, top-1%, top-decile mass.
3. JOIN to the seg axis. `ddm_bl1` found the seg×rate join ASYMMETRIC on the model span (seg errors
   are 2.01% of the expensive set but carry 5.27% of bits). Measure the same join on the token stream:
   do the expensive symbols coincide with the manufactured seg error `ddm_ms9`/`ddm_mst1` measured?
   The pattern-of-patterns claim ("what costs most is what hurts most", ar1b §) is either confirmed at
   a third granularity or REFUTED — say which, plainly.
4. NAME what the attribution makes targetable, and price nothing you have not measured. This arm's
   product is an INSTRUMENT and a MAP, not a lever. If it names candidate levers they are named as
   candidates with their falsifiers, never as projections.
5. Exchange rate 6.658590e-07 S/B — CITE `ddm_tx1_toolbox_crosswalk_20260819.md` §0, do NOT re-derive.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- NO Metal slot: jf1 holds it. $0, scorer-free, CPU only.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD (P0, DEF CON 1000): the per-symbol bit field is the deliverable payload.
  Persist it with sha256 + bytes to `/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/`.
  A scalar-only artifact when the field exists in memory is FORBIDDEN at the typing moment.
- Bit totals must RECONCILE to the measured 113,777 B stream. An attribution that does not sum to the
  real stream is an unsound instrument — report the residual explicitly, never silently.
- File-ownership: sibling arms ddm_na12/ddm_mf1/ddm_hr3 are LIVE — do not touch their surfaces.
  jf1's receipts under `.omx/tmp/arm_receipts_local/ddm_jf1_*` are SACRED. Shipped receiver bytes are
  CUSTODY — never edit in place.
- Every negative-existence claim states its SEARCH SCOPE or is not made (m53).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_r7_token_stream_coder_race` (owning memo for harness task #996) — THE CODER AXIS IS CLOSED: all four sections measured against their own memoryless
  bound. This arm is NOT a coder race and must not become one.
- `ddm_to2_token_ordering_race_20260822.md` — reordering LOST 196% on this stream.
  `ddm_ad2_addressing_cost_decomposition_20260822.md` — addressing is ALREADY FREE on dx2; the
  42,382 B must come out of pure payload. Both are aggregate results this attribution explains or not.
- `ddm_ef1_token_entropy_floor_20260822.md` — note MAIN's own recorded defect: that charter raced
  generic estimators against a domain-tuned learned model (the no-toy violation; owning memo `ddm_ef1_token_entropy_floor_20260822.md`, harness task #1202). Do not
  inherit its comparison shape.
- `ddm_cx3_context_axis_ceiling_20260822.md` — the context axis has a measured ceiling.
- `ddm_sv2` — SMEVR LOSES the IX2TOK01 token bulk by +5,183 B: the live coder pays for LZ MATCH
  STRUCTURE, not symbol rank. A per-symbol rank attribution must NOT be read as a byte prediction —
  that is exactly the wrong denominator (the reordering-substitutes-for-a-context-model law; owning memos `ddm_ad2_addressing_cost_decomposition_20260822.md` + `ddm_to2_token_ordering_race_20260822.md`, harness task #1201).
- **THE SHARP-OPTIMUM LAW** (five concordant arms): the HPAC model and field are jointly at a local
  optimum, SHARP in every measured direction. An attribution map does not escape it; it locates.

## OPTIMAL FORM

- **PROVENANCE PINS (verify before reading; do not work from memory):**
  - the residue decomposition: `ddm_ar1b_archive_residue_purchase_20260822.md`
  - the reference instrument: `ddm_bl1_per_position_bit_allocation_20260822.md` — this is the
    **reference** form (per-position bit attribution with Gini/top-1% concentration reporting),
    applied to the MODEL span; this arm ports it to the token stream
  - the coder-price law: `ddm_fs2_rc4_drop_carrier_resolve_20260820.md` (−log2 p is
    direction-dependent; ranker-based prices were 0.77–0.88× wrong)
  - current object: archive sha `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`
    (180,368 B); categorical field sha `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`
- SCOPE reductions LEGAL, declared per row: a pair subset with its selection STATED (never a bare
  prefix — m88: a prefix of a skewed population is a different population; rate-axis prefix bias is
  small and sign-variable per `ddm_na4`, but the selection must still be declared), a context-axis
  subset. MECHANISM reductions FORBIDDEN: estimating bit cost instead of reading it from the coder,
  or reporting summary statistics without persisting the field.
- **PRIOR-LAW PREDICTION (falsifiable).** bl1 measured extreme concentration on the model span
  (Gini 0.995) and ar1b's pattern-of-patterns held at two granularities. I predict the token stream is
  **ALSO concentrated (Gini > 0.8) and its expensive set OVERLAPS the manufactured-seg set measurably
  above independence.** FALSIFIER: Gini ≤ 0.8, or an expensive-set / manufactured-seg overlap within
  noise of independence — which would mean the token stream is flat and structureless, the
  pattern-of-patterns is granularity-bounded, and there is nothing here to target. Count it plainly
  either way; a refutation here retires a whole class of future targeting charters.

## DELIVERABLE

`.omx/research/ddm_tb2_token_bit_attribution_20260823.md` — typed rows: the per-axis attribution
tables with Gini/top-1%/top-decile mass; the reconciliation of attributed bits to the measured
113,777 B stream (with any residual stated); the seg×rate join measured against independence; named
candidate levers WITH falsifiers and no projections; the prior-law prediction adjudicated
CONFIRMED/REFUTED with its numbers; verdict_scope on the narrowest rung the evidence supports; the
persisted per-symbol field path + sha256 + bytes; a `STORES CONSULTED:` line (the contract's literal
key) naming what was loaded, honestly including "none" where none. Commit via the serializer. End with
the own-vehicle frontier line.
