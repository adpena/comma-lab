# ddm_bnd3 — decide the LEVEL of bnd2's closure: decompose the +1,416 B loss of the best segment recode into ADDRESS bytes vs OFFSET/CONTENT bytes on the shipped field, and test the two variants bnd2 left open (joint edge assignment, gap bridging); if the address term alone exceeds the whole attribution the token-level boundary code closes at FAMILY scope (charter, 2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (gpt-6-astra, xhigh) · Spawned by MAIN 2026-09-10. Sources: bnd2 gen 2 (`.omx/research/ddm_bnd2_boundary_segment_code_real_encode_20260910.md`, `d9a99ea88`; equation `boundary_segment_recode_price_v1`; retained payloads under `/Volumes/VertigoDataTier/pact/ddm_bnd2_segment_code/generation2/`; six greedy variants: best 121,200 B vs the 119,784 B envelope; masked control 129,316 B; median segment length one cell; all 235,044 mispredictions located), gs3 Addendum 19 (`6b9b3ad3c`), or1 (`ddm_or1_orthogonal_representation_regime_20260826.md:111–143`: row-start address packet 140,377 B vs 113,624 B), of1 (mean arclength 2.88, 18 % contour coverage), the verdict ladder (INSTANCE < FORMULATION < FAMILY < PARADIGM; declare the narrowest level the measurement supports). Axes: bytes `[exact, real encode, twins]`; `score_claim=false`.

## MANDATE
bnd2 closed six greedy grammars at formulation scope. The question the gestalt needs answered is whether ANY per-token address scheme can win: if the ADDRESS component alone of the best recode already exceeds the bytes the recode could ever save (the mispredicted population's share of the envelope, measured by bnd2's masked control as ≤ 0 — masking GREW the stream — so the true displaceable share must be re-measured as: envelope − bytes of the stream with those tokens PREDICTED-AND-SIGNALLED at zero cost, i.e. the ideal), then no offset/content coding can rescue it and the closure is FAMILY-level for token-level boundary description. Do three things by real encode on the shipped move-37 field: (1) split the best recode's 121,200 B into address bytes (segment starts / chain codes / lengths) and content bytes (offsets), twin-verified; (2) the two open variants — joint edge assignment (assign each mispredicted cell to the edge that minimises the global address cost, not greedily) and gap bridging (allow runs to span ≤ k correctly-predicted cells with a skip code) at k ∈ {1, 2, 4}; (3) the honest displaceable share: encode the envelope with the mispredicted tokens' surprise removed by SIGNALLING (a zero-cost oracle flag), so the saved bytes are an upper bound on what any boundary side-channel can displace. Then state the level: if min over all variants of (address bytes) ≥ displaceable share, FAMILY closure; else FORMULATION stays.

## PRIOR-LAW PREDICTION (m38)
- Address bytes ≥ 60 % of the best recode's 121,200 B; displaceable share (oracle-flag encode) ≤ 20 KB; joint assignment improves ≤ 5 %, gap bridging at k=4 ≤ 10 % — none crosses 119,784 B. Prediction: **FAMILY-level closure** for per-token-addressed boundary description on the shipped field.
- **FALSIFIER:** any variant ≤ 114,784 B twin-verified, or address bytes < 50 % of the displaceable share — then the family is open and bnd2's ITEM 2 draw is re-queued.

## SCOPE
Real encodes on the shipped field only; ≤ 2 procs; launcher with `--nice-best-effort` (priority is not a mechanism; the sandbox refuses setpriority); no scorer runs, no draw.

## HARD CONSTRAINTS
- `upstream/` READ-ONLY; never write into live trees (sj1/rp1/cmp2); your dir `/Volumes/VertigoDataTier/pact/ddm_bnd3_address_term/` (Vertigo ≈ 80 GiB free after vr5). Re-read `.omx/state/canonical_frontier_pointer.json` before every stage (sj1's move-38 candidate is on T4; if it promotes, the shipped field changes — state which field every number is on; do not mix).
- Every number a real encode with twins; no −log2 p sums as prices (m166); state the LEVEL beside every verdict (verdict_scope line mandatory).
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).
- `.py` = 2 visible review passes + ruff; serializer commits w/ post-edit `--expected-content-sha256`; if git object writes are refused, leave `landing.patch` + manifest and say so. Tokens `[no-triality] [p0-ledger-ok]`; NEVER a Co-Authored-By or AI-attribution trailer. Checkpoint `tools/subagent_checkpoint.py --subagent-id ddm_bnd3`.

## PRIOR NEGATIVE SIGNAL
- bnd2's four DEAD-ENDS (six greedy variants lose; attribution not detachable — masking grows the stream; pass-5 hash join fails; achieved lengths are not floors) # VERDICT_SCOPE_OK: cites bnd2's formulation-scope closure as the object under test; issues its own verdict only with a declared level
- or1: address cost is material; of1: runs are short — the priors this charter quantifies.

## OPTIMAL FORM
- Reference form: bnd2 gen 2 (`d9a99ea88`: real n600 twin encodes, independent decoder identity, registered equation). SCOPE reductions: none. MECHANISM reductions FORBIDDEN: no modelled address cost; no subset-n price; no greedy-only assignment presented as joint.
- **PRIOR-LAW PREDICTION (falsifiable):** as above.

## DELIVERABLE
Memo `.omx/research/ddm_bnd3_address_term_decomposition_20260910.md` with the decomposition table, the variant table, the displaceable-share bound, and the LEVEL verdict; anchor on `boundary_segment_recode_price_v1` or register `boundary_address_term_floor_v1`. Commit via the serializer. End with the live frontier line.
