# ddm_rt1 charter — decompose the 0.0282 S render→SegNet round-trip loss (the seg axis IS this)

Operator authority: 2026-08-16 full-authorization grant (#1074). Successor named by ddm_td1
(NEXT_IF_RESUMED H1). Owner of any fire: MAIN. This arm measures and decomposes; it never
launches training, never spends Modal.

## The measured premise (td1, do not re-derive)

On the hv1 ep0634 frontier archive (sha 80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e):
transmitted labels disagree with GT at **1,717 px** while the scored seg term is **34,930.6
flips** (d_seg 2.9611e-4 × 117,964,800 px — MAIN re-derived independently, matches to 0.1 flip).
So ~95% of the seg term — **0.028155 S, 2.9× the entire sub-0.15 gap** — is manufactured between
the label field and SegNet's argmax, not by the labels we ship. Robust ≥90% for any label-flip
amplification r ≤ 2. Receipts: td1 memo `.omx/research/ddm_td1_token_drop_schur_arithmetic_20260816.md`
(sha 9fc486e7eaacd00b94b1cf8b043b1dcbcb… — full: 9fc486e7eaacd00b94b1cf8b043b1dcbc1ab5ae1dcf6c18ccf9abbacc439ac44),
commits 616f0aae86 + db69d97d54, payloads (permutation map, label field, B=807/H=222,883/W=4
GT attribution) retained under `/Volumes/APDataStore/pact/ddm_td1_token_drop_schur_20260816/`.

## The question

WHERE between the transmitted label field and the scored argmax do the ~33,214 excess flips get
manufactured, and which stage's cure is cheapest? Candidate stages on this vehicle:

- S1 **neural render deviation**: the SemanticTokenRenderer's RGB output vs an ideal paint of its
  OWN transmitted label field (prototype colors). The renderer is trained; its output need not
  land in the argmax cell its own label says.
- S2 **paint→SegNet response**: even ideal prototype-color paint of the exact label field may not
  reproduce that field through R+SegNet (class-color margin, AA, texture absence). v14 measured
  this class of loss on the describe-line vehicle — this is the hv1 instance.
- S3 **resize/uint8 (R)**: camera-res → 512×384 bilinear + uint8 quantization.
- S4 **GT-side flicker floor**: SegNet(GT video) label instability vs our temporally-cleaner
  field (fl1 measured per-class GT-flicker floors — consult before attributing residual).

## Deliverables

1. **Instrument check first** (fail-closed): SegNet forward over the retained decoded frames must
   reproduce the scored d_seg 2.9611e-4 / 34,930.6 flips on the same advisory instrument before
   any decomposition row is claimed. Retained raw: wc1 workspace decode, raw sha e5539653… under
   `/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/runs/*/output/` (3.66 GB,
   retained per ALWAYS-KEEP-THE-PAYLOAD; cached token-decode path costs ~370 s if a re-decode is
   needed — `tools/` wc1 fast path per its admission receipt).
2. **The per-stage flip ledger** (n600, full field for every counted number; bounded pair subsets
   only for diagnostics and labeled with the m96 axis law): counterfactual swaps isolating S1
   (render vs ideal-paint-of-shipped-labels), S2 (ideal paint vs label field through R+SegNet),
   S3 (R on/off where separable), S4 (GT flicker floor join from fl1 receipts). Ledger rows in S
   units, additive-or-labeled-nonadditive, each with payload retained + sha.
3. **The named lever**: which stage holds the mass, and the cheapest cure candidate per stage
   with its family precedent cited (S1 → post-render label-reassert / margin-aimed finisher
   (menu1/qs machinery); S2 → margin-optimal prototype-color solve from the frozen head (v14's
   named fix) or texture carrier; S3 → pre-R placement (#149); S4 → floor, not fixable — say so).
   Include the 807 label-correction sites (td1 H3) as a priced footnote, not a headline.
4. **Routing memo + queued follow-ons**: td1-H2 (measure r via one ≥8-bit-set row) queued with
   fire-condition; successor charter skeleton ONLY if a stage cure needs a build arm.

## Prior negative signal this charter accounts for (bearing dead-ends)

- v14 measured the Movable projection loss as a FORMULATION-scoped wall on the describe-line
  vehicle (fixed-paint dependency named by a1) — this charter measures the TRAINED-renderer
  vehicle, where that precondition differs; do not transfer v14's stage magnitudes, only its
  taxonomy. fp1's f′=0.008305 flat-paint receiver floor is FORMULATION-scoped to flat paint
  (gc16-r2 adjudication #909) — the shipped family's live d_seg 2.9611e-4 already sits far below
  it, so no measured floor blocks this decomposition. Token-drop (td1), post-hoc weight edits
  (mp2/ns1 family), lossless recoding (mz2), and linear pose overlays (pk4) are all dead — this
  charter is the seg-axis successor precisely because those closures leave render fidelity as the
  only unmeasured seg mass. #149 pre-R placement measured favorably at $0 (camera-res placement
  survives D) — the S3 cure precedent.

## OPTIMAL FORM

- Reference form: ddm_v14_realization_fidelity (#624) per-stage loss diagnosis — exact-mask →
  painted RGB → R → uint8 → SegNet argmax on the describe-line vehicle. THIS charter is the
  declared hv1/HPAC-vehicle instance of that family methodology (semantic renderer replaces the
  worldsheet painter). Consult v14's memo for the stage taxonomy + its measured Movable
  projection-loss precedent before designing the swaps.
- Provenance pins: base archive sha256 80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e ·
  td1 memo sha256 9fc486e7eaacd00b94b1cf8b043b1dcbc1ab5ae1dcf6c18ccf9abbacc439ac44 (commits
  616f0aae86, db69d97d54) · wc1 decode raw sha prefix e5539653 (admission receipt
  `ddm_wc1_advisory_fast_path_admission.v1`) · ns1 memo sha256
  91741c062c38ab88ce7e225921a0024d6dc45dacc858658e02ee83ff08b2dba0.
- Instrument: frozen CPU-torch SegNet, deterministic, same advisory class as the mp2/base rows —
  [macOS-CPU advisory], NEVER a score; the exact row that converts any cure to a pointer claim is
  MAIN's fire.
- SCOPE reductions permitted + labeled: per-stage diagnostics on stratified subsets; every
  COUNTED ledger number is full n600. TOY-BRACKET: none — if a stage swap cannot run at full
  field in budget, report the wall, never substitute a subset extrapolation for a ledger row.

## Constraints (binding)

Scorer use = local advisory SegNet forwards only (the advisory slot is free; b2e is build-only);
no launches, no Modal, no n600 T4 spend. Serializer commits w/ --expected-content-sha256; 2
review passes on .py; ALWAYS KEEP THE PAYLOAD (every counterfactual frame set or its
deterministic regeneration recipe + sha); upstream/ read-only; SSD routing per policy (Vertigo
is near-full — use APDataStore). STORES CONSULTED must include: td1 memo · v14 memo · fl1
per-class flicker receipts · wc1 admission receipt · ns1 memo · the hv1 compose FINAL_RESULT.
Final message persisted with NEXT_IF_RESUMED; done receipt via the keeper.
