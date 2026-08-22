# ddm_dx2_cabac_receiver_fold — fold the measured −18 B CABAC dxi recode into the fx5_e1 body → seal → MAIN T4 (nineteenth-move candidate)

## MISSION
Complete the r012 rate-ceiling harvest on the live body. fx5_e1 (pointer, S 0.14823186109359
@ 180,386 B [contest-CUDA T4 n600], archive sha
4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841) harvested −70 B of the
measured 88 B composable ceiling (memo .omx/research/ddm_r012_rate_representation_20260821.md).
The remaining −18 B is dx1's MEASURED adaptive-ctx Rice (CABAC-prefix) dxi recode
(memo .omx/research/ddm_dx1_dxi_recode_and_fruit_sweep_20260820.md §3: payload 9,811 B vs 9,829,
decode-identical, clears the −3.5e-6 admission bar 3.4×; §5 recommendation = fold at the next
receiver-tree revision — that revision is NOW, on the fx5_e1 tree). Deliver: the CABAC-prefix
decoder implemented in the fx5_e1 receiver + the recoded dxi section spliced → byte-close with
decode-IDENTITY proof vs the fx5_e1 retained 0.raw (sha
6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883) → candidate seal
(tools/make_candidate_seal.py, single-axis waiver rationale = decode-identity) → READY;
MAIN fires the T4 row. Projected ΔS −1.198546e-05 (dx1-measured; re-derive at close).
NO Modal fire from the arm; NO heavy local launch (the governed slot is occupied by the jo1
solve — do not touch it).

## WORK ORDER
1. RECALL + custody: dx1 retained race artifacts at /Volumes/VertigoDataTier/pact/ddm_dx1/retained/
   (DX1_RECODE_RACE.json + every re-coded payload w/ sha256; coded-symbol array
   dx1_coded_symbols_U.int32.npy sha 0bfe31cf9586104f4308329fec8f76f748c56441ac5bd85b824dfcca3434db50).
   Verify the winning adaptive-ctx Rice payload's sha against the race receipt before use.
2. RECEIVER FOLD: implement the CABAC-prefix (adaptive-ctx Rice) decoder in the fx5_e1
   receiver tree (derive the working tree from the retained fx5_e1 runtime custody; recompute
   any inflate pin constants FROM the new archive bytes — the #1123-genus derived-runtime
   discipline, never hand-typed). Splice the 9,811 B dxi section; all other sections
   byte-identical.
3. BYTE-CLOSE + IDENTITY: fresh-process decode → 0.raw must be BYTE-IDENTICAL to fx5_e1's
   retained decode (sha 6bf8acf8…). Archive delta must be exactly −18 B (180,386 → 180,368);
   any other delta = STOP and report. Deterministic repeat.
4. SEAL: tools/make_candidate_seal.py on the new archive × new runtime tree → SEAL_VALID →
   final message names the seal path + FIRE command for MAIN. Serializer commits
   (post-edit --expected-content-sha256); .py = 2 genuine review passes. Keep every payload
   (sha+bytes persisted).

## OPTIMAL FORM
Family reference form + receipt: the dxi coder family's reference is dx1's 16-coder race at
optimal form (real bytes, decode-identity enforced, receipt
/Volumes/VertigoDataTier/pact/ddm_dx1/retained/DX1_RECODE_RACE.json; winner adaptive-ctx Rice
CABAC-prefix 9,811 B = −18 B vs shipped, memo §3 table). The fold must ship THAT winner's
exact payload semantics — re-encode from the retained coded-symbol array and assert payload
sha equality with the race artifact.
Provenance pin: .omx/research/ddm_dx1_dxi_recode_and_fruit_sweep_20260820.md=SEE-GIT (memo, commit-tracked)
and dx1_coded_symbols_U.int32.npy=0bfe31cf9586104f4308329fec8f76f748c56441ac5bd85b824dfcca3434db50.
SCOPE reductions (legal): none needed — the payload exists; this is a fold, not a search.
MECHANISM reductions: NONE — a re-raced or approximated coder is not the measured winner.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends, accounted)
- dx1 §4 fruit sweep: NOTHING transfers from the fruit families — do not re-open coder search;
  the axis is CLOSED at −18 B (memo .omx/research/ddm_dx1_dxi_recode_and_fruit_sweep_20260820.md).
- dx1 §8: ov1's −0.001190 ceiling WITHDRAWN (wrong object, 99.0% too large) — never cite it.
- rr2 device-scoped decode desync (memo .omx/research/…362463ca80 lineage, task #1096): a coder
  whose decode probabilities differ CPU-vs-CUDA refused S 27.83 — the CABAC decoder must be
  integer/deterministic, device-invariant; assert identity on the CPU decode AND document why
  the path is device-free.
- wd4 pin-bound-receiver refusals (memo .omx/research/ddm_wd4_warm_lineage_width_20260821.md):
  derive runtime pins from the file, never hand-type (#1123 genus).

## CONTEXT ANCHORS (memo-associated)
- Campaign sub-0.12 #1182 (memo .omx/research/ddm_r012_rate_representation_20260821.md).
- fx5_e1 pointer receipt /Volumes/APDataStore/pact/ddm_fx5/t4_row_r1/MODAL_REMOTE_RESULT.json
  (#877: recompute S from components, never the rounded final_score).
- Seal contract #1115 (tools/make_candidate_seal.py + fire_modal_auth_eval.py --seal).

## CONTRACT
upstream/ READ-ONLY; keep the payloads; end with final message stating archive sha/bytes,
identity proof, seal path, and the exact MAIN fire command.
