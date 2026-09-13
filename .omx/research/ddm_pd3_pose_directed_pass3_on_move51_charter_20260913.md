# ddm_pd3 — pose-directed token pre-distortion, pass 3 on the move-51 field: the tier-2 pairs and a deeper K (charter, MAIN 2026-09-13; operator "Continue frontier score lowering"; Opus)

## Why
Move 51 (pd2; S 0.1362333315680336 @ 179,285 B; d_seg 0.00010305; d_pose 4.29e-6; archive sha
42e47d0bae1b0647d08db8a5061fe3eec6368b2169d89eb62b14f832fdb0978f; tree `/Volumes/APDataStore/pact/ddm_pd2/candidate/candidate_runtime`; packet
`.omx/research/ddm_pd2_pose_directed_pass2_first_measurement_20260913_pointer_move_51_*.md`; memo
`.omx/research/ddm_pd2_pose_directed_pass2_on_move50_20260913.md`). pd2 walked 166 of the 588 non-floor pairs (tier 1 = pairs whose base
pose could pay at 16.98 bits/token; 10 of tier 2) at K_refine 8 and admitted 40; 125 credited pairs were left unadmitted by the sweep and
~420 pairs were never walked. Pass 3: (a) walk the unwalked tier-2 pairs (rank by base d_pose × the measured admitted fraction; declare the
cut from a timing smoke), (b) re-walk the 165 credited pairs at K_refine ≥ 12 (pd2 measured 33.3 s/refine; budget ~4 h on 8 shards —
declare K and the pair budget), (c) the 40 move-51 pairs re-searched from their new renders (pd2 measured that pose credits do NOT consume
local slack). Pre-registered band: −2e-5 … −6e-5 S net; falsifier: admitted set projects net > −2e-5 on the RESOLVED pose with real-encode
rate, or the composition realizes < 0.8 of the sum.

## Route (binding): NORMAL seal inheriting move 51's own minted leg
`/Volumes/APDataStore/pact/ddm_pd2/SEAL_ddm_pd2_pose_directed_pass2_contest_cuda.json.decode_wall_clock.json` (t4_direct, 1,164.9 s ≤ 1,260;
validator 0 problems; committed copy `.omx/research/ddm_pd2_packet_inputs_20260913/MOVE51_T4_DIRECT_LEG.decode_wall_clock.json`). Receiver
code byte-identical; only archive.zip + its two pins + MANIFEST.sha256 (Catalog #420 producer) change. pr18's behaviour digest 9f6e7168… must
match. `tools/make_candidate_seal.py … --inherit-decode-wall-clock <sidecar>` exactly as pd2 did (its seal inputs under
`/Volumes/APDataStore/pact/ddm_pd2/seal_inputs/` are the worked example). NO Modal, NO fire, NO packet (MAIN fires).

## Deliverable
Base = move 51 by parse-back (read-only; copy); per-pair pose base under the measured-tolerance gate; RLC1 pricer proved by byte-identical
repack of move 51's tail (119,055 B) and carrier; timing smoke → K and pair budget declared; search as above with pd2's producers
(`experiments/ddm_pd2_*.py`, `ddm_pd1_*.py`, `ddm_pp1_pose_actuation.py::cmd_search_b` — reuse, do not rewrite); three-leg Lagrange admission on
the RESOLVED pose (frame-0 repair inside; seg on the shipped-mode decode; real-encode rate, twins); composition by re-verification with the
realized fraction; if it nets ΔS < −2e-5 on exact bytes: byte-close on the move-51 base (prior/semantic member/receiver code byte-identical),
twins, cold n600 public parse-back (decoded scorer numbers = in-loop), manifest via the Catalog #420 producer, census, smokes candidate +
frontier (four native-library exports as inflate.sh sets them), retention ≤ 8 GiB with shas on APDataStore (Vertigo under its reserve — never
lower it; check free space before every heavy step: APDataStore is filling — report the number), seal as above. If it does not net, close
with the histogram and table. Memo `.omx/research/ddm_pd3_pose_directed_pass3_on_move51_20260913.md`; serializer commits (two visible review
passes per .py; `[no-triality] [p0-ledger-ok]`; NEVER a co-author trailer or AI attribution); lane
`ddm_pd3_pose_directed_pass3_on_move51_20260913` (claim it); checkpoint `ddm_pd3`. Heavy steps via `tools/launch_detached_process.py
--output-dir /Volumes/APDataStore/pact/ddm_pd3/<stage> --nice 0 --done-receipt …`; waits as background receipt-only until-loops;
`sys.dont_write_bytecode` against read-only trees.

## Boundaries
No Modal, no `authorize_*`, no `fire_modal_auth_eval.py`; never edit `upstream/`, the PR tree, sealed trees, contract code, receiver code,
the renderer (weights or per-pair frame_embed), the basis, or the prior; never lower a reserve; pd1/pd2/sj1/pp1/cb1/cr1/so2 directories
read-only; no ScheduleWakeup. Label MEASURED / DERIVED / INFERRED / ASSUMED; state the solver behind every pose number.

## OPTIMAL FORM
Reference forms: pd2's search/admission/pricer/seal inputs. Declared deltas: the walked set (tier 2) and K (SCOPE). Provenance pins: HEAD
(record); pd2 memo/packet shas (record); move 51 seal + leg.

## Prior negatives accounted (operator 2026-08-15)
pd1/pd2: the rate prior is a ranking (price by real encode; 15.4–17.0 bits/token measured); pass 8's same-field decay applies to SEG repairs
(pose credits re-open on re-render — measured); pp1's 12 floor pairs excluded; the 34.8 B lottery; pr19 (this route inherits ONCE from a
measured leg — move 51's is measured, so the inheritance is legal; move 52 will need its own leg minted at harvest or a re-measure).

Final message: K, budget and the timing smoke, the per-pair histogram, the three-leg table with exact bytes + sha, projected S and margins,
the seal path with file sha or the typed blocker, retained bytes and APDataStore free space, every boundary, ending with
`composition S 0.1362333315680336 @ 179,285 B [contest-CUDA T4 n600] (move 51)` — a new number only if MAIN's fire moved it.

<!-- # FORMALIZATION_PENDING: charter, not a finding; the row, if any, carries its equations leg through tools/pointer_move_packet.py --equations-leg at harvest -->
