---
schema: ddm_sl2_sq2_persist_compose_receipt.v1
date_utc: 2026-08-05
arm: sl2
axis: "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE"
score_claim: false
promotion_eligible: false
pointer_moved: false
tokens: [no-triality, p0-ledger-ok]
---

# SL2 - SQ2 persisted frames and composed terminal pose

## Answer First

No exact row was produced and the contest pointer did not move. SL2 reran the SQ2 selected n32 solve from the PU2 base, persisted the actual edited frame_1 bytes, then applied `terminal_pose_gn` to those persisted frames.

| leg | status | denominator | result | disposition |
|---|---|---:|---|---|
| Primary Leg 1: SQ2 persist then compose | MEASURED | 32/32 selected SQ1/SQ2 pairs; 6,291,456 SegNet sites | d_seg `0.0043002764 -> 0.0010015170`; d_pose mean `0.0580729881 -> 0.0082654368`; pose term `0.7620563503 -> 0.2874967265` | terminal pose repairs most, not all, SQ2 pose erosion; pair 514 remains high |
| Secondary F2 rung 3 | QUEUED-WITH-FIRE-ORDER | worst-3 SL1 F2 tail pairs | not run; primary full n32 + terminal consistency rerun consumed the budget | run pairs `151, 141, 111` at >=16,000 iters each, preserve plateau traces |

Score-like component accounting on the selected n32 object, not a full archive score:

| component | value |
|---|---:|
| `100*d_seg_solved` | `0.10015169779459636` |
| `sqrt(10*d_pose_pre_terminal)` | `0.7620563503396967` |
| `sqrt(10*d_pose_composed)` | `0.2874967265222887` |
| terminal packet bytes | `479` |
| terminal-packet-only rate term | `0.00031894643854552007` |
| composed packet-only action | `0.3879673707554306` |

The `479` bytes are the terminal-pose packet only (`sl2_terminal_pose_n32.tpgn`), not a complete shippable archive. The per-pair NPZ byte counts below are evidence/custody bytes, not archive-rate bytes.

## Scope

Subset selection came from `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_pair_selection.json`: `method = "stratified systematic on flips-sorted order, 32 strata, median of each"`, selected pairs `[0, 20, 32, 48, 115, 154, 170, 179, 180, 195, 196, 211, 214, 242, 261, 288, 357, 365, 370, 394, 400, 420, 433, 439, 471, 474, 485, 501, 504, 514, 521, 533]`, subset/population flip ratio `0.9973286607423718`. This is n32 stratified-systematic advisory evidence, not n600.

SQ2 used the landed SQ1/SQ2 solved-paint and trajectory-stop surfaces. All 32 prior SQ2 selected rows used `dec` as the best start, so SL2 ran the persisted rerun from `dec` only. No row hit an `iteration_cap_*` resource stop; stop census was 29 `marginal_below_bar` and 3 `converged_projected`. These are trajectory-stop endpoints, not exact-zero solve certificates.

ET1 block16 persistence was not run. The charter made SQ2 the priority, and SQ2 full n32 plus the terminal consistency rerun consumed this arm's budget. ET1 remains QUEUED if resumed: rerun its realizer only if it can persist actual edited frame bytes with full SHA custody and then compose terminal pose on those bytes.

## Artifacts

| artifact | bytes | SHA-256 | role |
|---|---:|---|---|
| `/Volumes/VertigoDataTier/pact/ddm_sl2_20260805/sl2_sq2_persist_n32.json` | 141,977 | `cc880649ba04cd27b3459e63193aa17026c873e93292de1559ce04fe650fadff` | stage-1 rows, aggregate, full per-pair NPZ SHA custody |
| `/Volumes/VertigoDataTier/pact/ddm_sl2_20260805/sl2_composed_terminal_pose_n32.json` | 62,371 | `5cf44a977189340636cb357022cfa593c18e96ab8020235f2f6f16826d5c6559` | composed terminal-pose rows and aggregate |
| `/Volumes/VertigoDataTier/pact/ddm_sl2_20260805/sl2_terminal_pose_n32.tpgn` | 479 | `77889f57d3d274f8f9bc90e132e31fdfe2fe3cf0475a1e209fbca4cc0386fbde` | terminal-section-only packet, not a full archive |
| `/Volumes/VertigoDataTier/pact/ddm_sl2_20260805/sq2_persisted_frames/` | 32 NPZs | full hashes in `sl2_sq2_persist_n32.json` | actual persisted frame_1 bytes used for terminal pose |

Primary command:

```bash
.venv/bin/python experiments/ddm_sl2_sq2_persist_and_compose.py --out-dir /Volumes/VertigoDataTier/pact/ddm_sl2_20260805 --seg-resource-step-bound 150 --seg-eval-every 5 --seg-convergence-patience-evals 3 --pose-relinearizations 2
```

Terminal consistency rerun, after detecting and correcting a small stale aggregate in the first terminal JSON:

```bash
.venv/bin/python experiments/ddm_sl2_sq2_persist_and_compose.py --out-dir /Volumes/VertigoDataTier/pact/ddm_sl2_20260805 --skip-stage1 --no-resume --pose-relinearizations 2
```

## Aggregate

| metric | value |
|---|---:|
| n_pairs | `32` |
| SegNet sites | `6,291,456` |
| flips before | `27,055` |
| flips solved | `6,301` |
| d_seg before | `0.004300276438395183` |
| d_seg solved | `0.0010015169779459636` |
| fixed / introduced | `21,799 / 1,045` |
| described-in-band flips | `23,450` |
| pooled eta_net | `0.8850319829424307` |
| d_pose pre-terminal mean | `0.05807298810930584` |
| d_pose composed mean | `0.008265436776103165` |
| terminal strict-improvement rows | `26 / 32` |
| all_terminal_done | `true` |

Consistency note: the final terminal JSON is bound to stage-1 SHA `cc880649ba04cd27b3459e63193aa17026c873e93292de1559ce04fe650fadff`, and its aggregate `d_seg_solved`/pre-terminal pose mean match the final stage-1 JSON. One helper row remains worth remembering: terminal helper row 514 reports initial pair d_pose `0.08082120587049998` while the stage-1 row records `0.08425009672379746`; the aggregate uses the stage-1 pre-terminal value, and the composed row uses the terminal final value `0.0803801109`.

## Per-Pair Results

Full NPZ hashes are in `sl2_sq2_persist_n32.json`; this table shows evidence bytes plus SHA-12 for readability.

| pair | flips before->solved | d_seg_solved | d_pose pre-term | d_pose composed | SQ2 stop | NPZ evidence |
|---:|---:|---:|---:|---:|---|---:|
| 0 | 765->188 | `0.0009562174` | `0.2064101068` | `0.0367374609` | `marginal_below_bar` | 4,287,218 B `d54f208fe5a9` |
| 20 | 772->174 | `0.0008850098` | `0.0033529373` | `0.0033529373` | `marginal_below_bar` | 4,246,405 B `a511a11e8200` |
| 32 | 781->224 | `0.0011393229` | `0.0211242332` | `0.0007313314` | `marginal_below_bar` | 4,159,992 B `0caf5c825b08` |
| 48 | 713->151 | `0.0007680257` | `0.1524017234` | `0.0627448878` | `marginal_below_bar` | 4,112,327 B `80d6c0163e1f` |
| 115 | 812->218 | `0.0011088053` | `0.0015975579` | `0.0015975579` | `marginal_below_bar` | 4,186,181 B `1bd6b5e277dc` |
| 154 | 962->188 | `0.0009562174` | `0.0127385389` | `0.0007776790` | `marginal_below_bar` | 4,175,487 B `5d7ede87a438` |
| 170 | 1051->313 | `0.0015920003` | `0.0423292026` | `0.0120165460` | `converged_projected` | 4,283,064 B `3fbcf7c24631` |
| 179 | 787->236 | `0.0012003581` | `0.0875685106` | `0.0020232445` | `marginal_below_bar` | 4,156,098 B `473b789d7568` |
| 180 | 747->164 | `0.0008341471` | `0.0310110909` | `0.0023258607` | `marginal_below_bar` | 4,202,661 B `d49fc023c416` |
| 195 | 722->174 | `0.0008850098` | `0.0182519621` | `0.0044030198` | `marginal_below_bar` | 4,163,516 B `be218a08cd6f` |
| 196 | 986->168 | `0.0008544922` | `0.0922320104` | `0.0121727934` | `marginal_below_bar` | 4,143,190 B `35ed14b7e83f` |
| 211 | 678->179 | `0.0009104411` | `0.0351240811` | `0.0024751513` | `converged_projected` | 4,077,761 B `43d1abe97c36` |
| 214 | 643->227 | `0.0011545817` | `0.0074632414` | `0.0055040765` | `marginal_below_bar` | 4,143,651 B `af34dc9858ae` |
| 242 | 703->197 | `0.0010019938` | `0.0954492856` | `0.0006829258` | `converged_projected` | 4,099,363 B `02168ff1f55b` |
| 261 | 736->165 | `0.0008392334` | `0.0305332888` | `0.0007638649` | `marginal_below_bar` | 4,036,593 B `9f7af29a9997` |
| 288 | 756->140 | `0.0007120768` | `0.0006758848` | `0.0006758848` | `marginal_below_bar` | 4,141,292 B `b0a0c01d5667` |
| 357 | 845->161 | `0.0008188883` | `0.0003651856` | `0.0003651856` | `marginal_below_bar` | 4,213,337 B `64b8dab9db30` |
| 365 | 661->111 | `0.0005645752` | `0.1623650592` | `0.0015112854` | `marginal_below_bar` | 4,145,055 B `916ee5c284ba` |
| 370 | 600->172 | `0.0008748372` | `0.0601698853` | `0.0049477305` | `marginal_below_bar` | 4,070,222 B `5fbc0a87161b` |
| 394 | 833->175 | `0.0008900960` | `0.1347369096` | `0.0044778155` | `marginal_below_bar` | 4,146,706 B `93316bf07c4a` |
| 400 | 799->157 | `0.0007985433` | `0.0054378735` | `0.0032133768` | `marginal_below_bar` | 4,164,968 B `b1ca3f97fb65` |
| 420 | 853->149 | `0.0007578532` | `0.0469393569` | `0.0021751176` | `marginal_below_bar` | 4,095,973 B `20555e01a5bf` |
| 433 | 690->158 | `0.0008036296` | `0.0182506722` | `0.0017336162` | `marginal_below_bar` | 4,376,312 B `6e395a93c397` |
| 439 | 871->244 | `0.0012410482` | `0.0509547106` | `0.0038389815` | `marginal_below_bar` | 4,144,208 B `14e7322b312e` |
| 471 | 919->203 | `0.0010325114` | `0.0016900180` | `0.0016900180` | `marginal_below_bar` | 4,151,072 B `e68fc3629354` |
| 474 | 941->185 | `0.0009409587` | `0.0137414723` | `0.0006069358` | `marginal_below_bar` | 4,178,110 B `741fa17853a1` |
| 485 | 822->179 | `0.0009104411` | `0.2023887929` | `0.0065117960` | `marginal_below_bar` | 4,256,855 B `7dc6b32a2836` |
| 501 | 1012->235 | `0.0011952718` | `0.0018787969` | `0.0018787969` | `marginal_below_bar` | 4,301,672 B `f8988b44d436` |
| 504 | 1091->211 | `0.0010732015` | `0.0025505233` | `0.0004641115` | `marginal_below_bar` | 4,143,947 B `034369c7fca2` |
| 514 | 1176->287 | `0.0014597575` | `0.0842500967` | `0.0803801109` | `marginal_below_bar` | 4,184,407 B `4ae957a13d86` |
| 521 | 1430->364 | `0.0018513997` | `0.0600674242` | `0.0007708431` | `marginal_below_bar` | 4,140,782 B `709b9733bbb4` |
| 533 | 898->204 | `0.0010375977` | `0.1742851865` | `0.0009430337` | `marginal_below_bar` | 4,299,697 B `a71d92770807` |

## RECALL EVIDENCE

| source searched | query / scope | finding | impact |
|---|---|---|---|
| Governing docs | `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | no-fake, no full-n600, no `evaluate.py`, SSD custody, protected-path, serializer, and pointer boundaries loaded | kept work to bounded local n32 advisory measurement and avoided protected files |
| Charter and common contract | `.omx/tmp/codex_runs/sl2_prompt.md`, `_common_contract.md` | SL2 is SL1 Leg-1 fire-order; deliverable and final line fixed | drove SQ2 priority, persisted-byte requirement, recall section, and commit constraints |
| SL1 and routing doctrine | `.omx/research/ddm_sl1_20260805/SL1_RECEIPT.md`, `.omx/research/ddm_seg_bank_routing_20260805.md` | SL1 blocked because SQ2/ET1 receipts lacked edited frame bytes; pose erosion before constraint is data, not verdict | reran SQ2 to persist bytes before applying terminal pose |
| SQ1/SQ2 receipts | `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_*`, search/read `sq1_stage_n32_uncap100_sq2.json`, `sq1_aggregate_n32_uncap100_sq2.json`, `sq1_pair_selection.json` | selected n32 pair set and selection mode; prior SQ2 best start was `dec` for all 32; prior rows were scalar-only | used exact pair list and did not manufacture terminal composition from old scalar summaries |
| Terminal pose helpers | `rg terminal_pose_gn`, `src/tac/optimization/terminal_pose_gn.py`, `tools/pb1_terminal_pose_gn_600.py`, `experiments/ddm_p3v2_optimal_form_pose_resolve.py` | `terminal_pose_gn` can operate on parent frames and enforces frame_1 unchanged; PB1 showed packet/score callback pattern | reused banked helper, did not rebuild pose machinery |
| Canonical equations registry | `.venv/bin/python tools/list_canonical_equations.py --json` | trajectory stopping law and score term structure surfaced | row stop reasons are recorded as trajectory-stop endpoints; score-like component accounting recomputed from components |
| Memory registry | `rg -n "(sl2|common_contract|#899|#904|required-component|declared-on-never-read|margin_targets)" /Users/adpena/.codex/memories/MEMORY.md` | no SL2-specific prior result; relevant Pact memory emphasized current-state reread, scoped absence, and evidence-backed apparatus | no plan change beyond stricter scoped labels and pointer honesty |

Nothing found beyond the listed scopes provided a cheaper pre-existing persisted SQ2 frame-byte artifact; bounded absence only, not a global nonexistence claim.

## Boundaries

No `upstream/evaluate.py` was run. No full n600 authority score, contest-CPU row, contest-CUDA row, remote dispatch, GPU launch, training launch, or archive submission was produced. MPS was not used as authority. `upstream/` was not edited. The SSD artifacts are measurement/evidence artifacts; the NPZ payloads are not claimed as shippable archive bytes. The terminal packet is terminal-section-only and not a complete archive. The contest pointer is borrowed/unmoved.

`experiments/ddm_sl2_sq2_persist_and_compose.py` is the new SL2 wrapper. It passed a one-pair smoke before the full run; final verification status is recorded in the commit/final response.

## NEXT_IF_RESUMED

```json
{
  "sl2_status": "primary_measured_no_score_claim",
  "primary_next": {
    "status": "MEASURED_ON_N32",
    "fire_order": "Use the persisted SQ2 frame bytes and terminal rows as the teacher/custody surface for a constrained/joint descent carrier; do not cite this as a shippable archive or n600 result."
  },
  "et1": {
    "status": "QUEUED_WITH_FIRE_ORDER",
    "fire_order": "If ET1 remains live, rerun ET1 block16 only if the realizer can persist actual edited frame bytes with per-pair SHA custody, then apply terminal_pose_gn to those exact bytes before any R8 claim."
  },
  "f2_rung3": {
    "status": "QUEUED_WITH_FIRE_ORDER_BUDGET_DEFERRED",
    "pairs": [151, 141, 111],
    "source": "/Volumes/VertigoDataTier/pact/ddm_sl1_20260805/sl1_f2_tail_confirmation.json",
    "fire_order": "Run one >=10x rung from the SL1 1600-iteration tail state, i.e. >=16000 iterations per listed pair, persist full traces, and require the plateau criterion before calling the tail convergence-closed."
  }
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
