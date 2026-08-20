JS7 completed and landed in commit `0dc3d4eb723c5818e805754d9f1a916bec2ac8c1`. The composed stack projects ΔS `−0.0005811216`, but this remains `[macOS-CPU advisory, stratified-random n32]`; no frontier moved.

| Family | Measured | Pose accepted | Robust improving | Bare admissions | Yield |
|---|---:|---:|---:|---:|---:|
| Boundary | 151 | 107 | 69 | 48 | 31.79% |
| Lane | 48 | 29 | 27 | 17 | 35.42% |
| Island | 1 | 1 | 0 | 0 | 0% |
| **Total** | **200** | **137** | **96** | **65** | **32.50%** |

The sweep measured the remaining 198 proposals and reused JS6’s two verified rows. The complete [acceptance table](/Volumes/APDataStore/pact/ddm_js7_20260812/ACCEPTANCE_TABLE.json) has SHA-256 `6274e6aa…52290`.

Joint composition admitted 44/65 singletons: 34 boundary offsets and 10 lane deltas, covering 58 sites across six frames. Twenty were rejected by the joint pose gate and one had zero marginal robust gain.

| Projected robust Δflips | Realized pose Δ | Packet | Container growth | Projected ΔS |
|---:|---:|---:|---:|---:|
| −1,133 | +1.3798192e-6 | 311 B, LZMA1-raw | +323 B | −0.0005811216 |

The retained [archive](/Volumes/APDataStore/pact/ddm_js7_20260812/container/retained/archive.zip) is 186,575 B, SHA-256 `465d3c58…1e9af4`, versus CP135’s 186,252 B. Its counted overlay is consumed by the adapted runtime. Independent expected/decoded 600-frame token planes are byte-identical, and the full independent repeat reproduced every selected ID, metric, packet, archive, and token hash. See the [repeat receipt](/Volumes/APDataStore/pact/ddm_js7_20260812/DETERMINISM_REPEAT.json) and [research receipt](/Users/adpena/Projects/pact/.omx/research/ddm_js7_acceptance_sweep_and_compose_20260812.md).

Verification: 22 tests passed, Ruff passed, payload-retention audit was clean, and two review passes covered all 57 Python entities. JS7 did not run MPS/Metal, an n600 scorer, `upstream/evaluate.py`, or Modal.

## NEXT_IF_RESUMED

- **Action:** measure the retained JS7 composed EC1 overlay on the full n600 scorer and exact contest-CUDA chain. **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN scorer-lane router. **Consumer store:** `/Volumes/APDataStore/pact/ddm_js7_20260812/main_n600_and_exact`. **Fire trigger:** MAIN owns the sole n600 scorer lane, confirms it is idle, revalidates the retained archive/runtime SHA records, then runs n600 in chunks no larger than 120 before any exact dispatch. The sealed [recipe](/Volumes/APDataStore/pact/ddm_js7_20260812/SEALED_MAIN_N600_RECIPE.json) names the exact bytes and sequence.

## LIVE-HYPOTHESES

- The 44-event stack will remain net-negative at n600 because its repeated n32 projection has `0.0005811216` score-unit headroom across six separated sampled frames.
- A pose-headroom-aware ordering may admit a better stack because 20 economical singletons failed only after interaction with the accepted prefix.
- Rejected boundary and lane events may support a second mutually exclusive stack because rejection against this prefix does not prove rejection against every prefix.

## DEAD-ENDS

- Summing singleton benefits is closed: joint remeasurement rejected 21/65 singleton admissions.
- Bare packet bytes are not a shipping receipt: 311 B became +323 B at the complete-container boundary.
- The available island event is closed only at INSTANCE scope: it passed pose but worsened projected robust flips by 19.
- Claiming JS7 as a score or pointer move is closed because no n600 or contest evaluation ran. Effective frontier remains CP135 `S=0.16195513827824176 @ 186,252 B [contest-CUDA T4,n600]`; own-vehicle frontier remains LC2 `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4,n600]`.