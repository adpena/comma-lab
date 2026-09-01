`BLOCKED(storage-reserve)` — partial receipts have Branch-1 shape, but the n=32 branch and Gate-1 verdict are not adjudicated.

APDataStore fell to 1,263,927,296 free bytes. Finishing while preserving the mandatory 1 GiB reserve requires at least 1,360,789,504 bytes, so execution stopped safely at 26/32 sealed rows.

| stride | sealed rows | byte-identical | median s/proposal |
|---:|---:|---:|---:|
| 200 | 26 | 26/26 | **478.573** |
| 300 | 26 | 26/26 | 714.279 |

Across both strides, all 52 comparisons had zero archive-delta error. This exact-vs-exact correlation is vacuous under the preregistered rule and carries no gate authority.

Terminal adaptive state reconverged in 0/26 rows across 147 arrays; 57–71 arrays still differed at frame 600. This closes terminal reconvergence only for the measured AFR1 instance and partial denominator.

Artifacts and verification:

- Blocker receipt: `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/BLOCKER.json`, SHA-256 `581a0768…d839b`
- Memo committed as `463d2cbe03`
- Focused tests: 16/16 passed
- `SCREEN.json` and `MANIFEST.json` correctly remain absent
- No scorer, CUDA evaluation, network, paid dispatch, deletion, or payload discard occurred
- Unrelated dirty `experiments/ddm_jg2_tail_reencode.py` was preserved

Pointer UNMOVED: AFR1 remains `S = 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]`, archive SHA-256 `cbb8d928…d405bf25`.

## NEXT_IF_RESUMED

- **Disposition:** QUEUED-WITH-A-FIRE-ORDER; **owner:** MAIN assigning the next RXC1 resume arm; **consumer store:** `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`; **fire trigger:** APDataStore has at least 1,400,000,000 free bytes with no concurrent decline; **action:** resume `--stage screen`, then run `--stage manifest` and perform the preregistered cost adjudication.

## LIVE-HYPOTHESES

- The remaining six rows may preserve exact identity because all 52 sealed comparisons are identical and checkpoint restoration is already proven.
- Batching several proposals per suffix encode may amortize the measured minutes-per-proposal cost because the expensive causal replay could potentially serve multiple candidate deltas.
- A coder with periodic state resets or bounded adaptation may admit a reconvergence splice because the observed nonreconvergence arises from persistent adaptive state in this specific HPAC/corrector instance.

## DEAD-ENDS

- Continuing on the current APDataStore free-space level is closed: it would breach the mandatory reserve.
- Treating exact-vs-exact correlation as Gate-1 evidence is closed by the preregistration; it is vacuous by construction.
- Claiming a completed `GATE-1-PARTIAL` verdict from 26/32 rows is closed because the named `SCREEN.json` denominator does not exist.
- Terminal frame-600 reconvergence on the sealed AFR1 rows is closed for this instance: 0/26 adaptive states matched baseline.