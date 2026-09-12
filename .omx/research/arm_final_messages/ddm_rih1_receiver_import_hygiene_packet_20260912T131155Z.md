The recommendation is: **clean the three fallback symbols, run a fresh T4 FIRST-MEASUREMENT chain, then publish after hosting compliance closes.**

| File:line | Offending import | Reachability |
|---|---|---|
| `runtime/rc3_shared_mixer.py:17` | `ddm_rc2_hpac_semistatic_mixing_codec as base` | Fallback-only; not taken on successful contest decode |
| `runtime/sm1_semantic_mixer.py:17` | `ddm_rc1_adaptive_section_codec as rc1` | Fallback-only |
| `runtime/sm1_semantic_mixer.py:18` | `ddm_rc2_hpac_semistatic_mixing_codec as fixed` | Fallback-only |

The minimal executable diff is 2 files, 3 insertions, 10 deletions, reducing receiver source by 275 bytes. Two manifest hashes must also be replaced, making the complete diff 3 files, 5 insertions, 12 deletions. `archive.zip` remains 179,111 B.

Measured results:

- Import check: PASS, `disallowed=[]`.
- Behavior digest v2: `9f6e7168… → d3295528…`.
- Timing identity: DIFFERENT; pr19 therefore forbids inheritance.
- Cold local decode: n600, cache disabled, no resume, 3,662,409,600 bytes.
- Full raw: byte-identical to move 48, SHA-256 `2b762eba…`.
- Seeded-random n120 subset: zero mismatches across 732,481,920 bytes, SHA-256 `b407935b…`.
- No scorer, Modal job, publication, hosting, or staged/live receiver edit occurred.

Options:

- Publish as-is: $0, but strict compliance remains 91/93.
- Clean + inherit: unavailable; the behavior digest and two receiver rows differ.
- Clean + FIRST-MEASUREMENT: approximately $0.30 and 20 T4 minutes; recommended.

Memo: [ddm_rih1_receiver_import_hygiene_decision_packet_20260912.md](/Users/adpena/Projects/pact/.omx/research/ddm_rih1_receiver_import_hygiene_decision_packet_20260912.md), SHA-256 `e49876443109428d5c15f4e9abf672eba70579487a6f7f9f28075d27aefc08d7`.

Commit: serializer `rc 17` from Git-object write denial; memo remains untracked and the index was untouched. Verified fallback bundle SHA-256 `b77700cd…`, fallback commit `ef24f77c…`.

Composition S `0.13638261682704697` @ `179,111 B` `[contest-CUDA T4 n600]` (move 48) unchanged and not published.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: operator + MAIN; consumer store: reviewed move-48 successor and its first-measurement custody directory; fire trigger: operator selects cleanup; apply the exact diff, rebind the manifest, authorize and fire one T4 FIRST-MEASUREMENT row, then restage.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: operator + MAIN; consumer store: `submissions/_staging_move48_pr140_swap/SWAP_COMMANDS.md`; fire trigger: explicit acceptance of the import exception plus publication confirmation; host and fetch back the archive while recording that 91/93 is not full strict compliance.

## LIVE-HYPOTHESES

- A cleaned T4 evaluation will reproduce move 48’s exact score components because the complete local n600 raw and archive are byte-identical. This remains untested on contest T4.

## DEAD-ENDS

- Clean + inherit is closed: the v2 digest and two behavior-bearing rows differ.
- Dynamic-import concealment is closed: it would evade detection without cleaning the dependency.
- Cached or prefix-only identity testing is closed: a cold full-n600 proof now exists.
- Treating the two compliance records as two symbols is closed: source inspection found three imported symbols.