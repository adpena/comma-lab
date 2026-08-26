Completed the GB1 route-table charter. The durable memo is [ddm_fb2_route_table_gb1_20260826.md](/Users/adpena/Projects/pact/.omx/research/ddm_fb2_route_table_gb1_20260826.md).

Key conclusions:

- Fixed-distortion sub-0.12 demand: **42,229 B**.
- Zero-distortion demand: **0 B**; GB1 has **3 B strict integer slack**. The inherited ~150 B shortfall is stale.
- Complete table: **8/21 open pairs**, **13/21 receipt-closed**; across every ≥2-axis subset, **90/120 open** and **30/120 closed**.
- Half-renderer plus full manufactured repair now has **562.67 B** margin, versus FB1’s ~410 B.
- D3B is **+360 B** against GB1’s live token subsystem, not +207 B.
- Top routes: aligned W96 `R+M`; born-small resolved carrier `B+C`; trained changed-object renderer `R+B`.
- Concurrent BS3 work byte-closed a repeat-identical **101,150 B** body, but Seg/Pose remain unmeasured.
- No scorer, training, Modal, MPS, or exact evaluation ran.

Validation:

- Arithmetic assertions passed.
- Canonical ledger validation: `status=valid`, with only pre-existing unreadable-history warnings.
- Focused ledger tests: **23 passed**.
- Three FB2 ledger notes were appended at physical lines 691–693.

Git object writes were denied as anticipated. Serializer-authored custody is ready:

- Memo fallback commit `5638c8bdbeeaafb86d356340024648eb71aeca3a`; [format patch](/Volumes/VertigoDataTier/pact/ddm_fb2_serializer_fallback/20260826T202613.830502Z-40824/intended-commit.format-patch), bundle SHA `d5b43ce9784a6a2c56839fe40973b65a55353b5b7a8274662084c25d1b929106`.
- Ledger fallback commit `07933ec9704d40e3185fb0193e8d5379e0a8e569`; [format patch](/Volumes/VertigoDataTier/pact/ddm_fb2_ledger_serializer_fallback/20260826T203357.261000Z-48936/intended-commit.format-patch), bundle SHA `4396aa27d9739df164022a80b22036bf967252e8cfe8971e6b6d6695803aa055`.
- Both bundles passed `git bundle verify`; neither is landed in HEAD.

Two incomplete AP fallback checkout trees were certified as rebuildable scratch and removed, recovering roughly 12 GiB. The failure receipt and [cleanup certificate](/Volumes/APDataStore/pact/ddm_fb2_serializer_fallback/CLEANUP_CERTIFICATE_20260826T203200Z.json) remain; the removed source-HEAD scratch can be reconstructed from pinned tree `26b27dce163fa2be966b980aa651d8b828e83f1e`.

Own-vehicle frontier: unchanged — `S=0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: Pact HEAD; fire trigger: Git object writes are available. Land the memo format patch, then the ledger format patch, and revalidate the ledger.
- **BLOCKED / QUEUED-WITH-A-FIRE-ORDER** — owner: W96B implementation successor, then MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/`; fire trigger: at least 45,521,567,744 free bytes and reviewed exact expected-flip objective, schedule, step-zero pose gate, and resume tests.
- **BLOCKED / QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN sole scorer-lane router`; consumer store: `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/`; fire trigger: explicit scorer ownership, no active full-n600 scorer job, and all `BODY_RESULT.json` identities revalidate.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN renderer-training successor`; consumer store: `/Volumes/APDataStore/pact/ddm_or1_renderer_born_small/`; fire trigger: a receiver-closed changed-object candidate ≤137,986 B with resumable training, step-zero pose, and no hidden video-derived code.

## LIVE-HYPOTHESES

- Aligned expected-flip W96 may unlock `R+M/R+P`: the exact configuration has never run, while existing negatives are OFF-config advisory screens.
- BS3’s exact fresh DX2 carrier may recover enough pose to exploit its 36,836 B rate headroom; QS5 makes this plausible, but the three-way random-n32 measurement is still absent.
- A genuinely trained smaller renderer on the changed object may jointly absorb renderer and carrier function; no reference-form candidate has tested it.

## DEAD-ENDS

- Do not reuse the DX2-era ~150 B zero-distortion demand; GB1 already has 3 B slack there.
- Do not resurrect RJ1’s retracted `3.51×/97.70%` figures or the withdrawn zero-byte manufactured-seg re-aim.
- Do not retry current-body token/coder/model perturbations: JT23, JF2, LM1, and D3B close the measured family; D3B loses by 360 B on GB1.
- Do not call OFF-config W96 screens an aligned-family closure, or launch the aligned window without its exact objective and retained-storage capacity.
- Do not score BS3’s inherited-carrier body as the requested member; BO2 already measures that stale-carrier failure.
- Do not use a pose-only frame-0 solve to rescue HG1: its segmentation damage alone is fatal.
- Do not retry DC1S sparse dictionaries or treat blocked constraint shipping as measured positive rate supply.