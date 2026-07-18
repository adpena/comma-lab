# Task #543 v10 power-diagram byte-close findings (2026-07-18)

`research_only=true` · lane `v10_power_diagram_byteclose_20260718` ·
`[macOS-CPU feature-pullback advisory]` · MAIN landing review required

Pointer `0.1910828242 [contest-CPU Linux x86_64]` is **UNMOVED**. No sacred
run, archive, score pointer, provider, GPU, or upstream file was mutated.

## Verdict first

**BLOCKED at INSTANCE/FORMULATION scope; family and paradigm remain OPEN.** A
governed batch-one n600 frozen-SegNet extraction stopped fail-closed at
canonical frame `195` because one near-boundary pixel had different labels
under the frozen CPU-Torch convolution and the generic float64 evaluation of
the serialized-float32 power target. The CPU-Torch forward still reproduced
cached `L*` exactly at that pixel. The repository has no RGB receiver and no
declared PDW1 receiver arithmetic, so neither arithmetic path may be silently
promoted to authority.

The preserved prefix `0..194/600` supports an honestly post-hoc diagnostic:
`899,388 / 38,338,560 = 0.023459097055288463`
`MEASURED_FEATURE_PULLBACK` mismatches for the fitted target. Its strict PDW1
payload is `314` raw bytes and `306` Brotli-Q11 bytes; the optimistic order-0
rounded-up ideal entropy estimate is `257` bytes under free-PMF and zero-overhead
assumptions. These are target coefficients only, not a spatial
quotient field or RGB realization. Therefore the comparison with the
`228,764 B` MDL contour, `235,974 B` contour+xi, and `225,272 B` sub-0.15 line
is **NON-EQUIVALENT** and does not close the stated 4.5 percent gap.

Factor 6 remains `HAVE | PARTIAL`; the candidate equation remains held outside
the shared registry; no score or pointer authority is created.

## Stores consulted and custody

- Delegated authority file, bytes `5,999`, SHA-256
  `44f67fc24d60121c4fc1c679dbf74dbad66e8419e2997e70c300dfacd6e888e3`.
- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, the
  fresh-eyes confound-hunt contract, v7.5/v8 operating contracts, v10 SSoT,
  frozen scorer factorization, factor-completeness matrix, and prior
  power-diagram memo.
- Real GT cache `gt_n600.npz`: `5,078,017,610 B`, SHA-256
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
- Frozen SegNet: `38,502,892 B`, SHA-256
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`.
- Upstream `modules.py` SHA-256
  `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`;
  `frame_utils.py` SHA-256
  `d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90`.
- Exact historical governed measurement source: `62,907 B`, SHA-256
  `be094a1540a94bf51aa98706b6d4515eec150bb569380f69b308ed66556cd7c9`,
  now preserved only inside a deterministic non-source gzip container:
  `16,187 B`, SHA-256
  `ee13d263b51f210fe7fd7bbfc6a21099260189573fce80715c0d69df0f2ef329`.
  Its v2 manifest is `1,130 B`, SHA-256
  `5873ad1f2145402a63a53f114f2845cd80f04c6458d4082366bb6f91b2e7b274`.
  The live tool path is a fail-closed tombstone, SHA-256
  `fb7114017c735c3ad38f4e4b81a60653910a9ed49dafa47ac47dd42fce05ce76`.
- Current source-custodied prefix harvester SHA-256
  `4afb88f72645f9bf43cf6765ae524a9d9263de1a5ae682757297c6eee13fd303`;
  frame-195 diagnostic SHA-256
  `fe95209817b8d7b2aba3bbdbeb19538d203ace1b1fa95fcb72d785cde7c191c1`;
  read-only evidence helper SHA-256
  `fe78ef1bc42d95f85ef4155f9b8e043c283012c96dfb4a2265969680ded428b1`.
- Preserved SSD quotient cache: `1,887,436,928 B`, SHA-256
  `59e96781aa1bac153bc8bb277cecdbd4b4e98fdfd41f50aa2294537b90390944`.
- Preserved blocked checkpoint: `4,940 B`, SHA-256
  `58656d231af5c63b12b3594d8eeeeccf0b2d0f25c09154ef3ef6da759e1fce4b`.
- Durable post-hoc receipt:
  `.omx/research/v10_power_diagram_byteclose_blocker_receipt_20260718.json`,
  SHA-256 `3c64eb2849ced6dd8eb4492437744fccd03f89977c244bce73cf5a5e30db6e2f`.
- Governed one-frame reproduction receipt:
  `.omx/research/v10_power_diagram_frame195_diagnostic_20260718.json`,
  SHA-256 `65d97194c6298a5502d0fcc792ee2fe3bf05599c69f1130d64c270dec5ec36ee`.
- Original launch-custody receipt:
  `.omx/research/v10_power_diagram_original_launch_custody_20260718.json`,
  SHA-256 `c2a5ba645a7fa4e9bdff6bdd6612a65b2fdb958b5d0c671f1d71f74fc0bf92e8`.
- Storage-waterfall plan:
  `.omx/research/v10_power_diagram_storage_plan_20260718.json`, SHA-256
  `4710959a4461d9746d6ba37cff2d29dc05375d1fb9eef420e6d2d7f53b6cc31f`;
  selected `/Volumes/VertigoDataTier/pact` for `3,000,000,000 B` projected
  scratch and authorized no GPU, score, or promotion claim.
- Canonical task status `task-543-v10-power-diagram-byteclose-20260718` is
  `blocked` with `test_status=green`; its blocker is the missing declared legal
  receiver arithmetic and complete spatial/RGB/uint8/archive realization, not
  the power-diagram family.

The original durable daemon registry preserves the exact inner argv and the
`12 GiB` system-admission projection. The `12,288 MiB` process-group RSS cap
and `2,400 s` timeout survive only in the tool transcript and are explicitly
`TRANSCRIPT_DERIVED_UNRECEIPTED`, not exact machine-custody argv. The terminal
lane observation records `136.09 s`, peak RSS `3,141 MiB`, and
`failed_positive_control`.

## PASS-1 containment of the retired cleanup path

Fresh-eyes review found that the historical tool's success cleanup certificate
could replay an existing output path and cleanup flag. No deletion occurred,
but that executable surface was not safe to retain. A first plaintext snapshot
was also still executable through an explicit Python interpreter despite its
suffix and mode. The checkpoint-pinned source bytes are therefore preserved
only inside deterministic, filename-free gzip evidence:
`.omx/research/evidence/measure_v10_power_diagram_generator_byteclose_be094a1540a94bf51aa98706b6d4515eec150bb569380f69b308ed66556cd7c9.source.gz`.
Validation decompresses only in memory and binds the `16,187 B` container
(`ee13d263...`) to the original `62,907 B` source (`be094a154...`). Direct
`python <container>` fails before import/argparse execution and leaves its
mutation sentinel unchanged. The live tool refuses execution, resume,
certification, and cleanup. All reusable parsing and math live in
`tools/v10_power_diagram_blocked_evidence.py`, which exposes no cleanup or
resume API. A subsequent clean-pass review found that two public helper
functions could still construct the retired checkpoint schema despite that
claim. Both builders and exports were removed from production; only test-local
fixtures can construct historical parser inputs. The production evidence
surface now validates immutable historical payloads but cannot construct or
serialize checkpoint/resume payloads. A later whole-scope pass also found that
the prefix receipt named its executing harvester in argv without binding the
harvester/helper bytes or runtime. The cured harvester now captures both
canonical sources before measurement, records path, bytes, SHA-256, device,
inode, mtime, and Python/platform/NumPy custody, and full-hash rechecks both sources
after compression. Receipt validation fails closed on absent, forged, or
current-source/runtime-drifted custody.

The next tooling pass found that both entrypoints resolved `--output` before
checking its final component for a symlink, allowing a broken in-tree link to
be interpreted as its nonexistent target path. Both run entrypoints now
validate the raw absolute output path before resolving any input or beginning
work. End-to-end tests cover broken links targeting both inside and outside the
research tree; neither target is created.

The first post-hoc receipt, SHA-256
`5cff28e602b2a9b51d986968d73f49e61df0ea8b26365b0c93422a64e197de71`,
is retained only at
`.omx/research/evidence/v10_power_diagram_byteclose_blocker_receipt_precontainment_5cff28e602b2a9b5.json`
as superseded, non-authorizing provenance. The current receipt revalidates the
historical container/manifest, live tombstone, checkpoint, full cache hash, GT,
model, source hashes, and post-run SSD immutability. The pre-seal receipt
`v10_power_diagram_byteclose_blocker_receipt_preseal_ae890fb8ef27effd.json`
(`ae890fb8ef27effd12572c1ec27e0faee008b042ccb165b8580bff946e57d747`)
and pre-seal diagnostic
`v10_power_diagram_frame195_diagnostic_preseal_61fd53dc344941b0.json`
(`61fd53dc344941b0f4e9fc34a832099006c5bd3685aad66ce164d1bcd95a461d`)
are also retained under `.omx/research/evidence/` as superseded,
non-authorizing provenance. The receipts from the failed checkpoint-API seal,
`v10_power_diagram_byteclose_blocker_receipt_preseal2_e66b3056a03912cd1e46009204fb1a094c365684cc81030431cde0b7f143e9e5.json`
and
`v10_power_diagram_frame195_diagnostic_preseal2_c9e3b37ce949e86fa78b3d494bc7b3670b1128a84db78ffadf64d5199633623e.json`,
retain exact SHA-256 identities in their names and are likewise superseded and
non-authorizing. The receipts from the failed execution-source-custody seal,
`v10_power_diagram_byteclose_blocker_receipt_preseal3_0a62dd1102f9b8462cbd12003891b852f4374fcf9c728a7a995e57e85b7031db.json`
and
`v10_power_diagram_frame195_diagnostic_preseal3_ce037fd1dd6b37ec209d730e10735ac4278c59200388da1fe103d833f88d52a2.json`,
are also preserved as superseded, non-authorizing evidence. The receipts from
the failed raw-output-symlink seal,
`v10_power_diagram_byteclose_blocker_receipt_preseal4_9b1b05bb65910433ddac0d1ff3e5e3e727f3d969999767544760ffe55d71a8d6.json`
and
`v10_power_diagram_frame195_diagnostic_preseal4_9ea3eccc38ab08e9d734c16b833d9be9065f3dcb5dcb369b055dcfb7c8bc187e.json`,
are also preserved as superseded, non-authorizing evidence. Both live
receipt writers now refuse any output outside an existing resolved
`REPO_ROOT/.omx/research/*.json` path, including source/main/SSD/transient and
symlink-escape targets; they never create parent directories or overwrite.

## Governed n600 attempt and preserved checkpoint

**[MEASURED local CPU-Torch, batch 1]** Frames were consumed in canonical
ascending order from `gt_f1[0]`; no shuffle or alternate class order was used.
Exactly frames `0..194` were committed before frame `195` triggered the stop.
The atomic checkpoint records:

- `status=blocked`, `next_canonical_frame=195`;
- `38,338,560 = 195*384*512` committed feature/label samples;
- class counts `(8,980,186, 224,892, 18,942,608, 429,289, 9,761,585)` in
  canonical class order `0..4`;
- one transformed power-target mismatch and zero CPU-Torch-forward mismatches;
- requested/effective Torch threads `6/6` and inter-op threads `18/18`;
- float32 device input, deterministic algorithms, ridge `1e-6`.

The preallocated cache and blocked checkpoint are preserved on
`/Volumes/VertigoDataTier/pact`. No resume, cleanup, truncation, or receiver
arithmetic override is authorized.

## Exact-vs-approximate numerical confound

**[MEASURED governed read-only local CPU diagnostic]** The sealed one-frame
reproducer ran through `tools/safe_run.py` with an explicit `6,144 MiB` RSS
cap, `1,200 s` timeout, and `6 GiB` admission projection; it exited `0` after
`6.249 s` with peak RSS `965 MiB`. The diagnostic receipt preserves the exact
inner argv, wrapper path/hash, and supplied limits; the active-lane ledger
records the terminal governed envelope. The diagnostic re-hashes every
verified input after inference. It
reproduced the first disagreement at
canonical frame `195`, pixel `(y=214,x=112)`, flat index `109680`:

- cached `L* = 0`; frozen CPU-Torch forward `= 0`;
- frozen logits are
  `(4.8681163787841797, 4.8681159019470215, -8.735725402832031,
  -5.273322105407715, -8.2166748046875)`;
- the CPU-Torch class-0 minus class-1 margin is exactly
  `4.76837158203125e-07`;
- the rank-4 quotient feature is
  `(0.0014007954159751534, 5.777266502380371,
  1.7700754404067993, 3.2382075786590576)`;
- generic float64 `power_scores` yields class 1 over class 0 by
  `2.5277826765e-07`;
- native float32 evaluation of the same serialized sites, weights, and
  quotient yields an exact class-0/class-1 tie at
  `6.278331756591797`; deterministic first-max selects class 0.

Direct float32 convolution of the row-difference filter and subtraction of two
separately reduced class convolutions differ in sign near this boundary. The
module's sampled parity diagnostic bounds the pair-score error at
`5.37073882256e-07`, larger than the real-arithmetic winner margin
`2.34714629954e-07`, and correctly labels this pixel tie-uncertain. The
rank remains four; the fifth singular value is `3.7304048124e-16`.

**[DERIVED]** This is floating-point non-associativity/reduction order plus the
already documented float32 serialization boundary, not SVD rank loss, cache
corruption, or a disproval of the real-arithmetic affine/power identity. An
epsilon, ULP nudge, or after-the-fact tie rule would be a fake fix. A new full
n600 run is justified only after a receiver explicitly declares deterministic
arithmetic and that receiver is tested on the failing pixel.

## Post-hoc committed-prefix measurement

**Selection warning:** this is `ADVISORY_POSTHOC_PREFIX_0_194_OF_600`, chosen
because the next frame exposed the blocker. It is neither a pre-registered
random subset nor an n600 result.

**[MEASURED_FEATURE_PULLBACK]** Reusing only the checkpoint's float64 ridge
sufficient statistics and the committed rank-4 cache:

- prefix frames: `195`; samples: `38,338,560`;
- observed adjacency:
  `(0,1) (0,2) (0,3) (0,4) (1,2) (1,3) (1,4) (2,3)`;
- fitted mismatch count: `899,388`;
- fitted feature-pullback `d_seg`: `0.023459097055288463`;
- held-out/generalization authority: **NONE**; fitting and measurement reuse
  the same post-hoc prefix.

**[MEASURED target bytes / DERIVED comparisons]** Strict encode -> fresh
decode -> encode is byte-identical:

| quantity | value | authority |
|---|---:|---|
| raw PDW1 | `314 B` | measured target payload |
| raw SHA-256 | `ac7915462724985c83a3b141cf0a2d9553132fe6ef5c379cca723701ba112c85` | measured bytes |
| Brotli-Q11 | `306 B` | measured target payload |
| Brotli SHA-256 | `7d3e5f90ea4cd85faf1ba291b5cca19f02914a93b341a6d05456972343a9c05e` | measured bytes |
| optimistic rounded-up ideal order-0 entropy | `257 B` | derived; free PMF and no model/header/termination overhead |
| vs `228,764 B` MDL contour | `-228,458 B` | derived, non-equivalent |
| vs `235,974 B` contour+xi | `-235,668 B` | derived, non-equivalent |
| vs `225,272 B` threshold | `-224,966 B` | derived, non-equivalent |

The apparent byte deltas are not savings. The 306-byte target omits the
`600*384*512*4` quotient field, RGB preimage, uint8 realization, inflate
receiver, Pose coupling, and exact archive. The contour references pay for
materially different realization surfaces.

## Round-1 adversarial self-review and named confound hunt

1. **Optimistic undercount versus realizable coding.** Brotli-Q11 is an actual
   compressed target payload. The order-0 value is only a rounded-up ideal
   entropy estimate under free-PMF/no-overhead assumptions; it is neither a
   realizable ceiling nor a lower bound on a legal witness archive. Neither
   quantity includes the dominant field/receiver bytes.
2. **Exact versus approximate cells.** The affine/power law is exact over the
   reals. PDW1 coefficients and the cached quotient are float32, while generic
   `power_assign` promotes to float64. The frame-195 disagreement proves that
   receiver arithmetic is load-bearing.
3. **Canonical real-GT order.** The fit uses real `gt_f1`/`lstars`, frames
   `0..194`, ascending, batch one, class order `0..4`. The subset is explicitly
   post-hoc and cannot be extrapolated to the other 405 frames.
4. **Positive control.** CPU-Torch forward versus cached `L*` remained exact at
   the blocker. Serialized target parity did not. Reporting only the former or
   silently choosing native float32 would hide the confound.
5. **Fit leakage.** The fitted mismatch is in-sample. It measures whether one
   global fitted power target can reproduce this prefix in the cached quotient,
   not generalization or receiver survival.
6. **Through-R premise falsification.** PDW1 is a channel-target certificate,
   not an RGB generator. With no spatial field and no receiver there is no
   image to pass through uint8/resize `R`; a through-R `d_seg` number would be
   fabricated.
7. **Comparator non-equivalence.** Comparing 306 target bytes to complete
   contour/xi realizations cannot answer whether the 4.5 percent score gap is
   closed. Correct answer: `NO_EQUIVALENT_RATE_OR_SCORE_VERDICT`.
8. **Cleanup-certificate replay.** The initial executable cleanup path was
   unsafe even though it was not exercised. Exact source bytes remain available
   for checkpoint lineage only; the live path is tombstoned and every new
   evidence tool is read-only with output-overwrite refusal.

## Triality and remaining work

- DAG: `.omx/research/power_diagram_byteclose_DAG_FEED_20260718.md`.
- Temporary equation candidate:
  `.omx/research/v10_power_diagram_equation_candidates_20260718.jsonl`, status
  `HELD_NOT_REGISTERED`.
- DSL: N/A for this read-only advisory target measurement. Any future receiver
  must enter through typed DSL and the canonical resume registry.
- Reactivation criterion: specify a deterministic legal PDW1 receiver
  arithmetic, reproduce the frame-195 boundary case, encode the spatial
  quotient field or an RGB inverse, parse back exact archive bytes, and measure
  realized uint8 through-R Seg/Pose on the same artifact.

Task #543 therefore lands a durable blocker and the first real-prefix
generator-target rate, not a completed factor or goal result. MAIN must review
the branch diff before any merge, shared-DAG fold, or equation registration.
