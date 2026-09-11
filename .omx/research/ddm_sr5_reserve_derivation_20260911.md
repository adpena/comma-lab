# A derivation for the SSD fail-closed reserve — proposal only, nothing changed

Axis: `[macOS filesystem custody; scorer-free]`. No score measured. **This memo changes no reserve.**
It proposes one, and says what the current number actually is.

## ANSWER FIRST

**The 40 GiB SSD reserve is not derived. It is a boot-volume swap floor that was copied onto two
external SSDs, where the failure mode it protects against does not exist.**

The canonical value is `src/comma_lab/storage_tiers.py:26`, `DEFAULT_RESERVE_FREE_GB = 40.0`, and it
carries no comment, no measurement and no rationale. The only 40 in this repo that carries a stated
reason is a *different* constant on a *different* volume: `tools/launch_detached_process.py:396`,
`BOOT_MIN_FREE_GIB = 40.0`, derived from the 2026-09-04 ENOSPC where **swap reached 72 GiB** on the
boot volume. Swap does not live on `/Volumes/VertigoDataTier` or `/Volumes/APDataStore`. The number
was transferred across regimes, and nobody re-derived it at the new scope.

Derived from what the reserve must actually cover, the SSD figure lands near **21 GiB**, or **24 GiB**
with a deliberate safety multiple. Today's 40 GiB is roughly **1.7–1.9× larger than its own job**, and
that gap is not free: it is the direct cause of this arm existing. On 2026-09-11 the guard refused a
**3 KB** copy on a volume holding **35.3 GiB** of genuinely free space — measured by `ddm_tmx1` at
22:05Z and relayed by MAIN, not by this arm.

## WHAT IS ACTUALLY IN THE TREE (MEASURED)

| constant | value | volume | derivation present? |
|---|---:|---|---|
| `comma_lab/storage_tiers.py:26 DEFAULT_RESERVE_FREE_GB` | **40.0 GiB** | SSD tiers (canonical) | **none** |
| `tools/launch_detached_process.py:396 BOOT_MIN_FREE_GIB` | 40.0 GiB | `/System/Volumes/Data` | **yes** — 2026-09-04 ENOSPC, swap peaked 72 GiB |
| `tools/cell_queue_driver.py:66 DEFAULT_RESERVE_BYTES` | 8 GiB | work tier | none |
| `witness_dsl/…_g_stream.py:55 DEFAULT_STORAGE_SAFETY_RESERVE_BYTES` | 8 GiB | work tier | none |
| `src/tac/payload_retention.py:61 _RESERVE_BYTES` | 2 GiB | payload tier | none |
| `tools/probe_segnet_exact_forward_transfer.py:55 STORAGE_METADATA_RESERVE_BYTES` | 8 MiB | metadata only | yes, scoped |
| `tools/measure_ddm_cb1_…py:117 minimum_free_bytes` | ≥ 40 GiB | SSD | none (copies canonical) |
| arm-local `RESERVE = 40 * 1024**3` | 40 GiB | SSD | none — `bnd2`, `bnd3`, `gdc1`, `dwc1`, `ntb2`, `rbf1` |

The arms re-hardcode 40 rather than importing the canonical constant, so the canonical value is not
even a single point of control. `experiments/ddm_rbf1_boundary_probe.py:167` shows the intended
semantics clearly: `required = 40 * 1024**3 + remaining * (per-chunk payload)`. The reserve is meant
to be **untouchable floor on top of the job's own need** — the job sizes itself separately. So the
reserve does not have to cover a whole job. It has to cover what a job cannot size in advance.

## WHAT THE RESERVE MUST COVER (MEASURED COMPONENTS)

| component | why the reserve must hold it | measured | source |
|---|---|---:|---|
| serializer fallback bundle | a Git-object denial must still be able to persist a landing | **16,521,329 B = 0.0154 GiB** (largest in the tree; the whole fallback dir 33 MB) | `ddm_vr7_20260910/serializer/…/intended-commit.format-patch` |
| one cold n600 raw | a decode in flight materialises a full raw | **3,662,409,600 B = 3.411 GiB** | `ddm_rlc5_cure_on_move43/public_rlc4/output/0.raw` |
| its in-progress twin | the decode keeps a resumable checkpoint beside the final file | **3.411 GiB** | `ntb2` cold manifest: `render_cpu_in_progress.raw` 3,662,409,600 B |
| its token stage file | written beside both | **0.110 GiB** | `tokens_cpu_stage_complete.u8` 117,964,800 B |
| one twin encode / candidate materialisation | an encode builds a candidate tree while the raw is held | **3.5 GB = 3.26 GiB** | `ddm_pc3_pose_carrier_curve/candidate` |
| scorer input caches touched by both | gt cache + token field must be resident | **0.220 GiB** | `gt_cache_dali.pt` 117,980,732 B + `subset6.u8` 117,964,800 B |
| candidate runtime tree | copied per candidate | **0.002 GiB** | `ddm_rlc5_cure_on_move43/candidate_runtime` 1.8 MB |
| **one arm's worst simultaneous footprint** | | **10.429 GiB** | sum of the above |

## THE PROPOSED DERIVATION

```
single-arm worst case                       10.429 GiB   (measured, table above)
× 2 concurrent arms on one tier               20.858 GiB   (the fleet routinely runs 3-4 arms;
                                                            two of them decoding at once is the
                                                            case the reserve exists for)
+ APFS metadata and directory slack            0.142 GiB   (round up)
-----------------------------------------------------------
proposed SSD reserve                          ~21 GiB
with a deliberate 1.15x safety multiple       ~24 GiB
```

Both figures are **well under 40**. The difference, 16–19 GiB per tier and 32–38 GiB across the two,
is capacity the fleet currently cannot use and that arms like this one are repeatedly spawned to
manufacture by deleting real payloads.

## WHY THE BOOT FIGURE DOES NOT TRANSFER

`BOOT_MIN_FREE_GIB = 40.0` is sound **for the boot volume**. Its derivation is explicit: during the
2026-09-04 near-OOM, macOS swap grew to 72 GiB, and swap lives on `/System/Volumes/Data`. A floor
there must absorb OS behaviour the fleet does not control.

Neither SSD tier hosts swap, hosts the OS, or takes Time Machine local snapshots (`tmutil` reported
none on Vertigo at 22:05Z). Their only writers are arms, and arms size their own payloads. The
uncontrolled-growth term that justifies 40 GiB on the boot volume is **absent** on the SSDs, so the
number has no support there. This is the cross-regime constant transfer the memory index already
names: re-derive a binding number at the scope where it binds.

## THE COST, MEASURED TODAY

The reserve is not a passive number; it refuses work.

- 2026-09-11 22:05Z: the guard **refused a 3 KB copy** with 35.3 GiB free — three instruments agreeing
  (`df` 35.32 GiB, `diskutil apfs` 37,921,574,912 B, `tmutil` no snapshots). MEASURED BY `ddm_tmx1`
  and relayed to this arm by MAIN; sr5 independently read 35.317 GiB from `df -k` at the same hour,
  which agrees, but the three-instrument cross-check is tmx1's, not mine.
- Earlier the same day, `tools/vertigo_certify_move.py` correctly refused a 10.9 GiB certified move of
  `ddm_hm1_20260810` because the destination would land at 38.2 GiB — 1.8 GiB under the floor.
- `experiments/ddm_ntb2_renderer_score.py:148` records the tier having "under 11 GiB usable above its
  40 GiB reserve", which is the same squeeze stated from the arm's side.
- This arm deleted **28.2 GiB of certified-rebuildable payload** and relocated a 6.27 GB tar to keep
  two tiers above a floor that, on this analysis, is roughly twice what it needs to be.

Every one of those refusals was the guard working exactly as written. The question this memo raises is
not whether the guard is correct. It is whether the number it enforces was ever chosen.

## WHAT WOULD MAKE THIS ACTIONABLE (not done here)

1. Give `DEFAULT_RESERVE_FREE_GB` a derivation comment naming the components and their measured sizes,
   the way `BOOT_MIN_FREE_GIB` already does. A number with a stated reason can be re-derived; a bare
   float cannot.
2. Make the six arms that hardcode `40 * 1024**3` import the canonical constant, so there is one point
   of control. Until then, lowering the canonical value changes nothing.
3. Re-measure the worst single-arm footprint whenever the decode or candidate shape changes, and treat
   the reserve as **derived output**, not configuration.
4. Only then consider a value. This memo proposes ~21 GiB derived, ~24 GiB with margin, and changes
   nothing.

## HONEST LIMITS OF THIS PROPOSAL

- The concurrency factor of 2 is a **judgement**, not a measurement. I did not measure the true
  maximum number of simultaneously-decoding arms; 3–4 arms run concurrently but they are not all
  decoding. A fleet that genuinely ran four cold decodes at once would need ~42 GiB, which would
  *vindicate* 40. Someone should measure that distribution before any number is changed.
- The single-arm footprint is assembled from components measured on **different** arms today, not from
  one instrumented decode. A direct high-water-mark measurement of one cold n600 decode plus encode
  would be stronger than this sum.
- I did not audit every reserve site in the tree; the table lists the named constants I found, not a
  proof that no other reserve exists.

<!-- # FORMALIZATION_PENDING: storage-policy proposal; no measured scientific row and no canonical equation governs filesystem reserves -->
