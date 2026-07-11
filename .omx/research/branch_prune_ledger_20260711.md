# Branch-prune ledger — origin/main becomes the SOLE branch (operator-directed 2026-07-11)

**Directive:** "Merge / cherry pick all branches and worktrees onto main and prune so origin/main is
the sole source of truth." **Pre-state (MEASURED):** local = main only · main == origin/main (0/0) ·
1 worktree (the main checkout) · 0 stashes · remote = origin/main + 4 `safety/stash-recovered-*`
branches. **Nothing required merging or cherry-picking** — the only non-main refs were 2-month-old
WIP-stash safety snapshots from superseded program eras (pre-PR95-race / pre-witness-pivot).

## Certify-or-block record (the 4 pruned refs; content certified SUPERSEDED, not merged)

| ref (origin/) | tip sha | base date | content | why superseded |
|---|---|---|---|---|
| safety/stash-recovered-20260505T052046Z-stash0 | 1d9e73294a | 2026-05-04 | "pre-rigor-pass safety stash" — 121-file WIP incl. uv.lock churn | the May-4 race work landed on main via its own commits; era pivoted twice since (HNeRV→witness→V9·CGauge) |
| safety/stash-recovered-20260505T052046Z-stash1 | 710bd3a23b | 2026-04-29 | compress_archive.py WIP (+69 lines) on the STRICT-checks-82/83 base | compress/archive path long rebuilt (L13 byte-close, levelset_byte_close_and_eval) |
| safety/stash-recovered-20260505T052046Z-stash2 | e8ca384e3f | 2026-04-26 | "yousfi_3_5_pending_greenup" on master — test_yousfi_5_uncertainty.py etc. | yousfi-3/5 era work landed/retired via its own history; branch base predates the pivot |
| safety/stash-recovered-20260505T052046Z-stash3 | c242e5dc5f | 2026-04-26 | "DEN-V2 partial: 4 layers of arch-drift fixed" — renderer.py WIP | DEN-V2 substrate dead/superseded by the witness line |

Reproducibility: tip shas above locate the commits in any pre-prune clone (local clone retains the
objects until GC); the branches were themselves the RECOVERY artifact of stashes already recovered
2026-05-05 — this ledger supersedes them as the durable record. Post-state target: origin holds
exactly ONE branch (main). [no-triality: repo hygiene]
