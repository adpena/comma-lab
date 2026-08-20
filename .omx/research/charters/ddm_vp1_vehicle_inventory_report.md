# ddm_vp1 — Re-target the v7→v10 inventory to the PR130 base. The table exists; its vehicle changed.

**Owner:** codex arm · **Base:** PR130 CPR1 · scorer-free · `[macOS-CPU advisory]` · `score_claim=false`

## OPTIMAL FORM (read first)

Reference form: `ddm_wl1`'s landed v1→v10 transfer table RE-SCORED against the PR130 base — pin
`0491d5df84fc70b6` (full sha in PROVENANCE), archive 191,052 B, commit `113b52fdb1`. Declared
reductions: SCOPE only — you may rank and cut the tail, but every PORT-NOW row you keep must carry a
PR130-base justification. MECHANISM reductions are TOY-BRACKET: re-reading wl1's verdicts as if the
vehicle were unchanged; a "port" whose evidence is a TR1/burn-vehicle number; a row promoted on
reputation instead of a measured PR130-base exchange rate.

**Authority:** full online research, full OSS, full internal leverage — our own code, docs, and
unwired modules are off the shelf to use, adapt, refactor, extend, or fork. Cite path + commit.

## WHY

Operator 2026-08-09: *"We can port inventories from past vehicles. V seven, eight, nine, ten, and up
to now if you think that they would be helpful."* They would — with one correction.

`ddm_wl1` (#967) ALREADY harvested v1–v10 into a ranked {PORT-NOW/RACE/LESSON-ONLY/DEAD/REOPEN}
table with named consumers. It was scored when the vehicle was **TR1**. The base then moved to
**PR130**, and #917 independently measured that our lever instruments still point at the retired
vehicle. So the table is an ASSET whose scoring column is stale — the cross-regime constant-transfer
genus at table scale.

WHAT CHANGED UNDER IT (all measured today, all must re-score the rows):
- pose representation CLOSED (`pk2`: baseline WON, basis already signed int5 −15…15)
- coder axis CLOSED (#996, all 4 sections vs their own memoryless bound)
- gauge family CLOSED (`pk4` `113b52fdb1`: 64 B vs a 2,000 B gate, 432 candidates)
- three byte wins banked: ANS tokens −2,120/−2,416 · sm3 semantic low-rank −6,272 · split-stream
  model pack −903 (real archive 190,149 B, parse-back byte-identical, RECEIVER-BLOCKED)
- rate is 73.9% of S; sub-0.15 by rate alone = −33,252 B (−17.4%)

Archive anatomy to score against (MEASURED, from the real archive):
```
archive.zip 191,052 = ZIP 100 + member p 190,952
p          = [u32][ LZMA(models_raw) 73,968 ][ HPAC tokens 116,980 ]
models_raw = 83,493 = 8 + semantic 40,252 + pose_carrier 23,054 + hpac_weights 20,179
```
NOTE: raw section sizes and leave-one-out marginals are DIFFERENT quantities. Do not slide between
them — say which you are using in every row.

## WHAT TO DO

1. **Load wl1's table, do not rebuild it.** Rebuilding is the rediscovery sin. Cite its rows.
2. **Re-score every PORT-NOW and REOPEN row against the PR130 base**, each landing in exactly one of:
   STILL-PORTS (name the PR130 slice it targets + its exchange rate vs 1,000 B ≈ 0.000666 S) ·
   DEAD-ON-THIS-BASE (name which of the four closures above kills it) ·
   NEEDS-REMEASURE (the TR1 number cannot transfer; name the measurement).
3. **Extend the sweep to "up to now"** per the operator: the TR1/burn era and today's PR130 arc are
   inside scope, and wl1 stopped before both.
4. **Rank the survivors by measured S-per-byte on THIS base** and emit the top 3 as build-ready
   charter stubs with provenance pins.
5. **Do NOT re-run any closed family.** pose-representation, coder-axis, gauge — all closed with
   receipts above. A row that proposes one of them is DEAD-ON-THIS-BASE by definition.

## HARD RULES

- Bulk → `/Volumes/VertigoDataTier/pact/ddm_vp1_20260810/`. No `/tmp` in evidence.
- Commits via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`, **no attribution trailer of any kind**.
- `.py`: 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never `REVIEW_GATE_OVERRIDE=1`
  with a `.py`. Run `tac.payload_retention_gate` on anything you write (P0, CLAUDE.md).
- Write ONLY under `ddm_vp1` paths. Sister arms' artifacts are APPEND-ONLY inputs.
- `upstream/` IMMUTABLE. Intake clones READ-ONLY. Every number carries its axis. No Modal, no scorer
  race — `ddm_ai1` holds the slot.
- The OLD-LINEAGE BAN still binds and is NOT what this arm ports: HNeRV/PR95/110/128 remain
  lessons-only. v7–v10 are OUR OWN lineage and are in scope.

## PROVENANCE PINS

- base archive `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip`
  191,052 B sha `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`
- wl1's landed transfer table (locate at source; do not reconstruct from memory)
- pk2 pose closure · pk4/pk3 gauge closure `113b52fdb1` · #996 coder-axis closure

## DELIVERABLE

The re-scored table with every row in exactly one of the three states and its PR130-base
justification · the "up to now" extension · top-3 ranked survivors as charter stubs. If the honest
answer is that most of v7–v10 is DEAD-ON-THIS-BASE, say so with the closure that kills each — a
short true table beats a long stale one, and it stops the campaign re-reading a table scored for a
vehicle we no longer run.
