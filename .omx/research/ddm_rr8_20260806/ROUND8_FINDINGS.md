# Round 8 recursive adversarial review findings

**Verdict:** NOT-CLEAN. Clean-pass counter remains **0/3**.

**Scope:** rr8 charter `.omx/tmp/codex_runs/rr8_prompt.md` plus `_common_contract.md`; prior review memos RR1..RR7; ET4 executed repair/final receipts; twelfth-move adjudication; OOM incident memory; HB1 resume caveat; MX1/MX1B fire scheduling surface.

**Authority labels:** ET4 numbers below are `[macOS-CPU advisory]` / `score_claim=false`; no contest-CPU/CUDA promotion claim is made. I did not launch a scorer job.

## RR8-F1 - HIGH - ET4 repair liveness fallback still fails open under denied process enumeration

**Finding:** `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_repair_rows_and_final.sh` claims that the `ps` fallback proves absence after `pgrep` enumeration failure, but its implementation collapses a failed/denied `ps` command into `0` candidates:

- `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_repair_rows_and_final.sh:21-28` enters fallback on `pgrep rc>=2`.
- `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_repair_rows_and_final.sh:24` runs `live=$(ps axo command 2>/dev/null | grep -c "ddm_et4_solve_within_cvp_n600" || true)`.
- Current managed-host probe: `pgrep_et4_rc=3`, `ps_axo_rc=126`, `ps_axo_msg=bash: /bin/ps: Operation not permitted`, while the script-shaped fallback produced `repair_fallback_live_value=0`.

That is not a proof of quiescence. If both enumerators are denied, the repair can dedupe and finalize while a writer is still alive.

**Damage check:** I did not find ET4 data corruption in the checked receipts. The final repaired log reports `n_rows=600`, `population=600`, `net_flip_reduction=78302`, `archive_bytes=11645079`, and `full_population_complete=true` (`final_stage_repaired.log:3-35,90-92`). Independent rows audit found the preserved backup at 751 rows / 600 unique pairs and post-dedupe rows at 600 rows / 600 unique pairs, with the sampled duplicate pairs matching the keep-last premise except elapsed-time metadata.

**Smallest cure:** make fallback fail closed on the fallback enumerator itself:

```bash
ps_out=$(ps axo command 2>&1) || { echo "REFUSE: pgrep rc=$prc and ps failed: $ps_out"; exit 4; }
live=$(printf '%s\n' "$ps_out" | grep -c "ddm_et4_solve_within_cvp_n600" || true)
```

Then rerun only the no-writer guard, or replace process enumeration with a durable launch-supervisor liveness receipt plus the existing ledger mtime guard.

## RR8-F2 - HIGH - Original ET4 detached driver still emitted a false success receipt

**Finding:** the original ET4 chain driver did not propagate final-stage failure to its detached `.done` receipt:

- `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_n600_driver.sh:32-33` runs the final stage and only echoes `$?`; it never exits with that status.
- `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/shards_rc.txt` records `shards rc: 0 0 0` and `final rc: 1`.
- `.omx/tmp/codex_runs/et4_chain_v3.done.done` nevertheless records `rc=0 elapsed=13392 detached-job`.

The repair path later produced a valid `rc=0` final receipt, so this did not create the final ET4 row by itself. It did, however, leave the executed fleet boundary able to announce success for a failed final stage.

**Smallest cure:** patch the original driver before any reuse:

```bash
rc=0
$PY $R --resume --threads 4 --build-archive --run-inflate --run-evaluate > "$LOG/final_stage.log" 2>&1 || rc=$?
echo "final rc: $rc" >> "$LOG/shards_rc.txt"
exit "$rc"
```

Watchers must also require the stage receipt or `shards_rc.txt`, not a detached `.done` receipt alone, for any prior affected run.

## RR8-F3 - HIGH - Metal fire scheduling is still a procedural law, not a structural gate

**Finding:** the crash was caused by a fire-order/scheduling surface that RR1..RR7 did not review. The OOM memory records the cause directly: bare `tools/launch_detached_process.py` fires have no memory gate, and a ticket is not a scheduling license (`concurrent_metal_fires_without_composed_preflight_oomed_the_machine_20260806.md:28-32`). The launcher confirms that shape: it parses generic argv and optional receipts (`tools/launch_detached_process.py:45-93`), then launches with `subprocess.Popen(start_new_session=True)` and writes PID/manifest (`tools/launch_detached_process.py:124-190`), with no Metal/MLX classification, global lock, load-phase peak receipt, or composed-memory check.

MX1B is the right instance-level cure direction, but it is not yet sufficient as a structural fleet gate:

- `.omx/tmp/codex_runs/mx1b_prompt.md:35-41` requires `--mode mem-probe`, `mem_probe_receipt_required: true`, and sequential Metal scheduling.
- Current source now has the intended ticket fields (`experiments/ddm_mx1_pr130_semantic_renderer.py:1033-1039`), but persisted launch tickets still lack them: `jq '.launch_ticket | {mem_probe_receipt_required, mem_probe_receipt_path, scheduling}' /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch_ticket_v2_regenerated.json` returns all `null`.
- The persisted v2 ticket still carries direct n32 `mlx-train` argvs (`launch_ticket_v2_regenerated.json:75-145`) and an `arm_selection_rule` to fire both n32 arms (`launch_ticket_v2_regenerated.json:147`), with no passed mem-probe receipt field.

**Residue check:** I did not find live MX1 orphan processes for the recorded PIDs: `kill -0 9700` and `kill -0 44122` both returned no such process. Because `ps`/`pgrep` are denied, this is bounded to those PID files. The run dirs remain stale residue: no `mx1_arm_cap_n32.done` or `mx1_arm_veh_n32.done`, logs are 1850 B and 925 B, and no checkpoints/results were present in either `launch_arm_*/n32_metal` directory.

**Smallest cure:** add a structural launch preflight outside the per-experiment ticket:

1. Require Metal/MLX workload classification before `tools/launch_detached_process.py` can launch a training argv, or route such launches through a governed wrapper.
2. Refuse if another Metal/MLX training launch is active unless a composed measured-peak projection proves headroom under the 116 GiB host ceiling.
3. Require a passed load-phase peak receipt for the exact config before a real Metal fire.
4. Regenerate and retire the stale MX1 launch tickets before any further dispatch.

**Named review surface owed by Round 8:** `fire_schedule_composition_surface`. Future recursive review of a launch wave must audit both the payload script and the scheduling/composition artifact that fires it. RR8 names the surface; it does not build it.

## RR8-F4 - LOW - Twelfth-move adjudication mixes rounded d_seg-derived flips with receipt integer flips

**Finding:** `.omx/research/ddm_et4_20260806/TWELFTH_MOVE_ADJUDICATION.md:25-33` states the solver fixed "approximately 78,304" flips, then uses those "same flips" for an "exact" W closure and patch B/flip. The receipt aggregate records an integer `net_flip_reduction=78302` (`final_stage_repaired.log:16-18`), while the rounded evaluate components imply `78303.84539074564` flips. Those are different scopes.

Independent recomputation from `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/byteclose_archive_receipt.json` and `et4_solve_within_cvp_summary.json`:

- receipt score delta: `+7.449588773954394 S`; component sum matches.
- aggregate net flips: `78302`.
- rounded-d_seg flip estimate: `78303.84539074564`.
- W break-even using aggregate flips: `99686.9194769287 B`, not the rounded `99689 B`.
- patch price using aggregate flips: `144.1435978646778 B/flip`.
- nnz per aggregate flip: `122.7733391228832`.
- patch over break-even: `113.22179538923534x`.

This does not change the row verdict: ET4 is still rate-dominated and not a pointer move.

**Smallest cure:** amend the adjudication to separate receipt integer flips from rounded evaluate-component flips, and replace "exactly" with the explicit scope:

```text
Aggregate receipt: 78,302 net flips; rounded evaluate d_seg components imply 78,303.845 flips.
W break-even on aggregate flips: 99,686.9 B; patch price: 144.1436 B/flip.
```

## Clean checks / non-findings

- ET4 archive custody is internally consistent: receipt archive bytes `11645079`, archive sha256 `34f0b769b67cb56c6b1a2f61519eb9303f4a7785d4edd5ccc63b986dc10d314b`, patch bytes `11286732`, total nnz `9613398`, raw patch bytes `57684002` (`byteclose_archive_receipt.json:3-27`).
- ET4 final repaired run completed full population through the archive path: `inflate` completed and the recomputed score is `8.203046586569972`; evaluate display `8.2` is rounded. Axis remains `[macOS-CPU advisory]`, not contest authority.
- HB1 restart is correct under the written caveat. The caveat says crash before about epoch 30 gets a clean restart, and zero `resume from latest` hits means form-clean (`RESUME_CAVEAT.md:8-12`). The live driver log shows the first run reached epoch 12, then a fresh stage2 start at `2026-08-06T23:49:34Z` and epoch 0, with no `resume from latest` hits.
- No new exact score row was measured by RR8. Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
