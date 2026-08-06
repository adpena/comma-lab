# ddm_q43a next if resumed

Current state: blocked before scorer load by receiver grammar mismatch.

## Resume From Here

Use the exact parent:

```text
/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes
sha256 b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06
bytes 357837
```

Do not use `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/tq1c_base/archive.zip`;
it is the prior `75df9cc3...` archive despite the matching byte count.

## Fire Order

1. Inspect the tq1c receiver used by the preserved aggregate:
   `ddm_tq1_qo1_inflate_runner_dcd4406aaa83.Decoder`, recorded in
   `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/stage_checkpoints/n600_scorer/move_0023_snap_r00_c12_L13/aggregate.json`.
2. Build a tq1c-specific QA43 adapter only if it can consume `0.bin` through
   the real receiver and apply a real pair-local correction in the shipped
   decode path. Do not map the archive into the v4d six-member grammar.
3. Run scorer-free validation first:

```bash
PYTHONPATH="$PWD:$PWD/src:$PWD/experiments" \
/Volumes/VertigoDataTier/pact/uv-envs/pact-main/bin/python \
experiments/ddm_su2_qa43_tail_solver.py validate \
  --program-kind warp-tail \
  --receiver-adapter <tq1c_adapter_module>:<factory> \
  --adapter-arg parent_archive=/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes
```

4. If validation passes, run solve on SSD with an absolute resume dir and a
   wall cap until baseline/profile is complete:

```bash
PYTHONPATH="$PWD:$PWD/src:$PWD/experiments" \
/Volumes/VertigoDataTier/pact/uv-envs/pact-main/bin/python \
experiments/ddm_su2_qa43_tail_solver.py solve \
  --program-kind warp-tail \
  --receiver-adapter <tq1c_adapter_module>:<factory> \
  --adapter-arg parent_archive=/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes \
  --top-k 56,112,200 \
  --resume-from /Volumes/VertigoDataTier/pact/ddm_q43a_20260806/tq1c_qa43_tail_solve_b35e756829 \
  --relinearizations 3 \
  --damping 0.001 \
  --coefficient-limit 7 \
  --max-seconds 1800 \
  --min-free-bytes 1073741824
```

5. If the tq1c adapter cannot be made real, route the q43a slot to wp1 /
   pose-in-training. Verdict scope should stay `INSTANCE`, not `FAMILY`.

## Required First Receipt After Resume

The next receipt must lead with:

- top-56/top-112/top-200 pose concentration on the tq1c parent;
- k=56 realized row if solved;
- whole-action B/admitted-pair;
- falsifier status;
- explicit d_seg collateral from the canonical decode chain.
