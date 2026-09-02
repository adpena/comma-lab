# semantic_joint_ctxmix

Task-aware compression of the challenge video to 180,002 bytes at 0.14797617125559104
on contest CUDA (recomputed from the evaluator's printed components; the report displays
0.15 at two decimals).

## Credits and lineage

The learned vehicle descends from public contest work, and I do not claim it:

- **PR #130** — the semantic-token renderer family: a dense per-pair semantic token
  grid decoded by a small learned renderer, integer HPAC arithmetic coding of the
  token stream, and the pose-carrier format.
- **PR #135** — the CPR1-polished form of that vehicle (0.162 on contest-CUDA), whose
  receiver structure this submission's runtime carries forward.
- **PR #133** — transitively in the ancestry via the above.

My contribution is the decision and lossless-representation layer on top:

1. **Joint edit admission** — per-pair segmentation-token edits admitted only when the
   measured joint delta (seg + pose + rate together) improves the score; 455 of 573
   solved pairs admitted.
2. **Pose-carrier re-solve** — Gauss–Newton re-fit of the stored pose coefficients
   against PoseNet on the rendered frames, after re-orienting the coefficient basis.
3. **Five lossless coder/container stages** — integer log-odds context mixing, group-
   and tile-conditioned token re-encodes, a joint re-encode collecting banked wins.
   Decoded output is byte-identical across all five.

## Reproduce

- **Inflate:** `./inflate.sh archive_dir output_dir file_list` — requires
  `linux-nvidia-t4` (CPU inflation exceeds the 30-minute budget, measured).
- **Compress:** `python compress.py` replays the five lossless stages from the pinned
  base archive (all inputs SHA-256-pinned inside the script) and refuses unless two
  complete runs reproduce the exact 180,002 bytes.

## Verify

- Archive SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`,
  180,002 bytes.
- Native/Python receiver identity: 600/600 pairs, 3,662,409,600 raw output bytes,
  0 differing.
- Full research receipts: <https://github.com/adpena/comma-lab> (evaluated commit
  `1c9fbbf5`).

Corrections welcome — if any attribution is incomplete, tell me and I will fix it.
