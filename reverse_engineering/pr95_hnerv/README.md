# PR95 HNeRV Family

PR95-family public submissions use a single stored ZIP member named `0.bin`.
The member payload is three length-prefixed Brotli streams:

1. metadata JSON;
2. decoder-weight payload;
3. uint8 latent rows encoded as low/high zigzag delta streams.

The reusable wire grammar is canonicalized in `src/tac/pr95_hnerv.py`.
Planning-only latent residual tools live in `src/tac/pr95_residual_atoms.py`.
The operator entry point is the thin wrapper:

```bash
python experiments/build_pr95_hnerv_residual_atom_plan.py \
  --archive experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/archive.pr95_repacked.zip \
  --exact-json <exact-cuda-adjudicated-json> \
  --output-dir experiments/results/pr95_hnerv_residual_atoms_<stamp>
```

The planner refuses sidecars, duplicate latent targets, no-op atoms, archive
SHA mismatches, member SHA mismatches, and component-trace mismatches. Any
candidate archive it emits remains `candidate_archive_requires_exact_cuda_eval`
until scored through exact CUDA auth eval.

## Known Custody Gap

The historical PR95 exact JSON paths referenced in the paper/research ledgers
may be absent from a fresh local tree if Lightning harvest artifacts were not
mirrored. Do not fake a baseline from paper text. Re-harvest or regenerate the
exact JSON before building a score-bearing residual-atom candidate.

## Related Files

- `src/tac/pr95_hnerv.py`
- `src/tac/pr95_residual_atoms.py`
- `src/tac/tests/test_pr95_residual_atoms.py`
- `experiments/profile_pr95_hnerv_muon_packing.py`
- `experiments/build_pr95_hnerv_residual_atom_plan.py`
- `.omx/research/public_pr95_pr96_no_dispatch_intake_20260504_codex.md`
- `.omx/research/public_hnerv_adapter_replays_20260504_codex.md`
