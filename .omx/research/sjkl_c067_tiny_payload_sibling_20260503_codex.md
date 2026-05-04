# SJ-KL C067 tiny-payload sibling candidate - 2026-05-03

## Local archive candidate

Built a deterministic local SJ-KL tiny-payload candidate from the existing
C067 frontier archive and the existing 250-byte shrink payload.

- Source archive:
  `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`
- Source archive bytes: `276214`
- Source archive SHA-256:
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- SJ-KL payload:
  `experiments/results/sjkl_c067_shrink_q6_minrpk1_20260502T_worker/repack/sjkl.bin`
- SJ-KL payload bytes: `250`
- SJ-KL payload SHA-256:
  `13f605bfd9ad950807d410c8371f20fd2b1c3d9c04bb59cc6bf07d474dcc78bb`
- Candidate archive:
  `experiments/results/sjkl_c067_tiny_payload_sibling_20260503_codex/pack/archive.zip`
- Candidate manifest:
  `experiments/results/sjkl_c067_tiny_payload_sibling_20260503_codex/pack/sjkl_c067_archive_manifest.json`
- Candidate archive bytes: `276556`
- Candidate archive SHA-256:
  `a576960be12fdcec1cc76257d5a49cd4102476c7e461150847c845ba0cceab6d`
- Delta vs C067: `+342` bytes
- Formula-only rate delta vs C067: `0.0002277237619677826`
- `score_claim=false`
- `promotion_eligible=false`

The archive layout is `top_level_sibling`: preserve the C067 `p` member bytes
exactly and add charged top-level `sjkl.bin`. Runtime logical members are
`renderer.bin`, `masks.mkv`, `optimized_poses.bin`, and `sjkl.bin` after
`unpack_renderer_payload.py` processes `p`.

## Compliance boundary

This is not a score claim. Promotion still requires exact CUDA auth eval through
`archive.zip -> inflate.sh -> upstream/evaluate.py`, preferably via
`experiments/contest_auth_eval.py --device cuda`.

The candidate does not use hidden sidecars. The basis and coefficients are
inside charged `sjkl.bin`; the source payload `p` is preserved byte-for-byte.
The builder manifest records `sidecars_required=false` and
`score_affecting_payload_charged_in_archive=true`.

No remote job was dispatched.

## Local commands

```bash
.venv/bin/python experiments/build_sjkl_c067_archive.py \
  --source-archive experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip \
  --sjkl-bin experiments/results/sjkl_c067_shrink_q6_minrpk1_20260502T_worker/repack/sjkl.bin \
  --output-dir experiments/results/sjkl_c067_tiny_payload_sibling_20260503_codex/pack \
  --archive-layout top_level_sibling \
  --sjkl-zip-compression stored \
  --max-sjkl-bytes 512 \
  --force
```
