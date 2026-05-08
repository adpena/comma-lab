# PR106 Entropy Floor Replication - Worker C - 2026-05-07

Owner: Worker C
Evidence grade: empirical planning / adversarial validation
Score claim: false
Dispatch attempted: false
Ready for exact-eval dispatch: false

## Scope

Replicate the PR101 entropy-floor style analysis on the current PR106/frontier
HNeRV substrate, without modifying PR101 tools and without inventing CLI
surface. The objective was to determine whether the claim "encoder side is
bounded at about 178KB without ML" is PR101-only or transferable to PR106.

## Real PR106 Decode Path

Public PR106 `belt_and_suspenders` archive:

- Archive: `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`
- Exact CUDA replay artifact: `experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z/contest_auth_eval.adjudicated.json`
- Archive bytes: `186239`
- Archive SHA-256: `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58`
- ZIP member: `0.bin`
- Member bytes: `186131`
- Member SHA-256: `7f2cc905b7611ae8d7bced72be24e2266b0aa341f90cfeccbb0854fd8fc01eb7`
- Payload grammar: `0xff | decoder_len:u24_le | decoder_packed_brotli | latents_and_sidecar_brotli`
- Runtime decode path: public `inflate.py -> parse_archive() -> parse_packed_archive()`
- Decoder section bytes: `170278`
- Decoder section SHA-256: `654999f81f0552fb7568e6977e73aa329661c10c79a6ab6cddc3171302352004`
- Decoder raw bytes: `229070`
- Decoder raw SHA-256: `f22eb6be56499fa5785f47f85d2bef7f71246f29674691fd3e06af733c8c0703`
- Fixed latent section bytes: `15849`
- Fixed latent section SHA-256: `94257b33cf3083c5daa0f3b1e127cb7c51bee42a6416b19763eea7bf9ecc3c32`
- Fixed latent raw bytes: `33712`
- Fixed latent raw SHA-256: `a38778c6304bacba39705cd9c45af337d73bc90c6b7b4ccf2563febfc312328e`

Current PR106x low-level Brotli repack substrate:

- Archive: `experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex/pr106x_hnerv_brotli_repack_candidate.zip`
- Exact CUDA replay artifact: `experiments/results/lightning_batch/exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506/contest_auth_eval.adjudicated.json`
- Archive bytes: `186080`
- Archive SHA-256: `b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`
- ZIP member: `x`
- Member bytes: `185980`
- Member SHA-256: `b6ba493aa37446143b003235eaeb3a49c0748a6d64392fb9a666a6c872629171`
- Payload grammar: same PR106 `0xff` packed HNeRV envelope
- Decoder section bytes: `170127`
- Decoder section SHA-256: `07725c39ff436195e319f258b1e033290de30e259bc3f103b1b487f21a698c5c`
- Decoder raw SHA-256: `f22eb6be56499fa5785f47f85d2bef7f71246f29674691fd3e06af733c8c0703`
- Fixed latent section bytes: `15849`
- Fixed latent section SHA-256: `94257b33cf3083c5daa0f3b1e127cb7c51bee42a6416b19763eea7bf9ecc3c32`

Active exact A++ rate boundary:

- Archive: `experiments/results/pr103_repack_pr106_standalone_20260507/exact_eval_static_release_surface/archive.zip`
- Exact CUDA replay artifact: `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json`
- Runtime: `submissions/pr103_pr106_final_runtime/inflate.py`
- Archive bytes: `185578`
- Archive SHA-256: `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- Decode path: `0xff` PR106 envelope, but the decoder section is PR103 arithmetic-coded closure, not legacy PR106 Brotli.

This active floor is the comparison boundary for rate-only PR106 work. It is
not decoded by the new legacy-Brotli floor probe.

## New Tool And Artifacts

Added `tools/pr106_entropy_floor_probe.py`.

The tool consumes either:

- `--archive` strict single-member PR106/frontier ZIP;
- `--payload-bin` raw `0xff` packed HNeRV payload;
- `--state-dict` decoded state_dict proxy input.

For archive/payload inputs it parses the charged substrate directly:

- `tac.hnerv_lowlevel_packer.parse_ff_packed_brotli_hnerv`
- `tac.hnerv_decoder_recode.parse_packed_decoder_brotli`
- fixed PR106 latent raw layout: `lo | fp16 mins | fp16 scales | hi`

It emits IID, Markov-1, Markov-2, and fixed-transform oracle floors for:

- decoder q-zz symbols plus f32 scales;
- fixed latent delta-zz bytes plus fp16 metadata;
- combined decoded payload sections without the fixed `0xff` header.

It also records unpriced Markov context counts/edges so low floors cannot be
mistaken for a real coder.

Generated artifacts:

- `experiments/results/pr106_entropy_floor_probe_20260507_worker_c/public_pr106_entropy_floor_probe.json`
- `experiments/results/pr106_entropy_floor_probe_20260507_worker_c/public_pr106_entropy_floor_probe.md`
- `experiments/results/pr106_entropy_floor_probe_20260507_worker_c/pr106x_entropy_floor_probe.json`
- `experiments/results/pr106_entropy_floor_probe_20260507_worker_c/pr106x_entropy_floor_probe.md`

Focused tests:

- `src/tac/tests/test_pr106_entropy_floor_probe.py`

## Probe Results

Public PR106:

| group | current bytes | identity IID floor | identity Markov-1 floor | identity Markov-2 floor | best transform | best Markov-2 floor | unpriced Markov-2 contexts / edges |
|---|---:|---:|---:|---:|---|---:|---:|
| decoder q-zz + f32 scales | 170278 | 167635 | 156485 | 89000 | delta_mod | 45660 | 105895 / 221578 |
| fixed latents + fp16 meta | 15849 | 15562 | 12710 | 1527 | delta_mod | 849 | 14030 / 16897 |
| decoded payload without header | 186127 | 183196 | 169194 | 90527 | delta_mod | 46508 | 119925 / 238475 |

PR106x low-level Brotli:

| group | current bytes | identity IID floor | identity Markov-1 floor | identity Markov-2 floor | best transform | best Markov-2 floor | unpriced Markov-2 contexts / edges |
|---|---:|---:|---:|---:|---|---:|---:|
| decoder q-zz + f32 scales | 170127 | 167635 | 156485 | 89000 | delta_mod | 45660 | 105895 / 221578 |
| fixed latents + fp16 meta | 15849 | 15562 | 12710 | 1527 | delta_mod | 849 | 14030 / 16897 |
| decoded payload without header | 185976 | 183196 | 169194 | 90527 | delta_mod | 46508 | 119925 / 238475 |

Interpretation:

- The identity IID floor reproduces the expected PR106-specific entropy gap: a
  zero-order model can see some byte headroom versus current Brotli, but that
  does not pay histogram/model overhead, stream headers, runtime code, ZIP
  overhead, or exact replay risk.
- The Markov and transform floors are oracle bounds only. The best delta/Markov
  rows are dominated by huge unpriced context models, e.g. `119925` Markov-2
  contexts and `238475` edges for the combined payload. They are useful for
  targeting context families, not for claiming a 46KB packet is feasible.
- Existing materialized evidence is stricter: the A++ PR103-on-PR106 archive is
  `185578` bytes, and the HDC2/HDM3 tranche recorded no below-active-floor
  candidate in `experiments/results/hnerv_hdm3_entropy_packet_20260507_codex/hdc2_hdm3_active_floor_blocker_classification.json`.

## Adversarial Transfer Verdict

The claim "encoder side is bounded at about 178KB without ML" is PR101-only at
the current evidence level. It does not transfer as a PR106/frontier bound.

Reasons:

- Public PR106 exact archive is `186239` bytes, which is `+7981` bytes above
  the PR101 `178258` exact reference.
- PR106x low-level Brotli exact archive is `186080` bytes, `+7822` above the
  PR101 reference.
- The active PR106 rate-only A++ floor is PR103-on-PR106 at `185578` bytes,
  still `+7320` above the PR101 reference.
- The only PR106 floor numbers far below `178KB` are oracle bounds with
  unpriced context tables and no emitted bitstream/runtime.

Therefore, PR101 entropy conclusions can motivate PR106 search directions, but
they cannot bound PR106 encoder-side bytes without a PR106-specific coder,
charged metadata accounting, runtime adapter, packet manifest, and exact CUDA
auth eval.

## Next Exact Substrate Work

Do not dispatch from the oracle floors. The next implementable PR106 entropy
work should start from the active `185578` byte boundary and produce one of:

1. A PR103-on-PR106 AC section parser/floor probe that consumes the closure in
   `submissions/pr103_pr106_final_runtime/inflate.py` or
   `src/tac/pr103_pr106_runtime_closure.py`, so the active floor itself can be
   anatomized.
2. A byte-accounted model-table compressor for the delta/Markov context rows,
   with explicit table bytes, stream bytes, raw equality, payload SHA diff, and
   runtime decode path.
3. A fixed-latent coder that first beats the existing `15849` byte latent
   section after all metadata and runtime costs, then stacks on the PR103-on-
   PR106 packet rather than the older PR106 Brotli packet.

## Verification

Commands run:

```text
.venv/bin/python -m pytest src/tac/tests/test_pr106_entropy_floor_probe.py -q
UV_CACHE_DIR=/tmp/uv-codex-pr106-entropy uv run --no-project --with pytest --with numpy --with brotli python -m pytest src/tac/tests/test_pr106_entropy_floor_probe.py -q
UV_CACHE_DIR=/tmp/uv-codex-pr106-entropy uv run --no-project --with ruff --with numpy --with brotli ruff check tools/pr106_entropy_floor_probe.py src/tac/tests/test_pr106_entropy_floor_probe.py
UV_CACHE_DIR=/tmp/uv-codex-pr106-entropy uv run --no-project --with numpy --with brotli python tools/pr106_entropy_floor_probe.py --archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip --json-out experiments/results/pr106_entropy_floor_probe_20260507_worker_c/public_pr106_entropy_floor_probe.json --md-out experiments/results/pr106_entropy_floor_probe_20260507_worker_c/public_pr106_entropy_floor_probe.md --pr101-reference-archive-bytes 178258 --active-floor-archive-bytes 185578 --active-floor-label pr103_on_pr106_a++
UV_CACHE_DIR=/tmp/uv-codex-pr106-entropy uv run --no-project --with numpy --with brotli python tools/pr106_entropy_floor_probe.py --archive experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex/pr106x_hnerv_brotli_repack_candidate.zip --json-out experiments/results/pr106_entropy_floor_probe_20260507_worker_c/pr106x_entropy_floor_probe.json --md-out experiments/results/pr106_entropy_floor_probe_20260507_worker_c/pr106x_entropy_floor_probe.md --pr101-reference-archive-bytes 178258 --active-floor-archive-bytes 185578 --active-floor-label pr103_on_pr106_a++
```

The `.venv` pytest command failed before running tests because this checkout's
`.venv/bin/python` has no `pytest` module. The `uv run --no-project` focused
test command passed: `4 passed, 1 warning`. The focused Ruff command passed:
`All checks passed!`.
