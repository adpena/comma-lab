# Current Focus - 2026-05-17 (L5 v2 + Rule #6 Rebaseline)

## Frontier

- Canonical scanner-derived best CPU anchor:
  `0.1920513168811056`
  `[contest-CPU; GHA Linux x86_64 1:1]`, archive
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`,
  lane `lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515`.
- Canonical scanner-derived best CUDA anchor:
  `0.20533002902019143`
  `[contest-CUDA T4]`, archive
  `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`,
  lane `lane_pr106_format0d_latent_score_table_20260516_contest_cuda`.
- A1 remains the Rule #6 control substrate, not the best current axis floor:
  `0.19284757743677347` `[contest-CPU; GHA Linux x86_64 1:1]` and
  `0.2263520234784395` `[contest-CUDA T4]`.
- A1 archive bytes/SHA-256:
  `178262` /
  `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`.
- Public medal band remains the immediate score target:
  PR101/PR102/PR103 around `0.193`/`0.195`/`0.195`, axis-specific and
  external until exact replay custody proves otherwise.

## Active Strategic Rebaseline

The May 17 T4 symposium supersedes the stale May 15 queue framing without
retiring the L5/L5-v2 staircase:

1. **Immediate frontier-breaking path**: Rule #6 bolt-ons on verified A1.
   Build small, byte-closed, PR101-style additions on the working A1 substrate
   before spending another wave on high-risk substrate-class guesses.
2. **L5 v2 / TT5L priority remains active**: TT5L side-info effect curve,
   L5-v2 probe gates, and architecture-lock custody remain the primary
   asymptotic campaign and must keep moving in parallel.
3. **High-risk substrate cluster**: 35-substrate per-pair-conditioning cluster
   is deferred pending SCORER-AWARENESS probes, not killed.
4. **Original Z6 FiLM path**: do not dispatch as-is; replace with
   per-frame-renderer-axis ego-motion conditioning.
5. **PR106/HNeRV local-basin work**: useful as forensic control and byte
   lessons only; do not let it crowd out Rule #6 or L5-v2 actuation.

## Current L5 v2 / TT5L State

- TT5L paired diagnostic exact eval is terminal and non-promotional:
  `3.8987840060549908` `[contest-CPU]` and
  `3.9007398365396795` `[contest-CUDA]` for archive
  `2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a`.
- Architecture lock remains forbidden. Current blocker class:
  missing complete L5-v2 gate evidence, missing C1/Z5/TT5L probe gate evidence,
  and missing paired CPU/CUDA side-info effect curve harvest.
- TT5L side-info Lightning paired-axis plan has 10 cells and dry-run custody,
  but non-dry-run provider execution still requires Lightning doctor, per-cell
  source manifests, active lane claims, exact harvest, terminal claim rows, and
  architecture-lock packet refresh.
- TT5L side-info Modal paired dispatch plan now consumes the shared exact
  dispatch authority gate. The live five-variant plan is intentionally blocked
  (`ready_work_unit_count=0`) until the submission runtime has `report.txt` and
  each variant archive has a matching `archive_manifest.json`. Ledger:
  `.omx/research/l5_v2_tt5l_exact_dispatch_authority_hardening_20260517_codex.md`.
- Lightning required-doctor plan exists at
  `.omx/research/l5_v2_tt5l_lightning_required_doctor_plan_20260517_codex.md`;
  it is planning-only and confers no dispatch or score authority.

## Active P0 Work

1. **Rule #6 A1 bolt-on #1**: Ballé-2018 hyperprior on A1 per-pair latent,
   with KL-on-logits `T=2.0` distillation from frozen A1 teacher.
   Existing Z3HV2 direct-residual export is not this implementation: it is now
   classified as a byte-negative direct-residual control with no active Ballé
   entropy residual decoder. See
   `.omx/research/rule6_z3v2_direct_residual_unwind_20260517_codex.md`.
2. **Rule #6 A1 bolt-on #2**: PR101-style per-tensor byte map plus
   Brotli/LZMA/Huffman sidecar on A1 weights/latents.
   Current A1 byte-escape profiler is saturated under the existing runtime:
   raw-LZMA latent sweep best equals source at `15387` bytes, current
   607-byte sidecar has only a 4-byte oracle entropy gap but no smaller
   runtime-supported representation for current semantics, and no candidate
   archive was emitted. Ledger:
   `.omx/research/a1_rule6_byte_escape_profile_20260517_codex.md`.
3. **Rule #6 A1 bolt-on #3**: VQ-codebook on A1 per-pair latent.
4. **TT5L side-info effect curve**: first materialize the missing `report.txt`
   and per-variant archive manifests flagged by exact-dispatch authority, then
   run Lightning doctor, then claim and execute the 10 paired CPU/CUDA cells
   only if doctor and source-manifest custody are green.
5. **SCORER-AWARENESS probe wave**: measure whether substrate distinguishing
   features reach scorer attention/argmax maps before deferring high-risk
   per-pair-conditioning substrates.
6. **Z6 replacement design**: per-frame-renderer-axis ego-motion variant,
   not FiLM-bottlenecked Z6.

## Dispatch Discipline

- No provider dispatch without `tools/claim_lane_dispatch.py claim`.
- No CPU/CUDA promotion without axis-labeled paired custody.
- No architecture lock until the shared authority predicate allows it.
- No score claim from planning, dry-run, macOS, proxy, or diagnostic anchors.
- Every result review must preserve failure class, custody, recomputed formula,
  and reactivation criteria.

## Parent-Scope OMX Markdown Scan

On 2026-05-17, the Markdown scan was widened from `.omx/research` to all
`.omx/**/*.md`, then repeated with `--hidden --no-ignore` so ignored
`.omx/auto_memory_snapshot_*` and `.omx/tmp` Markdown were not silently
excluded. Relevant non-research control surfaces checked:

- `.omx/state/current_focus.md` - refreshed by this file.
- `.omx/state/next_experiments.md` - refreshed alongside this file.
- `.omx/state/active_lane_dispatch_claims.md` - current source for dispatch
  conflict/terminal status.
- `.omx/auto_memory_snapshot_20260504T230223Z/*.md` - ignored historical
  Claude/OMX memory snapshot; no current L5/TT5L authority, but it preserves
  no-signal-loss, stack-order, entropy-coder, and remote-tarball lessons.
- `.omx/tmp/*.md` - ignored temporary appendices and detached clone READMEs;
  useful as forensic inputs only, not current score authority.
- `.omx/notepad.md` - stale April AV1/Track-B notebook, not current L5
  authority.
- `.omx/release_manifest_v0.2.0-rc1.md` - release hygiene context, not current
  score authority.
- `.omx/state/dispatch_queue.md` - historical HTD queue; not the May 17
  Rule #6/L5-v2 priority list.

Detailed scan ledger:
`.omx/research/l5_v2_omx_parent_markdown_scope_refresh_20260517_codex.md`.
No-ignore follow-up ledger:
`.omx/research/l5_v2_omx_parent_markdown_no_ignore_refresh_20260517_codex.md`.

## Required Refresh Cadence

- Refresh this file after any Rule #6 dispatch result, TT5L side-info harvest,
  L5-v2 architecture-lock packet change, or public frontier intake that changes
  the score target.
- Refresh `.omx/state/next_experiments.md` whenever the active P0 work order
  changes.
- Catalog #316 now checks this file, `reports/latest.md`, and
  `.omx/state/next_experiments.md` against `tac.frontier_scan` so stale
  frontier citations fail preflight instead of becoming hidden control-plane
  signal loss.
