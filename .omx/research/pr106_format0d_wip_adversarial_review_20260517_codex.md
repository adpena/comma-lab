# PR106 format0D WIP adversarial review

Date: 2026-05-17

Subject WIP:
`.omx/research/alien_tech_reverse_engineering_pr106_format0_family_20260517.md`

Related WIP read for signal and left unmodified:
`.omx/research/full_problem_space_reverse_engineering_cpu_gpu_both_20260517.md`

Status: `adversarial_review_of_partner_wip`

Authority:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

## Verdict

The WIP memo preserves a real mechanism signal, but its executive framing is
too strong for the evidence. Format0D is the current best **local
`[contest-CUDA T4]`** anchor, not the public leaderboard leader and not a
submission candidate. Its paired `[contest-CPU]` score is worse than the
current CPU anchor by `+0.03507460051721928`.

Do not edit or overwrite the partner WIP in place. Keep it as raw signal and
route corrections through this ledger unless the partner explicitly asks for a
rewrite.

## Evidence

Current scanner-derived anchors:

- best CPU: `0.1920513168811056` `[contest-CPU; GHA Linux x86_64 1:1]`,
  archive `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`.
- best CUDA: `0.20533002902019143` `[contest-CUDA T4]`, archive
  `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`.

Format0D paired eval evidence:

- CUDA: `0.20533002902019143` `[contest-CUDA]`, seg `0.00063042`,
  pose `0.00003188`, bytes `186876`.
- CPU: `0.22712591739832488` `[contest-CPU]`, seg `0.00062212`,
  pose `0.00016387`, bytes `186876`.
- Same archive and runtime tree, different inflated-output aggregate SHA.
- CUDA minus CPU score gap: `-0.021795888378133454`.
- Pose term accounts for `-0.022625888378133483`; seg term moves the opposite
  way by `+0.0008300000000000043`; rate is unchanged.

Format0D mechanism evidence:

- `src/tac/packet_compiler/pr106_sidecar_packet.py:1568` documents that
  format `0x0D` must be decoded as two correction passes.
- `src/tac/packet_compiler/pr106_sidecar_packet.py:1596` encodes base
  format0C payload + `extra_len(u16le)` + extra PR101 ranked/no-op payload +
  extra framing metadata.
- `src/tac/packet_compiler/pr106_sidecar_packet.py:4057` materializes
  format0D only from source format `0x0C`, computes same-dim / second-dim /
  into-base-noop diagnostics, and round-trips both passes.
- `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py:549` decodes
  the two streams, and `inflate.py:682` applies base corrections before extra
  corrections.
- Build metadata records `570` selected nonzero extra-pair corrections:
  `552` second-dimension pairs, `18` same-dimension out-of-format0C-vocabulary
  pairs, `0` into-base-noop pairs, `523` extra payload bytes, and `6` extra
  framing bytes.

## Findings

### F1 - Overclaim: "CURRENT LEADER"

Severity: high for control-plane routing.

The WIP states that format0D is the "CURRENT LEADER." Correct statement:
format0D is the current best **local `[contest-CUDA T4]`** anchor. The public
contest ranking axis is CPU, and format0D scored `0.22712591739832488`
`[contest-CPU]`, worse than the current best CPU anchor
`0.1920513168811056` by `+0.03507460051721928`.

Required wording for any landed derivative memo:

> Format0D is a local best `[contest-CUDA T4]` forensic/control anchor. It is
> not a `[contest-CPU]` frontier candidate and must not be submitted as-is.

### F2 - Overclaim: "dominates contest-CUDA leaderboard"

Severity: medium.

There is no public contest-CUDA leaderboard in the same sense as the public
CPU-ranked scoreboard. The evidence supports an internal CUDA-axis result:
format0D beats format0C on the same CUDA axis by
`-0.0009863575956184645`, and it beats an approximate PR101 CUDA replay value
around `0.22936` by about `-0.02403`. That is useful, but it must be described
as `[contest-CUDA T4]` replay/anchor evidence, not "leaderboard" evidence.

### F3 - Real mechanism: two-pass additive grammar is valid

Severity: positive signal.

The core mechanism is real: format0D is format0C exact-radix base plus a
second PR101 ranked/no-op correction stream. Runtime applies base then extra.
Build metadata shows the extra stream is not cosmetic: `570/600` pairs receive
extra nonzero corrections, and PacketIR records the extra payload and metadata
as score-affecting sections.

This signal should be harvested as a grammar/representation primitive, not as a
PR106 local-basin destination.

### F4 - The CPU/CUDA split is a primary phenomenon, not a footnote

Severity: high for next experiments.

The paired analysis proves same archive and runtime tree but different raw
outputs. The CPU/CUDA gap is dominated by PoseNet, not rate. The WIP correctly
mentions drift later, but any summary that says no CPU/GPU differential was
reported is stale relative to the paired drift artifact.

Do not classify the drift as "likely float accumulation order" without an xray.
The sidecar corrections are applied on CPU before `latents.to(device)` in the
runtime path; the raw-output divergence can still arise from decoder forward,
torch kernels, device precision, or batch behavior after transfer. The next
probe must localize which layer differs.

### F5 - Reproducibility is partial, not complete

Severity: medium.

The reusable format0D materializer exists:
`tools/materialize_pr106_latent_score_table_candidate.py`. The claim that
materialization exists only inside manifest JSON is stale. But full rebuild
still depends on the Kaggle-cached score table path and its custody fields.
Therefore the correct state is "locally rebuildable if cached score-table
custody is present," not "fully reproducible from source alone."

## Score-Lowering Implications

This review changes the next build/eval choices:

1. **Do not continue PR106-only local-basin polish as P0.** Format0D is useful
   as a donor primitive, but its CPU score is not near the submission target.
2. **Transplant the primitive, not the lane.** The score-lowering version is a
   Rule #6 byte-closed bolt-on on the verified A1/FEC6 CPU anchor: a small
   two-pass additive correction grammar with a CPU-aware objective and paired
   CPU/CUDA plan from byte zero.
3. **Before any new format0D retune, run xray.** Localize CPU/CUDA raw-output
   divergence at the layer level: sidecar decode/apply arrays, latent tensor
   after apply, decoder output before raw write, per-pair raw deltas, and
   scorer component deltas.
4. **Ablate the extra stream structurally.** Build zero-extra, shuffled-extra,
   sign-flipped-extra, same-dim-only, second-dim-only, and base-only controls
   against the same runtime. The useful donor is whichever submechanism moves
   score, not necessarily the whole 570-pair extra stream.
5. **Keep all reports axis-local.** Words like "leader", "frontier", "gold",
   "submission", and "beats" require an inline axis label and archive/runtime
   custody.

## Next Concrete Actions

P0:

- Build a tiny PR106-format0D xray probe that reads the byte-closed archive,
  emits base and extra correction arrays, emits latent tensor hashes after
  base-only and base+extra apply, and records same-dim/second-dim split counts.
  This should be scorer-free and local.

P1:

- Build a component ablation packet plan for format0D: base-only, zero-extra,
  shuffled-extra, sign-flipped-extra, same-dim-only, second-dim-only. The plan
  must be `score_claim=false` until paired exact auth eval artifacts exist.

P2:

- Design the Rule #6 transplant: A1/FEC6 two-pass additive micro-sidecar with
  CPU objective, archive grammar, no-op mutation proof, and paired CPU/CUDA
  dispatch plan. This is the score-lowering path; the PR106 family is only the
  donor primitive.

P3:

- If PR106 is revisited directly, build a CPU-biased score table and choose a
  Pareto row only if it improves CPU without destroying CUDA. A CUDA-only
  improvement is not sufficient for a submission candidate.

## State Disposition

- Partner WIP file left unmodified:
  `.omx/research/alien_tech_reverse_engineering_pr106_format0_family_20260517.md`.
- Related full-problem-space WIP left unmodified:
  `.omx/research/full_problem_space_reverse_engineering_cpu_gpu_both_20260517.md`.
  Its CPU/CUDA-axis and scorer-decomposition framing is compatible with this
  review's xray recommendation, but it is not treated as score or dispatch
  authority.
- This ledger is the current reviewed disposition for that WIP until it is
  superseded by a corrected version or by xray/ablation artifacts.
- PR106 format0D remains preserved as high-signal forensic/control evidence.
- Active score-lowering priority remains Rule #6 A1/FEC6 bolt-ons plus
  L5/TT5L side-info effect-curve execution, not PR106-only polish.
