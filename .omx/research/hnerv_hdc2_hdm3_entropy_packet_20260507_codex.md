# HNeRV HDC2/HDM3 Entropy Packet Classification (2026-05-07)

## Scope

This ledger preserves the HDM3/HDC2 entropy work product under
`experiments/results/hnerv_hdm3_entropy_packet_20260507_codex/` and classifies
it for frontier planning.

No contest score is claimed. No dispatch was attempted.

## Findings

HDC2 mixed-context recode is not a direct frontier replacement yet:

- HDC2 replacement stream bytes: `221381`
- Current frontier decoder section bytes: `170127`
- Net byte delta now: `+51254` bytes versus the frontier section
- Dispatch status: `ready_for_exact_eval_dispatch=false`

The bounded HDM3 fixed-schema stream is the actionable exact-byte closure:

- Candidate variant: `hdm3_q_brotli_split_fixed_schema_q_stream_plus_raw_scales`
- Candidate stream bytes: `170113`
- Candidate stream SHA-256:
  `149a41ecf4d1614757c9369838e3a7cb9f03a648fe9b61a2317e9b7f2996b256`
- Net byte delta versus frontier decoder section: `-14` bytes
- Raw equality closed: `true`
- Roundtrip valid: `true`
- Score claim: `false`

The combined entropy target remains planning-only:

- Model overhead target: `40840` bytes
- Payload entropy-gap target: `23979` bytes
- Projected combined target bytes: `156562`
- Projected rate-only score delta if made byte-equivalent:
  `-0.009032376699102255`
- Required before dispatch: actual runtime implementation, candidate archive
  manifest, runtime-tree parity manifest, strict pre-submission compliance,
  lane dispatch claim, exact CUDA auth eval.

## Interpretation

HNeRV remains the current exact substrate to exploit, but this does not make it
the final representation. The correct near-term use of this packet is:

1. Treat HDC2 as a negative/diagnostic entropy design until it becomes
   byte-closed and smaller than the current section.
2. Treat HDM3 as the smallest exact-byte archive-closure target already tied to
   a runtime adapter proof.
3. Feed the `-14` byte HDM3 atom into the meta-Lagrangian/Pareto planner as a
   rate-equivalent HNeRV recode only after archive manifest and runtime-tree
   parity are attached.
4. Keep exploring substitutive representations via alpha/beta/gamma/dezeta
   now that `CodecPipeline` can express substrate transforms mechanically.

## Tracked Artifacts

Small structured manifests are tracked. Binary streams remain local/generated
work products and are not promoted in git.

- `candidate_packet.json`
- `entropy_overhead_audit.json`
- `profile.json`
- `hdc2_combined_entropy_reduction_manifest.json`
- `hdc2_combined_entropy_reduction_manifest.md`
- `hdc2_stream_work_product/*.json`
