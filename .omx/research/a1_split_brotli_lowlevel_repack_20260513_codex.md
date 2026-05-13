# A1 Split-Brotli Low-Level Repack Candidate

Date: 2026-05-13
Author: codex
Scope: byte-closed A1/PR101-style split-brotli HNeRV low-level repack

## Summary

The HNeRV low-level repacker now handles the A1/PR101-style payload grammar:
`uint32 section_total + seven concatenated brotli decoder streams + latent/sidecar tail`.
The packer treats the seven decoder streams as one logical repackable section,
recompresses each stream independently, and preserves the A1 framing and tail
bytes.

This produced a local byte-smaller candidate:

- source archive: `submissions/a1/archive.zip`
- source SHA-256: `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- source bytes: `178262`
- candidate archive: `experiments/results/a1_lowlevel_repack_wide_20260513_codex/a1_hnerv_brotli_repack_candidate.zip`
- candidate SHA-256: `b8a9d60adda1c96e51c14dfc6307c962733c593eb3e2bb31a5156031844f06d9`
- candidate bytes: `178260`
- byte delta: `-2`
- formula byte-term delta if scorer components are unchanged:
  `-0.0000013317179062443427`

This is a tiny rate-only byte win, not a representation win. It is useful as a
PacketIR / low-level-packer correctness milestone and as proof that A1 still
has at least microscopic deterministic byte slack in the split-brotli decoder
section.

## Proofs Produced

Artifact directory:
`experiments/results/a1_lowlevel_repack_wide_20260513_codex/`

- `result.json`: candidate build manifest, `score_claim=false`,
  `ready_for_exact_eval_dispatch=false`, `ready_for_archive_preflight=true`.
- `inflate_shell_output_parity_local_cpu.json`: generic full-runtime
  `inflate.sh archive_dir output_dir file_list` parity probe. Source and
  candidate both returned `0`, emitted `0.raw`, and produced identical output:
  `3662409600` bytes, SHA-256
  `e63942793f963fa1e0f1ab195f9819519d8f63c067f9959cb5efc7879a4ef386`.

The parity probe also fixed a reusable harness bug: the previous
`--python-bin` shim only intercepted `python`, while A1's runtime uses
`${PYTHON:-python3}`. The tool now shims both `python` and `python3`, with a
regression test.

## Byte Anatomy

The accepted candidate changed only the A1 decoder split-brotli section:

- section: `decoder_packed_brotli`
- source bytes: `162164`
- candidate bytes: `162162`
- delta: `-2`
- brotli params: quality `11`, lgwin `16`, lgblock `18`
- raw decoder bytes: `229014`
- raw decoder SHA-256 before/after:
  `83598024bdb4d60463610db23934cdee60c3b6a81158a97e0dd55ea621833fcd`
- raw equality: true

The A1 latent/sidecar tail was not modified.

## Evidence Boundary

Known A1 source anchors from the axis-validation ledger:

- A1 source [contest-CPU Linux x86_64]: `0.19284757743677347`
- A1 source [contest-CUDA]: `0.2263520234784395`

Because source and candidate have full same-runtime output parity, the only
expected movement on either axis is the archive byte term:

- derived A1 candidate [contest-CPU] if exact auth eval confirms identical
  components: `0.19284624571886723`
- derived A1 candidate [contest-CUDA] if exact auth eval confirms identical
  components: `0.22635069176053326`

These derived values are not promoted as exact-auth-eval score claims in this
ledger. Exact candidate auth eval is deferred because the isolated delta is
only two bytes; it should be bundled with a larger A1 exact-evaluable packet
unless an operator explicitly wants a standalone micro-dispatch.

## Classification

- Legitimate byte movement: yes.
- Payload no-op: no; the decoder compressed bytes changed.
- Raw decoder equivalence: yes.
- Full-frame same-runtime local parity: yes.
- Exact contest score claim: no.
- Promotion/submission readiness: no; candidate exact auth eval and final
  compliance are not run.

## Follow-Up

1. Keep the A1 split-brotli grammar in `tac.hnerv_lowlevel_packer`; it is now
   reusable for A1/PR101-family low-level byte passes.
2. Use this candidate as a bundle component if a larger A1 byte/runtime
   candidate is produced.
3. Do not dispatch this two-byte candidate alone unless the objective is
   formal exact-eval closure rather than score-per-dollar efficiency.
4. Preserve the executable-bit fix on `submissions/a1/inflate.sh`; the generic
   parity harness and contest release checks require the source runtime shell
   entrypoint to be executable.
