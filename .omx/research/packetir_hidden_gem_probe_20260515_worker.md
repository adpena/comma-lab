# PacketIR hidden-gem probe - 2026-05-15 worker

research_only=true
score_claim=false
dispatch_attempted=false

## Scope

Read-only PacketIR hidden-gem pass over current PR106/PR101 packet payloads and
profile artifacts. Output artifacts are confined to
`experiments/results/packetir_hidden_gem_probe_20260515_worker/`.

## Inputs

- PR106 HDM9 exact-CUDA ledger:
  `.omx/research/pr106_hdm9_packetir_format09_20260515_codex.md`
- PR106 HDM10/HLM3 format0A ledger:
  `.omx/research/pr106_hdm10_hlm3_packetir_format0a_20260515_codex.md`
- PR106 HDM8 format06/07/08 ledgers:
  `.omx/research/pr106_hdm8_implicit_len_format06_packetir_20260515_codex.md`,
  `.omx/research/pr106_hdm8_headerless_format07_packetir_20260515_codex.md`,
  `.omx/research/pr106_hdm8_inner_headerless_format08_packetir_20260515_codex.md`
- PR106 PacketIR profiles:
  `experiments/results/pr106_hdm9_packetir_recode_20260515_codex/profile.with_proofs.json`,
  `experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/profile.with_proofs.json`
- PR101 FEC6 profile:
  `experiments/results/pr101_fec6_byte_escape_profile_20260515_codex/profile.json`

## Finding

No next candidate with `>50` realistic archive-byte delta remains in the
current local PacketIR byte-preserving family.

- PR106 current exact `[contest-CUDA]` PacketIR best is HDM9 format09:
  `186352` bytes, SHA
  `09bcd867c2778d38d5ac04b648d44cb3bdfcfd3e3db402beb8886826cced50e9`,
  score `0.20633303308963796`.
- PR106 format0A is the best runtime-implemented lossless row found in the
  latest profile: `186349` bytes, `-3` bytes versus format09, full-frame
  same-runtime parity proven, but exact CUDA missing. This is below any useful
  batch threshold by itself.
- PR106 profile rows contain no runtime-implemented, identity-passing candidate
  with archive delta `<= -50` bytes; the other rows are either worse, invalid
  for the source payload, or lack a runtime decoder.
- PR101 FEC6 profile reports a realistic same-frame byte-saving bound of `16`
  bytes and an optimistic wrapper-hardcode-included bound of `24` bytes. The
  selector payload is `249` bytes with recomputed entropy floor `241` bytes,
  leaving only `8` bytes in that surface.
- The PR101 FEC6 `[contest-CPU]` byte-only gap to `<0.192` is `78` bytes, and
  the paired `[contest-CUDA]` score is `0.22621002169349796`; CPU evidence does
  not promote to CUDA.
- Inspected PR101 and PR106 archives are single-member stored ZIPs named `x`
  with `100` bytes of ZIP overhead, so there is no compliant ZIP/header
  reservoir large enough to supply a `>50` byte candidate.

## Artifacts

- Probe script:
  `experiments/results/packetir_hidden_gem_probe_20260515_worker/probe_packetir_hidden_gem.py`
- Structured summary:
  `experiments/results/packetir_hidden_gem_probe_20260515_worker/packetir_hidden_gem_probe_summary.json`
- Markdown summary:
  `experiments/results/packetir_hidden_gem_probe_20260515_worker/packetir_hidden_gem_probe_summary.md`
- Commands run:
  `experiments/results/packetir_hidden_gem_probe_20260515_worker/commands_run.md`

## Dispatch decision

Do not dispatch another exact eval batch for current PacketIR byte-only
shavings. Format0A can be bundled opportunistically if another larger
component-changing packet is dispatched, but not alone.

Next step: abandon this local PacketIR basin and move to component-changing
CUDA-in-loop selector/waterfill work or a new trained substrate.
