# Master Gradient Raw-Byte Finite-Difference Adversarial Review - 2026-05-17

## Verdict

Composite **B + D**:

- **B:** the raw archive-byte gradient premise is cargo-culted for this packet
  class. A ZIP byte is not a smooth model coordinate; it may be a local header,
  central-directory field, CRC, compressed stream token, or codec grammar byte.
- **D:** the probe methodology must be changed. The useful object is not an
  `(N_archive_bytes, 3)` derivative tensor. It is an
  `(N_valid_mutation_operators, 3)` score-response matrix over packet-valid
  edits with inflate proof and axis labels.

This does **not** kill the master-gradient idea. It blocks the specific
raw-byte finite-difference dispatch route before it burns CPU/GPU time on a
mathematically ill-posed measurement.

## Trigger

Untracked partner WIP proposes a finite-difference master gradient by flipping
bits/bytes of the FEC6 archive and re-running inflate/evaluate:

- `.omx/research/cpu_frontier_master_gradient_campaign_plan_20260517.md`
- `.omx/research/grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md`

Those files remain unmodified partner WIP. This ledger is the current Codex
adversarial disposition for routing.

## Failure Mode

Raw byte finite differences violate the packet grammar:

1. Outer ZIP bytes include local headers, central directory records, sizes,
   filenames, flags, CRCs, and timestamps. Flipping one byte can make the
   container invalid or change metadata rather than decoded frames.
2. The inner payload is often entropy-coded or codec-structured. A one-bit flip
   is not a local semantic perturbation; it can corrupt a whole compressed
   stream or move decode state nonlocally.
3. Even if a mutated ZIP remains readable, the result is not a derivative of
   the model. It is a discontinuous accept/reject or grammar-corruption probe.
4. A failed inflate/eval cannot be interpreted as "this byte has high negative
   gradient." It may only prove that the mutation left the valid-packet
   manifold.

## Replacement Plan

Build a **score-response operator matrix**:

- rows are valid mutation operators, not raw bytes;
- each operator declares its section or grammar target;
- each operator emits a fresh archive with rebuilt ZIP metadata/CRC;
- each row proves `inflate.sh archive_dir output_dir file_list` succeeds;
- each row records `[contest-CPU]`, `[contest-CUDA]`, or paired-axis custody
  before it changes routing;
- each row is `score_claim=false` until exact eval result review lands.

Examples of valid row grains:

- section-conditioned entropy-coder choice for a known logical section;
- selector-table recoding with parser and byte-consumption proof;
- FEC6/A1 sidecar grammar replacement with no-op mutation detector;
- TT5L side-info variant cell with archive manifest and paired-axis harvest.

## Code Contract Landed

Added `tac.master_gradient_feasibility` and
`tools/audit_master_gradient_feasibility.py`.

The contract classifies:

- `raw_archive_byte` / `raw_archive_bit`: blocked with
  `blocked_raw_archive_gradient`;
- `zip_member_payload_byte` on entropy-coded streams: blocked with
  `blocked_entropy_stream_payload_gradient`;
- `grammar_aware_operator` with repack, ZIP header, CRC, inflate proof, and
  axis label: `operator_response_probe_ready`, but still
  `score_claim=false`, `promotion_eligible=false`,
  `ready_for_exact_eval_dispatch=false`.

## Routing Change

Do not dispatch the WIP raw-byte finite-difference master-gradient campaign as
written.

Proceed only after rewriting it as a valid mutation-operator response campaign.
The first local build should produce a small JSON manifest of operator rows and
run:

```bash
.venv/bin/python tools/audit_master_gradient_feasibility.py \
  --mutation-grain grammar_aware_operator \
  --axis-label paired_contest_cpu_cuda \
  --updates-zip-headers \
  --updates-crc \
  --repacks-archive \
  --proves-inflate-success
```

Authority:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`
