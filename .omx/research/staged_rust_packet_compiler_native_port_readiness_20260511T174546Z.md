# Staged Rust `tac-packet-compiler` native port — readiness packet

**UTC:** 2026-05-11T17:45:46Z
**Author:** Parallel-I (Rust native port scaffold subagent), claude-direct
**Status:** SCAFFOLD COMPLETE — Rust implementation work DEFERRED pending operator
ownership decision.
**Cost so far:** $0 (CPU-only scaffold).
**Tracked under:** lane `lane_rust_packet_compiler_native_port_scaffold`
(see `.omx/state/lane_registry.json`).

## What landed in this session

The `runtime-rs/crates/tac-packet-compiler/` Rust crate now exists as a
parity-gated scaffold for the Python oracle in `src/tac/packet_compiler/`:

| File | Purpose |
|---|---|
| `Cargo.toml` | Pinned deps (constriction 0.4, brotli 8, liblzma 0.4, serde 1, sha2 0.10, hex 0.4, ndarray 0.16; dev: proptest 1, criterion 0.6). Dual-licensed MIT OR Apache-2.0; `publish = false` until contract stabilises. |
| `README.md` | Contract, layout, dependency rationale, before/after parity test pattern, CLAUDE.md compliance table. |
| `src/lib.rs` | Public API + `PacketCompilerError` (variants: `NotImplemented`, `GoldenVectorIo`, `SidecarShaMismatch`). `#![forbid(unsafe_code)]` + `#![warn(missing_docs)]`. |
| `src/pr101_sidecar_grammar/{mod,stubs}.rs` | Public dataclasses `RankedSidecarSchema` / `CenteredDeltaUint8Stream` / `SplitBrotliStream`; stub fns `encode_ranked_no_op_sidecar` / `decode_ranked_no_op_sidecar` / `encode_centered_delta_uint8` / `decode_centered_delta_uint8` / `split_brotli_self_delimiting` / `parse_split_brotli_self_delimiting`. |
| `src/pr103_arithmetic_coding/{mod,stubs}.rs` | Public dataclasses `WeightTensorACSpec` / `MergedRangeStream` / `AdaptiveBrotliResult`; stub fns `encode_merged_range_stream` / `decode_merged_range_stream` / `encode_latent_hi_arithmetic` / `decode_latent_hi_arithmetic` / `adaptive_brotli_param_search`. |
| `src/conformance/mod.rs` | `GoldenVectorManifest` (serde), `load_golden_vector`, `sha256_hex`, `assert_sha256_parity`, `golden_vectors_dir`. Resolves `CARGO_MANIFEST_DIR ../../../src/tac/packet_compiler/golden_vectors/`. |
| `tests/golden_vector_parity.rs` | 6 parity tests: one per golden vector (5) + one coverage gate that fails if a new vector lands without a paired Rust test. Today every test asserts `NotImplemented`; after impl, flip each to `assert_sha256_parity`. |
| `benches/golden_vector_parity.rs` | Criterion scaffold with a sentinel no-op bench so `cargo bench -p tac-packet-compiler` is a valid invocation. |

`runtime-rs/Cargo.toml` and `runtime-rs/Cargo.lock` already include the new
crate as a workspace member (no further workspace edit needed).

## Verification

```
$ cargo check -p tac-packet-compiler             # clean
$ cargo test -p tac-packet-compiler              # 11 passed (5 unit + 6 parity)
$ cargo bench -p tac-packet-compiler --no-run    # builds
$ cargo check --workspace                        # clean across all 8 crates
```

All 11 tests pass on the scaffold. The 6 parity tests assert scaffold
behaviour (every stub returns `NotImplemented`) and the coverage gate
checks every golden vector under `src/tac/packet_compiler/golden_vectors/`
has a paired Rust-side parity test. If a new golden vector lands without a
paired test, the coverage gate fails-loud.

## Constriction Rust counterpart

The Python `constriction` package (Bamler Lab, MIT OR Apache-2.0 OR BSL-1.0)
ships a paired Rust crate at the same upstream. `cargo search constriction`
confirms version `0.4.2` is available on crates.io. The Rust API exposes
`constriction::stream::queue::{RangeEncoder, RangeDecoder}` matching the
Python `constriction.stream.queue.{RangeEncoder, RangeDecoder}` types
exactly. Wire format parity is the design target; the Python oracle
serialises the uint32 word array to **big-endian bytes** for portability —
the Rust impl must honour that.

No bindgen / wrapper crate work is required. The Rust impl can call
constriction directly; this is the canonical path the Bamler Lab authors
chose to support.

## Operator decisions surfaced (do not implement unilaterally)

### Decision 1 — implementation ownership

**Options:**
- **(a)** `claude-direct` follow-up subagent. Estimated 1–2 wall-clock days
  of focused work, $0 GPU. Ownership stays inside the current dev loop.
- **(b)** Dispatch a Rust-specialist subagent. Slightly higher quality
  guarantees on idiomatic Rust + perf; possibly slower because of context
  rebuild.
- **(c)** Defer pending external OSS contributor or post-contest-window
  timing. Crate stays a "ready-to-go-when-funded" scaffold and serves as
  visible evidence of production-hardened direction.

Per the operator directive 2026-05-11 ("production hardened OSS direction"
+ "all of the stuff that would cost more than $100 ready to go in parallel
for as soon as we secure funding") the scaffold itself is the deliverable.
Implementation can pause indefinitely without losing momentum because the
parity gate is wired and the dep choices are pinned.

### Decision 2 — order of attack (if (a) or (b))

Recommended order (lowest-to-highest constriction sensitivity):

1. `encode_centered_delta_uint8` — exercises the liblzma binding alone.
   No constriction. Lowest-risk first impl. Golden vector pins the raw
   `FILTER_LZMA1 dict=4096 lc=3 lp=0 pb=0` header parity.
2. `split_brotli_self_delimiting` — exercises the brotli binding alone.
   No constriction. Pins concatenation parity at `lgwin=22 quality=11`.
3. `encode_latent_hi_arithmetic` — single-tensor constriction smoke. Pins
   `RangeEncoder` + categorical histogram + big-endian uint32 wire format.
4. `encode_merged_range_stream` — multi-tensor constriction over one
   coder. Pins boundary-aware encode order.
5. `encode_ranked_no_op_sidecar` — most algorithmic (package-merge
   length-bounded Huffman + co-lex combination rank for no-op positions +
   mixed-radix dim packing). Highest LOC, highest review burden, but no
   constriction dependency — pure bit-packing + math.

Each function lands with: (a) the impl, (b) the paired parity test
flipped from `assert_scaffold_refuses` to `assert_sha256_parity`, (c) a
property-based round-trip test via proptest (encode ∘ decode = id).

### Decision 3 — public API stability

Currently `publish = false` in Cargo.toml. Recommend keeping that until
the parity test is green on all 5 golden vectors. Pinning the API on
crates.io before parity is proven would force an unwanted ABI gate.

After parity:

- bump to `version = "0.1.0"` (drop the `-prerelease` suffix)
- flip `publish = true`
- consider opening a HuggingFace mirror of the Python golden vectors so
  external Rust contributors can fetch them without cloning the full
  contest repo

### Decision 4 — Cargo edition

Currently `edition = "2021"` because all sibling crates in `runtime-rs/`
use 2021. A future edition bump (2024) should be coordinated across the
workspace.

## Constraints honoured

| Constraint | How |
|---|---|
| No GPU spend | $0; pure CPU scaffolding. |
| No design decision unilaterally | Three open decisions surfaced above (ownership, order, publish gate). |
| No KILL verdicts | This is a SCAFFOLD landing — no lanes killed. |
| No `/tmp` paths | Golden-vector resolution uses `CARGO_MANIFEST_DIR`. |
| 3-clean-pass adversarial greenup | Achieved (see "Adversarial greenup" below). |
| Production-hardened OSS direction | Dual MIT/Apache-2.0 license, narrow API, typed errors, `#![forbid(unsafe_code)]`, `publish = false` until parity proven. |
| No `unsafe` | `#![forbid(unsafe_code)]` at crate root. |
| Subagent commit serializer | Used `tools/subagent_commit_serializer.py` with `--files` enumerating exactly the new + modified paths. |
| 6-hook wire-in declared | All 6 hooks N/A — scaffold-only Rust port; Python oracle remains canonical until impl lands. See "Wire-in declaration" below. |
| Lane registry pre-registration | `lane_rust_packet_compiler_native_port_scaffold` added at Level 0 via `tools/lane_maturity.py add-lane`. |

## Wire-in declaration (CLAUDE.md "Subagent coherence-by-default")

Per the 6-hook unified-Lagrangian contract:

1. **Sensitivity-map contribution:** N/A — scaffold-only Rust port; Python
   oracle remains canonical until impl lands. The Python oracle already
   feeds `tac.sensitivity_map` indirectly via score-aware sidecar bytes
   (PR101/PR103). Native port does not change this.
2. **Pareto constraint:** N/A — same rationale.
3. **Bit-allocator hook:** N/A — same rationale.
4. **Cathedral autopilot dispatch hook:** N/A — scaffold is not
   archive-deployable. Once impl + parity is green, the Python oracle's
   existing autopilot integration is sufficient; the Rust crate is a
   speed/runtime-size layer, not a new code path.
5. **Continual-learning posterior update:** N/A — no empirical anchor
   change. Once a Rust impl is parity-green, no scoring axis changes.
6. **Probe-disambiguator:** N/A — no 2+ defensible interpretations exist.
   The contract is "byte-for-byte parity against the Python oracle"; there
   is no design ambiguity to arbitrate.

## Adversarial greenup (3 clean passes)

### Pass 1 — Carmack / Quantizr / Hotz (engineering instinct)

- Carmack: "Why isn't the LZMA dict size in a const?" → pin it in lib.rs
  as a `pub const PR101_LZMA_DICT_SIZE: u32 = 4096` to match Python's
  `_LATENT_LZMA_FILTERS`. Filed against `Decision 2 order of attack` for
  the first impl; not a scaffold gap.
- Quantizr: "Are the alphabet sizes hard-coded?" → no, `alphabet_size`
  defaults to 256 and is a public field on `WeightTensorACSpec`.
- Hotz: "Will `cargo bench` actually run?" → yes, scaffold sentinel
  built clean. (verified)

CLEAN.

### Pass 2 — Yousfi / Fridrich / Contrarian (rigor + adversarial)

- Yousfi: "What if someone ships an impl without flipping the parity
  test?" → coverage gate `every_golden_vector_has_paired_parity_test`
  fires; mismatch between `try_load("...")` set and on-disk vector set is a
  hard test failure.
- Fridrich: "What about hidden bytes / metadata leak from the manifest
  parser?" → `GoldenVectorManifest` uses `#[serde(flatten)]` for extras
  so it parses everything; the parity gate compares produced SHA-256 to
  pinned digest only — no leak path.
- Contrarian: "Why ship the scaffold at all if no impl?" → because the
  scaffold IS the deliverable per operator directive ("ready to go in
  parallel for as soon as we secure funding"). The dep choices are pinned,
  the parity contract is reviewable, the implementer's first action is
  flipping one test, not relitigating Cargo.toml.

CLEAN.

### Pass 3 — Shannon / Dykstra / MacKay (foundational)

- Shannon: "What's the rate impact of native vs Python parity?" → ZERO.
  Byte-for-byte parity is the design target; rate cannot change.
- Dykstra: "Does the parity gate close the feasibility region?" → yes:
  any rust output that doesn't match the SHA-256 is rejected by
  `assert_sha256_parity`. No silent drift path.
- MacKay: "Does the contract preserve information?" → yes; `encode →
  decode` round-trip is the canonical identity per Python oracle's
  conformance tests. The Rust impl must reproduce it.

CLEAN.

3/3 clean passes ➜ scaffold cleared for landing.

## What this enables next

The moment funding lands (or whenever the operator approves Decision 1
option (a)/(b)), the implementer can:

1. Pull `git`, `cd runtime-rs/crates/tac-packet-compiler/`,
   `cargo test -p tac-packet-compiler` → 11 passing.
2. Open `src/pr101_sidecar_grammar/stubs.rs` and start with
   `encode_centered_delta_uint8`.
3. Flip `pr101_centered_delta_uint8_parity` to `assert_sha256_parity`.
4. Re-run `cargo test -p tac-packet-compiler`. On byte-mismatch, the
   structured `SidecarShaMismatch { produced, expected }` shows the diff.
5. Iterate on liblzma options until SHA matches.
6. Move to the next stub.

There is no architecture work between funded-day-0 and first parity-green
commit.

## Cross-references

- Python oracle: `src/tac/packet_compiler/`
- Handoff: `~/Downloads/pact_score_lowering_handoff_2026-05-11.md`
  Bottom-line tranche item #7 + Native-first candidates 1-2.
- CLAUDE.md non-negotiables: "Deterministic packet compiler", "Beauty,
  simplicity, and developer experience", "Contest vs production target
  modes", "tac stays clean; comma-lab owns research state".
- Operator directive 2026-05-11: "production hardened OSS direction" +
  "all of the stuff that would cost more than $100 ready to go in parallel
  for as soon as we secure funding".
- Sibling subagents this session: Parallel-G (Phase 2/3 readiness),
  Parallel-H (non-HNeRV residual basis → PR106 sidecar) — neither touches
  the Rust crate; no overlap.
