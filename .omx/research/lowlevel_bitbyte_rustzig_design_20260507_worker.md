# Low-Level Bit/Byte Rust/Zig Lane Design - 2026-05-07

Worker: LowLevel-BitByte-RustZig
Repo: `/Users/adpena/Projects/pact`
Branch rule: `main` only

Scope: design a deterministic native systems lane for archive/container parsing,
packing, bitstream writing, entropy coding, and payload deconstruction. This
ledger is a research/control-plane artifact. It does not claim a score, does not
dispatch GPU or remote work, and does not make native code required by the
contest inflate path.

## Intake

Required starting checks were run before writing this ledger:

- `git status --short --branch`: workspace is on `main`, ahead of origin, with
  existing unrelated dirty and untracked work. No unrelated file was reverted or
  edited.
- Repo grep/listing for Rust/Zig/C/ASM/binary and archive/entropy tooling:
  `runtime-rs`, `*.rs`, `*.cpp`, `*.c`, `*.S`, `*.asm`, `*.zig`,
  `build.zig`, `zipfile`, `brotli`, `zstd`, `range`, `ANS`, `Huffman`,
  `arithmetic`, `local header`, `central`.
- Toolchain probe: `rustc 1.94.1`, `cargo 1.94.1`, `zig 0.16.0`.
- There is a `runtime-rs/Cargo.lock`, but no root `rust-toolchain.toml`,
  `rust-toolchain`, or root `.cargo/config.toml`.

## Sources Consulted

Local repo surfaces:

- `runtime-rs/Cargo.toml`
- `runtime-rs/crates/qma-codec/src/lib.rs`
- `runtime-rs/crates/stbm1br-codec/src/lib.rs`
- `runtime-rs/crates/inflate-cli/src/main.rs`
- `submissions/robust_current/range_mask_codec.cpp`
- `experiments/qma9_range_mask_cpp_profiler.cpp`
- `src/tac/hnerv_lowlevel_packer.py`
- `src/tac/bit_level_archive_optimizer.py`
- `src/tac/pr103_arithmetic_codec.py`
- `src/tac/lossless/range_coder.py`
- `src/tac/submission_archive.py`
- `scripts/pre_submission_compliance_check.py`
- `experiments/repack_single_member_archive.py`
- `src/tac/tests/test_hnerv_lowlevel_packer.py`
- `src/tac/tests/test_repack_single_member_archive.py`
- `src/tac/tests/test_submission_archive_safety.py`
- `.omx/research/pr103_pr106_ac_repack_exact_eval_20260507_codex.md`
- `.omx/research/hnerv_hdc2_hdm3_entropy_packet_20260507_codex.md`
- `.omx/research/hnerv_lowlevel_brotli_expanded_saturation_20260507_codex.md`
- `.omx/research/grand_council_pr106_substrate_findings_zig_default_20260507.md`
- `.omx/research/council_lane_bit_level_archive_design_20260430.md`
- `.omx/research/pr85_qma9_rust_bridge_lowering_20260504_codex.md`

Primary or implementation-authority online sources:

- PKWARE APPNOTE 6.3.9 ZIP File Format Specification:
  https://www.pkware.com/documents/casestudies/APPNOTE.TXT
- RFC 7932, Brotli Compressed Data Format:
  https://www.rfc-editor.org/rfc/rfc7932
- RFC 8878, Zstandard Compression:
  https://www.rfc-editor.org/rfc/rfc8878.html
- Zstandard reference format document:
  https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md
- Cargo profiles:
  https://doc.rust-lang.org/cargo/reference/profiles.html
- Cargo vendor:
  https://doc.rust-lang.org/cargo/commands/cargo-vendor.html
- Cargo source replacement:
  https://doc.rust-lang.org/cargo/reference/source-replacement.html
- rustc codegen options:
  https://doc.rust-lang.org/rustc/codegen-options/index.html
- Rust `std::arch` SIMD intrinsics:
  https://doc.rust-lang.org/std/arch/index.html
- Zig overview and cross-compilation notes:
  https://ziglang.org/learn/overview/
- `constriction` Rust entropy-coding docs:
  https://docs.rs/constriction/latest/constriction/index.html
- Duda, "Asymmetric numeral systems":
  https://arxiv.org/abs/0902.0271
- FiniteStateEntropy implementation repository:
  https://github.com/Cyan4973/FiniteStateEntropy

## Existing Native Surface

Rust already exists in this repo and should be extended before introducing a
new native build island:

- `runtime-rs` is a Cargo workspace with `qma-codec`, `stbm1br-codec`,
  `raw-writer`, `residual-codec`, `python-ast-indexer`, and an `inflate-cli`
  placeholder.
- `qma-codec` is a deterministic QMA9 parser/decoder with fail-closed errors,
  fixed arithmetic decode state, and Python fixture tests.
- `stbm1br-codec` parses and decodes STBM1BR-style payloads, uses Rust `brotli`,
  `bzip2`, and `flate2` with the Rust backend, and has internal arithmetic
  decode logic.
- `inflate-cli` is currently a placeholder, so it is not yet a reliable
  operator entrypoint for ZIP/container inspection.

C++ already exists in contest-facing and forensic paths:

- `submissions/robust_current/range_mask_codec.cpp` is an active contest runtime
  decoder for QMA9-style range-mask payloads.
- `experiments/qma9_range_mask_cpp_profiler.cpp` profiles the same family and
  emits JSON with bit/context accounting.

No Zig source or `build.zig` file was found in the repo. Zig is installed
locally, but adding Zig now would create a second native build system before the
Rust workspace has a byte-exact archive primitive.

## Current Byte Evidence Anchors

The low-level lane must route from measured archive bytes, not language
preference:

- Current strict HNeRV envelope repack floor in local custody is the
  PR103-on-PR106 arithmetic decoder repack at `185578` bytes with exact CUDA
  evidence in `.omx/research/pr103_pr106_ac_repack_exact_eval_20260507_codex.md`.
- Expanded HNeRV Brotli parameter search on PR106x is effectively saturated for
  generic recode work: one byte was found, candidate dispatch was false, and
  stricter archive/eval custody was still required.
- HDC2 mixed-context replacement is diagnostic for now: `221381` stream bytes
  versus `170127` current decoder section bytes. It is not a direct frontier
  replacement.
- HDM3 fixed-schema closure is small but real: `170113` stream bytes, `-14`
  bytes versus the current decoder section, raw equality true, no score claim.
- Older bit-level archive design already found that outer ZIP metadata is small
  compared with payload mass. ZIP work is still valuable for custody,
  deterministic writing, and small header/member-name savings, but not as a
  standalone Shannon-floor lane.

## Recommended Architecture

Build a Rust-first native toolkit inside `runtime-rs`, with Python as the
orchestrator until exact parity and strict compliance are proven.

Proposed crates and boundaries:

1. `runtime-rs/crates/zipwire`
   - Byte-exact ZIP reader and deterministic stored-member writer.
   - Reads EOCD, central directory, local file headers, data descriptors, ZIP64
     records when explicitly enabled, and raw member data without relying on
     Python `zipfile` normalization.
   - Fails closed on duplicate names, empty names, local/central name mismatch,
     zip-slip paths, hidden/resource names, unexpected extra fields, encrypted
     members, unsupported compression methods, data descriptor ambiguity, and
     CRC/size mismatch.
   - Writes deterministic single-member ZIPs with fixed timestamps, fixed
     permissions, no comments, no extras unless explicitly requested, no ZIP64
     for small files, stored payload by default, and stable member ordering.
   - Emits JSON manifests so Python tools can compare without linking native
     libraries into runtime code.

2. `runtime-rs/crates/bitbyte-core`
   - Fixed-endian integer helpers: u24/u32/u64 little-endian, varints only when
     schema-owned, and explicit bit order.
   - Bit writer/reader with golden vectors shared against
     `src/tac/lossless/range_coder.py` and Brotli/RFC-style packing examples.
   - No floating point in codec state. No host-endian transmute. No unchecked
     alignment assumptions.

3. `runtime-rs/crates/entropy-lab`
   - Static range coder and optional ANS/FSE experiments behind feature flags.
   - Canonical histogram normalization and table serialization with fixed
     endian, fixed tie-breaks, fixed overflow behavior, and explicit table-size
     accounting.
   - Initial mode should generate candidate streams and manifests only. It
     should not replace the PR103 Python/constriction runtime until a byte
     equivalence or exact archive closure exists.

4. `runtime-rs/crates/hnerv-scan`
   - HNeRV payload section scanner for `0xff + len24 + brotli decoder + brotli
     latents` payloads.
   - Computes raw byte histograms, section offsets, section SHA-256, entropy
     lower-bound estimates, brotli/zstd/range/ANS break-even tables, and section
     diff summaries for PR100-107 archives.
   - Optional SIMD accelerators only for scans and histograms, never for default
     contest decode. SIMD output must match scalar output byte-for-byte across
     macOS/Linux and x86_64/aarch64.

5. CLI layer
   - `cargo run --locked -p zipwire -- inspect archive.zip --json-out ...`
   - `cargo run --locked -p zipwire -- rewrite-single in.zip out.zip --member-name x`
   - `cargo run --locked -p hnerv-scan -- inspect archive.zip --json-out ...`
   - `cargo run --locked -p entropy-lab -- compare stream.bin --json-out ...`

Python integration rule: keep native commands opt-in until the JSON output has
golden parity tests against the existing Python validators. Native code should
not become part of `inflate.sh` or a score-lowering Python path without a
separate promotion review.

## Exact First Prototype

Implement only `runtime-rs/crates/zipwire` first.

Prototype commands:

```text
zipwire inspect <archive.zip> --json-out <manifest.json>
zipwire rewrite-single <input.zip> <output.zip> --member-name x --method stored
```

The first prototype replaces, in opt-in audit mode only, these Python hot paths:

- `src/tac/hnerv_lowlevel_packer.py::read_strict_single_member_zip`
- `src/tac/hnerv_lowlevel_packer.py::write_stored_single_member_zip`
- `experiments/repack_single_member_archive.py`
- The local/central header read performed by
  `scripts/pre_submission_compliance_check.py::_local_header_name`

It should not replace `src/tac/submission_archive.py` or
`scripts/pre_submission_compliance_check.py` by default. The first success
criterion is parity JSON, not automatic runtime adoption.

Minimum focused tests:

- Synthetic valid single-member stored ZIP round-trips to identical payload and
  stable output SHA-256 across two writes.
- Local header name differs from central directory name: fail closed.
- Duplicate central-directory member names: fail closed.
- Empty, absolute, parent-traversal, backslash, hidden, and resource-fork member
  names: fail closed.
- CRC mismatch and size mismatch: fail closed.
- Encrypted member or unsupported compression method: fail closed.
- ZIP64 is rejected unless explicitly enabled in the command/schema.
- Golden parity against Python `read_strict_single_member_zip` and
  `inspect_archive` on synthetic safe archives.
- Optional bridge test in Python can be gated on `PACT_ZIPWIRE_BIN`; absence of
  the binary must skip, not fail, the Python suite.

First real archive trial after synthetic tests:

1. Read current HNeRV PR106/PR106x archive with `zipwire inspect`.
2. Rebuild the same payload as member `x` with deterministic stored ZIP.
3. Compare payload SHA-256, archive bytes, central/local name parity, and Python
   validator output.
4. Record the result in a dated ledger. Do not dispatch exact eval from this
   worker lane.

## Entropy Coder Plan

Do not start by rewriting PR103 arithmetic coding in Rust. The current PR103
arithmetic repack has exact CUDA evidence and is a measured byte floor. A native
coder that changes stream bytes must first justify its header/table overhead
against `185578` charged archive bytes and must preserve raw decode equality.

Native entropy work should proceed in this order:

1. `entropy-lab hist`: deterministic byte/nibble/q8 histograms for HNeRV decoder
   and latent sections, with fixed table-size overhead estimates.
2. `entropy-lab compare`: compute closed-form and actual encoded sizes for
   static range, ANS/FSE, canonical Huffman, and generic Brotli/Zstd baselines.
3. `entropy-lab pack-static-range`: produce a self-describing candidate stream
   with fixed-endian table serialization, explicit raw length, encoded length,
   CRC/SHA, and bit tail length.
4. Python-only or native decode parity tests against raw tensors/sections.
5. Only after a candidate is smaller than the current exact floor and has a
   self-contained decoder should the lane move toward submission runtime.

Coder constraints:

- Range/ANS/Huffman models must account for their table bytes before claiming a
  win.
- Histogram normalization must be deterministic with stable tie-breaks and no
  floating point.
- All integer math must define overflow behavior and use checked or saturating
  paths in debug tests.
- Bitstream tail handling must include exact remaining bit count or a canonical
  terminator. Padding bits must be validated.
- Decoder must fail closed on impossible symbols, truncated streams, trailing
  data, bad CRC, and bad table normalization.

## SIMD And Assembly Plan

SIMD is a scanner/profiler accelerator first, not a decoder dependency.

Allowed first SIMD targets:

- Byte histograms for large HNeRV sections.
- Search for magic/length-prefix patterns and section offsets.
- CRC-like chunk checks if a portable scalar reference remains authoritative.
- Entropy-gap scanning over PR100-107 payloads.

Default implementation must be scalar. Optional SIMD should be gated by Cargo
features and runtime CPU detection. `std::arch` is architecture-specific, so
each accelerated path needs scalar parity tests on every supported target.

Assembly should not enter the default build. If used at all, it should be a
microbench-only experiment with identical scalar output and no contest inflate
dependency.

## Rust vs Zig vs C/ASM

Rust first:

- Fits the existing `runtime-rs` workspace.
- Strong failure typing and memory safety for hostile public archives.
- Good CLI/test tooling and Cargo.lock-based dependency custody.
- Can keep default CPU baseline generic and add feature-gated SIMD later.
- Best first target for ZIP/container and entropy-scanner code.

Zig second:

- Strong cross-compilation story and useful as a C toolchain.
- Potentially attractive for a tiny standalone inflate helper or reproducible C
  build harness.
- Adds a new build system and dependency-custody surface. Do not start here
  while `runtime-rs` lacks the ZIP primitive.

C/C++ only where already present:

- Existing QMA9 C++ runtime and profiler are valuable reference code.
- New C/C++ increases memory-safety and sanitizer burden for public archive
  parsing. Prefer Rust for new parsers.

ASM only as optional microbench:

- Not acceptable as a default contest inflate dependency without a scalar
  reference, per-target tests, and a deterministic dispatch rule.

## Deterministic Build Plan

Immediate prototype build:

```text
cd runtime-rs
cargo test --locked -p zipwire
cargo build --locked --release -p zipwire
```

Promotion build, once the prototype matters outside research:

- Add a local `rust-toolchain.toml` for `runtime-rs` or repo root only after
  review, pinning the exact stable Rust toolchain used for release builds.
- Keep `runtime-rs/Cargo.lock` authoritative and run `cargo vendor` for any
  promoted release surface that must build without network.
- Use `--locked` in all CI/operator commands and `--frozen` for release rebuild
  proofs when vendored sources are present.
- Record `rustc -vV`, `cargo -V`, target triple, `Cargo.lock` SHA-256, source
  tree SHA, binary SHA-256, and dynamic-link inspection (`ldd` on Linux,
  `otool -L` on macOS).
- Release profile should be evaluated with:
  `lto = "fat"` or `"thin"`, `codegen-units = 1`, `panic = "abort"`,
  `strip = "symbols"`, and baseline `target-cpu=generic`.
- Do not use `target-cpu=native` for portable release artifacts.
- Static Linux builds should be evaluated separately for T4 compatibility and
  binary size. `+crt-static` must be explicit and recorded if used.
- Zig release builds, if introduced, must pin the Zig version, target triple,
  optimization mode, libc choice, and binary SHA. Use Zig as a C toolchain only
  when it improves reproducibility over the existing build path.

## Risk Register

1. Python `zipfile` is convenient but not a byte-level archive authority. It can
   hide local/central mismatches and normalize details that matter for strict
   custody. Mitigation: native `zipwire inspect` parses raw headers and produces
   JSON parity reports.
2. Outer ZIP savings are small. Mitigation: treat ZIP work as compliance,
   custody, and deterministic rewrite infrastructure; route byte-mass effort to
   HNeRV decoder/latent entropy.
3. Native binary in contest inflate path increases custody risk. Mitigation:
   keep first native tools opt-in and off inflate path; require reproducible
   build manifests before runtime adoption.
4. Brotlipy, Rust `brotli`, and reference encoders may not emit identical bytes.
   Mitigation: compare raw decompressed equality and charged output bytes; never
   assume encoder-byte parity across implementations.
5. ANS/FSE/range table overhead can erase gains on small streams. Mitigation:
   model table bytes explicitly and compare against current exact archive bytes,
   not isolated entropy estimates.
6. SIMD can introduce CPU-feature drift. Mitigation: scalar is authoritative;
   SIMD is feature-gated and tested for byte-identical output.
7. Endianness and bit order are common failure classes. Mitigation: fixed-endian
   helpers, golden bitstream vectors, and cross-target CI before promotion.
8. Crate supply chain and licensing can block public release. Mitigation: prefer
   no-dependency core parsers, audit licenses, vendor dependencies only after a
   prototype earns promotion.
9. ZIP64/data descriptor behavior can create parser ambiguity. Mitigation:
   fail closed by default; only support ZIP64 or descriptors through explicit
   schema choices and tests.
10. PR103 exact evidence can be accidentally diluted by speculative rewrites.
    Mitigation: every native entropy result compares against `185578` bytes and
    remains non-promotable until exact archive closure and compliance pass.

## Integration Targets

Near-term:

- `src/tac/hnerv_lowlevel_packer.py` single-member read/write parity.
- `experiments/repack_single_member_archive.py` archive-surgery audit parity.
- `scripts/pre_submission_compliance_check.py` local/central header parity
  cross-check.
- `src/tac/tests/test_hnerv_lowlevel_packer.py` and
  `src/tac/tests/test_repack_single_member_archive.py` as Python bridge
  reference tests.

Medium-term:

- HNeRV PR101/PR103/PR106 archive surgery manifests.
- `src/tac/hnerv_hdm3_archive_candidate.py` and HDC2/HDM3 entropy manifests.
- Public PR100-107 payload deconstruction scorecard regeneration.
- `tools/all_lanes_preflight.py` or strict compliance checks, only after
  native JSON parity is stable and optional.

Out of scope for first prototype:

- Any active Python score-lowering file edit.
- Any default contest inflate path change.
- Any remote or GPU dispatch.
- Any exact score claim.

## Concrete Next Step

Create `runtime-rs/crates/zipwire` with an `inspect` command and deterministic
stored single-member `rewrite-single` command, plus focused Rust tests for
header parity, unsafe names, duplicate names, CRC/size mismatch, deterministic
output, and unsupported ZIP features. Then add an optional Python bridge test
gated on `PACT_ZIPWIRE_BIN` to compare native JSON with the existing Python
validators on synthetic archives.
