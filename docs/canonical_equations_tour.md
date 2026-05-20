# Canonical equations tour

The canonical equations registry is a small append-only ledger of empirically-calibrated equations that codify hard-earned anchors from the substrate-design loop as first-class artifacts. It exists to extinct tribal knowledge: when a session learns something quantitative about the system, the result is codified in the registry rather than left in chat logs and individual file docstrings.

Six equations populate the initial registry. Each one is grounded in a first-principles reference and backed by at least one empirical anchor on contest-1:1 hardware.

Related: [`docs/asymptotic_floor_candidate_inventory.md`](asymptotic_floor_candidate_inventory.md) Section E.1. Sister library: [`adpena/tac`](https://github.com/adpena/tac). Submission packet: [`commaai/comma_video_compression_challenge#110`](https://github.com/commaai/comma_video_compression_challenge/pull/110).

---

## What it is

The registry lives at `.omx/state/canonical_equations_registry.jsonl` (fcntl-locked append-only JSONL per the canonical state-discipline pattern). Each entry carries:

- `equation_id` (canonical name).
- `latex_form` (the formal equation).
- `domain_of_validity` (which archive families / codec families / measurement axes the equation applies to).
- `canonical_producers` (helpers that emit residuals against this equation).
- `canonical_consumers` (downstream code that ingests the equation's predictions).
- `empirical_anchors[]` (each anchor carries the input/output pair, the predicted-vs-empirical residual, and the canonical Provenance sub-object per the project's data-discipline contract).

The discipline is anti-tribal-knowledge: when a session empirically anchors a relationship (Brotli stream count vs differing-stream upper bound; per-byte leverage distribution on PR101 archives; MPS-vs-CUDA drift on a specific architecture class), the result is registered with first-principles citation rather than left as a docstring observation in one file.

## The six initial equations

### 1. Brotli decompression cascade is bounded per stream

`cascade_range(b, A) = |stream_i| where b ∈ stream_i`

A single-byte mutation to a Brotli-encoded archive affects at most one decoded stream's worth of post-decompress bytes. Anchored on PR101's OP7 diff (decoder blob = 7 streams, single-byte mutation differs in 1 stream of ~229014 bytes).

First-principles basis: Brotli (RFC 7932) is a stream-concatenated container; each stream decompresses independently. A mutation outside any stream's start/length headers is bounded by that stream's decompressed footprint.

### 2. MPS-vs-CUDA drift is architecture-class dependent

`drift(class) = empirical_per_class_constant` (NOT a single universal multiplier)

Local Apple Silicon MPS forward passes vs Linux x86_64 CUDA forward passes drift by an architecture-class-dependent factor. Anchored on PoseNet (FastViT-T12) at ~23x and SegNet (EfficientNet-B2 UNet) at ~2x. A naive "MPS is 10x off" rule is empirically wrong; the multiplier varies by 10x across architectures.

First-principles basis: forward-pass numerical drift accumulates across non-deterministic kernels (bicubic interpolation, fused-multiply-add ordering, half-precision rounding). The accumulation rate is architecture-class dependent because different kernels participate in different forward graphs.

The largest empirical residual in the registry sits here: ~30x on the C6 IBPS information-bottleneck substrate, recorded as a hard-earned anchor rather than smoothed away.

### 3. Per-byte leverage is approximately uniformly distributed (entropy-coded archives)

`cumulative_leverage(top-K bytes) ≈ K/N + O(small)`

On entropy-coded archives (Brotli + LZMA + arithmetic-coded streams), the top-1% of bytes by master-gradient L1 norm carry approximately 1% of the total leverage — not 90% as one would naively expect from "important bytes." Anchored convergently across PR101 (top-1% byte leverage = 6.4%), PR106, and FEC6 archives.

First-principles basis: entropy coders by construction equalize per-byte information density. Per-byte importance heterogeneity in the source (some pixels matter more than others to the contest scorer) gets smeared across the entropy-coded representation. Recovering per-source-pixel importance from per-archive-byte gradients requires inverting the entropy coder, which is non-trivial.

Operational consequence: per-byte optimization saturates quickly for entropy-coded archives; substrate-class shifts dominate per-byte edits at the entropy-coded layer.

### 4. Per-pair master-gradient Taylor + Cauchy-Schwarz upper bound

`|Δs_pair| ≤ ||g_pair||_2 · ||δ_pair||_2 + O(δ²)`

For a substrate that emits per-pair residuals, the per-pair score impact of a local perturbation is upper-bounded by the L2 product of the per-pair master-gradient norm and the per-pair perturbation norm. Anchored on FEC6 fixed-Huffman archive at the per-pair granularity.

First-principles basis: Cauchy-Schwarz inequality applied to the first-order Taylor expansion of the score around the operating point. The second-order term is bounded by the operator norm of the Hessian along the perturbation direction; for small perturbations within the eval-roundtrip uint8 quantization radius, the second-order term is dominated by the first.

Operational consequence: per-pair candidate ranking gets a principled upper bound rather than a heuristic. Per-pair-difficulty atlases (which pairs have the largest gradient norms) become a directly-actionable bit-allocation signal.

### 5. Master-gradient locality violation by codec

Raw per-byte master-gradient is NOT a valid score derivative for entropy-coded archives.

A single-byte mutation to an LZMA-compressed stream typically corrupts the inflate path far beyond a single semantic edit — the mutation propagates through the LZ77 dictionary back-reference chain and the arithmetic coder's state. A naive finite-difference master-gradient on raw archive bytes measures the corruption envelope, not the score gradient at the operating point.

Anchored on the FEC6 archive: byte-level finite-difference at the raw-archive layer produces gradient magnitudes ~3 orders larger than gradient magnitudes computed at the post-decompress / post-decode grain. The discrepancy is the codec locality violation.

First-principles basis: entropy coders are designed to have non-local byte-to-symbol mappings (that's how they achieve compression). The byte-level forward function is therefore non-Lipschitz at the per-byte granularity; the canonical Lipschitz domain is the post-decompress symbol space.

Operational consequence: master-gradient extraction must operate at the post-decompress grain (per-symbol on arithmetic-coded streams, per-coefficient on LZMA-coded streams, per-block on Huffman-coded streams) for the gradient to be a valid score derivative. The canonical extractor at `tools/extract_master_gradient.py` enforces this discipline.

### 6. Canonical frontier pointer

`frontier(axis, hardware, timestamp) = single source of truth`

The single canonical record of "the best score we have on the contest-CPU axis" and "the best score we have on the contest-CUDA axis" lives at `.omx/state/canonical_frontier_pointer.json`, auto-refreshed on every dispatch completion via a fcntl-locked write.

Not really an equation in the differential-calculus sense, but registered alongside the others because it codifies the same anti-tribal-knowledge discipline: the previous regime had the best-score numbers tracked in MEMORY.md, individual commit messages, and reports/latest.md, with drift across surfaces. The pointer is the structural extinction of that drift class.

First-principles basis: Shannon-style separation of concerns — the canonical state lives in one place; views over the state (operator briefings, dashboard, autopilot ranker input) read from the pointer rather than maintain parallel records.

## Canonical references

- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience. Chapter 2 (entropy and mutual information) is the canonical reference for the entropy-coding leverage equation (#3) and the cooperative-receiver / mutual-information framings the registry's downstream consumers depend on.
- Cauchy-Schwarz inequality (canonical; reproducible in any analysis text). Foundation for equation #4.
- [RFC 7932](https://datatracker.ietf.org/doc/html/rfc7932) — Brotli Compressed Data Format. Foundation for equation #1.

## Honest scope

Six equations is a small registry. The discipline matters more than the count: every future quantitative anchor that the session learns ought to land in the registry rather than in individual file docstrings or chat logs. The framework's value is structural — codifying empirical anchors as first-class artifacts means downstream consumers (the cathedral autopilot ranker, the per-pair Pareto envelope, the Wyner-Ziv deliverability proof builder) can audit which assumptions their predictions rest on.

The registry is not a closed-form score predictor. None of the six equations predicts the contest score directly; they predict relationships between operating-point quantities (Brotli stream counts, per-byte gradient norms, per-pair score impacts) that the substrate-design loop needs to reason about. Score prediction at the contest-archive granularity requires solving the full meta-Lagrangian / Pareto-feasibility problem at the substrate level, which the registry feeds into but does not replace.
