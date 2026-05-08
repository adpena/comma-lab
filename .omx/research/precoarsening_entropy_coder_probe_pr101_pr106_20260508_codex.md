# Pre-Coarsening Entropy-Coder Probe - PR101/PR106 - 2026-05-08

## Scope

Bounded lane: test whether a pre-coarsening static entropy coder should be
reactivated for PR101/PR106 HNeRV int8 renderer weight streams.

Evidence grade: CPU/proxy byte accounting only. No scorer load, no CUDA, no
candidate archive, no exact-eval dispatch, and no score claim.

Generated structured evidence:

- `experiments/results/precoarsening_entropy_probe_20260508_codex/precoarsening_entropy_probe.json`
- `experiments/results/precoarsening_entropy_probe_20260508_codex/precoarsening_entropy_probe.md`

Tool:

- `tools/probe_precoarsening_entropy_coders.py`

## Inputs

| target | archive bytes | archive sha256 | member | member bytes | member sha256 |
|---|---:|---|---|---:|---|
| PR101 `hnerv_ft_microcodec` | 178258 | `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e` | `x` | 178158 | `5f1948f9572e65f71c614d2ff15764ee416522e25cb1b06c8b1299c1306e8aaf` |
| PR106 `belt_and_suspenders` | 186239 | `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58` | `0.bin` | 186131 | `7f2cc905b7611ae8d7bced72be24e2266b0aa341f90cfeccbb0854fd8fc01eb7` |

Parser-proven logical sections:

| target | decoder section bytes | decoder sha256 | q symbols | scale bytes | source raw decoder bytes |
|---|---:|---|---:|---:|---:|
| PR101 | 162164 | `836d1876bffd74f77f30e387a3b4cac1dbb25929cc4d348830d36cfa2a6d48a6` | 228958 | 56 | 229014 |
| PR106 | 170278 | `654999f81f0552fb7568e6977e73aa329661c10c79a6ab6cddc3171302352004` | 228958 | 112 | 229070 |

## Method

The probe starts from exact archive payload bytes already in the repo:

- PR101: fixed offsets in member `x`; seven split Brotli streams for
  `decoder_blob`, then latent and sidecar bytes.
- PR106: `0xff + uint24 decoder_len`; one Brotli decoder stream followed by
  the latent/sidecar Brotli stream.

For each target, the tool reconstructs the pre-coarsening int8 weight symbols
under the fixed state schema. It then compares:

- exact/source decoder bytes;
- Brotli q11 on the source logical layout, which matched the exact source
  decoder section for both targets;
- Brotli q11 on a canonical fixed-schema int8 stream;
- a static AC/rANS/FSE proxy with full frequency-table, stream-length, packet
  header, and scale-byte accounting;
- an actual `constriction` range-coder bitstream with the same full headers.

The static coder packet charges:

- 8 bytes fixed packet header;
- 28 x 255 x 2 = 14280 bytes for smoothed uint16 frequency tables;
- 28 x 4 = 112 bytes for per-tensor compressed stream lengths;
- source scale bytes: 56 for PR101, 112 for PR106.
- per-tensor byte alignment for separate coded streams:
  `sum_t ceil(bits_t / 8)`, not `ceil(sum_t bits_t / 8)`.

It does not charge submission decoder code bytes, so a runtime candidate would
need stricter accounting before any archive work.

## Results

| target | source decoder | source q11 | canonical q11 | proxy payload floor | proxy overhead | proxy total | constriction payload | constriction total | delta vs reference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PR101 | 162164 | 162164 | 162395 | 160295 | 14456 | 174751 | 160344 | 174800 | +12636 |
| PR106 | 170278 | 170278 | 170226 | 167976 | 14512 | 182488 | 168020 | 182532 | +12254 |

`brotli` version: `1.2.0`.
`constriction` version: `0.4.2`.

Interpretation:

- PR101: the conservative static proxy loses by 12587 bytes and the practical
  constriction packet loses by 12636 bytes after full headers, even before
  decoder code bytes.
- PR106: canonical int8 Brotli q11 is 52 bytes smaller than source layout, but
  that is only a CPU byte-layout observation with no runtime adapter. The
  conservative static proxy loses by 12210 bytes and the practical constriction
  packet loses by 12254 bytes before decoder code.
- The practical constriction payload closely tracks the proxy floor, so this
  is not a Python-library overhead artifact. The table/model headers are the
  dominant blocker for this pre-coarsening zero-order screen.

Measured disposition:

- `measured_precoarsening_static_config_retired`
- `do_not_build_archive_from_precoarsening_static_entropy_probe`

Scope: this retires only the zero-order, per-tensor, pre-coarsening static
entropy-coder configuration measured here. It does not kill AC, rANS, FSE,
constriction, context-mixed coders, post-coarsening coders, table-sharing
coders, or runtime-integrated entropy coders.

## HStack/VStack Review

HStack status: decoder-stream-only horizontal candidate.

Synergy:

- It touches parser-proven renderer decoder bytes only, so it could compose
  horizontally with latent/sidecar repacks once a runtime adapter exists.
- It can be evaluated independently from latent/sidecar HStack lanes because
  the logical sections are disjoint.

Antagonism:

- It competes with PR101 split Brotli, HDM-style decoder recodes, and any
  decoder replacement that owns the same renderer weight stream.
- Any custom entropy runtime adds code bytes and dependency risk not charged in
  this probe.

VStack status: terminal entropy stage after representation and quantization.

Synergy:

- Best placed after byte-map/order derivation and after lossy coarsening,
  because those transforms change the symbol distribution and may reduce table
  cost.
- A post-coarsening rerun is valid if it starts from the new exact payload
  bytes and charges the same headers.

Antagonism:

- Pre-coarsening static coding ignores sequential context that Brotli already
  exploits.
- Proxy-only wins are insufficient for archive building; here even the actual
  constriction packet loses.

## Reactivation Criteria

Keep this measured pre-coarsening static configuration closed for archive work
unless a future probe satisfies all of:

1. Starts from the exact payload bytes or state dict of the active candidate.
2. Actual practical coder total beats the current source-layout Brotli
   reference after frequency/model headers, stream delimiters, scale bytes, and
   packet header are charged.
3. Decoder code bytes and dependency closure are either charged or proven
   already resident in the runtime.
4. Candidate archive bytes and payload SHA change are emitted with a runtime
   adapter and strict pre-submission compliance.
5. Exact CUDA auth eval lands before any score or promotion claim.

Next useful reruns: after lossy coarsening, after a context-mixing model, after
shared-table rANS/FSE design, or after another symbol-distribution change. Do
not rerun the same zero-order pre-coarsening configuration on the same streams.
