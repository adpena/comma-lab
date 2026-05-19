# Codex Findings - NSCS06 Seeded Chroma Hash-Seed Path - 2026-05-19

## Source

- Directive: `.omx/research/codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518.md` ITEM_9.
- Sidecar review: Ptolemy + Russell xhigh reviewers, both read-only.

## Landing

Implemented CH06 schema v3 for `nscs06_carmack_hotz_strip_everything`:

- v2 remains the default raw per-class chroma palette format.
- v3 stores an archive-contained 8-byte seed in the existing `CHROMA_BLOB` section.
- Inflate-time parse expands that seed into the per-class RGB palette without scorer loads, torch, or external `tac.*` imports in the vendored runtime.
- `experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py` now exposes opt-in `--chroma-seed-mode raw|hash-seed`, default `raw`.
- Full trainer provenance records schema version, carrier, seed hex, generated-palette SHA-256, raw-vs-seed byte delta, and remains non-promotional until exact eval.

## Premise Correction

The directive's older ~7.5 KB chroma-palette premise does not match current CH06 v2. Current NSCS06 v7 stores:

```text
NUM_SEGNET_CLASSES * CHROMA_BYTES_PER_CLASS = 5 * 3 = 15 bytes
```

Therefore this patch is a real archive-grammar and integration proof with a current CH06 payload reduction of:

```text
8-byte seed - 15-byte raw palette = -7 bytes
```

The higher-EV value is not the immediate 7-byte CH06 delta. It is the contest-compliant pattern: archive-charged seed bytes can replace larger deterministic codebook/palette surfaces when the generated substrate is score-acceptable or when seed search finds a better palette.

## Authority

The authority-preserving path is `archive_member_seed`: the seed is stored inside `0.bin`, charged in `archive.zip`, parsed by the self-contained inflate runtime, and consumed before RGB emission.

Rejected or high-risk variants remain out of this landing:

- literal `inflate.py` seed for score-affecting per-video payload;
- scorer-weight or scorer-load receiver paths;
- claiming equivalence to the GT-derived v2 median palette without an exact eval or seed-search proof.

## Tests

Focused verification:

```text
.venv/bin/python -m ruff check ... -> green
.venv/bin/python -m pytest \
  src/tac/substrates/nscs06_carmack_hotz_strip_everything/tests/test_codec_roundtrip.py \
  src/tac/substrates/nscs06_carmack_hotz_strip_everything/tests/test_oom_fix.py \
  src/tac/tests/test_procedural_codebook_generator.py -q
72 passed

.venv/bin/python experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py \
  --output-dir /tmp/nscs06_seed_smoke_raw --epochs 1 --device cpu --smoke
raw smoke passed

.venv/bin/python experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py \
  --output-dir /tmp/nscs06_seed_smoke_hash --epochs 1 --device cpu --smoke \
  --chroma-seed-mode hash-seed
hash-seed smoke passed

git diff --check -> clean
```

Coverage added:

- v3 writes schema byte 3 and replaces 15 raw palette bytes with 8 seed bytes.
- v3 parse expands `chroma_seed` into `chroma_rgb`.
- seed/palette mismatch is rejected instead of silently writing inconsistent archives.
- vendored CH06 seed helper is pinned against `tac.procedural_codebook_generator.hash_seed_codebook_generator`.
- mutating the archived seed changes the parsed palette.
- mutating the archived seed changes full inflated RGB bytes.
- trainer opt-in mode is pinned through argparse, `pack_archive`, and provenance.

## Next

This landing is not a score claim. The next score-moving step is seed search or a learned seed/generator family that beats the raw GT-derived 15-byte palette under exact auth eval despite the generated-palette substitution.
