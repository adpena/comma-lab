---
name: Compliance — Neural Artifacts Must Be Inside archive.zip
description: CRITICAL lesson — postfilter_int8.pt must be bundled inside archive.zip per contest rules. Rate penalty ~0.026 score.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
Contest rules: "External libraries and tools can be used and won't count towards
compressed size, unless they use large artifacts (neural networks, meshes, point
clouds, etc.), in which case those artifacts should be included in the archive
and will count towards the compressed size."

postfilter_int8.pt (45KB) is a neural network artifact. It MUST be inside archive.zip.

**Why:** Previous scoring (1.33) did not include the checkpoint in the archive,
making the rate artificially low. True compliant score is ~1.356 (+0.026).

**How to apply:**
- compress.sh now copies postfilter_int8.pt into the archive dir before zipping
- inflate_postfilter.py already checks `Path(archive_dir) / "postfilter_int8.pt"` first
- ALL future canonical scoring must use the compliant archive (with .pt inside)
- When promoting a new checkpoint, ALWAYS rebuild archive.zip with the new .pt included
- The rate term will always include ~38KB of compressed checkpoint bytes

**Lesson:** Always audit compliance against the LETTER of the rules, not just
whether the code works. This was caught in the 4th review round, not the 1st.
Run compliance audits early and often.
