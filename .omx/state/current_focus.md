# current focus

## status after colorspace-hardening promotion

- Upstream snapshot remains verified against `workspace/upstream_snapshot.json` at commit `ec82c291ffeae5212e9a38253791d58995518a80`.
- Track A remains alive under `current_workflow` at `0.00` with a `167` byte archive.
- Track B now holds a new promoted AV1 floor at **`2.12`** with `524x394 / libsvtav1 / preset0 / crf34 / film-grain22 / lanczos / unsharp=0.35 / explicit bt709/tv encode tags / explicit rgb24(pc) decode`.
- The installed rule-faithful payload under test is `archive.zip`, `inflate.sh`, `config.env`, `analyze_roi.py` = `897745` bytes.
- The prior implicit-colorspace floor at `2.18` is now the immediate comparison point.
- The local frontier story is now richer: compression, reconstruction, synthesis, geometry, and upscale-axis probes all have measured outcomes.
- The bug/rigor pass also fixed execution-contract issues: upstream-root packaging, stale-inflate cleanup, ROI codec guarding, ffmpeg/ffprobe propagation, live ROI_X_FRAC, ROI postfilter symmetry, explicit color handling, and stale-TMPDIR fallback.

## active priority

1. Preserve the promoted **`2.12`** floor on the canonical `robust_current` path.
2. Keep strengthening the writeup around the local frontier shape and the new rigor-cleanliness story.
3. Keep pixel-semantics checks disciplined and lightweight.
4. Only run another AV1 probe if the rationale is materially stronger than the recent one-axis losses.

## immediate next step

The highest-value next move is writeup-first: the repo now has a production-hardening win that materially improved score, which is unusually strong Best Write-up material.
