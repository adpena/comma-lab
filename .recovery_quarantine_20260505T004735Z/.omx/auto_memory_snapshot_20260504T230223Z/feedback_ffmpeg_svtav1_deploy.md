---
name: ffmpeg svtav1 + LD_LIBRARY_PATH on remote
description: Ubuntu 22.04 apt ffmpeg lacks libsvtav1; mask_codec needs upstream/ffmpeg-new + LD_LIBRARY_PATH for libSvtAv1Enc.so.2 (bundled in upstream/submissions/av1_roi_lanczos_unsharp/lib).
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
CANONICAL: when deploying mask-encoding code to a fresh Vast.ai instance,
the apt-installed `ffmpeg` is 4.4.2 (Ubuntu 22.04) which has `libaom` but
NOT `libsvtav1`. `mask_codec.encode_masks` uses `-svtav1-params`, so any
mask encoding crashes with `Unrecognized option 'svtav1-params'`.

**Why:** 2026-04-26 DEN-V2 deploy crashed at this exact spot AFTER 1h
of successful renderer training, wasting that GPU time + the operator's
attention. The upstream snapshot ships its own statically-built ffmpeg
n7.1 (`upstream/ffmpeg-new`, 24MB) WITH libsvtav1 + the bundled
`libSvtAv1Enc.so.2.3.0` (in `upstream/submissions/av1_roi_lanczos_unsharp/lib/`).
But ffmpeg-new is shipped without +x permission and the lib is in a
non-standard path so the loader can't find it without LD_LIBRARY_PATH.

**How to apply:**
- `_ffmpeg_binary()` in `src/tac/mask_codec.py` (added 2026-04-26)
  resolves the right binary: $TAC_FFMPEG → $TAC_UPSTREAM_DIR/ffmpeg-new
  → ./upstream/ffmpeg-new → system ffmpeg.
- `scripts/remote_train_bootstrap.sh` Stage 1b chmods ffmpeg-new,
  symlinks libSvtAv1Enc.so.2 → .so.2.3.0, exports LD_LIBRARY_PATH, and
  smoke-checks `-version` BEFORE training starts. This catches the
  library-loading failure at Stage 1, not Stage 5 after 1h of training.
- Any new deploy script must do the same Stage 1b setup, OR set both
  TAC_UPSTREAM_DIR and LD_LIBRARY_PATH before invoking pipeline.py.
- Commit reference: 2acbc25b (mask_codec resolver + bootstrap hardening).
