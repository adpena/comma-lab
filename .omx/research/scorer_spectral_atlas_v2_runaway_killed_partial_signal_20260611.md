# Scorer spectral-sensitivity atlas v2: runaway KILLED, partial signal preserved (2026-06-11)

**Authority:** `[macOS-CPU advisory]` / mechanism_update_eligible only (NOT a score row; per the tool's
own contract). Pointer UNMOVED at 0.19109982.

## What it was
`tools/measure_scorer_spectral_sensitivity.py v2` — the scorer TRANSFER FUNCTION (the "arbitrariness cure",
operator 2026-06-09): perturb each radial frequency band of frame1 at fixed relative energy, score
perturbed-vs-source through the EXACT DistortionNet, and read H_seg(k)/H_pose(k) = which spatial
frequencies SegNet/PoseNet actually react to → so the carrier's frequency budget (SIREN ω / Fourier-feature
bandwidth / the capstone `grid_pe_num_freqs`) is DERIVED, not guessed.

## Why it was killed (not respawned as-is)
A RUNAWAY: 3200 cells × 12 pairs × 3 phases = 115,200 exact-scorer-forward groups on the CPU scorer.
Started Jun 9 18:31; at 02:33 Jun 11 it was at **cell 800/3200 (25%)** after **31 hours** (~2.3 min/cell →
**~5 days** total ETA), with **no incremental atlas output** (only `daemon.log`; killing loses the
structured json — the log per-cell H values are preserved at
`.omx/research/scorer_spectral_atlas_v2_partial_20260609/daemon_partial_cells_1_800.log`). It was an
ORPHAN (PPID=1, no owning session) contending with the decisive capstone daemon. Advisory-only + 5-day ETA
+ no checkpoint + contending → kill + preserve was the right call. The CPU is now freed for the daemon.

## The partial signal IS already actionable (the isotropic all-band sweep, cells 1–625, COMPLETED)
- **PoseNet is most sensitive to LOW spatial frequencies + HORIZONTAL orientation.** Strongest responses:
  band0 horizontal a=2.0 rgb → H_pose **0.587**; band1 horizontal a=4.0 yuv:y → **0.408**; band1
  horizontal a=8.0 yuv:v → 0.086. (Isotropic responses are much weaker — the orientation matters: dashcam
  ego-motion is dominantly horizontal, so the pose head reacts to horizontal low-freq structure.)
- **SegNet is broadly WEAK across all bands** (max H_seg ~0.009 at band1 yuv:y frame1 a=8; most cells
  H_seg < 0.003). The argmax map is robust to band-limited perturbations — consistent with d_seg being a
  boundary-flip phenomenon, not a broadband one.
- **Implication for the carrier frequency budget:** concentrate at LOW frequencies (small ω / few low-freq
  Fourier features). This **validates** the capstone's low `grid_pe_num_freqs=4` as principled, not
  arbitrary-wrong — the scorer's energy is at the low end. A high-frequency budget would be wasted.

## Recommendation on "should we respawn it"
**Not the 3200-cell version.** The core actionable signal (band-sensitivity curve + the low-freq/horizontal
pose finding) is captured. If a full structured atlas is wanted to refine the grid-PE frequency bands, run
a RIGHT-SIZED version (~5–10× smaller, ~1–2h, incremental output + marker-on-exit) AFTER the decisive
capstone daemon finishes (the atlas feeds the DOWNSTREAM grid-PE-ON daemon, not the current critical path):

```bash
# right-sized: drop the amplitude/channel/phase cross-product to the signal-bearing minimum.
# (verify these flags against the tool's argparse first — grep "add_argument".)
nohup bash -c '.venv/bin/python tools/measure_scorer_spectral_sensitivity.py v2 \
  --n-pairs 12 --n-bands 8 --band-spacing log \
  --amplitudes-lsb 2,8 --orientations isotropic,horizontal \
  --frame-incidences frame1_only,both_opposite --channel-bases yuv --yuv-channels y \
  --rgb-channels all --n-phase-samples 1 \
  --work-dir /Volumes/VertigoDataTier/pact/scorer_spectral_atlas_v2_rightsized_<UTC> \
  --out /Volumes/VertigoDataTier/pact/scorer_spectral_atlas_v2_rightsized_<UTC>/atlas.json; \
  echo "EXIT=$?" > /Volumes/VertigoDataTier/pact/scorer_spectral_atlas_v2_rightsized_<UTC>/DONE.marker' \
  < /dev/null > /dev/null 2>&1 & disown
```
That cuts 3200 cells → ~256 cells (2 amplitudes × 2 orientations × 2 incidences × 1 channel × 8 bands ×
1 phase × ... ≈ ~1–2h) while keeping the bands + the two orientations + the two amplitudes that carry the
signal. Lower priority than the capstone daemon → int8-reality → exact eval critical path.

## Process lesson (durable-daemon discipline, CLAUDE.md "session-watcher trap")
This atlas had NO marker-on-exit, NO incremental checkpoint, and became a PPID=1 orphan no session tracked
— exactly the anti-pattern. Any future long sweep MUST: (a) write incremental results (resumable), (b)
marker-on-exit, (c) be sized so the ETA is hours not days for an ADVISORY measurement.
