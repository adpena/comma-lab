---
name: Partial-tarball deploys lose critical sidecar files (config.env, parent-level artifacts)
description: 2026-04-27 Two deploy failures from incomplete tarballs — Spain Lane M+N missing submissions/robust_current/config.env (auth eval fell to ffmpeg fallback expecting 0.mkv); Nevada Lane F-V2 missing experiments/results/lane_a_landed/optimized_poses.pt at parent level (script anchor failed). Both were "minor" missing files but cost ~10-15 min of recovery debug each. Prefer git-based deploys; if using tarball, audit the include set against script anchors EXPLICITLY.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Rule: tarball-based deploys MUST include every file referenced by lane scripts AND by inflate.sh's config.env loader.**

**Why (today's incidents):**

1. **Spain Lane M+N (35716364) missing `submissions/robust_current/config.env`:** The bootstrap that produced the Spain instance excluded `submissions/robust_current/config.env`. Without it, `inflate.sh` couldn't find `PYTHON_INFLATE=renderer` and fell through to the ffmpeg fallback that expects `${stem}.mkv` (e.g., `0.mkv`) instead of `masks.mkv`. The auth eval crashed at `ffmpeg: Error opening input file ...0.mkv`. Fix: `scp config.env` + re-run auth eval.

2. **Nevada Lane F-V2 (35719867) missing `experiments/results/lane_a_landed/optimized_poses.pt` at parent level:** The tarball only included `experiments/results/lane_a_landed/iter_0/` and `experiments/results/lane_a_landed/extracted/masks.mkv`, but the lane script anchored on `experiments/results/lane_a_landed/optimized_poses.pt` (parent level). Pre-flight check failed: "FATAL: missing experiments/results/lane_a_landed/optimized_poses.pt". Fix: `cp iter_0/optimized_poses.pt parent/optimized_poses.pt` on remote.

**The unifying pattern:** scripts reference paths derived from local layout, but deploys preserve only a subset of paths. Sidecar configs (config.env), parent-level artifacts (optimized_poses.pt at lane_a_landed root), and meta-files (provenance.json schema files) all silently drop.

**How to apply:**

1. **Prefer git-based deploys for new lane work.** SSH in, `git fetch + git reset --hard <commit>` guarantees exact HEAD parity. The cost: needs git credentials on remote (or a bare-repo intermediary). Defer until canonical_remote_bootstraps gets git-pull support.

2. **For tarball deploys, audit explicitly:**
   - Before launching, `grep -E "experiments/results/.*\.pt|submissions/.*config|/.*\.json" scripts/remote_<lane>.sh` to enumerate path references.
   - Confirm every referenced path is in the tarball's `tar -tzf` listing.
   - Pre-flight that on the remote BEFORE setup_full.sh runs (cost: 1 second, saves 10-15 min of debug).

3. **Add a preflight check `check_lane_script_paths_in_deploy_manifest`:** AST-grep lane scripts for path literals starting with `experiments/`, `submissions/`, `upstream/`. Cross-reference with the deploy manifest (whatever the canonical bootstrap uses). Warn if any path isn't covered.

4. **For inflate.sh specifically:** ensure `submissions/robust_current/config.env` is in EVERY deploy manifest. It's load-bearing for the renderer dispatch path. Memory note: any auth eval that falls to the ffmpeg fallback is a silent symptom of missing config.env.

5. **For the lane_a_landed artifacts (and any other lane's "anchor" dir):** put renderer.bin, optimized_poses.pt, and masks.mkv at a SINGLE canonical layout (e.g., always inside `iter_0/`) and update lane scripts to point there. Don't have parent-level + iter_0 copies of the same file with different deploy coverage.

**Cost of these incidents:** ~$0.10 GPU + ~25 min total debug across both lanes. The pattern would cost more on lanes with longer compute envelopes (e.g., Lane D's 5h training).

**Related memories:**
- `feedback_canonical_remote_bootstraps` — the canonical deploy approach.
- `feedback_zip_dep_bootstrap_trap` — earlier deploy bug (no `zip` in PyTorch container) — same root cause class.
- `feedback_remote_code_parity_required` — another "deploy state ≠ HEAD" pattern, $10+ wasted SHIRAZ run.
