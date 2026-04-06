# Notepad

## Priority Context
Track B canonical floor: 2.12 @ 524x394 SVT-AV1 p0 CRF34 film-grain22 lanczos+unsharp with explicit bt709/tv encode tags and explicit rgb24(pc) decode. Rule-faithful runtime payload: 897745 bytes -> 2.1418040615200598. Track A exact_current remains the only explicitly non-rule-faithful lane. Canonical robust summary: reports/raw/robust_current-current_workflow-cpu-summary.json .

## Working Memory
- [2026-04-06T14:12:54.037734+00:00] 2026-04-06 dashboard-coherence fix: corrected local-frontier microscope to label 522x392 as geometry instead of CRF, and corrected the AV1 mechanism card to show lanczos + unsharp for the historical 2.18 floor. Redeployed corrected site. Latest Wrangler-confirmed preview: https://80c389c1.comma-lab.pages.dev .
- [2026-04-06T14:03:16.917002+00:00] 2026-04-06 Lanczos-upscale cycle: smoke gate passed, scorer result and canonical regression both measured 2.18 at unchanged 864,455 bytes, so the AV1 floor promoted from 2.19 to 2.18 by changing only bicubic -> lanczos on the upscale axis. This is the cleanest recent win because bytes stayed flat while both pose and seg improved. Latest Wrangler-confirmed preview: https://5a45e8a4.comma-lab.pages.dev .
- [2026-04-06T14:02:14.929839+00:00] 2026-04-06 Lanczos-upscale cycle: one-axis upscale-kernel probe from bicubic to Lanczos passed the smoke gate and promoted the honest Track B floor from 2.19 to 2.18 at unchanged 864,455 bytes. Pose and seg both improved slightly. Latest Wrangler-confirmed preview: https://5a45e8a4.comma-lab.pages.dev .
- [2026-04-06T13:01:53.094603+00:00] 2026-04-06 geometry cycle: 522x392 smoke gate passed, scorer result was 2.23 at 862,238 bytes after estimated 2.17–2.22 outcome, and was rejected. Restored canonical 2.19 floor. This adds a geometry-side near-miss to the local frontier story: crf35 loss, unsharp030 loss, film-grain0 loss, and 522x392 loss. Latest Wrangler-confirmed preview: https://ee5bbfe6.comma-lab.pages.dev .
- [2026-04-06T12:59:45.131173+00:00] 2026-04-06 geometry cycle: 522x392 passed smoke gate but scored 2.23 at 862,238 bytes and was rejected. Estimate before run: 2.17–2.22. Bytes fell only slightly while both pose and seg worsened. Canonical 2.19 floor preserved. Latest Wrangler-confirmed preview: https://ee5bbfe6.comma-lab.pages.dev .
- [2026-04-06T07:15:18.112606+00:00] 2026-04-06 filmgrain0 cycle: smoke gate passed on both the live 2.19 floor and the film-grain0 candidate. Film-grain0 was estimated at 2.16–2.24 but measured 3.33 at 719,096 bytes, with PoseNet distortion exploding. Canonical 2.19 floor preserved. Latest Wrangler-confirmed preview: https://a9c110e6.comma-lab.pages.dev .
- [2026-04-06T07:14:15.740145+00:00] 2026-04-06 film-grain0 cycle: film-grain0 candidate passed smoke gate, then scored 3.33 at 719,096 bytes and was rejected. This is strong evidence that AV1 film-grain synthesis is functionally important in this evaluator regime. Live canonical floor remains 2.19 at CRF34 / film-grain22 / unsharp0.35. Latest Wrangler-confirmed preview after refreshing the frontier-shape story: https://a9c110e6.comma-lab.pages.dev .
- [2026-04-06T06:16:37.114639+00:00] 2026-04-06 unsharp030 cycle: live 2.19 AV1 floor passed the new smoke gate; one-axis reconstruction trim from unsharp 0.35 to 0.30 measured 2.20 at unchanged 864,455 bytes and was rejected. This strengthens the writeup story that the local frontier is tight from both the compression side (crf35 lost) and the reconstruction side (softer unsharp lost). Latest Wrangler-confirmed preview: https://2f1800e7.comma-lab.pages.dev .
- [2026-04-06T06:14:47.363878+00:00] 2026-04-06 unsharp030 cycle: one-axis AV1 probe reduced decoder-side unsharp from 0.35 to 0.30. Estimate before run: 2.18–2.20. Measured 2.20 at 864,455 bytes. No byte change, slight pose/seg regression, so hypothesis failed. Live canonical floor remains 2.19 at crf34/unsharp0.35. Latest Wrangler-confirmed preview after updating the frontier-shape story: https://2f1800e7.comma-lab.pages.dev .
- [2026-04-06T05:41:03.068089+00:00] 2026-04-06 smoke-gate cycle: implemented src/comma_lab/smoke.py and CLI command `python3 -m src.comma_lab.cli smoke-submission robust_current --package`. Live 2.19 AV1 floor passed with exact raw file cardinality, frame count 1200, and expected total bytes 3,662,409,600. Also ran one more one-axis AV1 probe at crf35: estimated 2.17–2.20, measured 2.21 at 808,036 bytes, rejected. Restored canonical 2.19 floor and redeployed site. Latest Wrangler-confirmed preview: https://de095e94.comma-lab.pages.dev .
- [2026-04-06T05:29:44.366311+00:00] 2026-04-06 continuation: ran one more one-axis AV1 probe at crf35 after the 2.19 crf34 floor. Estimate before run: about 2.17–2.20. Measured result: 2.21 at 808,036 bytes. Bytes improved further but pose/seg rose too much, so hypothesis failed. Restored live config.env and archive.zip to the canonical 2.19 crf34 floor. Latest Wrangler-confirmed preview after updating the writeup with the CRF34 win / CRF35 loss frontier shape: https://457010ad.comma-lab.pages.dev .
- [2026-04-06T05:27:59.101511+00:00] CRF35 AV1 probe rejected. Estimate was 2.17–2.20; measured 2.21 at 808,036 bytes. Bytes kept falling, but pose/seg rose too far. This strengthens the writeup: CRF34 appears near the local knee, CRF35 crosses past it. Live floor remains 2.19. Latest preview after this refresh: https://457010ad.comma-lab.pages.dev .

- [2026-04-06T15:00:32.758252+00:00] 2026-04-06 comprehensive bug/rigor pass: corrected rule-faithful accounting to the installed runtime payload under test (`archive.zip`, `inflate.sh`, `config.env`, `analyze_roi.py`) = 892,472 bytes -> 2.196195252141633 (historical pre-hardening estimate). Fixed: requested-upstream packaging, package-without-sync rejection, stale `inflated/` cleanup before eval, AV1+ROI fail-fast guard, ROI metadata ffmpeg/ffprobe propagation, live `ROI_X_FRAC`, ROI-side `INFLATE_POSTFILTER`, root `.gitignore`, and transient scratch cleanup. Fresh evidence: reports/raw/2026-04-06-rigor-pass/robust_current-rigor-smoke.json and reports/raw/2026-04-06-rigor-pass/roi-rigor-checks.txt . Git history cleanup was not possible because this workspace is not a git repository.

- [2026-04-06T15:48:41.055937+00:00] 2026-04-06 colorspace/range hardening promotion: explicit bt709/tv encode tags plus explicit rgb24(pc) decode improved the scorer-backed Track B floor from 2.18 to 2.12 at 864,486 bytes. Corrected installed-runtime-payload rule-faithful estimate is 897,745 bytes -> 2.1418040615200598. Evidence: reports/raw/2026-04-06-hardening/encoded-ffprobe.json, .../robust_current-hardening-smoke.json, and .../robust_current-hardening-current_workflow-cpu-summary.json .

- [2026-04-06T15:54:48.646151+00:00] 2026-04-06 deployed updated static site for the 2.12 production-hardened floor. Latest Wrangler-confirmed preview: https://4faa3d95.comma-lab.pages.dev .

- [2026-04-06T16:01:39.598655+00:00] 2026-04-06 recorded speculative AV1+ROI parity lane in docs/speculative_lanes.md and .omx/state/next_experiments.md. Required plan: codec-agnostic ROI encode abstraction; AV1 params for base/ROI/ROI2; matching AV1-aware metadata ROI path; matching inflate/smoke/scorer parity checks; fresh scorer-backed evidence. This lane remains explicitly non-canonical until measured.

- [2026-04-06T16:48:41.839297+00:00] 2026-04-06 final packet/site sync complete. Fresh exact_current regression stayed at 0.00 / 167 bytes, current robust_current floor remains 2.12 / 864,486 bytes with rule_faithful 897,745 -> 2.1418040615200598. Latest Wrangler-confirmed preview: https://ab2fa023.comma-lab.pages.dev .

- [2026-04-06T17:01:15.922308+00:00] 2026-04-06 final remaining-risk sweep closed. Promotion accounting now includes the 2.12 floor, exact_current re-regressed to 0.00 / 167 bytes, and the latest Wrangler-confirmed preview is https://ee631431.comma-lab.pages.dev .

- [2026-04-06T17:38:18.929863+00:00] 2026-04-06 frontend/editorial pass: reduced the landing page from a generic dashboard shell toward a tighter technical brief. Changes included a simpler header, a thinner summary strip, corrected scatter semantics, boxed-but-decluttered lineage notes, lighter operational note styling, and reduced lower-page density. Latest Wrangler-confirmed preview: https://805e7d79.comma-lab.pages.dev .

- [2026-04-06T17:42:34.610074+00:00] 2026-04-06 frontend cleanup iteration: reduced card density, simplified the hero to a brief, clarified scatter semantics, omitted the 97.45 outlier from the main plot, trimmed the page to overview scope, and redeployed. Latest Wrangler-confirmed preview: https://287dff0d.comma-lab.pages.dev .

- [2026-04-06T17:44:35.179899+00:00] 2026-04-06 frontend cleanup iteration: further reduced chrome, removed remaining winner-language labels, and tightened the brief-style landing page. Latest Wrangler-confirmed preview: https://131d3d84.comma-lab.pages.dev .

- [2026-04-06T17:50:35.013871+00:00] 2026-04-06 frontend cleanup iteration: converted the explanatory section from side-by-side notes into a numbered walkthrough, kept the page in overview mode, and redeployed. Latest Wrangler-confirmed preview: https://3464cd88.comma-lab.pages.dev .

## MANUAL
- Persistent lab memory:
  - Only Track A / exact_current is explicitly non-rule-faithful.
  - Honest Track B promotion authority is the local CPU scorer.
  - The canonical Track B floor is whatever is recorded in:
    - reports/raw/robust_current-current_workflow-cpu-summary.json
    - reports/latest.md
    - .omx/state/current_focus.md
  - Rule-faithful accounting must charge only the installed runtime payload under test:
    - archive.zip
    - inflate.sh
    - config.env
    - analyze_roi.py
  - Promotion gate: docs/promotion_gate.md
  - Latest promotion review lives at:
    - reports/promotion_reviews/2026-04-06-av1-colorspace-hardening-promotion-review.md
  - FFmpeg / hardening review: reports/ffmpeg_path_review.md
  - Canonical Pages URL: https://comma-lab.pages.dev


## WORKING MEMORY
[2026-04-06T18:16:32.820Z] 2026-04-06 site/frontend pass: added generated experiment manifest, code callouts, browser-preview comparison media with synced controls, contest/about/last-updated context, GitHub link (adpena/comma-lab), comparison posters, and mobile-safe horizontal scroll for the frontier table. Verified with Playwright desktop+iPhone screenshots against reports/graphs/site/index.html.

[2026-04-06T18:20:27.646Z] 2026-04-06 deployed writeup/frontend pass preview: https://d911022c.comma-lab.pages.dev (commit 572122b). Includes contest/about/last-updated context, experiment manifest, code callouts, comparison media with posters + sync controls, and mobile-safe frontier table scrolling.
[2026-04-06T18:39:10.002Z] 2026-04-06 player/scatter coherence pass: fixed comparison mode-switch bug (hidden videos no longer keep playing, shared playhead preserved across full/zoom), replaced brittle 2.18→2.12 SVG rows with semantic HTML, added published-baseline delta to summary, flipped failure detour downward in lineage graph, and added a focused operating-range scatter view with smaller markers and hover/focus detail.
[2026-04-06T18:40:01.925Z] 2026-04-06 deployed player/scatter coherence pass preview: https://78803cbe.comma-lab.pages.dev (commit afa1017). Includes published-baseline delta, cleaned context strip, downward failure detour, semantic 2.18→2.12 comparison, focused scatter view, and fixed comparison-player state handling.