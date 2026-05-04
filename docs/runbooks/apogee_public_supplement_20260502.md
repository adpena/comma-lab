# Apogee Public Supplement Plan

Purpose: ship a public supplement that strengthens the contest PR without
changing the contest score path. The score authority remains the exact archive
bytes evaluated through `archive.zip -> inflate.sh -> upstream/evaluate.py`.

## Source Constraints

- Contest PR contract: public PR with a download link to `archive.zip`,
  `inflate.sh`, `report.txt`, GPU-required answer, optional compression script,
  and additional comments.
- Contest runtime contract: official evaluation has a 30 minute limit; GPU
  inflation runs on T4-class hardware.
- Contest artifact contract: external tools are allowed, but large
  score-affecting artifacts must be included in `archive.zip` and charged.
- Current exact frontier for the public supplement: PR100 HNeRV-LC-v2 adapter
  replay, exact Tesla T4 A++ score `0.22826947142244708`, `178981` bytes,
  archive SHA-256
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`, and
  runtime tree SHA-256
  `ef6323533666c9cac1c204a9d3f7054157d44a185b16fc859fb3f0438ccd1832`.
- Lightning Studios are useful for notebooks, GPU-backed demos, training, and
  hosted app work; public-port/app URLs must be intentionally published through
  a sanitized manifest, not copied from private Studio state.
- Cloudflare Pages Direct Upload is the static writeup host: prebuilt assets can
  be deployed by Wrangler or dashboard upload. Wrangler expects a folder, and
  each static asset must stay under the current 25 MiB Pages asset limit.
- `reports/graphs/site/` is an internal generated bundle and may contain
  historical private custody paths. Deploy only the sanitized
  `reports/graphs/public_site/` bundle built by
  `reports/graphs/build_public_site_bundle.py`.
- The public bundle builder omits assets over the Pages limit by default and
  records them in `public_site_manifest.json`; final polish should replace any
  omitted media with smaller public-facing clips.

## Public Artifact Names

Use `Apogee` for the PR-facing submission name and lower-case `apogee` for
files, IDs, and deployment names.

- PR title: `Apogee - contest-faithful scorer-aligned video compression`
- submission name field: `Apogee`
- archive URL placeholder: `${APOGEE_ARCHIVE_ZIP_URL}`
- release manifest placeholder: `${APOGEE_RELEASE_MANIFEST}`
- sanitized release manifest: `reports/graphs/site/apogee_release_manifest.json`
- Cloudflare project: `apogee-comma-video`
- Cloudflare deploy directory: `reports/graphs/public_site/`
- Lightning notebook: `notebooks/apogee_lightning_supplement.ipynb`
- Lightning public URL placeholder: `${LIGHTNING_SUPPLEMENT_URL}`
- Cloudflare public URL placeholder: `${CLOUDFLARE_PAGES_URL}`
- submission packet output:
  `experiments/results/submission_packet_c067_20260502/automated_packet/`

## Notebook Design

The Lightning notebook should be a reproducible, public, no-secret supplement:

1. **Frontier card.** Load the release manifest or PR100 packet, print score,
   bytes, SHA, component distances, sample count, device, and evidence grade.
2. **Formula audit.** Recompute
   `100 * seg_dist + sqrt(10 * pose_dist) + 25 * archive_bytes / 37545489`
   from published fields and fail closed on mismatch.
3. **Byte anatomy.** Plot mask/model/pose/zip-overhead contributions and the
   unchanged-distortion sub-0.300 byte gap.
4. **External-source attribution.** Show the PR100 public-source attribution,
   C067/PR67 historical mask segment attribution, and distinguish charged reuse
   from sidecar/script-payload exploits.
5. **Negative evidence map.** Visualize scoped exact negatives as useful
   design constraints, not broad method kills.
6. **Yousfi-Fridrich floor view.** Present the atom-waterfill/Lagrangian view:
   every byte, mask atom, pose atom, renderer block, residual, and packer
   decision is an atom with charged byte cost, component benefit, confidence,
   and interaction risk.
7. **Reproduction checklist.** Link the exact artifact paths in the release
   manifest and the contest PR template fields. Do not link private provider
   job pages.
8. **Next-wave roadmap.** Label hidden-gem work as roadmap or research signal:
   HPM1/HPAC parity, native action atoms, HNeRV adapter replay hardening,
   scorer-gradient atoms, byte self-compression, and field-policy waterfill.
   None of these can rank without charged bytes and exact CUDA auth eval.

The notebook should have no executed outputs before publication unless outputs
were regenerated from sanitized release artifacts. It must not contain absolute
operator paths, API keys, private Studio URLs, Vast SSH endpoints, raw `.omx`
state, or job transcripts.

## Cloudflare Site Design

The Cloudflare site remains static and judge-friendly:

- front page: current frontier, score formula, exact-evidence badge, archive
  SHA/bytes, and links to `report.txt`, PR body, notebook, release manifest;
- visuals: byte anatomy, score timeline, hard-pair/atom density, exact negative
  evidence, final stack composition, and generated GIFs
  `comma_comparison.gif` / `comma_comparison_full.gif`;
- appendices: PR100 attribution, C067 attribution, exploit quarantine,
  preflight hardening, deterministic archive construction, month-long research
  process, Grand Council/Skunkworks AI-assisted workflow, meta-Lagrangian
  method, hidden-gem roadmap, and next-tranche roadmap;
- security: keep `_headers` in the output directory with strict static-site
  headers; do not use Pages Functions unless a later feature requires them.

## Publish Gate

Before any public upload:

```bash
.venv/bin/python reports/graphs/build_public_site_bundle.py
.venv/bin/python - <<'PY'
from pathlib import Path
from tac.preflight import check_public_release_hygiene

check_public_release_hygiene(
    repo_root=Path("."),
    strict=True,
    scan_paths=[
        "docs/submission_template.md",
        "docs/runbooks/apogee_public_supplement_20260502.md",
        "notebooks/apogee_lightning_supplement.ipynb",
        "reports/latest.md",
        "reports/writeup_working.md",
        "reports/graphs/public_site",
    ],
)
PY
```

Then rebuild and deploy:

```bash
.venv/bin/python reports/graphs/refresh_site.py
.venv/bin/python reports/graphs/build_public_site_bundle.py
npx wrangler pages deploy reports/graphs/public_site --project-name apogee-comma-video
```

The deploy command must only run after Cloudflare auth is intentionally present
in the operator environment; no token or account identifier belongs in the repo.

## Research Sources

- Contest README and rules:
  https://github.com/commaai/comma_video_compression_challenge
- Contest leaderboard:
  https://comma.ai/leaderboard
- Upstream PR template:
  https://raw.githubusercontent.com/commaai/comma_video_compression_challenge/master/.github/pull_request_template.md
- Lightning Studios overview:
  https://lightning.ai/docs/pytorch/stable/clouds/lightning_ai.html
- Lightning public ports docs:
  https://lightning.ai/docs/overview/ai-studio/deploy-on-public-ports
- Cloudflare Pages Direct Upload:
  https://developers.cloudflare.com/pages/get-started/direct-upload/
- Cloudflare Pages headers:
  https://developers.cloudflare.com/pages/configuration/headers/
- Cloudflare Pages limits:
  https://developers.cloudflare.com/pages/platform/limits/
