# Apogee Public Supplement And Submission Naming Plan - 2026-05-02 Codex

## Decision

Use `Apogee` as the PR-facing submission name and `apogee` as the lower-case
artifact/deployment prefix. The public supplement is split into two surfaces:

- Lightning.ai notebook/app supplement for interactive, GPU-capable, executable
  explanation and artifact inspection.
- Cloudflare Pages static site for fast judge-facing narrative, figures, and
  release-manifest links.

Neither surface is a score authority. The only score authority remains exact
CUDA/T4-equivalent auth eval of the exact archive bytes through
`archive.zip -> inflate.sh -> upstream/evaluate.py`.

## Online Research Refresh

Contest docs checked on 2026-05-02:

- The public PR must include a download link to `archive.zip`, `inflate.sh`,
  `report.txt`, GPU-required answer, optional compression script, and comments.
- The official challenge states a 30 minute evaluation limit; GPU inflation
  runs on T4-class hardware.
- Large score-affecting artifacts must be charged in `archive.zip`; external
  tools are allowed, but model/mesh/point-cloud-like payloads cannot be free.
- The current visible leaderboard top row is `qpose14` at rounded `0.32`.
  Apogee/C067 is currently an internal exact A++ result until public PR
  submission.

Hosting docs checked on 2026-05-02:

- Lightning Studios are appropriate for cloud notebooks, GPU experiments, and
  hosted apps; public links must be sanitized and intentionally published.
- Cloudflare Pages Direct Upload supports prebuilt assets through Wrangler or
  dashboard upload. Wrangler deploys a folder. Current Pages single-asset limit
  is 25 MiB, so public media must be compressed/split or hosted elsewhere.
- The existing generated `reports/graphs/site/` bundle is an internal site
  build, not a publish-safe bundle; strict hygiene found historical local
  absolute paths in generated timeline/report JSON. The deployable artifact is
  now `reports/graphs/public_site/` after sanitizer build.
- Cloudflare `_headers` can set static-site security and cache headers from the
  build output directory.

## Naming Conventions

- PR title: `Apogee - contest-faithful scorer-aligned video compression`
- PR submission name: `Apogee`
- archive URL placeholder: `${APOGEE_ARCHIVE_ZIP_URL}`
- report artifact: final exact T4/equivalent `report.txt`
- notebook path: `notebooks/apogee_lightning_supplement.ipynb`
- Cloudflare project: `apogee-comma-video`
- Cloudflare deploy directory: `reports/graphs/public_site/`
- public release manifest placeholder: `${APOGEE_RELEASE_MANIFEST}`
- source public site bundle: `reports/graphs/site/`

## Public Hygiene Gate

The durable guard is now strict-publish-surface first:

```bash
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
        "reports/graphs/site",
    ],
)
PY
```

This scan intentionally excludes raw `.omx/state`, provider logs, and
forensic custody files. Those remain private unless sanitized into a public
release manifest.

## Notebook Contents

The Lightning notebook should be executable from sanitized public artifacts and
should fail closed when the manifest is absent or component math does not
recompute. Sections:

1. exact frontier card;
2. score formula recomputation;
3. byte anatomy plot;
4. attribution and exploit quarantine;
5. exact negative evidence map;
6. atom-waterfill / Lagrangian floor explanation;
7. reproduction checklist and release links.

## Implementation Landed

- `docs/submission_template.md` now mirrors the upstream PR template fields and
  uses Apogee placeholders.
- `docs/runbooks/apogee_public_supplement_20260502.md` records the public
  supplement plan and source-backed hosting constraints.
- `reports/graphs/deploy_cloudflare_pages.md` now uses the Apogee project name,
  Cloudflare Direct Upload constraints, and strict hygiene scan.
- `reports/graphs/build_public_site_bundle.py` now copies
  `reports/graphs/site/` to `reports/graphs/public_site/`, redacts local paths,
  private Lightning links, Vast endpoints, and token-like strings, enforces the
  25 MiB asset limit, and runs the strict public-release hygiene guard on the
  sanitized bundle.
- `src/tac/preflight.py` now includes `notebooks` in the default public release
  scan surface.
- `notebooks/apogee_lightning_supplement.ipynb` is a sanitized notebook
  skeleton with no executed outputs and no private provider links.

## Publish-Bundle Greenup

`reports/graphs/build_public_site_bundle.py` was run after the sanitizer landed:

- output: `reports/graphs/public_site/`
- files: `74`
- redactions: `474`
- strict public-release hygiene violations after sanitize: `0`
- omitted oversized assets: `comparison/comparison.gif`, `114724141` bytes

The omitted asset exceeds Cloudflare Pages' current 25 MiB single-asset limit
and must be replaced by a smaller clip or separate release asset during the
final judges' polish pass.

## Sources

- https://github.com/commaai/comma_video_compression_challenge
- https://comma.ai/leaderboard
- https://raw.githubusercontent.com/commaai/comma_video_compression_challenge/master/.github/pull_request_template.md
- https://lightning.ai/docs/pytorch/stable/clouds/lightning_ai.html
- https://lightning.ai/docs/overview/ai-studio/deploy-on-public-ports
- https://developers.cloudflare.com/pages/get-started/direct-upload/
- https://developers.cloudflare.com/pages/configuration/headers/
- https://developers.cloudflare.com/pages/platform/limits/
