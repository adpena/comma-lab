# Cloudflare Pages Deployment

This is the public writeup/supplement surface for Apogee. It is not a score
authority and must not include private provider state, local absolute paths,
secrets, raw `.omx/state`, or unredacted job URLs.

## Pages-Ready Directory

The internal site bundle is generated into:

- `reports/graphs/site/`

Do not deploy that directory directly. First build the sanitized public bundle:

```bash
.venv/bin/python reports/graphs/build_public_site_bundle.py
```

The deployable bundle is:

- `reports/graphs/public_site/`

That directory contains:

- `index.html`
- `apogee_release_manifest.json`
- dashboard data/json assets
- markdown/source packet files
- `_headers`
- `_redirects`

Cloudflare's Direct Upload path accepts a prebuilt folder through Wrangler or a
folder/zip through dashboard upload. Wrangler deploys a folder, not a zip, and
the deployed project is served at `<PROJECT_NAME>.pages.dev`.

## Prepublish Hygiene

Run the strict publish-surface guard before upload:

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

## Deploy With Wrangler

If Cloudflare auth is already configured in this environment:

```bash
npx wrangler pages deploy reports/graphs/public_site --project-name apogee-comma-video
```

Use preview branches for dry runs:

```bash
npx wrangler pages deploy reports/graphs/public_site \
  --project-name apogee-comma-video \
  --branch preview
```

## Deploy Manually

You can also upload the contents of `reports/graphs/public_site/` to a
Cloudflare Pages project via the dashboard.

## Current Constraints

- The public bundle builder omits assets over Cloudflare Pages' current 25 MiB
  asset limit and records them in `public_site_manifest.json`; replace omitted
  media with smaller GIF/MP4 clips or separate public release assets before the
  final judges' polish pass.
- Keep `_headers` in the final output directory. It currently sets conservative
  static-site security headers and cache rules.
- Keep all final public URLs in a sanitized release manifest:
  `reports/graphs/public_site/apogee_release_manifest.json` until
  `${APOGEE_RELEASE_MANIFEST}` is intentionally published.
- This deployment package is for the **Best Write-up** and reproducibility
  supplement tracks. It does not change the authoritative compression-evaluation
  path.
