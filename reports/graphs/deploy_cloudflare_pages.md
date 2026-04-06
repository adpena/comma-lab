# cloudflare pages deployment

## pages-ready directory

The static site bundle is generated into:

- `reports/graphs/site/`

That directory contains:

- `index.html`
- dashboard data/json assets
- markdown/source packet files
- `_headers`
- `_redirects`

## deploy with wrangler

If Cloudflare auth is already configured in this environment:

```bash
npx wrangler pages deploy reports/graphs/site --project-name comma-lab
```

## deploy manually

You can also upload the contents of `reports/graphs/site/` to a Cloudflare Pages project via the dashboard.

## note

This deployment package is for the **Best Write-up** track. It does not change the authoritative compression-evaluation path.
