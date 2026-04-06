# frontend audit

## design direction

- understated, dark, editorial layout
- system-font stack only
- no client-side framework
- no runtime JavaScript on the page body
- simple information hierarchy: headline -> frontier -> evidence -> narrative

## accessibility

- semantic landmarks (`header`, `main`, sections, tables)
- skip link present
- focus-visible styles present
- reduced-motion support present
- color contrast intentionally high in the dark palette
- no motion-heavy hero / no autoplay / no hidden interaction traps

## performance

- static HTML output
- no application JS bundle
- no external fonts
- no tracking scripts
- small number of static assets
- short cache on `index.html`, longer cache on data artifacts
- site bundle built under `reports/graphs/site/`

## security

- CSP set in `_headers`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- strict referrer policy
- permissions policy strips camera/microphone/geolocation
- COOP / CORP headers present

## seo

- canonical URL set to `https://comma-lab.pages.dev/`
- title and description metadata present
- Open Graph metadata present
- schema.org `WebSite` JSON-LD present
- crawlable static HTML

## caveats

- no Lighthouse trace was captured in this cycle
- Cloudflare deployment verification is Wrangler-confirmed rather than shell-HTTP-verified from this machine
