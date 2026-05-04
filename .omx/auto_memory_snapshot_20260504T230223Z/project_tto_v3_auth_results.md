---
name: TTO v3 Auth Eval — Score 0.7400 (embedding loss frames)
description: Auth eval on v3 TTO frames. 0.74 auth. Rate term added 0.10. Proxy-auth gap confirmed.
type: project
originSessionId: 47bf3dd8-df75-4271-9ce1-428c19c2eb32
---
## TTO v3 Auth Eval (2026-04-15)
- Auth score: **0.7400**
- PoseNet distortion: 0.01830 (contribution: 0.4278)
- SegNet distortion: 0.00213 (contribution: 0.2127)  
- Rate: 0.00401 (contribution: 0.1004)
- Proxy was: 0.6219 (no rate term)

## Key findings
- Proxy-to-auth gap: 0.12 (0.62 proxy → 0.74 auth)
- The gap is almost entirely from RATE (0.1004) — archive.zip adds bytes
- Rate contribution formula: 25 * (archive_bytes / gt_bytes) = 25 * 0.00401 = 0.1004
- Archive: 150,715 bytes, GT: 37,545,489 bytes

## v3 vs v1 comparison
- v3 auth: 0.7400 (embedding loss, worse proxy)
- v1 auth: STILL UNKNOWN (proxy was 0.5896, but auth eval crashed twice)
- Renderer-only auth: 0.87

## Next
- Get v1 auth eval (better proxy, should have better auth)
- The GT-sparse TTO path bypasses rate entirely (start from GT, no archive needed)
