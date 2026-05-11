# PR106 r2 + PR101 grammar exact-CUDA validation (Codex adversarial review, 2026-05-11)

## Verdict

**Validated as an internal contest-CUDA score claim on the CUDA axis.**

The landed `submissions/pr106_latent_sidecar_r2_pr101_grammar/` packet is a
rate-only rewrite of the PR106 r2 latent sidecar:

- archive SHA-256:
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`
- archive bytes: `186780`
- single ZIP member: `0.bin`, stored, `186672` bytes, member SHA-256
  `8c9f259e497033e0e22971ebd9f3b899fe644ae234998664f3edc3a790706b0e`
- auth-eval axis: `cuda`
- hardware: Linux x86_64 Tesla T4, `gpu_t4_match=true`
- samples: `600`
- canonical score from auth-eval JSON: `0.2066181354574151`
- strict formula recompute from components and exact archive bytes:
  `0.20661802072157426`
- absolute recompute delta: `1.147e-7`

The small score-value delta is below decision relevance and is explained by the
auth-eval record's rounded component fields. The rate term is exact from charged
archive bytes.

## No-op / payload-closure proof

Local parser-only verification on the committed archives:

```text
new_format 0x2 pr106_len 186131 side_len 527 meta_len 6
new_corrections 600
new dim_sha a55bd9bdecb4a462e166cf0f8363bcb5578e8be61339a97639859173aa5098c3
new dq_sha  616ee8fbee1e4770ee5574372efc8637fbf8b6f4c450c3c4c9cd47ee20ad6a53

base_format 0x1 pr106_len 186131 side_len 575
base_corrections 600
base dim_sha a55bd9bdecb4a462e166cf0f8363bcb5578e8be61339a97639859173aa5098c3
base dq_sha  616ee8fbee1e4770ee5574372efc8637fbf8b6f4c450c3c4c9cd47ee20ad6a53

pr106_base_bytes_match True
dims_match True
delta_q_match True
```

This confirms the scored bytes differ through the sidecar grammar, while the
PR106 base payload and decoded correction arrays match the `format_id=0x01`
baseline. The observed CUDA movement is therefore the predicted rate-only
change:

```text
42 bytes * 25 / 37,545,489 = 0.00002797 score points
```

## Compliance / custody

The strict pre-submission compliance rerun passed with:

```text
scripts/pre_submission_compliance_check.py ... --contest-final --strict
```

Important custody nuance: the full Modal runtime tree SHA in
`contest_auth_eval.json` includes Modal-side custody files such as `report.txt`.
The committed `report.txt` is a human report and does **not** have the same SHA
as the Modal runtime-manifest `report.txt`; this is expected and is accepted by
the strict gate through the portable runtime hash:

```text
portable_runtime_tree_sha256_without_custody_files =
c7072669a58c5865ad9db57a382800243bb221b1eaea82e3f62c60c6198ebaf6
```

The executable runtime files do match the auth-eval manifest exactly:

```text
inflate.py          60055bced3ab608d0e93ba83e18fa5bc662746cfa273ad50d5960c34028d1fb3
inflate.sh          bfcceb491c01a97c1f5ee46919abd3ce921040a1e0b7d431eb2fbe8184369fe4
src/codec.py        2abb3ae103e92f65677e8ad0178b9015651ba58f4d5364e59b396aef9fe8dda1
src/model.py        e63b04ad3df4942b9bc1e31afd8ec84177dfbe83827f67cf7c5a682b05c1b46b
src/pr101_grammar.py 04c9fb70172aa3e7013fc17f584a91df48d6e521bf52cf8b726c803585d054c4
```

## Axis interpretation

This result must not be used to infer that CUDA is universally better or CPU is
universally better. The device-axis matrix now shows packet-specific behavior:

- A1 / PR101-derived score-gradient cluster: CPU wins.
- PR106 latent-sidecar cluster: CUDA wins.

Each submission family needs paired CPU/CUDA evidence before ranking,
submission-policy review, or kill decisions. This result validates the PR106
CUDA axis only; the paired CPU row for the `format_id=0x02` archive remains a
follow-up unless another exact CPU artifact lands.

## Remaining risks

- `archive.zip` is intentionally ignored by git (`submissions/*/archive.zip`);
  local custody is through the manifest and auth-eval JSON. External release
  needs a hosted artifact manifest if this packet is published.
- The exact score is **not** a public leaderboard claim by itself. The auth-eval
  JSON marks `promotion_eligible=false`; public submission remains policy-gated.
- Runtime source should not be edited after this score without rerunning exact
  CUDA, because any executable-runtime hash change invalidates the sealed
  custody packet.

