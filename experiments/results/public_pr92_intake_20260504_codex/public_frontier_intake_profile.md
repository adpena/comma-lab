# Public Frontier Archive Intake

- label: `PR92_qzs3_range_joint_r258`
- evidence_grade: `external_archive_byte_intake_only`
- score_claim: `False`
- archive bytes: `236516`
- archive sha256: `f0dedeb7ad3c019ab3f4e2dea317f3192408e47a573897adb208030709d01490`
- strict ZIP valid: `True`
- side-info bytes: `386`

This report is byte-only. It does not inflate videos, load scorers, dispatch jobs, promote methods, or claim a contest score.

## Members

| name | bytes | compressed | local name match | sha256 |
|---|---:|---:|---|---|
| x | 235952 | 235952 | True | 89f14d331063125c88db0b4e3e51a92f21d2edc175a64d5c9cb6f873130763d8 |
| a | 386 | 386 | True | 5422d47b4092e7304649cc49f4d4c8c7efa9c3d5a4fc7d39ab63cf2518e0897e |

## Primary PR85-Family Bundle

- member: `x`
- format: `pr85_v5_micro_24bit_lengths_fixed_bias_region`

| segment | bytes | codec | sha256 |
|---|---:|---|---|
| mask | 159011 | QMA9_range_mask | 4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179 |
| model | 57074 | opaque_pr85_segment | c9194c19d44d3c32c4e588bbe4f9986ed0c562779078f0f4d6b073f0799b5dbc |
| pose | 1487 | opaque_pr85_segment | 2b6e03e02fb17f662e3f3f9ce47daf2fc33bf1f76c96e34ffac005f1c5c4054c |
| post | 1400 | opaque_pr85_segment | c3dc88e0fb5a1e48aec49eeacefeea20685e3c02598933deb1ff22dc09892575 |
| shift | 226 | opaque_pr85_segment | 48a6ab5972eb594112c5e9c8ef2a35a757a3df818d489c893e30772f148e4dc4 |
| frac | 106 | opaque_pr85_segment | a5039f64850da646535df761447cecd580703d1cc0c56040006ce1a50bc9a992 |
| frac2 | 149 | opaque_pr85_segment | 3a01c7546d89d9d3ca846da37a20add7d456421e9211a6339850d946b856026b |
| frac3 | 154 | opaque_pr85_segment | d725b01b604566c5a76a06b821c644c8d5d5bb9a57ab0597d3bdabc8389858b5 |
| bias | 223 | opaque_pr85_segment | 8910af036b9fa9f9bee2dbf241dede5afc9720ff9e7e5bcef784a7a3b3e4309b |
| region | 273 | opaque_pr85_segment | 4a7c16094fab281c30e27621a58577c095c4b1153c232518c9adbaba2a2b2392 |
| randmulti | 15825 | RMB1_side_info_backed_randmulti | 4b10018eab64d8755da3def355881f51f2d450c9f19dd1457a6ec813cddd6f7c |

## Charged Side Info

| member | bytes | magic | sha256 |
|---|---:|---|---|
| a | 386 | RSB1X... | 5422d47b4092e7304649cc49f4d4c8c7efa9c3d5a4fc7d39ab63cf2518e0897e |

## Baseline Diffs

### PR85

- archive delta bytes: `188`
- rate score delta: `0.00012518148318696823`

| segment | delta bytes | baseline codec | candidate codec | same sha256 |
|---|---:|---|---|---|
| randmulti | -276 | opaque_pr85_segment | RMB1_side_info_backed_randmulti | False |
### PR85_STBM1BR

- archive delta bytes: `6760`
- rate score delta: `0.004501206523105879`

| segment | delta bytes | baseline codec | candidate codec | same sha256 |
|---|---:|---|---|---|
| mask | 6572 | STBM1BR_lossless_mask_recode | QMA9_range_mask | False |
| randmulti | -276 | opaque_pr85_segment | RMB1_side_info_backed_randmulti | False |
