# Endgame Archive Decision Profile

- schema: `endgame_archive_decision_profile_v1`
- evidence_grade: `byte_level_decision_support_only`
- score_claim: `False`
- frontier_label: `PR85_STBM1BR`

This report is byte-only. It does not inflate videos, load scorers, dispatch jobs, promote methods, or claim contest score.

## Archives

| label | bytes | sha256 | strict ZIP | decision-valid | side bytes |
|---|---:|---|---|---|---:|
| PR85 | 236328 | eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e | True | True | 0 |
| PR85_STBM1BR | 229756 | c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6 | True | True | 0 |
| PR91_HPM1 | 222404 | 4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f | True | True | 0 |
| PR92_RSB1 | 236516 | f0dedeb7ad3c019ab3f4e2dea317f3192408e47a573897adb208030709d01490 | True | True | 386 |
| STBM1BR_RMB1 | 229480 | f8d2dff12004fe15bdedefcd3f9574fab97f22c302fa1417a265c325468ad774 | True | True | 0 |

## PR85 Segments

| segment | bytes | codec | validation | sha256 |
|---|---:|---|---|---|
| mask | 159011 | QMA9_range_mask | ok | 4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179 |
| model | 57074 | brotli_qh_model | ok | c9194c19d44d3c32c4e588bbe4f9986ed0c562779078f0f4d6b073f0799b5dbc |
| pose | 1487 | brotli_p1d1_pose | ok | 2b6e03e02fb17f662e3f3f9ce47daf2fc33bf1f76c96e34ffac005f1c5c4054c |
| post | 1400 | brotli_pr85_sidechannel | ok | c3dc88e0fb5a1e48aec49eeacefeea20685e3c02598933deb1ff22dc09892575 |
| shift | 226 | brotli_pr85_sidechannel | ok | 48a6ab5972eb594112c5e9c8ef2a35a757a3df818d489c893e30772f148e4dc4 |
| frac | 106 | brotli_pr85_sidechannel | ok | a5039f64850da646535df761447cecd580703d1cc0c56040006ce1a50bc9a992 |
| frac2 | 149 | brotli_pr85_sidechannel | ok | 3a01c7546d89d9d3ca846da37a20add7d456421e9211a6339850d946b856026b |
| frac3 | 154 | brotli_pr85_sidechannel | ok | d725b01b604566c5a76a06b821c644c8d5d5bb9a57ab0597d3bdabc8389858b5 |
| bias | 223 | brotli_pr85_sidechannel | ok | 8910af036b9fa9f9bee2dbf241dede5afc9720ff9e7e5bcef784a7a3b3e4309b |
| region | 273 | brotli_pr85_sidechannel | ok | 4a7c16094fab281c30e27621a58577c095c4b1153c232518c9adbaba2a2b2392 |
| randmulti | 16101 | brotli_pr85_sidechannel | ok | c624372e60a0851c4c427dc333a60dcc5d6657ba8ed56951612e7c9d7be7629f |

## PR85_STBM1BR Segments

| segment | bytes | codec | validation | sha256 |
|---|---:|---|---|---|
| mask | 152439 | STBM1BR_lossless_mask_recode | ok | 1b1ec60b64e284aae11e838dc3d9996bce00125df5712a8ba9c3e8f739c9d313 |
| model | 57074 | brotli_qh_model | ok | c9194c19d44d3c32c4e588bbe4f9986ed0c562779078f0f4d6b073f0799b5dbc |
| pose | 1487 | brotli_p1d1_pose | ok | 2b6e03e02fb17f662e3f3f9ce47daf2fc33bf1f76c96e34ffac005f1c5c4054c |
| post | 1400 | brotli_pr85_sidechannel | ok | c3dc88e0fb5a1e48aec49eeacefeea20685e3c02598933deb1ff22dc09892575 |
| shift | 226 | brotli_pr85_sidechannel | ok | 48a6ab5972eb594112c5e9c8ef2a35a757a3df818d489c893e30772f148e4dc4 |
| frac | 106 | brotli_pr85_sidechannel | ok | a5039f64850da646535df761447cecd580703d1cc0c56040006ce1a50bc9a992 |
| frac2 | 149 | brotli_pr85_sidechannel | ok | 3a01c7546d89d9d3ca846da37a20add7d456421e9211a6339850d946b856026b |
| frac3 | 154 | brotli_pr85_sidechannel | ok | d725b01b604566c5a76a06b821c644c8d5d5bb9a57ab0597d3bdabc8389858b5 |
| bias | 223 | brotli_pr85_sidechannel | ok | 8910af036b9fa9f9bee2dbf241dede5afc9720ff9e7e5bcef784a7a3b3e4309b |
| region | 273 | brotli_pr85_sidechannel | ok | 4a7c16094fab281c30e27621a58577c095c4b1153c232518c9adbaba2a2b2392 |
| randmulti | 16101 | brotli_pr85_sidechannel | ok | c624372e60a0851c4c427dc333a60dcc5d6657ba8ed56951612e7c9d7be7629f |

## PR91_HPM1 Segments

| segment | bytes | codec | validation | sha256 |
|---|---:|---|---|---|
| mask | 145087 | HPM1_hpac_mask | ok | a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc |
| model | 57074 | brotli_qh_model | ok | c9194c19d44d3c32c4e588bbe4f9986ed0c562779078f0f4d6b073f0799b5dbc |
| pose | 1487 | brotli_p1d1_pose | ok | 2b6e03e02fb17f662e3f3f9ce47daf2fc33bf1f76c96e34ffac005f1c5c4054c |
| post | 1400 | brotli_pr85_sidechannel | ok | c3dc88e0fb5a1e48aec49eeacefeea20685e3c02598933deb1ff22dc09892575 |
| shift | 226 | brotli_pr85_sidechannel | ok | 48a6ab5972eb594112c5e9c8ef2a35a757a3df818d489c893e30772f148e4dc4 |
| frac | 106 | brotli_pr85_sidechannel | ok | a5039f64850da646535df761447cecd580703d1cc0c56040006ce1a50bc9a992 |
| frac2 | 149 | brotli_pr85_sidechannel | ok | 3a01c7546d89d9d3ca846da37a20add7d456421e9211a6339850d946b856026b |
| frac3 | 154 | brotli_pr85_sidechannel | ok | d725b01b604566c5a76a06b821c644c8d5d5bb9a57ab0597d3bdabc8389858b5 |
| bias | 223 | brotli_pr85_sidechannel | ok | 8910af036b9fa9f9bee2dbf241dede5afc9720ff9e7e5bcef784a7a3b3e4309b |
| region | 273 | brotli_pr85_sidechannel | ok | 4a7c16094fab281c30e27621a58577c095c4b1153c232518c9adbaba2a2b2392 |
| randmulti | 16101 | brotli_pr85_sidechannel | ok | c624372e60a0851c4c427dc333a60dcc5d6657ba8ed56951612e7c9d7be7629f |

## PR92_RSB1 Segments

| segment | bytes | codec | validation | sha256 |
|---|---:|---|---|---|
| mask | 159011 | QMA9_range_mask | ok | 4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179 |
| model | 57074 | brotli_qh_model | ok | c9194c19d44d3c32c4e588bbe4f9986ed0c562779078f0f4d6b073f0799b5dbc |
| pose | 1487 | brotli_p1d1_pose | ok | 2b6e03e02fb17f662e3f3f9ce47daf2fc33bf1f76c96e34ffac005f1c5c4054c |
| post | 1400 | brotli_pr85_sidechannel | ok | c3dc88e0fb5a1e48aec49eeacefeea20685e3c02598933deb1ff22dc09892575 |
| shift | 226 | brotli_pr85_sidechannel | ok | 48a6ab5972eb594112c5e9c8ef2a35a757a3df818d489c893e30772f148e4dc4 |
| frac | 106 | brotli_pr85_sidechannel | ok | a5039f64850da646535df761447cecd580703d1cc0c56040006ce1a50bc9a992 |
| frac2 | 149 | brotli_pr85_sidechannel | ok | 3a01c7546d89d9d3ca846da37a20add7d456421e9211a6339850d946b856026b |
| frac3 | 154 | brotli_pr85_sidechannel | ok | d725b01b604566c5a76a06b821c644c8d5d5bb9a57ab0597d3bdabc8389858b5 |
| bias | 223 | brotli_pr85_sidechannel | ok | 8910af036b9fa9f9bee2dbf241dede5afc9720ff9e7e5bcef784a7a3b3e4309b |
| region | 273 | brotli_pr85_sidechannel | ok | 4a7c16094fab281c30e27621a58577c095c4b1153c232518c9adbaba2a2b2392 |
| randmulti | 15825 | RMB1_bitmask_randmulti | ok | 4b10018eab64d8755da3def355881f51f2d450c9f19dd1457a6ec813cddd6f7c |

### Side Info

| member | bytes | codec | validation | sha256 |
|---|---:|---|---|---|
| a | 386 | RSB1_router_side_actions | ok | 5422d47b4092e7304649cc49f4d4c8c7efa9c3d5a4fc7d39ab63cf2518e0897e |

## STBM1BR_RMB1 Segments

| segment | bytes | codec | validation | sha256 |
|---|---:|---|---|---|
| mask | 152439 | STBM1BR_lossless_mask_recode | ok | 1b1ec60b64e284aae11e838dc3d9996bce00125df5712a8ba9c3e8f739c9d313 |
| model | 57074 | brotli_qh_model | ok | c9194c19d44d3c32c4e588bbe4f9986ed0c562779078f0f4d6b073f0799b5dbc |
| pose | 1487 | brotli_p1d1_pose | ok | 2b6e03e02fb17f662e3f3f9ce47daf2fc33bf1f76c96e34ffac005f1c5c4054c |
| post | 1400 | brotli_pr85_sidechannel | ok | c3dc88e0fb5a1e48aec49eeacefeea20685e3c02598933deb1ff22dc09892575 |
| shift | 226 | brotli_pr85_sidechannel | ok | 48a6ab5972eb594112c5e9c8ef2a35a757a3df818d489c893e30772f148e4dc4 |
| frac | 106 | brotli_pr85_sidechannel | ok | a5039f64850da646535df761447cecd580703d1cc0c56040006ce1a50bc9a992 |
| frac2 | 149 | brotli_pr85_sidechannel | ok | 3a01c7546d89d9d3ca846da37a20add7d456421e9211a6339850d946b856026b |
| frac3 | 154 | brotli_pr85_sidechannel | ok | d725b01b604566c5a76a06b821c644c8d5d5bb9a57ab0597d3bdabc8389858b5 |
| bias | 223 | brotli_pr85_sidechannel | ok | 8910af036b9fa9f9bee2dbf241dede5afc9720ff9e7e5bcef784a7a3b3e4309b |
| region | 273 | brotli_pr85_sidechannel | ok | 4a7c16094fab281c30e27621a58577c095c4b1153c232518c9adbaba2a2b2392 |
| randmulti | 15825 | RMB1_bitmask_randmulti | ok | 4b10018eab64d8755da3def355881f51f2d450c9f19dd1457a6ec813cddd6f7c |

## Frontier Comparisons

### PR85 vs PR85_STBM1BR

- archive delta bytes: `6572`
- primary member delta bytes: `6572`
- side-info delta bytes: `0`
- ZIP overhead delta bytes: `0`

| segment | delta bytes | frontier codec | candidate codec | validation |
|---|---:|---|---|---|
| mask | 6572 | STBM1BR_lossless_mask_recode | QMA9_range_mask | ok |

#### Transplant Estimates

| surface | est. delta bytes | side info | advice |
|---|---:|---|---|
| mask | 6572 | False | do_not_dispatch_rate_only |
### PR91_HPM1 vs PR85_STBM1BR

- archive delta bytes: `-7352`
- primary member delta bytes: `-7352`
- side-info delta bytes: `0`
- ZIP overhead delta bytes: `0`

| segment | delta bytes | frontier codec | candidate codec | validation |
|---|---:|---|---|---|
| mask | -7352 | STBM1BR_lossless_mask_recode | HPM1_hpac_mask | ok |

#### Transplant Estimates

| surface | est. delta bytes | side info | advice |
|---|---:|---|---|
| mask | -7352 | False | byte_positive_but_requires_contract_gates |
### PR92_RSB1 vs PR85_STBM1BR

- archive delta bytes: `6760`
- primary member delta bytes: `6296`
- side-info delta bytes: `386`
- ZIP overhead delta bytes: `78`

| segment | delta bytes | frontier codec | candidate codec | validation |
|---|---:|---|---|---|
| mask | 6572 | STBM1BR_lossless_mask_recode | QMA9_range_mask | ok |
| randmulti | -276 | brotli_pr85_sidechannel | RMB1_bitmask_randmulti | ok |

#### Transplant Estimates

| surface | est. delta bytes | side info | advice |
|---|---:|---|---|
| randmulti | 188 | True | do_not_dispatch_rate_only |
| mask | 6572 | False | do_not_dispatch_rate_only |
### STBM1BR_RMB1 vs PR85_STBM1BR

- archive delta bytes: `-276`
- primary member delta bytes: `-276`
- side-info delta bytes: `0`
- ZIP overhead delta bytes: `0`

| segment | delta bytes | frontier codec | candidate codec | validation |
|---|---:|---|---|---|
| randmulti | -276 | brotli_pr85_sidechannel | RMB1_bitmask_randmulti | ok |

#### Transplant Estimates

| surface | est. delta bytes | side info | advice |
|---|---:|---|---|
| randmulti | -276 | False | byte_positive_but_requires_contract_gates |

## Ranked Actions

| candidate | surface | est. delta bytes | advice | blockers |
|---|---|---:|---|---|
| PR91_HPM1 | mask | -7352 | byte_positive_but_requires_contract_gates | mask_codec_runtime_or_parity_change_required |
| STBM1BR_RMB1 | randmulti | -276 | byte_positive_but_requires_contract_gates |  |
| PR92_RSB1 | randmulti+side_info | 188 | do_not_dispatch_rate_only | rate_only_not_byte_positive |
