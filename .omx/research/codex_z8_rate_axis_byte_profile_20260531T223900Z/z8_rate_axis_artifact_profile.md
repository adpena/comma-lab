# Z8 Rate-Axis Byte Profile

Byte-only advisory profile. No scorer or promotion authority.

## Contest Rate Targets

| Rate term | Archive bytes |
|---:|---:|
| 0.05 | 75,090 |
| 0.10 | 150,181 |
| 0.20 | 300,363 |
| 1.00 | 1,501,819 |

## Artifact Ranking

| Label | paid bytes | paid rate term | 0.bin bytes | dominant section | wavelet share | zip custody |
|---|---:|---:|---:|---|---:|---|
| quantized_inner_original | 10,195,155 | 6.7885 | 10,195,155 | wavelet_blob | 99.71% | None |
| repaired_quantized | 10,289,674 | 6.8515 | 10,195,155 | wavelet_blob | 99.71% | True |
| q0156_valid_zip | 23,376,927 | 15.5657 | 23,278,362 | wavelet_blob | 99.87% | True |

## Largest Sections

### repaired_quantized

| Section | bytes | share of 0.bin | rate term if direct |
|---|---:|---:|---:|
| wavelet_blob | 10,165,099 | 99.71% | 6.7685 |
| wyner_ziv_blob | 22,804 | 0.22% | 0.0152 |
| indices_blob | 5,400 | 0.05% | 0.0036 |
| meta_blob | 1,725 | 0.02% | 0.0011 |
| z8hpc1_header | 62 | 0.00% | 0.0000 |
| decoder_blob | 34 | 0.00% | 0.0000 |
| dreamer_state_blob | 31 | 0.00% | 0.0000 |

Detail codec methods: {"qi16_zero_rle": 7200}

### q0156_valid_zip

| Section | bytes | share of 0.bin | rate term if direct |
|---|---:|---:|---:|
| wavelet_blob | 23,248,258 | 99.87% | 15.4801 |
| wyner_ziv_blob | 22,804 | 0.10% | 0.0152 |
| indices_blob | 5,400 | 0.02% | 0.0036 |
| meta_blob | 1,773 | 0.01% | 0.0012 |
| z8hpc1_header | 62 | 0.00% | 0.0000 |
| decoder_blob | 34 | 0.00% | 0.0000 |
| dreamer_state_blob | 31 | 0.00% | 0.0000 |

Detail codec methods: {"qi16_constriction_range": 5974, "zigzag_u16_byteplane": 1226}

### quantized_inner_original

| Section | bytes | share of 0.bin | rate term if direct |
|---|---:|---:|---:|
| wavelet_blob | 10,165,099 | 99.71% | 6.7685 |
| wyner_ziv_blob | 22,804 | 0.22% | 0.0152 |
| indices_blob | 5,400 | 0.05% | 0.0036 |
| meta_blob | 1,725 | 0.02% | 0.0011 |
| z8hpc1_header | 62 | 0.00% | 0.0000 |
| decoder_blob | 34 | 0.00% | 0.0000 |
| dreamer_state_blob | 31 | 0.00% | 0.0000 |

Detail codec methods: {"qi16_zero_rle": 7200}

## Opportunities

| Priority | Surface | Finding |
|---:|---|---|
| 0 | contest_zip_bytes_are_authority | Rank candidates by archive.zip bytes when a valid sibling ZIP exists; inner 0.bin bytes can be the wrong objective when outer ZIP exploits structured zeros or is blocked by already-random entropy-coded payloads. |
| 0 | z8_rate_gap_to_competitive_range | The best custody-valid Z8 ZIP is still far above the byte range where rate stops dominating. Incremental entropy-mode polishing cannot close this alone; the next lever must remove or generate most residual wavelet coefficients, not merely recode them. |
| 0 | smallest_inner_packet_missing_zip_receiver_proof | The smallest observed inner Z8 packet is not yet a custody-valid contest ZIP in this profile. Highest-EV next action is package, inflate-proof, and full-replay it before inventing a new coder. |
| 1 | wavelet_blob_dominates_0bin | The Z8 rate axis is still a wavelet-pyramid byte problem; decoder, Dreamer state, indices, and meta are second-order. |
| 1 | wavelet_blob_dominates_0bin | The Z8 rate axis is still a wavelet-pyramid byte problem; decoder, Dreamer state, indices, and meta are second-order. |
| 1 | wavelet_blob_dominates_0bin | The Z8 rate axis is still a wavelet-pyramid byte problem; decoder, Dreamer state, indices, and meta are second-order. |
| 2 | runtime_payload_overhead_future_dominates | The Python runtime bundle is small relative to today's wavelet blob, but it is already larger than a 0.05 rate-term budget. Once residual bytes collapse, runtime tree-shaking or a thinner adapter becomes first-order. |
| 2 | runtime_payload_overhead_future_dominates | The Python runtime bundle is small relative to today's wavelet blob, but it is already larger than a 0.05 rate-term budget. Once residual bytes collapse, runtime tree-shaking or a thinner adapter becomes first-order. |
