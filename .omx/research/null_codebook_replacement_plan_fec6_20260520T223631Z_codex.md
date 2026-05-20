# Null-Space Seed Replacement Plan

- Schema: `null_space_seed_replacement_plan_v1`
- Score claim: `false`
- Promotion eligible: `false`
- Input bytes: `178417`
- Null bytes: `16292`
- Seed bytes per candidate: `8`
- Runtime header bytes per candidate: `0`
- Candidate count: `2`
- Positive candidates: `2`
- Best net saved inner bytes: `16238`
- Greedy disjoint rate-delta upper bound: `-0.010812217681`

| rank | source | section | range | original bytes | net saved | rate delta upper bound |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `contiguous_null_run` | `source_payload+selector_len_hdr+selector_payload` | `162171-178417` | 16246 | 16238 | -0.010812217681 |
| 2 | `whole_null_section` | `selector_payload` | `178168-178417` | 249 | 241 | -0.000160472008 |

This is not a score artifact. It is a packet-compiler target list for
archive-charged seed replacement candidates.
