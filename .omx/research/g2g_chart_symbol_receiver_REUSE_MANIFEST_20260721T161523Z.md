# G2g chart-symbol receiver — REUSE MANIFEST

| Required surface | Disposition | Exact use |
|---|---|---|
| `predictor_upgrade_xi_chart.py` | EXTENDED IN PLACE | Added G2CS1 packet, strict parser, decoded LaneLine coefficient application, and replay receipt; no fork. |
| Openpilot lane chart | REUSED | Hash-pinned LBND2 base `d2b2a62e...`; 121,128 total base counted bytes; centerline coefficient is the actuator. |
| #402 fail-closed receiver | REUSED/EXTENDED | Exact consumption, canonical re-encode, CRC, address validation, and independent double decode. |
| `predict_project_receiver` | REUSED | Exact-R realization, receiver/scorer input custody, and native frozen hard-oracle path. |
| Seed coder/accounting discipline | REUSED | Actual emitted G2CS1 payload length is charged; no pixel estimate and no receiver LOC is counted. Existing seed schema has no LaneLine pair/line/coeff address, so the minimal typed extension was required. |
| #549 solved target cells/tube | REUSED | Frozen target cells and declared pose tube are the hard admission authority. |
| G2f candidates | REUSED WITHOUT REOPENING TRUST | Chart-only `[0,34,37,46]` first; overlap `[22,30]` second; both signs at the selected rung. |
| Motion constants | REUSED BY LAWREF | `dsl_custodied_scalar_identity_v1` plus hash-pinned G1 motion receipt; no invented flags or constants. |
| Rate threshold | REUSED BY LAWREF | `realization_breakeven_bytes_v1`, numerically `25/37,545,489`. |

## New-code failed-search justification

Repository search found no existing counted packet that addresses a decoded lane coefficient by pair, line, and coefficient index. The existing seed packet cannot express that actuator, and charging a raster/pixel proxy would violate the requested alphabet. `G2CS1` is therefore the smallest required extension: a generic free decoder and a video-derived counted delta payload. The established G2 measurement CLI was extended rather than forked.

No receiver code, rasterizer table, homography code, scorer state, target cells, or per-frame RGB table is charged or smuggled in the packet. Only the video-specific coefficient address/delta rows are counted. `MAIN_REVIEW_REQUIRED=true`.
