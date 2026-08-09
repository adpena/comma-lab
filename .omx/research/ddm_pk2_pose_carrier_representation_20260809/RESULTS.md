# DDM PK2 pose-carrier representation results

All numbers below are [macOS-CPU advisory], `score_claim=false`. The exact contest pointer did not move.

## Measured result

Best n120 row: `baseline_cpr1_int5` at 191052 B, d_pose 2.01484579947e-05, d_seg 0.00028962030774, delta S +0.
Composed A+B row: 195844 B, d_pose 0.00367187695892, d_seg 0.00028962030774, delta S +0.180617692269.

## Ranked scorer table

| rank | row | arm | archive B | delta B | d_pose | d_seg | delta S | scope |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `baseline_cpr1_int5` | control | 191052 | +0 | 2.0148458e-05 | 0.000289620308 | +0 | MEASURED_CONTROL |
| 2 | `coeff_lowrank_r04_q06_res01` | A_low_rank_plus_residual | 195368 | +4316 | 2.0148458e-05 | 0.000289620308 | +0.00287384724 | MEASURED_OPTIMAL_FORM_PACKET |
| 3 | `coeff_lowrank_r06_q06_res04` | A_low_rank_plus_residual | 194860 | +3808 | 4.03691302e-05 | 0.000289620308 | +0.00843313512 | MEASURED_OPTIMAL_FORM_PACKET |
| 4 | `coeff_lowrank_r04_q06_res04` | A_low_rank_plus_residual | 193924 | +2872 | 4.81344593e-05 | 0.000289620308 | +0.00965738734 | MEASURED_OPTIMAL_FORM_PACKET |
| 5 | `coeff_lowrank_r04_q06_res16` | A_low_rank_plus_residual | 191644 | +592 | 0.000156857916 | 0.000289620308 | +0.0258049541 | MEASURED_OPTIMAL_FORM_PACKET |
| 6 | `coeff_lowrank_r06_q06_res16` | A_low_rank_plus_residual | 192364 | +1312 | 0.000175874001 | 0.000289620308 | +0.0286164148 | MEASURED_OPTIMAL_FORM_PACKET |
| 7 | `basis_dct_keep75` | B_spatial_DCT | 191028 | -24 | 0.00225570165 | 0.000289620308 | +0.135979428 | TOY_BRACKET_FULL_ARRAY_CPR1 |
| 8 | `basis_perplane_int4_p99_5` | B_per_plane_precision | 191544 | +492 | 0.00367187696 | 0.000289620308 | +0.177754499 | MEASURED_FORMULATION |
| 9 | `compose_best_measured_A_B` | A_plus_B_composed_optimal_packet | 195844 | +4792 | 0.00367187696 | 0.000289620308 | +0.180617692 | MEASURED_OPTIMAL_FORM_PACKET_COMPOSITION |
| 10 | `basis_plane_rank24` | B_cross_plane_rank | 191256 | +204 | 0.0042835371 | 0.000289620308 | +0.192908386 | TOY_BRACKET_FULL_ARRAY_CPR1 |
| 11 | `basis_int4_p99_5` | B_precision | 189128 | -1924 | 0.00578585807 | 0.000289620308 | +0.225062468 | MEASURED_FORMULATION |
| 12 | `coeff_rank11` | A_low_rank | 191044 | -8 | 0.0139164778 | 0.000289620308 | +0.358848104 | TOY_BRACKET_FULL_ARRAY_CPR1 |
| 13 | `basis_perplane_int4_p100_0` | B_per_plane_precision | 190296 | -756 | 0.0143097221 | 0.000289620308 | +0.363584018 | MEASURED_FORMULATION |
| 14 | `basis_int3_p99_0` | B_precision | 185652 | -5400 | 0.0171839259 | 0.000289620308 | +0.396744827 | MEASURED_FORMULATION |
| 15 | `basis_int4_p100_0` | B_precision | 186916 | -4136 | 0.0189977932 | 0.000289620308 | +0.418916061 | MEASURED_FORMULATION |
| 16 | `basis_perplane_int3_p99_0` | B_per_plane_precision | 187520 | -3532 | 0.0217368706 | 0.000289620308 | +0.44968183 | MEASURED_FORMULATION |
| 17 | `compose_coeff_rank11_basis_int4` | A_plus_B_composed | 186908 | -4144 | 0.0285464775 | 0.000289620308 | +0.517335191 | TOY_BRACKET_FULL_ARRAY_CPR1 |
| 18 | `basis_lowrank_packet_r18_q10` | B_cross_plane_low_rank_packet | 199564 | +8512 | 0.0354824668 | 0.000289620308 | +0.587144876 | MEASURED_OPTIMAL_FORM_PACKET |
| 19 | `basis_plane_rank18` | B_cross_plane_rank | 191392 | +340 | 0.0368269912 | 0.000289620308 | +0.592884329 | TOY_BRACKET_FULL_ARRAY_CPR1 |
| 20 | `capacity_drop_dim01` | C_capacity | 189192 | -1860 | 0.0556489342 | 0.000289620308 | +0.730549109 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 21 | `basis_dct_keep50` | B_spatial_DCT | 191116 | +64 | 0.0555188738 | 0.000289620308 | +0.730957972 | TOY_BRACKET_FULL_ARRAY_CPR1 |
| 22 | `capacity_drop_dim11` | C_capacity | 189208 | -1844 | 0.0620878527 | 0.000289620308 | +0.772536085 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 23 | `coeff_rank10` | A_low_rank | 191052 | +0 | 0.0713334585 | 0.000289620308 | +0.830396845 | TOY_BRACKET_FULL_ARRAY_CPR1 |
| 24 | `capacity_drop_dim06` | C_capacity | 189392 | -1660 | 0.0784141731 | 0.000289620308 | +0.870217923 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 25 | `capacity_drop_dim00` | C_capacity | 189168 | -1884 | 0.0825675314 | 0.000289620308 | +0.893217774 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 26 | `compose_coeff_rank10_basis_int4` | A_plus_B_composed | 186912 | -4140 | 0.08427584 | 0.000289620308 | +0.901067554 | TOY_BRACKET_FULL_ARRAY_CPR1 |
| 27 | `capacity_drop_dim05` | C_capacity | 189124 | -1928 | 0.0846454051 | 0.000289620308 | +0.904551072 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 28 | `capacity_drop_measured_nested02` | C_capacity | 187248 | -3804 | 0.0854573832 | 0.000289620308 | +0.907704173 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 29 | `basis_perplane_int3_p100_0` | B_per_plane_precision | 185736 | -5316 | 0.0979173545 | 0.000289620308 | +0.97179775 | MEASURED_FORMULATION |
| 30 | `capacity_drop_dim04` | C_capacity | 189264 | -1788 | 0.11289039 | 0.000289620308 | +1.04711381 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 31 | `capacity_drop_dim09` | C_capacity | 189220 | -1832 | 0.120039877 | 0.000289620308 | +1.08021273 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 32 | `capacity_drop_dim02` | C_capacity | 189132 | -1920 | 0.124851964 | 0.000289620308 | +1.10189878 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 33 | `basis_int3_p100_0` | B_precision | 183412 | -7640 | 0.143392609 | 0.000289620308 | +1.17818484 | MEASURED_FORMULATION |
| 34 | `capacity_drop_dim07` | C_capacity | 189128 | -1924 | 0.146751308 | 0.000289620308 | +1.19593391 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 35 | `capacity_drop_nested02` | C_capacity | 187112 | -3940 | 0.164377179 | 0.000289620308 | +1.26527863 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 36 | `basis_plane_rank12` | B_cross_plane_rank | 191404 | +352 | 0.181262614 | 0.000289620308 | +1.3323779 | TOY_BRACKET_FULL_ARRAY_CPR1 |
| 37 | `capacity_drop_dim03` | C_capacity | 189348 | -1704 | 0.192548058 | 0.000289620308 | +1.37228772 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 38 | `basis_lowrank_packet_r12_q08` | B_cross_plane_low_rank_packet | 190576 | -476 | 0.19324095 | 0.000289620308 | +1.37559985 | MEASURED_OPTIMAL_FORM_PACKET |
| 39 | `capacity_drop_measured_nested03` | C_capacity | 185228 | -5824 | 0.198656615 | 0.000289620308 | +1.39138349 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 40 | `coeff_rank08` | A_low_rank | 191144 | +92 | 0.197755756 | 0.000289620308 | +1.39212331 | TOY_BRACKET_FULL_ARRAY_CPR1 |
| 41 | `basis_dct_packet_k256_q08` | B_spatial_DCT_packet | 199144 | +8092 | 0.265346231 | 0.000289620308 | +1.62013876 | MEASURED_OPTIMAL_FORM_PACKET |
| 42 | `capacity_drop_dim10` | C_capacity | 189296 | -1756 | 0.279257754 | 0.000289620308 | +1.65573692 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 43 | `capacity_drop_dim08` | C_capacity | 189228 | -1824 | 0.387971967 | 0.000289620308 | +1.95429135 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 44 | `gauge_rotation_best` | beyond_seed_gauge_rotation | 191084 | +32 | 0.442203029 | 0.000289620308 | +2.08868918 | MEASURED_FORMULATION_SEEDED_SEARCH |
| 45 | `capacity_drop_nested03` | C_capacity | 185136 | -5916 | 0.515115575 | 0.000289620308 | +2.25148202 | MEASURED_EXISTING_CARRIER_RESPONSE |
| 46 | `basis_dct_keep25` | B_spatial_DCT | 191516 | +464 | 0.580270122 | 0.000289620308 | +2.39499409 | TOY_BRACKET_FULL_ARRAY_CPR1 |
| 47 | `basis_lowrank_packet_r06_q08` | B_cross_plane_low_rank_packet | 184432 | -6620 | 0.90632132 | 0.000289620308 | +2.99191459 | MEASURED_OPTIMAL_FORM_PACKET |
| 48 | `basis_dct_packet_k128_q08` | B_spatial_DCT_packet | 189592 | -1460 | 1.57976297 | 0.000289620308 | +3.95945654 | MEASURED_OPTIMAL_FORM_PACKET |
| 49 | `basis_dct_packet_k064_q08` | B_spatial_DCT_packet | 184056 | -6996 | 6.90692989 | 0.000289620308 | +8.29194124 | MEASURED_OPTIMAL_FORM_PACKET |

## Byte-only table

These rows have real archive bytes and receiver parse-back, but were not scorer rows.

| row | arm | archive B | delta B | parse-back | scope |
|---|---|---:|---:|---|---|
| `basis_dct_packet_k064_q10` | B_spatial_DCT_packet | 184544 | -6508 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `basis_lowrank_packet_r06_q10` | B_cross_plane_low_rank_packet | 185264 | -5788 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `basis_dct_packet_k128_q10` | B_spatial_DCT_packet | 190588 | -464 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_rank06` | A_low_rank | 191208 | +156 | PASS | TOY_BRACKET_FULL_ARRAY_CPR1 |
| `coeff_rank04` | A_low_rank | 191268 | +216 | PASS | TOY_BRACKET_FULL_ARRAY_CPR1 |
| `coeff_lowrank_r05_q06_res16` | A_low_rank_plus_residual | 192040 | +988 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `basis_lowrank_packet_r12_q10` | B_cross_plane_low_rank_packet | 192392 | +1340 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r11_q06_res16` | A_low_rank_plus_residual | 192500 | +1448 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r07_q06_res16` | A_low_rank_plus_residual | 192588 | +1536 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r04_q08_res16` | A_low_rank_plus_residual | 192676 | +1624 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r08_q06_res16` | A_low_rank_plus_residual | 192792 | +1740 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_predictor_ar3` | A_reversible_predictor | 192856 | +1804 | PASS | MEASURED_EXACT_REPRESENTATION |
| `coeff_predictor_ar1` | A_reversible_predictor | 192860 | +1808 | PASS | MEASURED_EXACT_REPRESENTATION |
| `coeff_predictor_ar2` | A_reversible_predictor | 192860 | +1808 | PASS | MEASURED_EXACT_REPRESENTATION |
| `coeff_predictor_first` | A_reversible_predictor | 192864 | +1812 | PASS | MEASURED_EXACT_REPRESENTATION |
| `coeff_predictor_ar4` | A_reversible_predictor | 192880 | +1828 | PASS | MEASURED_EXACT_REPRESENTATION |
| `coeff_lowrank_r10_q06_res16` | A_low_rank_plus_residual | 192884 | +1832 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r09_q06_res16` | A_low_rank_plus_residual | 192992 | +1940 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_predictor_linear_knot30` | A_reversible_predictor | 193068 | +2016 | PASS | MEASURED_EXACT_REPRESENTATION |
| `coeff_predictor_linear_knot60` | A_reversible_predictor | 193096 | +2044 | PASS | MEASURED_EXACT_REPRESENTATION |
| `coeff_predictor_cubic_knot60` | A_reversible_predictor | 193116 | +2064 | PASS | MEASURED_EXACT_REPRESENTATION |
| `coeff_predictor_cubic_knot30` | A_reversible_predictor | 193128 | +2076 | PASS | MEASURED_EXACT_REPRESENTATION |
| `coeff_predictor_second` | A_reversible_predictor | 193284 | +2232 | PASS | MEASURED_EXACT_REPRESENTATION |
| `coeff_lowrank_r05_q08_res16` | A_low_rank_plus_residual | 193308 | +2256 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r11_q08_res16` | A_low_rank_plus_residual | 193344 | +2292 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r04_q10_res16` | A_low_rank_plus_residual | 193352 | +2300 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r11_q10_res16` | A_low_rank_plus_residual | 193500 | +2448 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r06_q08_res16` | A_low_rank_plus_residual | 193868 | +2816 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r05_q10_res16` | A_low_rank_plus_residual | 194112 | +3060 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r07_q08_res16` | A_low_rank_plus_residual | 194376 | +3324 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r05_q06_res04` | A_low_rank_plus_residual | 194412 | +3360 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r04_q08_res04` | A_low_rank_plus_residual | 194480 | +3428 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r11_q06_res04` | A_low_rank_plus_residual | 194652 | +3600 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r08_q08_res16` | A_low_rank_plus_residual | 194776 | +3724 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r06_q10_res16` | A_low_rank_plus_residual | 194800 | +3748 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r10_q08_res16` | A_low_rank_plus_residual | 194816 | +3764 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r04_q10_res04` | A_low_rank_plus_residual | 194948 | +3896 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r07_q06_res04` | A_low_rank_plus_residual | 195140 | +4088 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r05_q08_res04` | A_low_rank_plus_residual | 195164 | +4112 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r10_q06_res04` | A_low_rank_plus_residual | 195188 | +4136 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r09_q08_res16` | A_low_rank_plus_residual | 195200 | +4148 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r08_q06_res04` | A_low_rank_plus_residual | 195304 | +4252 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r07_q10_res16` | A_low_rank_plus_residual | 195476 | +4424 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r09_q06_res04` | A_low_rank_plus_residual | 195536 | +4484 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r06_q08_res04` | A_low_rank_plus_residual | 195772 | +4720 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r05_q10_res04` | A_low_rank_plus_residual | 195800 | +4748 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r10_q10_res16` | A_low_rank_plus_residual | 195852 | +4800 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r04_q08_res01` | A_low_rank_plus_residual | 195876 | +4824 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r05_q06_res01` | A_low_rank_plus_residual | 195904 | +4852 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r08_q10_res16` | A_low_rank_plus_residual | 195992 | +4940 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r04_q10_res01` | A_low_rank_plus_residual | 196132 | +5080 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r11_q10_res04` | A_low_rank_plus_residual | 196152 | +5100 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r07_q08_res04` | A_low_rank_plus_residual | 196252 | +5200 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r11_q08_res04` | A_low_rank_plus_residual | 196420 | +5368 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r06_q06_res01` | A_low_rank_plus_residual | 196440 | +5388 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r09_q10_res16` | A_low_rank_plus_residual | 196524 | +5472 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `basis_lowrank_packet_r18_q08` | B_cross_plane_low_rank_packet | 196564 | +5512 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r06_q10_res04` | A_low_rank_plus_residual | 196572 | +5520 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r05_q08_res01` | A_low_rank_plus_residual | 196596 | +5544 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r08_q08_res04` | A_low_rank_plus_residual | 196676 | +5624 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r07_q06_res01` | A_low_rank_plus_residual | 196848 | +5796 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r05_q10_res01` | A_low_rank_plus_residual | 196936 | +5884 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r09_q08_res04` | A_low_rank_plus_residual | 197128 | +6076 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r10_q08_res04` | A_low_rank_plus_residual | 197136 | +6084 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r08_q06_res01` | A_low_rank_plus_residual | 197216 | +6164 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r11_q06_res01` | A_low_rank_plus_residual | 197216 | +6164 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r06_q08_res01` | A_low_rank_plus_residual | 197260 | +6208 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r07_q10_res04` | A_low_rank_plus_residual | 197268 | +6216 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `basis_haar_lifting_exact` | B_reversible_spatial_transform | 197412 | +6360 | PASS | MEASURED_EXACT_REPRESENTATION |
| `coeff_lowrank_r09_q06_res01` | A_low_rank_plus_residual | 197564 | +6512 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r10_q06_res01` | A_low_rank_plus_residual | 197612 | +6560 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r06_q10_res01` | A_low_rank_plus_residual | 197700 | +6648 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r07_q08_res01` | A_low_rank_plus_residual | 197852 | +6800 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r08_q10_res04` | A_low_rank_plus_residual | 197868 | +6816 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r08_q08_res01` | A_low_rank_plus_residual | 198388 | +7336 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r07_q10_res01` | A_low_rank_plus_residual | 198444 | +7392 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r10_q10_res04` | A_low_rank_plus_residual | 198456 | +7404 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r09_q10_res04` | A_low_rank_plus_residual | 198500 | +7448 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r11_q08_res01` | A_low_rank_plus_residual | 198648 | +7596 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r09_q08_res01` | A_low_rank_plus_residual | 198924 | +7872 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r10_q08_res01` | A_low_rank_plus_residual | 199008 | +7956 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r08_q10_res01` | A_low_rank_plus_residual | 199148 | +8096 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r11_q10_res01` | A_low_rank_plus_residual | 199592 | +8540 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r09_q10_res01` | A_low_rank_plus_residual | 199844 | +8792 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `coeff_lowrank_r10_q10_res01` | A_low_rank_plus_residual | 200308 | +9256 | PASS | MEASURED_OPTIMAL_FORM_PACKET |
| `basis_dct_packet_k256_q10` | B_spatial_DCT_packet | 201500 | +10448 | PASS | MEASURED_OPTIMAL_FORM_PACKET |

## RECALL EVIDENCE

Full-corpus recall found the #140 low-rank method on a different object, AM1's exact residual-predictor crosswalk, ddm_na2's measured pose-prefix bias, and a 20260808 Candidate-A surface-fit task that reused the short ddm_pk2 identifier but is not this PR130 experiment. That changed the implementation to direct counted factor/residual and transform packets, while retaining seeded stratified n120 selection. No ancestor result was transferred as a PR130 number.

## Boundaries

- Not measured: contest CPU/CUDA or a retrained smaller carrier.
- No non-control row advanced to n600: the n120 gate selected unchanged CPR1, so firing a full scorer on the losing composed packet would not be a promotion measurement.
- Measured: exact CPR1 anatomy, real candidate archives, receiver parse-back, seeded stratified n120 frozen CPU-torch scorer rows.
- Unchanged-byte outer-coder races were not repeated.

## NEXT_IF_RESUMED

- `ddm_pk2_rate_aware_gauge_qat`; disposition=QUEUED-WITH-A-FIRE-ORDER; owner=future PR130 pose-carrier training owner; consumer_store=.omx/state/codex_arm_queue.next_if_resumed.jsonl; fire_trigger=a scorer-free counted PK2R preflight projects at least 2000 full-archive bytes saved with carrier-product MSE below 2.5e-6, after which only a resumable quantization-aware training run and seeded stratified n120 row may fire.
- `ddm_pk2_full_n600_promotion`; disposition=FOLDED; owner=ddm_pk2; consumer_store=/Volumes/VertigoDataTier/pact/ddm_pk2_20260809/FINAL_RECEIPT.json; fire_trigger=none in this run because no non-control n120 row had delta S below zero.

## LIVE-HYPOTHESES

- A learned rate-aware gauge can still outperform the 64 random rotations: C @ B is invariant before quantization, while the random search optimized neither entropy nor scorer sensitivity. The preflight threshold above prevents this from becoming another unconstrained search.
- Quantization-aware retraining may make per-plane low precision usable because per-plane scaling reduced post-hoc int4 pose damage relative to the shared-scale row. This remains weak evidence: the measured absolute distortion is still far outside break-even.

## DEAD-ENDS

- Bare Brotli/LZMA/Zstd/ANS recoding of unchanged CPR1 pose bytes: the prior real-byte race was +4 B and the current runner did not reopen it.
- Calling int7/int6/int5 a precision ladder: deployed source values are already signed int5, so those labels are one control row.
- Treating projected full arrays as a low-rank or transform family verdict: those rows remain toy brackets and direct factor packets carry the actual verdict surface.
- Exact first/second/AR/spline coefficient predictors on this instance: all reconstructed the deployed coefficients exactly but enlarged the archive by 1804 to 2232 bytes.
- Direct low-rank coefficient factors plus residuals on this instance: the exact row was 4316 bytes larger, and every lossy shortlisted row worsened both rate or pose enough to lose.
- Post-hoc dimension dropping on the existing carrier: every single dimension and both measured-order nested drops were decisively outside break-even.
- The seeded 64-rotation random gauge formulation: its best materialized row was larger and had d_pose 0.442203.
