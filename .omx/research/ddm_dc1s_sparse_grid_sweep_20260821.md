# ddm_dc1s sparse-grid full-population sweep — honest closure

Date: 2026-08-21  
Axis: `[macOS-CPU advisory / scorer-free real-fx5 full-frame rate measurement]`  
Selection: full population, 600/600 frames, all 190 HPAC groups, block sizes {1,2,4,8}  
Disposition: **CLOSED(FAMILY A at measured full-frame FX5 scope)**

## Verdict

The actual retained sparse-grid question packet is **388,326 B**, versus the real FX5 HPAC member's **113,777 B**. It loses **274,549 B** on the member-to-member comparison and loses **274,549.821 B** against the slightly more favorable 910,209.432-bit HPAC ideal baseline after every header, group tax, and final byte boundary. The required +3 KiB fire gate therefore fails by 277,621 B.

Search compute is not the blocker: the chosen paths examined 618,965 candidates in 8.690 local CPU seconds. The conservative native projection is 8.690 added seconds, below both the inherited 211.471 s cold headroom and 691.471 s warm governed headroom. This is a **rate failure**, not a wall-time failure.

No receiver was edited, no candidate archive was built, no scorer was used, and no score is claimed. The stop rule binds. The full-frame variable-block-size Family A representation is closed on this exact FX5 body.

## RECALL EVIDENCE

Before implementation I searched the bounded original-research corpus `.omx/research` and live state/task stores for `finite-target`, `rank mismatch`, `combinatorial hash`, `hash sieve`, `sparse-grid`, `hash-constrained`, `question coding`, `ddm_dc1`, and `dc1s`, and queried the canonical-equation registry. I did not find, in that scope, a pre-existing full-population Family A result or a cheaper operational hash-question representation beyond the dc1 seed memo/prototype, its final-message/charter copies, and the NA4 rate-axis sampling calibration. NA4 did not substitute for this n600 measurement.

The beyond-seed fact that changed execution was the real production loop in `experiments/ddm_jg2_tail_reencode.py`: FX5 uses the previous-frame boundary buckets, calls `FreeCorrector.end_frame`, and carries the full adaptive corrector state across frames. The first n600 attempt inherited dc1's frame-0-only omission and failed closed at frame 1. The corrected sweep checkpoints all 127 adaptive-state arrays and exactly matches the retained ledger on every one of 600 frames.

The canonical-equation search returned general conditional/joint rate-distortion laws, but no byte-defined hash-sieve packet that displaced this measurement. I therefore executed the charter's real HPAC law and real deterministic search rather than transferring a projection.

## Source and custody pins

- dc1 source commit: `badc6e2e9b`; prototype SHA-256 `51d537ba0b1ac4db835e4c162ec624932498a2ea5c176a260fe12b8a7b18e8bd`.
- FX5 archive: 180,386 B, SHA-256 `4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841`.
- Retained token field: 117,964,800 B, SHA-256 `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`.
- Ideal-bit ledger: SHA-256 `0585b0d98ba2958be3e20021641dd0a74bc61714d1434cab16efb005320418df`; sum 910,209.4321425341 bits.
- Real-coder control: SHA-256 `b5a2668f499bc7060f5c09fa36b8435fd98bae80e62d4d5a0fc2ddc3713c2685`; retained `byte_identical=true`, n600.
- dc1 packet/decoded controls: `3e688f115333…` / `cbb99eb650b3…`.
- Sweep implementation: `experiments/ddm_dc1s_sparse_grid_sweep.py`, SHA-256 `2e4ca2576abac8259d11b6bfaf3deb92940ee5434feff3a2a82e23b7ba94813c`.
- Two renewed tracker passes cover 46 Python entities: `search_wire_resume_state` and `authority_payload_determinism_negative`. Ruff, bytecode compilation, heap-vs-exhaustive self-tests, packet/Elias-Fano self-tests, and the bounded payload-retention census pass.
- Storage preflight passed with 30,208,819,200 free bytes, an 8 GiB sweep estimate, and an 8 GiB reserve.
- Serializer secret preflight produced exactly two `generic-api-key` false positives: both are the pinned public token-field content SHA-256 `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`, once in the executed source and once in the failed-run custody JSON. Direct unredacted scans found one row in each file and zero in this memo. A deliberate `TAC_SECRETS_WAIVE=1` retry preserved the byte-exact executed source instead of changing its provenance hash; no credential is present. That retry then failed at `git add` because this managed checkout cannot create Git index/object temporaries (`Operation not permitted`). The verified artifacts are unstaged and **not committed**; no serializer-commit claim is made.

## Retention receipts

Canonical consumer store: `/Volumes/APDataStore/pact/ddm_dc1_decode_time_compute/full_frame_sparse_sweep/` (288 MiB retained).

- 30 distinct 20-frame NPZ checkpoints cover frames 0–599. Each contains source probability rows, targets, ranks, minimal hash lengths, target and trace digests, search counts/timing, group bit totals, and the complete production corrector state.
- 190 per-group NPZ bundles retain all four block-size option bodies, the chosen body, sparse rank/search tables, and four seeded repeat searches per group.
- Actual packet and independently reconstructed repeat: 388,326 B each, identical SHA-256 `9ca6e59e789abdd0c02c70c3d5d52d2b0da917518f03f792b3bbcc31c30fa839`.
- A completed resume-only rerun revalidated all checkpoint hashes, reconstructed the same packet SHA, and reproduced the same accounting.
- Two-frame stop/resume control: `/Volumes/APDataStore/pact/ddm_dc1_decode_time_compute/full_frame_sparse_sweep_smoke_n2_state_v2/`; frame 0 and frame 1 both match the ledger, packet/repeat SHA `4d53327c8f2cc2f3e68dc29e2b9eee3fabb9218d44bc7a1ae1b583b41b3a0059`.
- The failed frame-1 attempt was losslessly moved, not deleted, to `full_frame_sparse_sweep_failed_frame1_state_v1/`. Its custody record is `.omx/research/ddm_dc1s_failed_frame1_state_v1_custody_20260821.json`.

## Full-population accounting

| quantity | bits | bytes |
| --- | ---: | ---: |
| direct HPAC ideal baseline | 910,209.432 | 113,776.179 |
| top header | 80 | 10 |
| 190 group headers/tax | 9,120 | 1,140 |
| sparse position fields | 1,819,325 | 227,415.625 |
| per-block hash-length fields | 814,021 | 101,752.625 |
| hash-prefix questions | 464,062 | 58,007.750 |
| meaningful sparse body | 3,097,408 | 387,176 |
| actual byte packet | 3,106,608 | 388,326 |
| honest credit vs ideal | -2,196,398.568 | **-274,549.821** |
| credit vs 113,777 B FX5 member | — | **-274,549** |

Positions alone cost twice the complete HPAC member. Hash-length metadata adds another 101.8 KB, and hash questions add 58.0 KB. Every one of the 190 groups is negative after its 6-byte tax; the least-negative group is -97.18 B and the worst is -2,730.95 B.

Variable selection chose b=8 for 154 groups, b=4 for 22, b=1 for 13, and b=2 for 1. Fixed-size packet outcomes were: b=1 433,126 B; b=2 432,364 B; b=4 417,301 B; b=8 389,760 B. Variable selection saves only 1,434 B against fixed b=8 and remains 274,549 B behind the shipped member. The adjacent-group branch was not treated as a free factorization: later conditional rows depend on earlier decoded observations, so a legal joint search would need to rerun the adaptive corrector for each branch and cannot erase the dominant sparse-position census.

## Per-group chosen wire table

`direct` is the retained HPAC ideal. `positions`, `length table`, and `hash prefixes` are actual meaningful packet fields. `wire` includes the 48-bit group header/tax; credit is `direct - wire`.

| g | b | direct | positions | length table | hash prefixes | wire incl. 48-bit tax | credit B |
| -: | -: | -: | -: | -: | -: | -: | -: |
| 0 | 8 | 1685.03 | 1646 | 1495 | 752 | 3941 | -282.00 |
| 1 | 8 | 576.52 | 683 | 380 | 243 | 1354 | -97.18 |
| 2 | 8 | 1677.74 | 2285 | 1101 | 904 | 4338 | -332.53 |
| 3 | 8 | 922.74 | 1521 | 648 | 371 | 2588 | -208.16 |
| 4 | 8 | 1632.88 | 2635 | 1568 | 720 | 4971 | -417.26 |
| 5 | 1 | 1210.92 | 2957 | 0 | 582 | 3587 | -297.01 |
| 6 | 8 | 1894.04 | 3144 | 1796 | 793 | 5781 | -485.87 |
| 7 | 4 | 1266.13 | 2494 | 876 | 584 | 4002 | -341.98 |
| 8 | 8 | 2126.49 | 3539 | 1984 | 924 | 6495 | -546.06 |
| 9 | 8 | 1483.90 | 2615 | 1368 | 654 | 4685 | -400.14 |
| 10 | 4 | 2157.26 | 4343 | 1572 | 893 | 6856 | -587.34 |
| 11 | 8 | 1717.64 | 3027 | 1176 | 725 | 4976 | -407.29 |
| 12 | 8 | 2329.74 | 4178 | 1695 | 1205 | 7126 | -599.53 |
| 13 | 8 | 1656.20 | 3188 | 1600 | 890 | 5726 | -508.72 |
| 14 | 4 | 2257.03 | 4820 | 1680 | 1175 | 7723 | -683.25 |
| 15 | 8 | 1874.83 | 3672 | 1848 | 1036 | 6604 | -591.15 |
| 16 | 8 | 2523.62 | 4793 | 2520 | 1454 | 8815 | -786.42 |
| 17 | 4 | 2038.80 | 4702 | 1581 | 953 | 7284 | -655.65 |
| 18 | 8 | 2509.09 | 4785 | 2440 | 1497 | 8770 | -782.61 |
| 19 | 8 | 2115.16 | 4322 | 2148 | 1043 | 7561 | -680.73 |
| 20 | 8 | 2834.74 | 5570 | 2888 | 1583 | 10089 | -906.78 |
| 21 | 8 | 2569.27 | 5012 | 2516 | 1177 | 8753 | -772.97 |
| 22 | 8 | 3554.36 | 6504 | 3436 | 1821 | 11809 | -1031.83 |
| 23 | 8 | 3053.25 | 5856 | 3004 | 1534 | 10442 | -923.59 |
| 24 | 8 | 4022.75 | 7337 | 4895 | 2088 | 14368 | -1293.16 |
| 25 | 8 | 3395.90 | 6503 | 3360 | 1725 | 11636 | -1030.01 |
| 26 | 8 | 4044.37 | 7509 | 3956 | 2292 | 13805 | -1220.08 |
| 27 | 8 | 4086.78 | 7431 | 3904 | 2455 | 13838 | -1218.90 |
| 28 | 8 | 4553.55 | 8438 | 4500 | 2376 | 15362 | -1351.06 |
| 29 | 4 | 4082.13 | 8835 | 4084 | 1747 | 14714 | -1328.98 |
| 30 | 8 | 4851.26 | 8652 | 4568 | 2167 | 15435 | -1322.97 |
| 31 | 8 | 4625.18 | 9048 | 4832 | 2190 | 16118 | -1436.60 |
| 32 | 8 | 5168.38 | 9377 | 4976 | 2653 | 17054 | -1485.70 |
| 33 | 8 | 4829.84 | 9257 | 4896 | 2515 | 16716 | -1485.77 |
| 34 | 8 | 5362.44 | 10011 | 5324 | 2957 | 18340 | -1622.20 |
| 35 | 1 | 4871.08 | 13464 | 0 | 2542 | 16054 | -1397.86 |
| 36 | 8 | 5471.93 | 10322 | 5456 | 3118 | 18944 | -1684.01 |
| 37 | 8 | 5137.65 | 10022 | 5256 | 2799 | 18125 | -1623.42 |
| 38 | 8 | 5528.14 | 10590 | 5560 | 2804 | 19002 | -1684.23 |
| 39 | 8 | 5397.54 | 10350 | 5400 | 2795 | 18593 | -1649.43 |
| 40 | 4 | 5785.81 | 12352 | 4281 | 2514 | 19195 | -1676.15 |
| 41 | 8 | 5616.03 | 10967 | 5736 | 2969 | 19720 | -1763.00 |
| 42 | 8 | 6075.03 | 11235 | 5840 | 3082 | 20205 | -1766.25 |
| 43 | 8 | 5519.47 | 10743 | 5512 | 2736 | 19039 | -1689.94 |
| 44 | 8 | 6148.56 | 11636 | 6032 | 3353 | 21069 | -1865.05 |
| 45 | 8 | 5897.01 | 11588 | 6000 | 3425 | 21061 | -1895.50 |
| 46 | 8 | 6821.65 | 12576 | 6584 | 3538 | 22746 | -1990.54 |
| 47 | 8 | 6597.14 | 12516 | 6544 | 3291 | 22399 | -1975.23 |
| 48 | 8 | 7143.42 | 13271 | 6972 | 3581 | 23872 | -2091.07 |
| 49 | 8 | 6940.51 | 13325 | 7008 | 3778 | 24159 | -2152.31 |
| 50 | 8 | 7765.88 | 14223 | 7532 | 4152 | 25955 | -2273.64 |
| 51 | 8 | 7373.21 | 13797 | 7248 | 3976 | 25069 | -2211.97 |
| 52 | 8 | 8210.02 | 15278 | 8160 | 4580 | 28066 | -2482.00 |
| 53 | 8 | 7562.21 | 14276 | 7492 | 4199 | 26015 | -2306.60 |
| 54 | 8 | 8486.18 | 15738 | 8392 | 4431 | 28609 | -2515.35 |
| 55 | 8 | 7950.33 | 14910 | 7840 | 4054 | 26852 | -2362.71 |
| 56 | 8 | 8483.96 | 15887 | 8416 | 4620 | 28971 | -2560.88 |
| 57 | 8 | 7908.59 | 14543 | 7520 | 4147 | 26258 | -2293.68 |
| 58 | 8 | 9056.12 | 16281 | 8604 | 5331 | 30264 | -2650.98 |
| 59 | 8 | 8049.51 | 14967 | 7728 | 3633 | 26376 | -2290.81 |
| 60 | 8 | 9283.43 | 16988 | 9000 | 5095 | 31131 | -2730.95 |
| 61 | 8 | 8196.64 | 15092 | 7736 | 4205 | 27081 | -2360.54 |
| 62 | 8 | 9241.86 | 17010 | 8940 | 4783 | 30781 | -2692.39 |
| 63 | 1 | 8158.47 | 22473 | 0 | 4194 | 26715 | -2319.57 |
| 64 | 8 | 8629.60 | 16188 | 8392 | 4578 | 29206 | -2572.05 |
| 65 | 8 | 7548.85 | 15018 | 7612 | 3649 | 26327 | -2347.27 |
| 66 | 4 | 7768.97 | 17334 | 5886 | 3291 | 26559 | -2348.75 |
| 67 | 8 | 6930.43 | 13728 | 6816 | 3467 | 24059 | -2141.07 |
| 68 | 8 | 6813.69 | 13364 | 6608 | 3595 | 23615 | -2100.16 |
| 69 | 8 | 6731.90 | 13672 | 6784 | 3608 | 24112 | -2172.51 |
| 70 | 1 | 6600.30 | 17990 | 0 | 3238 | 21276 | -1834.46 |
| 71 | 1 | 6232.71 | 17920 | 0 | 4836 | 22804 | -2071.41 |
| 72 | 8 | 6350.28 | 12769 | 6268 | 2585 | 21670 | -1914.97 |
| 73 | 8 | 6260.99 | 12440 | 6080 | 3495 | 22063 | -1975.25 |
| 74 | 8 | 6143.26 | 12454 | 6088 | 3131 | 21721 | -1947.22 |
| 75 | 8 | 6089.87 | 12090 | 5880 | 3455 | 21473 | -1922.89 |
| 76 | 8 | 5813.76 | 11747 | 5684 | 2592 | 20071 | -1782.16 |
| 77 | 8 | 5584.80 | 10753 | 5116 | 2584 | 18501 | -1614.53 |
| 78 | 8 | 5387.40 | 10942 | 5224 | 2434 | 18648 | -1657.57 |
| 79 | 8 | 5438.68 | 10858 | 5176 | 2848 | 18930 | -1686.41 |
| 80 | 8 | 5291.63 | 10718 | 5096 | 2422 | 18284 | -1624.05 |
| 81 | 8 | 5578.59 | 11411 | 5492 | 3057 | 20008 | -1803.68 |
| 82 | 8 | 5319.67 | 10592 | 5024 | 2894 | 18558 | -1654.79 |
| 83 | 8 | 5423.45 | 11278 | 5416 | 3071 | 19813 | -1798.69 |
| 84 | 8 | 5078.99 | 10221 | 4812 | 2428 | 17509 | -1553.75 |
| 85 | 4 | 5261.32 | 11968 | 3813 | 1983 | 17812 | -1568.84 |
| 86 | 8 | 5417.10 | 10851 | 5172 | 2881 | 18952 | -1691.86 |
| 87 | 8 | 5030.15 | 10256 | 4832 | 2319 | 17455 | -1553.11 |
| 88 | 8 | 5258.29 | 10564 | 5008 | 2579 | 18199 | -1617.59 |
| 89 | 8 | 5095.57 | 10263 | 4836 | 2562 | 17709 | -1576.68 |
| 90 | 8 | 5221.91 | 10606 | 3774 | 2853 | 17281 | -1507.39 |
| 91 | 8 | 5092.69 | 10431 | 4932 | 2419 | 17830 | -1592.16 |
| 92 | 8 | 5816.24 | 10935 | 5220 | 2944 | 19147 | -1666.35 |
| 93 | 8 | 5634.17 | 10921 | 5212 | 2625 | 18806 | -1646.48 |
| 94 | 8 | 5825.13 | 11243 | 5396 | 4064 | 20751 | -1865.73 |
| 95 | 8 | 5821.98 | 11096 | 5312 | 2302 | 18758 | -1617.00 |
| 96 | 8 | 5685.36 | 11124 | 5328 | 2633 | 19133 | -1680.95 |
| 97 | 8 | 5397.02 | 10676 | 5072 | 2247 | 18043 | -1580.75 |
| 98 | 8 | 5538.32 | 10571 | 5012 | 2342 | 17973 | -1554.33 |
| 99 | 4 | 5942.95 | 13032 | 4212 | 2421 | 19713 | -1721.26 |
| 100 | 8 | 5744.64 | 11558 | 5576 | 3156 | 20338 | -1824.17 |
| 101 | 8 | 5744.76 | 11250 | 5400 | 2872 | 19570 | -1728.16 |
| 102 | 8 | 5745.95 | 11593 | 5596 | 2752 | 19989 | -1780.38 |
| 103 | 8 | 5795.23 | 11565 | 5580 | 2877 | 20070 | -1784.35 |
| 104 | 8 | 5593.14 | 10858 | 5176 | 2565 | 18647 | -1631.73 |
| 105 | 4 | 5433.86 | 12536 | 4026 | 2062 | 18672 | -1654.77 |
| 106 | 4 | 5783.41 | 12920 | 4170 | 2676 | 19814 | -1753.82 |
| 107 | 8 | 5547.89 | 11180 | 5360 | 2408 | 18996 | -1681.01 |
| 108 | 8 | 5593.67 | 11285 | 5420 | 3093 | 19846 | -1781.54 |
| 109 | 8 | 5757.82 | 11467 | 5524 | 2876 | 19915 | -1769.65 |
| 110 | 4 | 5813.12 | 12544 | 4029 | 2246 | 18867 | -1631.74 |
| 111 | 8 | 5904.31 | 11600 | 5600 | 2875 | 20123 | -1777.34 |
| 112 | 8 | 5986.18 | 11789 | 5708 | 2596 | 20141 | -1769.35 |
| 113 | 8 | 6387.85 | 12370 | 6040 | 3504 | 21962 | -1946.77 |
| 114 | 8 | 6430.55 | 12314 | 6008 | 3263 | 21633 | -1900.31 |
| 115 | 8 | 6340.99 | 12566 | 6152 | 3634 | 22400 | -2007.38 |
| 116 | 8 | 6697.99 | 12776 | 6272 | 3043 | 22139 | -1930.13 |
| 117 | 8 | 6519.46 | 12657 | 6204 | 3275 | 22184 | -1958.07 |
| 118 | 8 | 6667.74 | 13077 | 6444 | 3103 | 22672 | -2000.53 |
| 119 | 4 | 6557.00 | 14640 | 4815 | 2849 | 22352 | -1974.37 |
| 120 | 8 | 6827.51 | 12993 | 6396 | 3837 | 23274 | -2055.81 |
| 121 | 8 | 6287.20 | 12342 | 6024 | 2831 | 21245 | -1869.73 |
| 122 | 8 | 6417.14 | 12524 | 6128 | 3163 | 21863 | -1930.73 |
| 123 | 8 | 6319.50 | 12251 | 5972 | 3465 | 21736 | -1927.06 |
| 124 | 8 | 6190.17 | 12314 | 6008 | 3367 | 21737 | -1943.35 |
| 125 | 8 | 6075.62 | 11642 | 5624 | 2725 | 20039 | -1745.42 |
| 126 | 8 | 6516.31 | 12839 | 6308 | 3038 | 22233 | -1964.59 |
| 127 | 8 | 6191.26 | 12223 | 5956 | 2914 | 21141 | -1868.72 |
| 128 | 8 | 5707.39 | 11460 | 5552 | 4001 | 21061 | -1919.20 |
| 129 | 8 | 6028.20 | 12132 | 5936 | 3194 | 21310 | -1910.23 |
| 130 | 8 | 5617.78 | 11306 | 5496 | 2498 | 19348 | -1716.28 |
| 131 | 8 | 5724.66 | 11278 | 5480 | 2888 | 19694 | -1746.17 |
| 132 | 8 | 5676.64 | 11390 | 5576 | 3202 | 20216 | -1817.42 |
| 133 | 8 | 6235.91 | 11803 | 5812 | 2983 | 20646 | -1801.26 |
| 134 | 8 | 5879.02 | 11711 | 5792 | 2692 | 20243 | -1795.50 |
| 135 | 8 | 5968.41 | 11914 | 5908 | 3055 | 20925 | -1869.57 |
| 136 | 8 | 5833.29 | 11424 | 5660 | 2982 | 20114 | -1785.09 |
| 137 | 8 | 6011.61 | 11816 | 5884 | 2990 | 20738 | -1840.80 |
| 138 | 8 | 5603.74 | 11067 | 5488 | 2439 | 19042 | -1679.78 |
| 139 | 4 | 5736.88 | 12919 | 4296 | 2255 | 19518 | -1722.64 |
| 140 | 2 | 5546.31 | 13539 | 4044 | 2118 | 19749 | -1775.34 |
| 141 | 8 | 5780.56 | 11025 | 5496 | 2614 | 19183 | -1675.31 |
| 142 | 8 | 5558.83 | 10618 | 5296 | 2869 | 18831 | -1659.02 |
| 143 | 8 | 5513.12 | 10667 | 5324 | 2707 | 18746 | -1654.11 |
| 144 | 1 | 5246.55 | 14450 | 0 | 3954 | 18452 | -1650.68 |
| 145 | 8 | 5556.62 | 10784 | 5464 | 3800 | 20096 | -1817.42 |
| 146 | 8 | 5228.65 | 10155 | 5120 | 2291 | 17614 | -1548.17 |
| 147 | 8 | 5249.93 | 10317 | 5228 | 2570 | 18163 | -1614.13 |
| 148 | 8 | 4960.05 | 9623 | 4840 | 2353 | 16864 | -1487.99 |
| 149 | 8 | 5064.38 | 9995 | 5088 | 3005 | 18136 | -1633.95 |
| 150 | 8 | 4677.84 | 8944 | 4468 | 2474 | 15934 | -1407.02 |
| 151 | 8 | 4666.00 | 9294 | 4696 | 2337 | 16375 | -1463.63 |
| 152 | 8 | 4421.25 | 8570 | 4288 | 2320 | 15226 | -1350.59 |
| 153 | 8 | 4197.89 | 8195 | 4072 | 1961 | 14276 | -1259.76 |
| 154 | 8 | 3999.47 | 8013 | 4000 | 2287 | 14348 | -1293.57 |
| 155 | 4 | 4186.58 | 9389 | 3156 | 2195 | 14788 | -1325.18 |
| 156 | 8 | 3622.10 | 7187 | 3560 | 1720 | 12515 | -1111.61 |
| 157 | 8 | 3636.42 | 7096 | 3508 | 1685 | 12337 | -1087.57 |
| 158 | 8 | 3565.22 | 6885 | 3420 | 1687 | 12040 | -1059.35 |
| 159 | 8 | 3490.57 | 6794 | 3368 | 1836 | 12046 | -1069.43 |
| 160 | 1 | 3277.16 | 8944 | 0 | 1620 | 10612 | -916.85 |
| 161 | 8 | 3055.07 | 5961 | 2924 | 1426 | 10359 | -912.99 |
| 162 | 8 | 2888.54 | 5702 | 2808 | 1351 | 9909 | -877.56 |
| 163 | 8 | 3149.26 | 6309 | 3156 | 1607 | 11120 | -996.34 |
| 164 | 8 | 2726.39 | 5765 | 2876 | 1400 | 10089 | -920.33 |
| 165 | 1 | 2880.77 | 8465 | 0 | 2334 | 10847 | -995.78 |
| 166 | 4 | 2641.40 | 6173 | 0 | 3445 | 9666 | -878.08 |
| 167 | 1 | 2736.32 | 7740 | 0 | 2130 | 9918 | -897.71 |
| 168 | 8 | 2740.38 | 5342 | 2736 | 1462 | 9588 | -855.95 |
| 169 | 4 | 2678.39 | 6187 | 2121 | 995 | 9351 | -834.08 |
| 170 | 8 | 2819.85 | 5397 | 2848 | 1535 | 9828 | -876.02 |
| 171 | 8 | 2772.89 | 5229 | 2736 | 1775 | 9788 | -876.89 |
| 172 | 8 | 2625.41 | 5021 | 2672 | 1253 | 8994 | -796.07 |
| 173 | 4 | 2290.52 | 5290 | 1833 | 929 | 8100 | -726.18 |
| 174 | 8 | 2351.92 | 4152 | 1626 | 1026 | 6852 | -562.51 |
| 175 | 8 | 2071.30 | 4086 | 2124 | 1109 | 7367 | -661.96 |
| 176 | 4 | 2129.40 | 4477 | 1581 | 979 | 7085 | -619.45 |
| 177 | 4 | 2013.89 | 4211 | 1467 | 788 | 6514 | -562.51 |
| 178 | 8 | 1654.19 | 3183 | 1672 | 974 | 5877 | -527.85 |
| 179 | 8 | 1701.58 | 3147 | 1236 | 712 | 5143 | -430.18 |
| 180 | 1 | 1387.17 | 3668 | 0 | 690 | 4406 | -377.35 |
| 181 | 8 | 1612.12 | 2861 | 1532 | 819 | 5260 | -455.99 |
| 182 | 8 | 1328.04 | 2334 | 942 | 668 | 3992 | -332.99 |
| 183 | 8 | 1264.44 | 2178 | 1152 | 841 | 4219 | -369.32 |
| 184 | 1 | 936.97 | 2543 | 0 | 735 | 3326 | -298.63 |
| 185 | 4 | 896.75 | 1857 | 651 | 389 | 2945 | -256.03 |
| 186 | 1 | 594.87 | 1539 | 0 | 292 | 1879 | -160.52 |
| 187 | 8 | 710.11 | 1239 | 507 | 348 | 2142 | -178.99 |
| 188 | 1 | 385.19 | 995 | 0 | 196 | 1239 | -106.73 |
| 189 | 4 | 403.60 | 764 | 0 | 372 | 1184 | -97.55 |

## Wall projection

| measure | value |
| --- | ---: |
| chosen non-MAP blocks | 221,374 |
| candidates examined | 618,965 |
| retained local search time | 8.689549 s |
| local throughput | 71,230.967 candidates/s |
| dc1 reference throughput | 363,456.964 candidates/s |
| conservative projection throughput | 71,230.967 candidates/s |
| projected added decode time | 8.689549 s |
| cold native headroom | 211.470886 s — PASS |
| warm governed headroom | 691.470886 s — PASS |

The 8.690 s value is a `[macOS-CPU advisory]` projection from scorer-free search work, not a T4 runtime measurement. It is sufficient only to show that time did not cause this closure.

## Stop-rule disposition

**CLOSED(FAMILY A at full-population n600 FX5 scope).** The real packet fails the +3 KiB gate by 277,621 B and is worse than direct HPAC in every group. Do not build the receiver/archive chain for this representation. Reopening requires a materially different representation that removes the explicit non-MAP position/length census; retuning block size or hash width inside this packet family is not enough.

Family B/C remains outside this charter and blocked on the jo1u harvest. This run does not promote or kill those families.

## LIVE-HYPOTHESES

- A representation that makes surprise locations implicit in a shared task-cell certificate could still remove the 227.4 KB position field; that is plausible because location metadata, not hash compute, is the largest measured debt. It is not a continuation of this sparse-position Family A packet.
- Quotient/certificate coding that amortizes hash-length choices globally could remove the 101.8 KB length table; that is plausible only if the receiver derives widths from counted shared structure rather than storing one width per block.
- Family B/C may remain viable after the jo1u materializer harvest because they attack the representation boundary rather than optimizing this closed hash-question packet; no evidence from this sweep prices them.

## DEAD-ENDS

- Monolithic hash/free-run: inherited dead at the 26.2–27.9 governed reachable-rank-bit wall.
- Sparse-grid Family A with real positions, per-block lengths, SHA-prefix questions, and b in {1,2,4,8}: closed by the full n600 packet, -274,549 B versus the shipped member.
- Block-size retuning within {1,2,4,8}: closed; the best variable mixture gains only 1,434 B over fixed b=8.
- Cheap independent factorization across adjacent HPAC groups: invalid because the production conditional law and adaptive corrector depend on earlier decoded observations.
- Receiver/archive construction for dc1s: stopped by the mandatory +3 KiB credit gate.

**OWN-VEHICLE FRONTIER — UNMOVED:** fx5_e1, S=0.14823186109359, archive=180,386 B, SHA-256 `4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841` `[contest-CUDA T4 n600]`.
