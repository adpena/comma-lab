# PR101 Seeded Selector Adapter Profile

- Schema: `pr101_seeded_selector_adapter_profile_v1`
- Score claim: `false`
- Ready for exact eval dispatch: `false`
- Selector codes: `600`
- FEC6 selector payload bytes: `249`
- Candidate count: `28`
- Best candidate: `seeded_selector::constant::code0::60294ecaa6ca`
- Best candidate bytes: `326`
- Best saving vs FEC6 selector: `-77`
- Can meet target: `false`

## Order Entropy

- Global entropy floor bytes: `241`
- First-order transition floor bytes plus first symbol: `227`
- First-order transition byte-closed floor with u16 counts: `739`
- Run count: `505`
- Run mean length: `1.188`
- Run max length: `7`

| context_mod | zero-model floor bytes | u16 model bytes | byte-closed floor bytes |
| ---: | ---: | ---: | ---: |
| 100 | 193 | 3200 | 3393 |
| 50 | 212 | 1600 | 1812 |
| 25 | 221 | 800 | 1021 |
| 16 | 229 | 512 | 741 |
| 8 | 236 | 256 | 492 |
| 4 | 239 | 128 | 367 |
| 2 | 240 | 64 | 304 |

Zero-model floors are diagnostic only; byte-closed candidates must charge the context/transition model bytes or derive them from an allowed runtime prior.

## Charged Candidates

| candidate | mode | bytes | saving | seed bytes | model bytes | residual bytes | mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seeded_selector::constant::code0::60294ecaa6ca | constant | 326 | -77 | 0 | 1 | 308 | 466 |
| seeded_selector::constant::code2::0ee9c65b698b | constant | 329 | -80 | 0 | 1 | 311 | 471 |
| seeded_selector::constant::code13::f34ecbfe5073 | constant | 347 | -98 | 0 | 1 | 329 | 508 |
| seeded_selector::constant::code7::83ea9a547d44 | constant | 358 | -109 | 0 | 1 | 340 | 529 |
| seeded_selector::seed_mod16::pcg64_seed2::026e6fb8317b | seed_mod16 | 366 | -117 | 2 | 0 | 347 | 544 |
| seeded_selector::seed_mod16::pcg64_seed1::5e20da170376 | seed_mod16 | 367 | -118 | 1 | 0 | 349 | 548 |
| seeded_selector::seed_mod16::pcg64_seed4::a643ab7c3b31 | seed_mod16 | 368 | -119 | 4 | 0 | 347 | 543 |
| seeded_selector::histogram_shuffle::pcg64_seed2::63e8e3e73d0c | histogram_shuffle | 370 | -121 | 2 | 32 | 319 | 487 |
| seeded_selector::histogram_shuffle::pcg64_seed1::3c7c0bb1b6ed | histogram_shuffle | 372 | -123 | 1 | 32 | 322 | 494 |
| seeded_selector::seed_mod16::pcg64_seed8::10a248ff0e33 | seed_mod16 | 372 | -123 | 8 | 0 | 347 | 543 |
| seeded_selector::histogram_shuffle::pcg64_seed4::5f6d77a5ea6c | histogram_shuffle | 374 | -125 | 4 | 32 | 321 | 491 |
| seeded_selector::constant::code1::9609c0552d37 | constant | 376 | -127 | 0 | 1 | 358 | 565 |
| seeded_selector::histogram_shuffle::pcg64_seed8::579d58abefc3 | histogram_shuffle | 377 | -128 | 8 | 32 | 320 | 489 |
| seeded_selector::seed_mod16::pcg64_seed16::05bd79be9a92 | seed_mod16 | 379 | -130 | 16 | 0 | 346 | 542 |
| seeded_selector::histogram_shuffle::pcg64_seed16::fc5f9d7e5f0c | histogram_shuffle | 381 | -132 | 16 | 32 | 316 | 481 |
| seeded_selector::constant::code4::43cddbe54fc7 | constant | 381 | -132 | 0 | 1 | 363 | 575 |
| seeded_selector::constant::code9::5eca230541b2 | constant | 381 | -132 | 0 | 1 | 363 | 576 |
| seeded_selector::constant::code14::c45bcda199a8 | constant | 385 | -136 | 0 | 1 | 367 | 583 |
| seeded_selector::constant::code11::c202065c1f18 | constant | 385 | -136 | 0 | 1 | 367 | 584 |
| seeded_selector::constant::code5::368616cd38ff | constant | 387 | -138 | 0 | 1 | 369 | 587 |
| seeded_selector::constant::code6::9dcafe8c9715 | constant | 388 | -139 | 0 | 1 | 370 | 589 |
| seeded_selector::constant::code8::149c7f27c1bb | constant | 388 | -139 | 0 | 1 | 370 | 590 |
| seeded_selector::constant::code3::ec53d36458e1 | constant | 389 | -140 | 0 | 1 | 371 | 591 |
| seeded_selector::constant::code10::4164b42e8e34 | constant | 390 | -141 | 0 | 1 | 372 | 593 |
| seeded_selector::constant::code12::f41e26545e4a | constant | 390 | -141 | 0 | 1 | 372 | 594 |
| seeded_selector::constant::code15::275596be43cc | constant | 393 | -144 | 0 | 1 | 375 | 599 |
| seeded_selector::seed_mod16::pcg64_seed32::15a629ad1770 | seed_mod16 | 395 | -146 | 32 | 0 | 346 | 541 |
| seeded_selector::histogram_shuffle::pcg64_seed32::980596d34a3b | histogram_shuffle | 397 | -148 | 32 | 32 | 316 | 481 |

## Blocker

- blocked: `true`
- reason: Seed-derived selector priors still require enough residual overrides that the charged adapter payload does not beat the current FEC6 selector payload.
- reactivation_criteria: Reopen with a predictor whose residual override payload plus all charged model/seed bytes is smaller than the current FEC6 selector payload, then materialize a runtime adapter and run runtime-consumption/no-op plus exact CPU/CUDA eval.
