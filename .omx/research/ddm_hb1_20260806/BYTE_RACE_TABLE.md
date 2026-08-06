# ddm_hb1 2026-08-06 byte race table

Scope: n600 5-class semantic label planes only. Axis is byte-only and
scorer-free. `upstream/evaluate.py` was not run and no SegNet/PoseNet forwards
were run. Rate term uses `W = 37545489` bytes.

## Winner Per Payload

| payload | best measured incumbent | incumbent bytes | incumbent S_rate | HPAC total on OUR payload | HPAC vs incumbent bytes | HPAC vs incumbent S_rate | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| tq1c parent argmax labels | PP1 KT temporal context-arith | 142001 | 0.094552637 | not measured | not measured | not measured | HPAC queued, no local CUDA/trained OUR-label checkpoint |
| GT `lstars` labels | PP1 KT temporal context-arith | 173617 | 0.115604434 | not measured | not measured | not measured | HPAC queued, no local CUDA/trained OUR-label checkpoint |

External PR130 anchor, not an HB1 result: PR130's retained CPR1 HPAC stream is
116980 token bytes plus a 15164 byte packed HPAC model blob, total 132144 bytes
for PR130's own semantic payload. That would be 9857 bytes below the tq1c KT
incumbent, equal to `-0.006563372` S_rate, but it was not trained or encoded on
our tq1c or GT label payloads in this unit.

## Required Matrix

| payload | coder | token/stream bytes | counted model bytes | total bytes | S_rate | vs incumbent bytes | vs incumbent S_rate | decode equality status | wall-clock status |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| tq1c parent argmax labels | PR130 integer HPAC | not measured | not measured | not measured | not measured | not measured | not measured | BLOCKED: no trained OUR-label HPAC checkpoint; no substitution | not run locally; CUDA unavailable |
| tq1c parent argmax labels | PP1 KT temporal context-arith | 142001 | 0 | 142001 | 0.094552637 | 0 | 0.000000000 | PARTIAL: n600 closed-form length; n=6 bit-exact range proof; no persisted full n600 stream | imported TK1 aggregate run 171.402 s |
| tq1c parent argmax labels | Brotli q11 raw uint8 | 368760 | 0 | 368760 | 0.245542148 | 226759 | 0.150989510 | PASS: whole-stream decompress equality | imported TK1 aggregate run 171.402 s |
| tq1c parent argmax labels | LZMA1 preset 9 extreme raw uint8 | 354900 | 0 | 354900 | 0.236313342 | 212899 | 0.141760705 | PASS: whole-stream decompress equality | imported TK1 aggregate run 171.402 s |
| tq1c parent argmax labels | SMEVR | not measured | not measured | not measured | not measured | not measured | not measured | BLOCKED: existing R7 SMEVR path is token-tensor scoped and capped below full n600 label-map size | not run locally |
| GT `lstars` labels | PR130 integer HPAC | not measured | not measured | not measured | not measured | not measured | not measured | BLOCKED: no trained OUR-label HPAC checkpoint; no substitution | not run locally; CUDA unavailable |
| GT `lstars` labels | PP1 KT temporal context-arith | 173617 | 0 | 173617 | 0.115604434 | 0 | 0.000000000 | PARTIAL: n600 closed-form length; n=6 bit-exact range proof; no persisted full n600 stream | imported TK1 aggregate run 171.402 s |
| GT `lstars` labels | Brotli q11 raw uint8 | 424728 | 0 | 424728 | 0.282808941 | 251111 | 0.167204508 | PASS: whole-stream decompress equality | imported TK1 aggregate run 171.402 s |
| GT `lstars` labels | LZMA1 preset 9 extreme raw uint8 | 409989 | 0 | 409989 | 0.272994846 | 236372 | 0.157390412 | PASS: whole-stream decompress equality | imported TK1 aggregate run 171.402 s |
| GT `lstars` labels | SMEVR | not measured | not measured | not measured | not measured | not measured | not measured | BLOCKED: existing R7 SMEVR path is token-tensor scoped and capped below full n600 label-map size | not run locally |

## Extra Imported Baselines

| payload | coder | total bytes | S_rate | vs incumbent bytes | vs incumbent S_rate | decode equality status |
|---|---|---:|---:|---:|---:|---|
| tq1c parent argmax labels | bz2-9 raw uint8 | 285394 | 0.190032150 | 143393 | 0.095479513 | PASS: whole-stream decompress equality |
| tq1c parent argmax labels | zlib-9 raw uint8 | 504452 | 0.335893881 | 362451 | 0.241341243 | PASS: whole-stream decompress equality |
| GT `lstars` labels | bz2-9 raw uint8 | 338593 | 0.225455181 | 164976 | 0.109850747 | PASS: whole-stream decompress equality |
| GT `lstars` labels | zlib-9 raw uint8 | 581266 | 0.387041170 | 407649 | 0.271436736 | PASS: whole-stream decompress equality |

