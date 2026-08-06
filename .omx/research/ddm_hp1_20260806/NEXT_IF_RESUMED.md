# ddm_hp1 next if resumed

Do not rerun the <=10K static-context HP1 race on tq1c unless the token stream
changes. It is negative on the measured object:

- shipped `IX2TOK01`/Brotli-q11: 341,296 B.
- forced `IX2TOK01`/LZMA1: 349,811 B.
- best HP1 learned+model: 456,166 B.
- delta vs shipped: +114,870 B, `dS_rate=+0.076487217945`.

The useful successor is not another hash-table pass. The only measured headroom
is the high-cardinality prev+spatial empirical floor (289,076 ideal bytes), which
requires a model that generalizes those contexts without storing a large
video-derived table. Fire only if a successor can name how its counted model
captures that structure under a real byte budget and still decode causally.

Fire order for any successor:

1. Use the same tq1c extraction code and token sha
   `1a46a51909b150bc1fc320cb6f66f52cc53472e6f830c911c2ea7bbec2bbdcc3`.
2. Keep it scorer-free until a net byte win exists after counted model bytes.
3. Count every video-trained weight/table byte in the candidate frame.
4. Require exact decode equality and canonical re-encode equality.
5. Race against 341,296 B shipped IX2 and 349,811 B forced IX2/LZMA before any
   receiver or composition work.

#918 wording to preserve: explicit LZ/rank/basis token-coder closure stands; HP1
measures the learned-conditional-prior reopening and finds this <=10K formulation
negative on the live stream.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
