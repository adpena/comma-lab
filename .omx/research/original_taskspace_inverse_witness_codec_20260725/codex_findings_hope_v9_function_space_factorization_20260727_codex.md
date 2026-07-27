# Codex findings — HOPE as a V9 post-training factorization probe

Date: 2026-07-27  
Source: Mobahi and Bartlett, *Hilbert Operator for Progressive Encoding
(HOPE)*, arXiv:2607.21366  
Scope: executor/adversarial applicability check; no score, candidate, lane
promotion, or pointer claim

## Verdict

Useful as a design principle and post-training rate probe; invalid as a
drop-in neuron-merge algorithm for the current V9 decoder.

HOPE ranks pruning, neuron merging, and macro-block eviction by functional
operator distortion instead of raw parameter magnitude. That directly attacks
a Pact failure mode: parameter-space size or norm is not the same as
receiver/scorer value. It also offers one common arbitration surface for
micro-neuron and macro-layer reductions.

The paper's closed-form construction assumes positively homogeneous
activations and uses BatchNorm statistics for its data-free surrogate. V9 uses
a FiLM-conditioned `tanh(beta * sin(wx))` HOSC trunk and has no compatible BN
surrogate. A ReLU/BN parent-neuron merge copied literally would therefore
change the function family and has no production authority.

## Contest-native adaptation

After the first real G111 stage exists, define each hidden unit by its function
over the exact frozen contest measure:

- all physical odd-Y1 modulation codes from that G112 stage;
- the exact public polar coordinate grid and HOSC activation;
- the parsed public-wire quantization/realization boundary;
- optional scorer-cell weighting from G109/G120 telemetry.

Build an empirical functional Gram operator over complete FiLM-conditioned
unit contributions, not isolated incoming weights. Candidate actions may then
include:

1. merge a pair of hidden units while jointly transforming incoming weights,
   outgoing heads, and every associated FiLM row;
2. project a hidden layer to a smaller shared functional subspace;
3. evict or fuse a macro layer only when a source-compatible HOSC/FiLM
   projection exists;
4. re-entropy-code the projected exact G105 tensors.

Every action is projected back into a legal versioned V9 packet and arbitrated
by exact whole-object marginal value:

`Delta action = 100*Delta d_seg + Delta V_pose + 25*Delta bytes/37_545_489`.

The initial probe may omit `Delta V_pose` only as a conservative screen:
retain every action whose Seg-plus-rate result remains below the live target,
then send all survivors through G119. Magnitude, parameter count, Hilbert
distance, or training loss alone never admits a change.

## Why it is high leverage

The exact G111 structural preflight assigns 72,430 bytes to model tensors and
38,400 bytes to raw odd-Y1 codes before packet/container overhead. The shared
model is therefore a major rate home. Train-high-then-functionally-factor can
search a better distortion basin first and remove functionally redundant
capacity afterward; training a smaller model from scratch need not reach the
same basin.

## Activation gate

Do not delay the first real G111/G120/G121/G119 whole-object row. Activate this
probe only when a physical retained G112 stage exists. The smallest real
deliverable is a deterministic merge/evict proposal ledger that:

- binds the physical G112 receipt and public runtime tree;
- reports predicted functional distortion only as a proposal;
- recompiles every proposal through exact G105;
- measures exact public-wire `d_seg` and archive bytes;
- preserves all non-obstructed proposals for G119;
- records a scoped negative per action rather than killing the family.

This is a factorization child of the current codec, not a replacement for its
authority closure.

