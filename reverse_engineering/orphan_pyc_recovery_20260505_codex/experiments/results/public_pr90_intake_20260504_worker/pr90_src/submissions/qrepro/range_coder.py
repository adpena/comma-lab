# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``41:13: cannot assign to literal here. Maybe you meant '==' instead of '='?``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``range_coder.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/results/public_pr90_intake_20260504_worker/pr90_src/submissions/qrepro/range_coder.py'
__recovery_spec__ = 'range_coder.recovery_spec.json'
__recovery_ast_error__ = "41:13: cannot assign to literal here. Maybe you meant '==' instead of '='?"

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: range_coder.cpython-312.pyc (Python 3.12)

'''Pure-Python range coder for seg_targets compression.

Stdlib-only (no constriction, no numpy needed for the coder core -- numpy
used only for input-prep helpers outside the hot loop). Shipped in the
scoring archive.

Bitstream format (standard carry-less 32-bit range coder):
- 32-bit state (low, high), byte-level renormalization
- frequency tables are fixed-point uint16 (total_freq = 2^precision, typically 1<<16)
- symbols are uint8 in range [0, n_classes)

This is the Witten/Neal/Cleary arithmetic-coding algorithm rewritten with
plain 32-bit ints and E1/E2/E3 renormalization. Implementation is
byte-identical lossless by construction (encode/decode share arithmetic).
'''
from typing import List, Sequence
TOP = 0xFFFFFFFF
HALF = 0x80000000
QUARTER = 1073741824
THREE_QUARTER = 0xC0000000

class RangeEncoder:
    '''Encode symbols under per-symbol CDFs into a byte stream.'''
    
    def __init__(self = None):
        self.low = 0
        self.high = TOP
        self.pending = 0
        self.out = bytearray()
        self._byte_buf = 0
        self._byte_bits = 0

    
    def _emit_bit(self = None, bit = None):
        self._byte_buf = self._byte_buf << 1 | bit
        if self._byte_bits == 8:
            self.out.append(self._byte_buf & 255)
            0 = self, self._byte_bits += 1, ._byte_bits
            self._byte_bits = 0
            return None

    
    def _emit_bit_and_pending(self = None, bit = None):
        self._emit_bit(bit)
        neg = 1 - bit
        for _ in range(self.pending):
            self._emit_bit(neg)
        self.pending = 0

    
    def encode_symbol(self = None, cum_low = None, cum_high = None, total = ('cum_low', int, 'cum_high', int, 'total', int, 'return', None)):
        '''Encode one symbol defined by its cumulative range [cum_low, cum_high).'''
        rng = (self.high - self.low) + 1
        self.high = self.low + rng * cum_high // total - 1
        self.low = self.low + rng * cum_low // total
        if self.high < HALF:
            self._emit_bit_and_pending(0)
        elif self.low >= HALF:
            self._emit_bit_and_pending(1)
        elif self.low >= QUARTER and self.high < THREE_QUARTER:
            pass
        else:
            return None
        self.low << 1 & TOP = self, self.high -= QUARTER, .high
        self.high = (self.high << 1 | 1) & TOP
        continue

    
    def finish(self = None):
        '''Flush remaining state; returns the encoded byte string.'''
        if self._byte_bits > 0:
            self.out.append(self._byte_buf << 8 - self._byte_bits & 255)
            0 = None if self.low < QUARTER else self, self.pending += 1, .pending
            self._byte_bits = 0
        return bytes(self.out)



class RangeDecoder:
    '''Decode symbols from a byte stream using per-symbol CDFs.

    Matches RangeEncoder exactly; constructor consumes the first 32 bits.
    '''
    
    def __init__(self = None, data = None):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0
        self.low = 0
        self.high = TOP
        self.code = 0
        for _ in range(32):
            self.code = (self.code << 1 | self._read_bit()) & TOP

    
    def _read_bit(self = None):
        if self.byte_pos >= len(self.data):
            return 0
        byte = self.data[self.byte_pos]
        bit = byte >> 7 - self.bit_pos & 1
        if self.bit_pos == 8:
            0 = self, self.bit_pos += 1, .bit_pos
        return bit

    
    def decode_target(self = None, total = None):
        \"\"\"Return the 'offset' in [0, total) that the current state decodes to.

        Caller must then look up which symbol owns this offset under the same
        CDF as the encoder used, and pass cum_low/cum_high to advance().
        \"\"\"
        rng = (self.high - self.low) + 1
        return (((self.code - self.low) + 1) * total - 1) // rng

    
    def advance(self = None, cum_low = None, cum_high = None, total = ('cum_low', int, 'cum_high', int, 'total', int, 'return', None)):
        '''Narrow the decoder state to match the consumed symbol.'''
        rng = (self.high - self.low) + 1
        self.high = self.low + rng * cum_high // total - 1
        self.low = self.low + rng * cum_low // total
        if self.high < HALF:
            pass
        elif self.low >= HALF:
            pass
        elif self.low >= QUARTER and self.high < THREE_QUARTER:
            pass
        else:
            return None
        self.low << 1 & TOP = self, self.code -= QUARTER, .code
        self.high = (self.high << 1 | 1) & TOP
        self.code = (self.code << 1 | self._read_bit()) & TOP
        continue



def cdfs_from_freqs(freqs = None):
    '''Build cumulative frequency array [0, f0, f0+f1, ..., total].'''
    cdf = [
        0]
    acc = 0
    for f in freqs:
        acc += f
        cdf.append(acc)
    return cdf


"""
