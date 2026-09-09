"""RC3H counted 24-weight shared logistic mixers; deterministic integer decode.

Both families preserve IHS1 exactly. The initial int8 weights and the learning
rate byte are counted in the rider. Every subsequent state update uses decoded
bits, known row geometry and integer arithmetic only. No fitted table is code.
"""
from __future__ import annotations

import struct
from collections import defaultdict

import numpy as np

try:
    from . import rc2_hpac_semistatic_mixing as base
except ImportError:
    from experiments import ddm_rc2_hpac_semistatic_mixing_codec as base

MAGIC = b'RC3H'
HEADER = struct.Struct('<4sBBBBHIII')
FAMILIES = {1:'position_sibling', 2:'order2'}
WEIGHT_COUNT = 24
WEIGHT_Q = 65536


def bucket(value: int | None) -> int:
    return 3 if value is None else (0 if value < 0 else 1 if value == 0 else 2)


class Features:
    """A causal predictor bank shared by encoder and receiver."""

    def __init__(self, family: int):
        if family not in FAMILIES:
            raise ValueError('unknown RC3 family')
        self.family = family
        self.base = base._AdaptiveExperts()
        self.banks = [{} for _ in range(15)]
        self.history = defaultdict(list)
        self.group = -1
        self.width = None
        self.sibling = None
        self.current = []
        self.depth = 0

    def start_row(self, count: int, depth: int) -> None:
        if count != self.width:
            self.group += 1
            self.width = count
            self.sibling = None
        self.depth = depth
        self.current = []

    def predict(self, pos: int, node: int, bitpos: int) -> tuple[list[int], list, list]:
        d = self.depth
        h = self.history[d]
        prev, prev2 = (h[-1] if h else None), (h[-2] if len(h)>1 else None)
        sib = self.sibling[pos] if self.sibling is not None else None
        ps,p2s,ss = bucket(prev),bucket(prev2),bucket(sib)
        quart = min(3,4*pos//self.width)
        keys = self.base.keys(d,node,bitpos,prev)
        x = [base.stretch(p) for p in self.base.predictions(keys)]
        common = [(d,node)]*5
        if self.family == 1:
            extra = [(d,node,quart),(d,node,self.group),(d,node,pos%3),(d,node,pos%8),
                     (d,node,ss),(d,node,bucket(self.current[-1]) if pos else 3),
                     (bitpos,node),(d,node,prev,quart),(d,node,prev,self.group),(d,node,sib)]
        else:
            extra = [(d,node,ps,p2s),(d,node,prev,p2s),(d,node,ps,prev2),
                     (d,node,abs(prev or 0).bit_length(),abs(prev2 or 0).bit_length()),
                     (d,node,prev,prev2),(d,node,ps),(bitpos,node,ps,p2s),
                     (d,node,ss),(d,node,quart),(d,node,prev2)]
        for j,key in enumerate(common+extra):
            bank=self.banks[j]
            if j == 4 or (self.family == 2 and j == 10):
                z,n=bank.get(key,(0,0))
                p=max(1,min(4095,((2*z+1)*4096)//(2*n+2)))
            else:
                p=bank.get(key,2048)
            x.append(base.stretch(p))
        x.append(4096)
        return x,keys,common+extra

    def update(self, keys: list, extra: list, bit: int) -> None:
        self.base.update(keys,bit)
        for j,key in enumerate(extra):
            bank=self.banks[j]
            if j == 4 or (self.family == 2 and j == 10):
                z,n=bank.get(key,(0,0))
                bank[key]=(z+int(bit == 0),n+1)
            else:
                p=bank.get(key,2048)
                bank[key]=base._updated_probability(p,bit,(3,4,6,7)[j] if j<4 else 5)

    def symbol(self, value: int) -> None:
        self.current.append(value)
        self.history[self.depth].append(value)

    def finish_row(self) -> None:
        self.sibling=self.current


def walk(counts: list[int], depths: np.ndarray, family: int, weights: np.ndarray,
         learning_shift: int, *, source_rows: list[np.ndarray] | None = None,
         payload: bytes | None = None, trace: bool = False) -> tuple[bytes | None, list[np.ndarray], list]:
    """Encode or decode one row stream; online learning follows each coded bit."""
    if weights.shape != (24,) or weights.dtype != np.int8:
        raise ValueError('RC3 initial weights must be 24 int8 values')
    if learning_shift not in (0,18,20,22,24):
        raise ValueError('unsupported RC3 counted learning rate')
    if (source_rows is None) == (payload is None):
        raise ValueError('exactly one of source rows and coded payload is required')
    encoder = base._RangeEncoder() if source_rows is not None else None
    decoder = base._RangeDecoder(payload) if payload is not None else None
    state=Features(family)
    w=[int(v)*2048 for v in weights.tolist()]
    result,events=[],[]
    for ri,(count,depth) in enumerate(zip(counts,depths.tolist(),strict=True)):
        state.start_row(count,int(depth))
        row=[]
        for pos in range(count):
            unsigned,node=0,1
            for bitpos,shift in enumerate(reversed(range(int(depth)))):
                x,keys,extra=state.predict(pos,node,bitpos)
                p=base.squash(base._round_div_signed(sum(a*b for a,b in zip(w,x,strict=True)),WEIGHT_Q))
                if encoder is not None:
                    bit=(int(source_rows[ri][pos])>>shift)&1
                    encoder.encode_bit(p,bit)
                else:
                    bit=decoder.decode_bit(p)
                if trace:
                    events.append(x)
                if learning_shift:
                    error=(4096 if bit==0 else 0)-p
                    for j in range(24):
                        delta=base._round_div_signed(error*x[j],1<<learning_shift)
                        w[j]=min(8*WEIGHT_Q,max(-8*WEIGHT_Q,w[j]+delta))
                state.update(keys,extra,bit)
                unsigned=(unsigned<<1)|bit
                node=2*node+bit
            sign=1<<(int(depth)-1) if depth else 0
            value=unsigned-(1<<int(depth)) if depth and unsigned>=sign else unsigned
            state.symbol(value)
            row.append(value)
        state.finish_row()
        result.append(np.asarray(row,dtype=np.int16))
    return (encoder.finish() if encoder is not None else None),result,events


def encode(body: bytes, counts: list[int], family: int, weights: np.ndarray,
           learning_shift: int, *, trace: bool = False) -> tuple[bytes, dict]:
    rows,depths=base.unpack_rows(body,counts)
    prefix,packed,tail,_=base.split_ihs1(body,counts)
    if base.pack_rows(rows,depths)!=packed:
        raise ValueError('noncanonical IHS1 input')
    payload,restored,events=walk(counts,depths,family,weights,learning_shift,source_rows=rows,trace=trace)
    if any(not np.array_equal(a,b) for a,b in zip(rows,restored,strict=True)):
        raise ValueError('encoder walk changed source symbols')
    params=bytes([learning_shift])+weights.tobytes()
    header=HEADER.pack(MAGIC,1,family,24,0,len(counts),len(prefix),len(payload),len(tail))
    return header+prefix+params+payload+tail,{'payload':payload,'parameters':params,'events':events}


def restore_hpac(rider: bytes, counts: list[int]) -> bytes:
    if len(rider)<HEADER.size:
        raise ValueError('short RC3 header')
    magic,version,family,nweights,flags,nrows,prefix_n,payload_n,tail_n=HEADER.unpack_from(rider)
    if (magic,version,nweights,flags,nrows)!=(MAGIC,1,24,0,len(counts)) or family not in FAMILIES:
        raise ValueError('invalid RC3 header')
    if prefix_n!=4+(len(counts)+1)//2 or payload_n<4:
        raise ValueError('invalid RC3 component lengths')
    if len(rider)!=HEADER.size+prefix_n+25+payload_n+tail_n:
        raise ValueError('RC3 byte census mismatch')
    off=HEADER.size
    prefix=rider[off:off+prefix_n]
    off+=prefix_n
    depths=base._depths(prefix,len(counts))
    rate=rider[off]
    weights=np.frombuffer(rider[off+1:off+25],dtype=np.int8)
    off+=25
    _,rows,_=walk(counts,depths,family,weights,rate,payload=rider[off:off+payload_n])
    return prefix+base.pack_rows(rows,depths)+rider[off+payload_n:]
