import sys, math, time
sys.path.insert(0,'src'); sys.path.insert(0,'experiments')
import numpy as np
from tac.optimization import ddm_ix2_archive_container as IX2
from tac.optimization.pfs1_warp_receiver import ST_GRID
TOK=np.load('/Volumes/VertigoDataTier/pact/ddm_br1_20260803/cx1_tokens.npy')
t0=time.time()
def bitpack(codes,bits):
    c=np.asarray(codes,dtype=np.uint8).reshape(-1)
    b=((c[:,None]>>np.arange(bits-1,-1,-1))&1).astype(np.uint8).reshape(-1)
    return np.packbits(b).tobytes()
for L in (16,14,8):
    cL=np.rint((TOK/15.0)*(L-1)).astype(np.uint8) if L!=16 else TOK
    b,d=IX2._factor_mode_delta(cL,L); res=np.ascontiguousarray(np.transpose(d,(1,2,3,0))).reshape(-1)
    bpc=max(1,math.ceil(math.log2(L)))
    nib=IX2._pack_nibbles(res); _,nc=IX2.code_block(nib)
    tp=bitpack(res,bpc); _,tc=IX2.code_block(tp)
    print('DEPTH L=%d bpc=%d nibble_raw=%d nibble_coded=%d tight_raw=%d tight_coded=%d gain=%d'%(L,bpc,len(nib),len(nc),len(tp),len(tc),len(nc)-len(tc)))
fmt,coded=IX2.encode_exact_table(ST_GRID)
print('STGRID knots=%d fmt=%s table_bytes=%d comb(11,5)=%d comb(11,6)=%d'%(len(ST_GRID),['f16','f32','f64','scaled_int'][fmt],len(coded),math.comb(11,5),math.comb(11,6)))
print('lean3_done t',round(time.time()-t0,1))
