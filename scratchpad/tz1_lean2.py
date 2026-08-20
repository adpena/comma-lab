import sys, json, math, time
sys.path.insert(0,'src'); sys.path.insert(0,'experiments')
import numpy as np
from tac.optimization import ddm_ix2_archive_container as IX2
from tac.witness_dsl.ax1_pool_a_levers_20260730 import margin_coupled_level_map
TOK=np.load('/Volumes/VertigoDataTier/pact/ddm_br1_20260803/cx1_tokens.npy')
FIELD=np.load('/Volumes/VertigoDataTier/pact/ddm_sg1_20260731/cell_flip_mass.npy')
t0=time.time()
base=len(IX2.encode_token_frame(TOK,levels=16))
# coder winner on live blocks
b16,d16=IX2._factor_mode_delta(TOK,16)
res=np.ascontiguousarray(np.transpose(d16,(1,2,3,0)))
cr,_=IX2.code_block(IX2._pack_nibbles(res.reshape(-1))); cb,_=IX2.code_block(IX2._pack_nibbles(b16.reshape(-1)))
print('CODER-WINNER residual=%s base=%s'%(IX2.CODER_NAMES[cr],IX2.CODER_NAMES[cb]),'t',round(time.time()-t0,1))
# helpers
def c2t(c): return c.astype(np.float64)/15.0*2-1
def t2c(t): return np.rint((np.clip(t,-1,1)+1)*0.5*15).astype(np.uint8)
def apply_map(tok,lm):
    t=np.clip(c2t(tok),-1,1); L=(lm.astype(np.float64)-1)[None,:,:,None]; x=(t+1)*0.5
    return t2c(np.round(x*L)/L*2-1)
def act(tok):
    b,_=IX2._factor_mode_delta(tok,16); return (tok!=b[None]).sum(axis=(0,3)).astype(float)
A=act(TOK)
def mapcost(lm):
    vals=np.unique(lm); rm={int(v):i for i,v in enumerate(vals)}
    idx=np.vectorize(rm.get)(lm).astype(np.uint8).reshape(-1); _,cd=IX2.code_block(idx.tobytes()); return len(cd)
# ARM B adaptive: ladder [16,12,8,4], derived(activity) vs margin(flip-mass)
for ladder in ([16,12,8,4],[16,8]):
    bl,ml,nt=max(ladder),min(ladder),len(ladder)
    dmap=margin_coupled_level_map(A,base_levels=bl,min_levels=ml,n_tiers=nt)
    cd=apply_map(TOK,dmap); bd=len(IX2.encode_token_frame(cd,levels=16)); rtd=np.array_equal(IX2.decode_token_frame(IX2.encode_token_frame(cd,levels=16)),cd)
    mmap=margin_coupled_level_map(FIELD.astype(float),base_levels=bl,min_levels=ml,n_tiers=nt)
    cm=apply_map(TOK,mmap); bm=len(IX2.encode_token_frame(cm,levels=16)); rtm=np.array_equal(IX2.decode_token_frame(IX2.encode_token_frame(cm,levels=16)),cm)
    mc=mapcost(mmap)
    print('ADAPT ladder=%s DERIVED saved=%d(0-map,rt=%s) hist=%s | MARGIN saved_gross=%d map_cost=%d net=%d(rt=%s) hist=%s'%(
        ladder, base-bd, rtd, dict(zip(*[x.tolist() for x in np.unique(dmap,return_counts=True)])),
        base-bm, mc, base-bm-mc, rtm, dict(zip(*[x.tolist() for x in np.unique(mmap,return_counts=True)]))))
# global-L reference (fast) with break-evens
for L in (15,14,12,10,8):
    cL=np.rint((TOK/15.0)*(L-1)).astype(np.uint8); bL=len(IX2.encode_token_frame(cL,levels=L)); print('GLOBAL-L L=%d saved=%d'%(L,base-bL))
print('lean2_done t',round(time.time()-t0,1))
