import sys, json, lzma, zlib, math, time
sys.path.insert(0,'src'); sys.path.insert(0,'experiments')
import numpy as np
from tac.optimization import ddm_ix2_archive_container as IX2
from tac.witness_dsl.ax1_pool_a_levers_20260730 import margin_coupled_level_map
TOK=np.load('/Volumes/VertigoDataTier/pact/ddm_br1_20260803/cx1_tokens.npy')
t0=time.time()
base=len(IX2.encode_token_frame(TOK,levels=16)); print('BASE',base,'t',round(time.time()-t0,1))
# ARM D clamp mass (coder-free, instant)
h=np.bincount(TOK.reshape(-1),minlength=16).astype(float); tot=h.sum()
print('CLAMP lvl0=%.4f lvl15=%.4f extremes=%.4f'%(h[0]/tot,h[15]/tot,(h[0]+h[15])/tot))
perch={}
for k in range(TOK.shape[3]):
    hk=np.bincount(TOK[...,k].reshape(-1),minlength=16).astype(float); tk=hk.sum()
    perch['ch%d'%k]=round((hk[0]+hk[15])/tk,4)
print('CLAMP per-channel extremes',perch)
# ARM E lzma filter race (no brotli; bounded)
b16,d16=IX2._factor_mode_delta(TOK,16)
res=np.ascontiguousarray(np.transpose(d16,(1,2,3,0))); 
payloads={'residual':IX2._pack_nibbles(res.reshape(-1)),'base':IX2._pack_nibbles(b16.reshape(-1))}
lz=lambda pay,lc,lp,pb: len(lzma.compress(pay,format=lzma.FORMAT_RAW,filters=[{'id':lzma.FILTER_LZMA1,'dict_size':1<<24,'lc':lc,'lp':lp,'pb':pb}]))
E={}
for nm,pay in payloads.items():
    shipped=lz(pay,3,0,0); bestv=None;bb=None
    for lc in range(5):
        for lp in range(3):
            if lc+lp>4: continue
            for pb in range(3):
                v=lz(pay,lc,lp,pb)
                if bb is None or v<bb: bb=v; bestv=(lc,lp,pb)
    E[nm]={'shipped_lzma_lc3lp0pb0':shipped,'best':bestv,'best_bytes':bb,'gain_vs_shipped':shipped-bb}
    print('LZMA',nm,'shipped',shipped,'best',bestv,bb,'gain',shipped-bb)
print('lean_done t',round(time.time()-t0,1))
json.dump({'base':base,'clamp':{'lvl0':h[0]/tot,'lvl15':h[15]/tot,'ext':(h[0]+h[15])/tot,'perch':perch},'lzma':E},open('/Volumes/VertigoDataTier/pact/ddm_tz1_20260804/tz1_lean.json','w'),indent=1)
