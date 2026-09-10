/* RLC1 Q16 causal moment geometry. All content-selected knobs arrive in cfg.
 * Generic constants: public H/W/K, 64-cell HPAC slots, six sufficient moments,
 * Q16 arithmetic, integer bit/cardinality operations. No floating arithmetic. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define H 384
#define W 512
#define K 5
#define S 8
#define Q 65536

typedef __int128 wide;
typedef struct {
  int64_t cfg[17];
  int64_t now[6][H][S], old[6][H][S];
  int lo[H][S], hi[H][S], plo[H][S], phi[H][S], ambiguous[H][S];
  uint64_t other[H][S];
} Geometry;

static int64_t floor_div(wide a, wide b) {
  wide q=a/b, r=a%b;
  return (int64_t)(q-(r<0));
}
static uint64_t root(uint64_t n) {
  uint64_t a=0, bit=(uint64_t)1<<62;
  while(bit>n) bit>>=2;
  while(bit) {
    if(n>=a+bit) { n-=a+bit; a=(a>>1)+bit; }
    else a>>=1;
    bit>>=2;
  }
  return a;
}
static int64_t nearest_even(int64_t a) {
  int64_t q=a/Q, r=a%Q;
  return q+(r>Q/2 || (r==Q/2 && (q&1)));
}
void *rlc_new(const int64_t *cfg, const uint8_t *previous) {
  Geometry *g=calloc(1,sizeof(*g));
  if(!g) return NULL;
  memcpy(g->cfg,cfg,sizeof(g->cfg));
  for(int y=0;y<H;y++) for(int s=0;s<S;s++) {
    g->lo[y][s]=g->plo[y][s]=W;
    g->hi[y][s]=g->phi[y][s]=-1;
    int runs=0, active=0;
    for(int t=0;t<W/S;t++) {
      int x=s*(W/S)+t, hit=previous && previous[y*W+x]==cfg[0];
      if(hit) {
        int64_t v[6]={1,y,y*y,x,x*y,x*x};
        for(int j=0;j<6;j++) g->old[j][y][s]+=v[j];
        if(x<g->plo[y][s]) g->plo[y][s]=x;
        g->phi[y][s]=x;
        runs+=!active;
      }
      active=hit;
    }
    g->ambiguous[y][s]=runs>1;
  }
  return g;
}
void rlc_free(void *p) { free(p); }
void rlc_observe(void *p, int n, const int64_t *positions, const int64_t *symbols) {
  Geometry *g=p;
  for(int i=0;i<n;i++) {
    int y=positions[i]/W, x=positions[i]%W, s=x/(W/S);
    if(symbols[i]==g->cfg[0]) {
      int64_t v[6]={1,y,y*y,x,x*y,x*x};
      for(int j=0;j<6;j++) g->now[j][y][s]+=v[j];
      if(x<g->lo[y][s]) g->lo[y][s]=x;
      if(x>g->hi[y][s]) g->hi[y][s]=x;
    } else g->other[y][s]|=(uint64_t)1<<(x%(W/S));
  }
}
void rlc_contexts(void *p, int n, const int64_t *positions, uint8_t *out) {
  Geometry *g=p;
  int64_t center[H][S], width[H][S];
  uint8_t valid[H][S];
  memset(valid,0,sizeof(valid));
  for(int s=0;s<S;s++) {
    int bad[H], badsum=0;
    int64_t sums[6]={0};
    for(int y=0;y<H;y++) {
      int lo=g->lo[y][s], hi=g->hi[y][s];
      uint64_t between=0;
      if(hi>lo+1) {
        int l=lo%(W/S), h=hi%(W/S);
        between=(((uint64_t)1<<h)-1)^(((uint64_t)1<<(l+1))-1);
      }
      int bothlo=lo<g->plo[y][s]?lo:g->plo[y][s];
      int bothhi=hi>g->phi[y][s]?hi:g->phi[y][s];
      bad[y]=g->ambiguous[y][s] || (g->other[y][s]&between) || bothhi-bothlo>g->cfg[10];
      int64_t c=sums[0], sy=sums[1], syy=sums[2], sx=sums[3], sxy=sums[4], sxx=sums[5];
      int64_t d=c*syy-sy*sy, b=c*sxy-sy*sx;
      if(c>=g->cfg[5]*g->cfg[4] && d>0 && !badsum && b<=g->cfg[7]*d && b>=-g->cfg[7]*d) {
        wide r=(wide)d*((wide)c*sxx-(wide)sx*sx)-(wide)b*b;
        wide den=(wide)c*c*d;
        if(r<=g->cfg[6]*den) {
          center[y][s]=floor_div(((wide)sx*d+(wide)b*(y*c-sy))*Q,(wide)c*d);
          uint64_t squared=r>0?(uint64_t)(r*g->cfg[9]*Q*Q/den):0;
          int64_t w=(int64_t)root(squared), minimum=Q/g->cfg[8];
          width[y][s]=w>minimum?w:minimum;
          valid[y][s]=1;
        }
      }
      for(int j=0;j<6;j++) {
        sums[j]+=g->cfg[4]*g->now[j][y][s]+g->old[j][y][s];
        if(y>=g->cfg[3]) sums[j]-=g->cfg[4]*g->now[j][y-g->cfg[3]][s]+g->old[j][y-g->cfg[3]][s];
      }
      badsum+=bad[y];
      if(y>=g->cfg[3]) badsum-=bad[y-g->cfg[3]];
    }
  }
  for(int i=0;i<n;i++) {
    int y=positions[i]/W,x=positions[i]%W;
    out[i]=8;
    if(y<g->cfg[1] || y>=g->cfg[2]) continue;
    int64_t best=INT64_MAX;
    for(int s=0;s<S;s++) if(valid[y][s]) {
      int64_t a=(int64_t)x*Q-center[y][s]+width[y][s];
      int64_t b=(int64_t)x*Q-center[y][s]-width[y][s];
      if(a<0) a=-a;
      if(b<0) b=-b;
      int64_t v=nearest_even(a<b?a:b);
      if(v<best) best=v;
    }
    if(best==INT64_MAX) out[i]=7;
    else { int j=0; while(j<6 && best>g->cfg[11+j]) j++; out[i]=(uint8_t)j; }
  }
}
