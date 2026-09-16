#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define H 384
#define W 512
#define S 8
#define Q 65536

/* Two-limb signed arithmetic: exact ordered operations without compiler extensions. */
typedef struct { uint64_t hi, lo; } Wide;
static Wide number(int64_t a) { return (Wide){a<0?UINT64_MAX:0,(uint64_t)a}; }
static Wide plus(Wide a, Wide b) { uint64_t l=a.lo+b.lo; return (Wide){a.hi+b.hi+(l<a.lo),l}; }
static Wide negate(Wide a) { return plus((Wide){~a.hi,~a.lo},number(1)); }
static Wide minus(Wide a, Wide b) { return plus(a,negate(b)); }
static int below(Wide a, Wide b) { return a.hi<b.hi || (a.hi==b.hi && a.lo<b.lo); }
static int signed_below(Wide a, Wide b) { return (a.hi>>63)!=(b.hi>>63)?(int)(a.hi>>63):below(a,b); }
static Wide times(Wide a, Wide b) {
  uint64_t a0=(uint32_t)a.lo,a1=a.lo>>32,b0=(uint32_t)b.lo,b1=b.lo>>32;
  uint64_t w0=a0*b0,t=a1*b0+(w0>>32),w1=(uint32_t)t,w2=t>>32;
  w1+=a0*b1;
  return (Wide){a.hi*b.lo+a.lo*b.hi+a1*b1+w2+(w1>>32),(w1<<32)+(uint32_t)w0};
}
static Wide quotient(Wide a, Wide b, int *remainder) {
  Wide q={0,0},r={0,0};
  for(int i=127;i>=0;i--) {
    uint64_t bit=i>=64?(a.hi>>(i-64))&1:(a.lo>>i)&1;
    r=(Wide){(r.hi<<1)|(r.lo>>63),(r.lo<<1)|bit};
    if(!below(r,b)) { r=minus(r,b); if(i>=64) q.hi|=(uint64_t)1<<(i-64); else q.lo|=(uint64_t)1<<i; }
  }
  *remainder=r.hi!=0 || r.lo!=0; return q;
}
static int64_t floor_div(Wide a, Wide b) {
  int negative=(int)(a.hi>>63),remainder; if(negative) a=negate(a);
  Wide q=quotient(a,b,&remainder);
  return negative?-(int64_t)q.lo-remainder:(int64_t)q.lo;
}
typedef struct {
  int64_t cfg[17];
  int64_t now[6][H][S], old[6][H][S];
  int lo[H][S], hi[H][S], plo[H][S], phi[H][S], ambiguous[H][S];
  uint64_t other[H][S];
} Geometry;

static uint64_t root(uint64_t n) {
  uint64_t a=0, bit=(uint64_t)1<<62;
  while(bit>n) bit>>=2;
  while(bit) {
    if(n>=a+bit) { n-=a+bit; a=(a>>1)+bit; }
    else a>>=1;
    bit>>=2;
  }
  return a; }
static int64_t nearest_even(int64_t a) {
  int64_t q=a/Q, r=a%Q;
  return q+(r>Q/2 || (r==Q/2 && (q&1)));
}
void *geometry_new(const int64_t *cfg, const uint8_t *previous) {
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
  return g; }
void geometry_free(void *p) { free(p); }
void geometry_observe(void *p, int n, const int64_t *positions, const int64_t *symbols) {
  Geometry *g=p;
  for(int i=0;i<n;i++) {
    int y=positions[i]/W, x=positions[i]%W, s=x/(W/S);
    if(symbols[i]==g->cfg[0]) {
      int64_t v[6]={1,y,y*y,x,x*y,x*x};
      for(int j=0;j<6;j++) g->now[j][y][s]+=v[j];
      if(x<g->lo[y][s]) g->lo[y][s]=x;
      if(x>g->hi[y][s]) g->hi[y][s]=x;
    } else g->other[y][s]|=(uint64_t)1<<(x%(W/S));
  } }
void geometry_contexts(void *p, int n, const int64_t *positions, uint8_t *out) {
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
        Wide r=minus(times(number(d),minus(times(number(c),number(sxx)),times(number(sx),number(sx)))),times(number(b),number(b)));
        Wide den=times(times(number(c),number(c)),number(d));
        if(!signed_below(times(number(g->cfg[6]),den),r)) {
          center[y][s]=floor_div(times(plus(times(number(sx),number(d)),times(number(b),number(y*c-sy))),number(Q)),times(number(c),number(d)));
          int remainder; uint64_t squared=signed_below(number(0),r)?quotient(times(times(times(r,number(g->cfg[9])),number(Q)),number(Q)),den,&remainder).lo:0;
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
  } }
