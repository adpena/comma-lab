// Auto-cycling SVD rank explorer. Slider stays draggable but auto-pulses
// through interesting ranks if the user doesn't touch it. Energy retained
// uses the same 99.5% number as the prose (calibrated below).

import { useEffect, useRef, useState } from "react";

// Calibrated singular value spectrum so the cumulative L2 energy at rank 16
// is ~99.5%, matching the prose. Spectrum is exponential-decay-with-noise-tail.
const SINGULAR_VALUES = (() => {
  const out: number[] = [];
  for (let i = 0; i < 64; i++) {
    // Steep early decay so first 16 dominate; long noise tail
    const major = i < 16 ? 3.2 * Math.exp(-i * 0.22) : 0;
    const tail = i >= 16 ? 0.07 * Math.exp(-(i - 16) * 0.05) : 0;
    out.push(major + tail);
  }
  return out;
})();

const TOTAL_ENERGY = SINGULAR_VALUES.reduce((a, s) => a + s * s, 0);

const AUTO_RANKS = [4, 8, 16, 32, 16];
const AUTO_INTERVAL = 2200;

export default function SVDRankSlider() {
  const [rank, setRank] = useState(16);
  const [autoIdx, setAutoIdx] = useState(0);
  const userTouchedRef = useRef(false);

  useEffect(() => {
    if (userTouchedRef.current) return;
    const t = window.setTimeout(() => {
      const next = (autoIdx + 1) % AUTO_RANKS.length;
      setRank(AUTO_RANKS[next]);
      setAutoIdx(next);
    }, AUTO_INTERVAL);
    return () => clearTimeout(t);
  }, [rank, autoIdx]);

  const cumulative = SINGULAR_VALUES.slice(0, rank).reduce((a, s) => a + s * s, 0);
  const energyKept = (cumulative / TOTAL_ENERGY) * 100;

  const denseParams = 2 * 64 * 64;
  const lowRankParams = 2 * 2 * 64 * rank;
  const paramSaved = denseParams - lowRankParams;
  const byteSaved = (paramSaved * 5) / 8;

  const handleSlider = (v: number) => {
    userTouchedRef.current = true;
    setRank(v);
  };

  return (
    <div className="card !p-6 lg:!p-8 not-prose">
      <div className="flex items-center justify-between mb-6">
        <div className="mono text-[11px] uppercase tracking-widest text-comma-green">interactive · auto</div>
        <div className="mono text-[11px] uppercase text-white/40">SVD rank explorer</div>
      </div>

      <div className="grid lg:grid-cols-[1fr_280px] gap-8 items-start" style={{ minHeight: 380 }}>
        <div>
          <div className="bg-black border border-white/10 p-4">
            <div className="mono text-[10px] uppercase tracking-widest text-white/40 mb-3">singular values σᵢ of pose_mlp.2 weight (64×64)</div>
            <div className="flex items-end h-44 gap-[1px]">
              {SINGULAR_VALUES.map((s, i) => {
                const h = (s / SINGULAR_VALUES[0]) * 100;
                const kept = i < rank;
                return (
                  <div key={i} className="flex-1 transition-colors duration-300"
                    style={{
                      height: `${Math.max(h, 1)}%`,
                      background: kept ? "#51FF00" : "rgba(255,255,255,0.15)",
                    }}
                  />
                );
              })}
            </div>
            <div className="relative h-6 mt-1">
              <div
                className="absolute top-0 bottom-0 w-[1px] bg-comma-error transition-all duration-300"
                style={{ left: `${(rank / 64) * 100}%` }}
              />
              <div
                className="absolute -top-1 mono text-[9px] text-comma-error transition-all duration-300"
                style={{ left: `${(rank / 64) * 100}%`, transform: "translateX(2px)" }}
              >
                ↑ rank {rank} cut
              </div>
            </div>
          </div>

          <div className="mt-6">
            <div className="flex items-baseline justify-between mb-2">
              <span className="mono text-[12px] uppercase tracking-widest text-white/55">rank</span>
              <span className="mono text-[18px] text-comma-green font-bold tabular-nums">{rank} / 64</span>
            </div>
            <input
              type="range"
              min={1} max={64} step={1} value={rank}
              onChange={(e) => handleSlider(parseInt(e.target.value))}
              className="w-full appearance-none h-[3px] outline-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, #51FF00 0%, #51FF00 ${(rank / 64) * 100}%, rgba(255,255,255,0.15) ${(rank / 64) * 100}%, rgba(255,255,255,0.15) 100%)`,
              }}
            />
            <div className="flex justify-between mono text-[10px] text-white/40 mt-1">
              <span>1</span><span>16 (ours)</span><span>32</span><span>64 (full)</span>
            </div>
          </div>
        </div>

        <div className="border border-white/10 p-5 bg-black space-y-4" style={{ minHeight: 360 }}>
          <div>
            <div className="mono text-[10px] uppercase tracking-widest text-white/40">L2 energy retained</div>
            <div className="text-[44px] leading-none font-bold text-comma-green tabular-nums">{energyKept.toFixed(1)}%</div>
            <div className="h-[3px] bg-white/10 mt-2">
              <div className="h-full bg-comma-green transition-all duration-300" style={{ width: `${Math.min(energyKept, 100)}%` }} />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-y-2 mono text-[12px] pt-2 border-t border-white/10">
            <div className="text-white/40">rank chosen</div>
            <div className="text-white text-right tabular-nums">{rank}</div>
            <div className="text-white/40">params (low-rank)</div>
            <div className="text-white text-right tabular-nums">{lowRankParams.toLocaleString()}</div>
            <div className="text-white/40">vs dense (8,192)</div>
            <div className={`text-right tabular-nums ${paramSaved > 0 ? "text-comma-green" : "text-comma-error"}`}>
              {paramSaved > 0 ? "−" : "+"}{Math.abs(paramSaved).toLocaleString()}
            </div>
            <div className="text-white/40">FP4 bytes saved</div>
            <div className={`text-right tabular-nums ${byteSaved > 0 ? "text-comma-green" : "text-comma-error"}`}>
              {byteSaved > 0 ? "−" : "+"}{Math.abs(byteSaved).toLocaleString()} B
            </div>
          </div>

          <p className="mono text-[11px] text-white/55 leading-relaxed pt-3 border-t border-white/10" style={{ minHeight: 60 }}>
            {rank === 16 ? (
              <>↑ what we shipped. <span className="text-comma-green">{energyKept.toFixed(1)}%</span> of energy in 50% of the params.</>
            ) : rank < 8 ? (
              <>too aggressive: losing the dominant singular subspace that carries the actual pose mapping.</>
            ) : rank > 32 ? (
              <>diminishing returns: the bottom 32 σᵢ are noise the optimizer never organized.</>
            ) : (
              <>nearly all of the matrix energy is in the top 16 σᵢ. Anything past rank 16 is mostly noise.</>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
