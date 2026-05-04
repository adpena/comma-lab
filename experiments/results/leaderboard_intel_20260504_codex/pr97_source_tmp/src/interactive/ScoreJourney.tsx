// Auto-playing 4.39 → 0.229 score counter. Loops forever, no buttons.

import { useEffect, useRef, useState } from "react";

interface Stage {
  name: string;
  score: number;
  delta?: number;
  blurb: string;
  category: "baseline" | "training" | "model" | "codec" | "sidecar";
}

const STAGES: Stage[] = [
  { name: "baseline_fast",                score: 4.39,    blurb: "Repo example. ffmpeg + libx265, no model.", category: "baseline" },
  { name: "Autoresearch best",            score: 1.36,    delta: -3.03, blurb: "192 self-contained 5-min experiments. LLM agent picks the architecture.", category: "training" },
  { name: "A100 12h training",            score: 0.41,    delta: -0.95, blurb: "Three-stage curriculum with QAT.", category: "training" },
  { name: "3090 4h joint+ continue",      score: 0.33,    delta: -0.08, blurb: "Lower LR, longer joint stage. Found by EMA still trending.", category: "training" },
  { name: "Targeted fine-tune",           score: 0.30,    delta: -0.03, blurb: "lr=2e-6 EMA=0.99 finds a flatter local minimum that FP4 rounds onto cleanly.", category: "model" },
  { name: "H3 LowRank pose_mlp",          score: 0.29,    delta: -0.01, blurb: "Replace 64×64 Linears with rank-16 LowRank, SVD warm-start.", category: "model" },
  { name: "Codec optimizations",          score: 0.2345,  delta: -0.0555, blurb: "Lossless mask codec + flat-FP4 model packing + per-dim pose quant.", category: "codec" },
  { name: "Sidecar patches",              score: 0.22878, delta: -0.0057, blurb: "2.4 KB of per-pair corrections that invert the SegNet/PoseNet oracles.", category: "sidecar" },
];

const CAT_COLOR: Record<Stage["category"], string> = {
  baseline: "#9E9E9E",
  training: "#3b82f6",
  model: "#8b5cf6",
  codec: "#16a34a",
  sidecar: "#51FF00",
};

const STEP_MS = 1100;
const HOLD_END_MS = 3500;

export default function ScoreJourney() {
  const [idx, setIdx] = useState(0);
  const [hasStarted, setHasStarted] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Start playing when scrolled into view
  useEffect(() => {
    if (hasStarted) return;
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { setHasStarted(true); obs.disconnect(); } });
    }, { threshold: 0.3 });
    obs.observe(el);
    return () => obs.disconnect();
  }, [hasStarted]);

  // Drive the playback loop
  useEffect(() => {
    if (!hasStarted) return;
    const isEnd = idx >= STAGES.length - 1;
    const t = window.setTimeout(() => {
      setIdx(isEnd ? 0 : idx + 1);
    }, isEnd ? HOLD_END_MS : STEP_MS);
    return () => clearTimeout(t);
  }, [idx, hasStarted]);

  const cur = STAGES[idx];
  const maxScore = STAGES[0].score;

  return (
    <div className="card !p-6 lg:!p-8 not-prose" ref={ref}>
      <div className="flex items-center justify-between mb-6">
        <div className="mono text-[11px] uppercase tracking-widest text-comma-green">interactive · auto</div>
        <div className="mono text-[11px] uppercase text-white/40">score journey · 4.39 → 0.229</div>
      </div>

      <div className="text-center py-10" style={{ minHeight: 320 }}>
        <div className="mono text-[12px] uppercase tracking-widest text-white/40 mb-3 min-h-[16px]">{cur.name}</div>
        <div className="text-[120px] leading-none font-bold tabular-nums" style={{ color: CAT_COLOR[cur.category] }}>
          {cur.score.toFixed(cur.score < 1 ? 4 : 2)}
        </div>
        <div className="mono text-[16px] mt-3 min-h-[24px]" style={{ color: cur.delta && cur.delta < 0 ? "#51FF00" : "transparent" }}>
          {cur.delta !== undefined && (cur.delta > 0 ? "+" : "") + cur.delta.toFixed(4)}
        </div>
        <p className="text-[14px] text-white/70 mt-4 max-w-[560px] mx-auto" style={{ minHeight: 40 }}>{cur.blurb}</p>
      </div>

      <div className="space-y-2">
        {STAGES.map((s, i) => {
          const w = (s.score / maxScore) * 100;
          const active = i === idx;
          return (
            <div
              key={i}
              className={`grid grid-cols-[170px_1fr_70px] gap-3 items-center text-left transition-opacity ${active ? "opacity-100" : "opacity-40"}`}
            >
              <div className="mono text-[10px] uppercase tracking-wider text-white truncate">{s.name}</div>
              <div className="relative h-5 bg-black border border-white/10">
                <div
                  className="absolute inset-y-0 left-0 transition-all duration-700 ease-out"
                  style={{ width: `${Math.max(w, 1.5)}%`, background: CAT_COLOR[s.category] }}
                />
              </div>
              <div className="mono text-[11px] text-right tabular-nums" style={{ color: CAT_COLOR[s.category] }}>
                {s.score.toFixed(s.score < 1 ? 4 : 2)}
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-5 mono text-[11px] text-white/40 text-right">≈ 19× total reduction</div>
    </div>
  );
}
