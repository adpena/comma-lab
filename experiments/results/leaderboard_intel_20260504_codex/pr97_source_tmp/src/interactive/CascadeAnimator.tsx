// Auto-playing cascade walkthrough. Cycles through scenarios with no buttons,
// behaves like a rendered loop. Fixed-height layout so nothing reflows.

import { useEffect, useRef, useState } from "react";

const COLORS = ["#1A1A1A", "#51FF00", "#FF4133", "#facc15", "#3b82f6"];

interface Scenario {
  label: string;
  desc: string;
  prevFrame: number[][];
  curFrame: number[][];
  pixel: [number, number];
}

const SCENARIOS: Scenario[] = [
  {
    label: "exits on UP",
    desc: "Current pixel matches the one directly above. 1 bit emitted, cascade ends after step 1. ~70% of dashcam pixels take this path.",
    prevFrame: [[1,1,1,1,1],[1,1,1,1,1],[1,1,0,0,0],[0,0,0,0,0]],
    curFrame:  [[1,1,1,1,1],[1,1,1,1,1],[1,1,0,0,0],[0,0,0,0,0]],
    pixel: [2, 2],
  },
  {
    label: "exits on LEFT",
    desc: "UP doesn't match (sky to road transition), but LEFT does. 2 bits emitted: 0 (UP=no) then 1 (LEFT=yes).",
    prevFrame: [[1,1,1,1,1],[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0]],
    curFrame:  [[1,1,1,1,1],[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0]],
    pixel: [2, 1],
  },
  {
    label: "exits on PREV",
    desc: "Spatial neighbors disagree but the same pixel in the previous frame matches. 3 bits emitted.",
    prevFrame: [[1,1,1,1,1],[1,1,1,1,1],[1,1,2,1,0],[0,0,0,0,0]],
    curFrame:  [[1,1,1,1,1],[1,1,1,1,1],[1,1,2,1,0],[0,0,0,0,0]],
    pixel: [2, 2],
  },
  {
    label: "fallback (multi-symbol)",
    desc: "Nothing matches. Encoder falls through to the 5-class coder, costs ~2.3 bits. ~5% of pixels.",
    prevFrame: [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,0,0],[0,0,0,0,0]],
    curFrame:  [[1,1,1,1,1],[1,1,1,1,1],[1,1,3,0,0],[0,0,0,0,0]],
    pixel: [2, 2],
  },
];

type StepKind = "checkUp" | "checkLeft" | "checkPrev" | "fallback" | "done";

const STEP_DELAY = 850;
const SCENARIO_HOLD = 2200;

export default function CascadeAnimator() {
  const [scenarioIdx, setScenarioIdx] = useState(0);
  const [step, setStep] = useState<StepKind>("checkUp");
  const [bits, setBits] = useState<string[]>([]);
  const timeoutRef = useRef<number | null>(null);

  const scn = SCENARIOS[scenarioIdx];
  const [pr, pc] = scn.pixel;
  const cur = scn.curFrame[pr][pc];
  const up = pr > 0 ? scn.curFrame[pr - 1][pc] : -1;
  const left = pc > 0 ? scn.curFrame[pr][pc - 1] : -1;
  const prev = scn.prevFrame[pr][pc];

  // Drive the cascade
  useEffect(() => {
    const tick = () => {
      if (step === "checkUp") {
        if (cur === up) { setBits(["1"]); setStep("done"); }
        else { setBits(["0"]); setStep("checkLeft"); }
      } else if (step === "checkLeft") {
        if (cur === left) { setBits((b) => [...b, "1"]); setStep("done"); }
        else { setBits((b) => [...b, "0"]); setStep("checkPrev"); }
      } else if (step === "checkPrev") {
        if (cur === prev) { setBits((b) => [...b, "1"]); setStep("done"); }
        else { setBits((b) => [...b, "0"]); setStep("fallback"); }
      } else if (step === "fallback") {
        setBits((b) => [...b, "??", "?"]); setStep("done");
      } else if (step === "done") {
        // Hold then advance to next scenario
        setScenarioIdx((i) => (i + 1) % SCENARIOS.length);
        setStep("checkUp");
        setBits([]);
      }
    };
    timeoutRef.current = window.setTimeout(tick, step === "done" ? SCENARIO_HOLD : STEP_DELAY);
    return () => { if (timeoutRef.current) clearTimeout(timeoutRef.current); };
  }, [step, cur, up, left, prev]);

  // Reset on scenario change
  useEffect(() => {
    setStep("checkUp");
    setBits([]);
    return () => { if (timeoutRef.current) clearTimeout(timeoutRef.current); };
  }, [scenarioIdx]);

  const stepLabel: Record<StepKind, { color: string; label: string }> = {
    checkUp:     { color: "#facc15", label: "step 1: is current class == UP neighbor?" },
    checkLeft:   { color: "#facc15", label: "step 2: is current class == LEFT neighbor?" },
    checkPrev:   { color: "#facc15", label: "step 3: is current class == PREV-frame pixel?" },
    fallback:    { color: "#FF4133", label: "fallback: emit log₂(5) ≈ 2.3 bits via class coder" },
    done:        { color: "#51FF00", label: `done: emitted ${bits.length} bit${bits.length === 1 ? "" : "s"}` },
  };

  return (
    <div className="card !p-6 lg:!p-8 not-prose">
      <div className="flex items-center justify-between mb-6">
        <div className="mono text-[11px] uppercase tracking-widest text-comma-green">interactive · auto</div>
        <div className="mono text-[11px] uppercase text-white/40">cascade walkthrough</div>
      </div>

      <div className="grid lg:grid-cols-[300px_1fr] gap-8" style={{ minHeight: 420 }}>
        <div className="space-y-3">
          <MaskGrid
            grid={scn.prevFrame}
            highlight={[pr, pc]}
            highlightKind={step === "checkPrev" ? "active" : "neighbor-prev"}
            label="PREV frame"
          />
          <MaskGrid
            grid={scn.curFrame}
            highlight={[pr, pc]}
            highlightKind="current"
            extraHighlights={
              step === "checkUp" ? [{ rc: [pr - 1, pc], kind: "active" }] :
              step === "checkLeft" ? [{ rc: [pr, pc - 1], kind: "active" }] : []
            }
            label="CURRENT frame"
          />
        </div>

        <div className="flex flex-col">
          {/* Scenario tag, fixed height row */}
          <div className="mb-3 flex flex-wrap gap-2 min-h-[36px]">
            {SCENARIOS.map((s, i) => (
              <div key={i}
                className={`mono text-[10px] uppercase tracking-wider px-2 py-1.5 border transition-colors ${i === scenarioIdx ? "border-comma-green text-comma-green" : "border-white/15 text-white/40"}`}
              >{s.label}</div>
            ))}
          </div>

          {/* Description: fixed height */}
          <div className="mb-5" style={{ minHeight: 70 }}>
            <div className="mono text-[10px] uppercase text-white/40 mb-1">scenario</div>
            <p className="text-[14px] text-white/85 leading-snug">{scn.desc}</p>
          </div>

          {/* Cascade state: fixed height */}
          <div className="border border-white/10 p-3 bg-black mb-3" style={{ minHeight: 70 }}>
            <div className="mono text-[10px] uppercase tracking-widest text-white/40 mb-1">cascade state</div>
            <div className="mono text-[13px]" style={{ color: stepLabel[step].color }}>
              {stepLabel[step].label}
            </div>
          </div>

          {/* Emitted bits: fixed height, monospace renders bits */}
          <div className="border border-white/10 p-3 bg-black" style={{ minHeight: 84 }}>
            <div className="mono text-[10px] uppercase tracking-widest text-white/40 mb-2">emitted bits</div>
            <div className="mono text-[26px] tracking-[0.3em] text-comma-green font-bold leading-none">
              {bits.length > 0 ? bits.join(" ") : <span className="text-white/15">···</span>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MaskGrid({ grid, highlight, highlightKind, extraHighlights = [], label }: {
  grid: number[][];
  highlight: [number, number];
  highlightKind: "current" | "active" | "neighbor-prev";
  extraHighlights?: { rc: [number, number]; kind: "active" }[];
  label: string;
}) {
  return (
    <div>
      <div className="mono text-[10px] uppercase tracking-widest text-white/40 mb-2">{label}</div>
      <div className="inline-grid p-1" style={{ gridTemplateColumns: `repeat(${grid[0].length}, 30px)`, gap: 2, background: "#0a0a0a", border: "1px solid rgba(255,255,255,0.1)" }}>
        {grid.map((row, ri) =>
          row.map((cls, ci) => {
            const isCur = highlight[0] === ri && highlight[1] === ci;
            const extra = extraHighlights.find(h => h.rc[0] === ri && h.rc[1] === ci);
            const isPrevHL = highlightKind === "neighbor-prev" && isCur;
            return (
              <div key={`${ri}-${ci}`}
                className="w-[30px] h-[30px]"
                style={{
                  backgroundColor: COLORS[cls],
                  outline: isCur ? "2px solid #51FF00" : extra ? "2px solid #facc15" : isPrevHL ? "2px solid #facc15" : "none",
                  outlineOffset: 1,
                  opacity: cls === 0 ? 0.85 : 1,
                }}
              />
            );
          })
        )}
      </div>
    </div>
  );
}
