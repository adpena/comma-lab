import DataGrid from "./DataGrid";
import PixelText from "./PixelText";

export default function Hero() {
  return (
    <section className="relative w-full overflow-hidden bg-black crt-scanlines">
      {/* Animated grid background, briefly washed bright on load */}
      <div className="grid-boot absolute inset-0">
        <DataGrid rows={20} cols={48} spacing={4} color="#51FF00" duration={6} opacityMin={0.04} opacityMax={0.45} />
      </div>

      {/* One-shot scanline sweep (cosmetic boot effect) */}
      <div className="scan-sweep" aria-hidden />

      <div className="relative z-10 w-full border-b border-white/15 bg-black/60 backdrop-blur-sm">
        <div className="max-w-[1200px] mx-auto px-6 py-3 flex items-center justify-between text-[12px] mono">
          <span className="text-white/60">/// VIBE_CODER_FINAL_BOSS &mdash; comma video compression challenge writeup</span>
          <span className="text-comma-green">FINAL SCORE 0.22878</span>
        </div>
      </div>

      <div className="relative z-10 max-w-[1200px] mx-auto px-6 pt-14 pb-12 lg:pt-20 lg:pb-16 flex flex-col items-center text-center">
        <div className="flex flex-col items-center gap-4 mb-10">
          <PixelText
            text="VIBE CODER" pixelSize={11} color="#FFFFFF"
            letterSpacing={1} wordSpacing={3}
            showOff offOpacity={0.04}
            reveal revealDuration={1200} revealStartDelay={200}
          />
          <PixelText
            text="FINAL BOSS" pixelSize={11} color="#51FF00"
            letterSpacing={1} wordSpacing={3}
            showOff offOpacity={0.04}
            reveal revealDuration={1300} revealStartDelay={500}
          />
        </div>

        <p className="max-w-[640px] text-[17px] md:text-[19px] text-white/75 leading-snug">
          A writeup of the comma 2026 video compression challenge.
          37.5 megabytes of dashcam frames in, <span className="text-comma-green">197 kilobytes</span> out,
          decoded by a 92K-parameter generator.
        </p>

        <div className="mt-10 grid grid-flow-col auto-cols-fr gap-x-10 mono">
          <Stat label="archive.zip" value="197,160 B" />
          <Stat label="reduction" value="190.4×" />
          <Stat label="seg dist" value="0.000272" />
          <Stat label="pose dist" value="0.000495" />
          <Stat label="final score" value="0.22878" highlight />
        </div>

        <div className="mt-10 flex flex-wrap items-center justify-center gap-3">
          <a href="#story" className="bg-comma-green text-black px-10 py-4 font-semibold tracking-wider text-[16px] uppercase hover:bg-white transition-colors">
            READ THE STORY
          </a>
          <a href="#archive" className="border border-white/30 text-white px-10 py-4 font-semibold tracking-wider text-[16px] uppercase hover:border-comma-green hover:text-comma-green transition-colors">
            ARCHIVE BREAKDOWN
          </a>
        </div>
      </div>
    </section>
  );
}

function Stat({ label, value, highlight = false }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex flex-col items-center min-w-[100px]">
      <div className={`text-[22px] md:text-[28px] font-bold ${highlight ? "text-comma-green" : "text-white"}`}>{value}</div>
      <div className="mt-1 text-[10px] uppercase tracking-[0.25em] text-white/45">{label}</div>
    </div>
  );
}
