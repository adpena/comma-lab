// Animated grid background. Pulse-from-center cells with mouse-follow glow.
// Adapted from the DataGridHero pattern, ported to TypeScript + Tailwind +
// framer-free CSS animations.
import { useEffect, useRef } from "react";

interface Props {
  rows?: number;
  cols?: number;
  spacing?: number;
  color?: string;
  duration?: number;
  opacityMin?: number;
  opacityMax?: number;
  mouseGlow?: boolean;
  className?: string;
}

export default function DataGrid({
  rows = 28,
  cols = 48,
  spacing = 4,
  color = "#51FF00",
  duration = 6,
  opacityMin = 0.04,
  opacityMax = 0.5,
  mouseGlow = true,
  className = "",
}: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = ref.current;
    if (!container) return;
    container.innerHTML = "";
    container.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    container.style.gridTemplateRows = `repeat(${rows}, 1fr)`;
    container.style.gap = `${spacing}px`;
    const total = rows * cols;
    const cR = Math.floor(rows / 2);
    const cC = Math.floor(cols / 2);
    for (let i = 0; i < total; i++) {
      const cell = document.createElement("div");
      const r = Math.floor(i / cols);
      const c = i % cols;
      const dr = r - cR;
      const dc = c - cC;
      const delay = Math.sqrt(dr * dr + dc * dc) * 0.18;
      cell.style.backgroundColor = color;
      cell.style.opacity = String(opacityMin);
      cell.style.setProperty("--opacity-min", String(opacityMin));
      cell.style.setProperty("--opacity-max", String(opacityMax));
      cell.style.animation = `cellPulse ${duration}s infinite alternate`;
      cell.style.animationDelay = `${delay.toFixed(3)}s`;
      container.appendChild(cell);
    }
  }, [rows, cols, spacing, color, duration, opacityMin, opacityMax]);

  useEffect(() => {
    if (!mouseGlow) return;
    const el = ref.current;
    if (!el) return;
    const handler = (e: MouseEvent) => {
      const r = el.getBoundingClientRect();
      el.style.setProperty("--mouse-x", `${e.clientX - r.left}px`);
      el.style.setProperty("--mouse-y", `${e.clientY - r.top}px`);
    };
    window.addEventListener("mousemove", handler);
    return () => window.removeEventListener("mousemove", handler);
  }, [mouseGlow]);

  return (
    <div
      className={`absolute inset-0 ${className}`}
      style={{
        WebkitMask: mouseGlow
          ? "radial-gradient(circle 320px at var(--mouse-x, 50%) var(--mouse-y, 50%), #000 0%, rgba(0,0,0,0.7) 60%, rgba(0,0,0,0.3) 100%)"
          : undefined,
        mask: mouseGlow
          ? "radial-gradient(circle 320px at var(--mouse-x, 50%) var(--mouse-y, 50%), #000 0%, rgba(0,0,0,0.7) 60%, rgba(0,0,0,0.3) 100%)"
          : undefined,
      }}
    >
      <div
        ref={ref}
        className="absolute inset-0 grid"
        style={{ pointerEvents: "none" }}
        aria-hidden
      />
    </div>
  );
}
