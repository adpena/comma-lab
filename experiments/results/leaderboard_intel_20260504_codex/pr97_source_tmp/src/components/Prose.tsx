// Reusable prose helpers. Inter, max-width body, spacing tuned for reading.
import React from "react";

export function Prose({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div
      className={`max-w-[760px] text-white/90 text-[18px] leading-[1.7] space-y-6 ${className}`}
      style={{ letterSpacing: "-0.012em" }}
    >
      {children}
    </div>
  );
}

export function SubHeading({ id, kicker, children }: { id?: string; kicker?: string; children: React.ReactNode }) {
  return (
    <div className="mt-12 mb-5" id={id}>
      {kicker && <div className="mono text-[11px] uppercase tracking-widest text-comma-green mb-2">{kicker}</div>}
      <h3 className="h-display text-[28px] md:text-[36px] text-white">{children}</h3>
    </div>
  );
}

export function Code({ children }: { children: React.ReactNode }) {
  return <code className="mono text-[14px] bg-white/5 px-2 py-[2px] text-comma-green">{children}</code>;
}

export function Pull({ children, label }: { children: React.ReactNode; label?: string }) {
  return (
    <div className="my-8 border-l-2 border-comma-green pl-5 py-2">
      {label && <div className="mono text-[11px] uppercase tracking-widest text-comma-green mb-1">{label}</div>}
      <div className="text-[20px] text-white/95 italic font-light leading-snug">{children}</div>
    </div>
  );
}

export function Table({ headers, rows }: { headers: string[]; rows: (string | React.ReactNode)[][] }) {
  return (
    <div className="not-prose my-8 overflow-x-auto">
      <table className="w-full border-collapse mono text-[13px]">
        <thead>
          <tr className="border-b border-white/20">
            {headers.map((h, i) => (
              <th key={i} className="text-left py-2 px-3 text-comma-green uppercase text-[11px] tracking-widest font-normal">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} className="border-b border-white/8 hover:bg-white/[0.02]">
              {row.map((cell, ci) => (
                <td key={ci} className="py-2.5 px-3 text-white/85">{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function FigCaption({ children }: { children: React.ReactNode }) {
  return <div className="mono text-[12px] text-white/45 italic mt-3 text-center">{children}</div>;
}

export function Figure({ src, alt, caption }: { src: string; alt: string; caption?: React.ReactNode }) {
  return (
    <figure className="not-prose my-8">
      <img src={src} alt={alt} className="w-full max-w-[1000px] mx-auto" loading="lazy" />
      {caption && <FigCaption>{caption}</FigCaption>}
    </figure>
  );
}
