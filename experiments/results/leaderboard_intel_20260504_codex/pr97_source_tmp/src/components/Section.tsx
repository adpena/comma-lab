import React from "react";

interface Props {
  id: string;
  number: string;
  kicker?: string;
  title: string;
  lede?: React.ReactNode;
  children: React.ReactNode;
  background?: "black" | "surface";
}

export default function Section({ id, number, kicker, title, lede, children, background = "black" }: Props) {
  const bg = background === "black" ? "bg-black" : "bg-comma-surface";
  return (
    <section id={id} className={`relative ${bg} border-t border-white/10`}>
      <div className="max-w-[920px] mx-auto px-6 lg:px-10 py-20 lg:py-28">
        <header className="mb-12 lg:mb-16">
          <div className="mono text-comma-green text-[12px] tracking-[0.25em] mb-3">
            §{number}{kicker && <span className="text-white/35"> &nbsp;·&nbsp; {kicker}</span>}
          </div>
          <h2 className="h-display text-[40px] md:text-[56px] lg:text-[68px] text-white leading-[0.95]">{title}</h2>
          {lede && <p className="mt-6 text-white/65 text-[18px] md:text-[20px] leading-[1.55]">{lede}</p>}
        </header>
        {children}
      </div>
    </section>
  );
}
