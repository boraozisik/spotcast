// The SpotCast mark: a raked windsurf sail with a wishbone boom over a short board and waves.
// Drawn for this project (same drawing as app/icon.svg and the favicon).
export function Logo({ className = "h-8 w-8" }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 64" className={className} aria-hidden>
      <defs>
        <linearGradient id="spotcast-sea" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#0c4a6e" />
          <stop offset="1" stopColor="#0e7490" />
        </linearGradient>
        <clipPath id="spotcast-disc">
          <circle cx="32" cy="32" r="32" />
        </clipPath>
      </defs>
      <circle cx="32" cy="32" r="32" fill="url(#spotcast-sea)" />
      <g clipPath="url(#spotcast-disc)">
        <path d="M-2 52 C 8 47, 16 56, 26 52 S 44 47, 54 52 S 62 53, 66 51" fill="none" stroke="#7dd3fc" strokeWidth="3.2" strokeLinecap="round" />
        <path d="M10 59 C 18 55.5, 26 62, 34 58.5 S 48 55.5, 56 59" fill="none" stroke="#38bdf8" strokeWidth="2.6" strokeLinecap="round" opacity="0.7" />
      </g>
      <path d="M12 45.2 Q 11 42.8, 14 42.6 L 45 41 Q 51 40.8, 55 42.4 Q 51 44.4, 45 44.8 L 14 46.4 Q 12 46.4, 12 45.2 Z" fill="#ffffff" />
      <g transform="translate(30 42) rotate(12) scale(1.14) translate(-30 -42)">
        <path d="M28 8 C 38 14, 45 26, 45 39 L 28 40.5 Z" fill="#ffffff" />
        <path d="M28 20 L 39.6 18.2 C 41 20, 42.2 22, 43.2 24.2 L 28 26.5 Z" fill="#fbbf24" />
        <path d="M28 29 Q 36 25.5, 44.5 31 Q 36 33.5, 28 29 Z" fill="none" stroke="#0c4a6e" strokeWidth="1.8" strokeLinejoin="round" />
      </g>
    </svg>
  );
}
