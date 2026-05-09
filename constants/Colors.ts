// ─── Global Design Tokens ────────────────────────────────────────────────────
// Sky Blue + Black palette used across all screens

export const Colors = {
  // Backgrounds
  bg: '#000000',
  bgSoft: '#080C10',
  surface: '#0D1117',
  surfaceElevated: '#131920',
  surfaceCard: '#0A0F15',

  // Borders
  border: '#1a2535',
  borderFocused: '#38BDF8',
  borderSubtle: '#111c28',

  // Text
  textPrimary: '#FFFFFF',
  textSecondary: '#94A3B8',
  textMuted: '#3D5068',

  // Sky Blue accents
  accent: '#38BDF8',          // sky-400
  accentBright: '#7DD3FC',    // sky-300 (lighter for highlights)
  accentDeep: '#0EA5E9',      // sky-500 (deeper for buttons)
  accentSoft: '#38BDF815',    // translucent fill
  accentGlow: '#38BDF830',    // for glows / orbs
  accentDim: '#38BDF860',     // mid-opacity

  // Gradient stops (sky → deep blue → black)
  gradStart: '#87CEEB',       // classic sky blue
  gradMid: '#0EA5E9',         // sky-500
  gradEnd: '#000000',         // pure black
};

// Gradient definitions for LinearGradient / inline use
export const Gradients = {
  // Main hero gradient: sky → black
  hero: ['#87CEEB', '#0EA5E9', '#000000'] as string[],
  heroAngles: { start: { x: 0, y: 0 }, end: { x: 1, y: 1 } },

  // Button gradient
  button: ['#38BDF8', '#0EA5E9'] as string[],

  // Card top-edge shimmer
  cardShimmer: ['#38BDF820', '#38BDF800'] as string[],

  // Glow orb
  orb: ['#38BDF840', '#0EA5E900'] as string[],
};
