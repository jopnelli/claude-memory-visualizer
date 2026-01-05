// Color utilities for point cloud visualization

import * as THREE from 'three';

// Color gradient for time-based coloring (old → new)
export const TIME_GRADIENT = [
  new THREE.Color(0x3b82f6), // blue (oldest)
  new THREE.Color(0x06b6d4), // cyan
  new THREE.Color(0x10b981), // green
  new THREE.Color(0xfbbf24), // yellow
  new THREE.Color(0xf97316), // orange
  new THREE.Color(0xef4444), // red (newest)
];

// Category colors for demo dataset
export const CATEGORY_COLORS: Record<string, THREE.Color> = {
  geography: new THREE.Color(0x3b82f6), // blue
  programming: new THREE.Color(0x10b981), // green
  ai: new THREE.Color(0xa855f7), // purple
  food: new THREE.Color(0xf97316), // orange
  music: new THREE.Color(0xec4899), // pink
  random: new THREE.Color(0x6b7280), // gray
  // Outlier categories - varied colors
  science: new THREE.Color(0x14b8a6), // teal
  trivia: new THREE.Color(0xeab308), // yellow
  art: new THREE.Color(0xf43f5e), // rose
  history: new THREE.Color(0x8b5cf6), // violet
  astronomy: new THREE.Color(0x0ea5e9), // sky blue
  physics: new THREE.Color(0x22c55e), // lime
  medicine: new THREE.Color(0xef4444), // red
  biology: new THREE.Color(0x84cc16), // lime green
  technology: new THREE.Color(0x06b6d4), // cyan
  economics: new THREE.Color(0xd946ef), // fuchsia
  engineering: new THREE.Color(0xf59e0b), // amber
  weather: new THREE.Color(0x64748b), // slate
};

// Special colors
export const HIGHLIGHT_COLOR = new THREE.Color(0xffffff);
export const SELECTED_COLOR = new THREE.Color(0x00ff00);
export const DIM_FACTOR = 0.15;

// Interpolate through gradient
export function getGradientColor(t: number): THREE.Color {
  t = Math.max(0, Math.min(1, t));
  const segments = TIME_GRADIENT.length - 1;
  const segment = Math.min(Math.floor(t * segments), segments - 1);
  const segmentT = t * segments - segment;

  const color = new THREE.Color();
  color.lerpColors(TIME_GRADIENT[segment], TIME_GRADIENT[segment + 1], segmentT);
  return color;
}
