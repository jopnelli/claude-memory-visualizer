// Animation utilities

import { state, threeState } from '../state';

// Easing function for smooth transitions
export function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

// Animate position transitions
export function animatePositionTransition(): void {
  if (
    !threeState.points ||
    !state.data ||
    state.animationProgress >= 1 ||
    !state.previousPositions ||
    !state.projectedPositions
  ) {
    return;
  }

  state.animationProgress = Math.min(1, state.animationProgress + 0.02); // ~50 frames
  const easedProgress = easeOutCubic(state.animationProgress);

  const geometry = threeState.points.geometry;
  const positions = geometry.getAttribute('position') as THREE.BufferAttribute;

  for (let i = 0; i < state.data.documents.length; i++) {
    const px =
      state.previousPositions[i * 3] +
      (state.projectedPositions[i * 3] - state.previousPositions[i * 3]) *
        easedProgress;
    const py =
      state.previousPositions[i * 3 + 1] +
      (state.projectedPositions[i * 3 + 1] - state.previousPositions[i * 3 + 1]) *
        easedProgress;
    const pz =
      state.previousPositions[i * 3 + 2] +
      (state.projectedPositions[i * 3 + 2] - state.previousPositions[i * 3 + 2]) *
        easedProgress;

    positions.setXYZ(i, px, py, pz);
  }

  positions.needsUpdate = true;
}

// Start a new transition animation
export function startTransition(): void {
  if (state.projectedPositions) {
    state.previousPositions = new Float32Array(state.projectedPositions);
  }
  state.animationProgress = 0;
}

// Check if animation is complete
export function isAnimationComplete(): boolean {
  return state.animationProgress >= 1;
}
