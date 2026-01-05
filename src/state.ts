// Centralized application state
import * as THREE from 'three';
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { FeatureExtractionPipeline } from '@huggingface/transformers';
import type { DataSet, TimeRange } from './types';

// Application state - single source of truth
export const state = {
  // Data
  data: null as DataSet | null,
  timeRange: null as TimeRange | null,

  // Projections
  projectedPositions: null as Float32Array | null,
  previousPositions: null as Float32Array | null,
  animationProgress: 1, // 0 to 1, 1 = complete

  // Selection
  hoveredIndex: null as number | null,
  selectedIndex: null as number | null,
  selectedIndices: new Set<number>(),
  matchedIndices: new Map<number, number>(), // index -> similarity score

  // Box selection
  isSelecting: false,
  selectionStart: null as { x: number; y: number } | null,

  // Colors
  baseColors: null as Float32Array | null,
};

// Three.js objects - kept separate as they're not serializable
export const threeState = {
  scene: null as THREE.Scene | null,
  camera: null as THREE.PerspectiveCamera | null,
  renderer: null as THREE.WebGLRenderer | null,
  controls: null as OrbitControls | null,
  raycaster: null as THREE.Raycaster | null,
  mouse: new THREE.Vector2(),
  points: null as THREE.Points | null,
  gridHelper: null as THREE.GridHelper | null,
  axesGroup: null as THREE.Group | null,
  selectionBox: null as HTMLDivElement | null,
};

// Embedding state
export const embeddingState = {
  embedder: null as FeatureExtractionPipeline | null,
  embedderPromise: null as Promise<FeatureExtractionPipeline | null> | null,
  localEmbedAvailable: null as boolean | null,
  ollamaAvailable: null as boolean | null,
};

// Reset selection state
export function clearSelection() {
  state.selectedIndex = null;
  state.selectedIndices.clear();
}

// Reset search state
export function clearSearch() {
  state.matchedIndices.clear();
}
