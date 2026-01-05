// Point cloud creation and management

import * as THREE from 'three';
import { state, threeState } from '../state';
import {
  CATEGORY_COLORS,
  HIGHLIGHT_COLOR,
  SELECTED_COLOR,
  DIM_FACTOR,
  getGradientColor,
} from './colors';

// Create circular point texture
function createCircleTexture(): THREE.Texture {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  // Draw circle with soft edge
  const gradient = ctx.createRadialGradient(
    size / 2,
    size / 2,
    0,
    size / 2,
    size / 2,
    size / 2
  );
  gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
  gradient.addColorStop(0.7, 'rgba(255, 255, 255, 1)');
  gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

// Check if dataset uses category-based coloring
export function hasCategoryData(): boolean {
  if (!state.data || state.data.documents.length === 0) return false;
  return typeof state.data.documents[0].metadata?.category === 'string';
}

// Create Three.js point cloud with circular points
export function createPointCloud(): void {
  if (!state.data || !state.projectedPositions || !threeState.scene) return;

  // Remove existing points
  if (threeState.points) {
    threeState.scene.remove(threeState.points);
    threeState.points.geometry.dispose();
    (threeState.points.material as THREE.PointsMaterial).dispose();
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(state.projectedPositions, 3)
  );

  const colors = new Float32Array(state.data.documents.length * 3);
  const useCategories = hasCategoryData();

  if (useCategories) {
    // Category-based coloring
    state.timeRange = null;
    for (let i = 0; i < state.data.documents.length; i++) {
      const doc = state.data.documents[i];
      const category = doc.metadata?.category as string;
      const color = CATEGORY_COLORS[category] || new THREE.Color(0x666666);

      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
  } else {
    // Time-based coloring
    const timestamps: (number | null)[] = [];
    let minTime = Infinity;
    let maxTime = -Infinity;

    for (let i = 0; i < state.data.documents.length; i++) {
      const doc = state.data.documents[i];
      const timestamp = doc.metadata?.timestamp as string | undefined;

      if (timestamp) {
        const time = new Date(timestamp).getTime();
        if (!isNaN(time)) {
          timestamps.push(time);
          minTime = Math.min(minTime, time);
          maxTime = Math.max(maxTime, time);
        } else {
          timestamps.push(null);
        }
      } else {
        timestamps.push(null);
      }
    }

    // Store time range for legend
    if (minTime !== Infinity && maxTime !== -Infinity) {
      state.timeRange = { min: new Date(minTime), max: new Date(maxTime) };
    } else {
      state.timeRange = null;
    }

    // Assign colors based on time (or default gray if no timestamp)
    const timeRangeMs = maxTime - minTime || 1;
    const defaultColor = new THREE.Color(0x666666);

    for (let i = 0; i < state.data.documents.length; i++) {
      const time = timestamps[i];
      let color: THREE.Color;

      if (time !== null && state.timeRange) {
        const t = (time - minTime) / timeRangeMs;
        color = getGradientColor(t);
      } else {
        color = defaultColor;
      }

      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
  }

  state.baseColors = new Float32Array(colors);
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: 0.6,
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    alphaTest: 0.5,
    map: createCircleTexture(),
  });

  threeState.points = new THREE.Points(geometry, material);
  threeState.scene.add(threeState.points);

  state.selectedIndex = null;
  state.matchedIndices.clear();
}

// Update point colors based on selection and search
export function updatePointColors(): void {
  if (!threeState.points || !state.data || !state.baseColors) return;

  const geometry = threeState.points.geometry;
  const colors = geometry.getAttribute('color') as THREE.BufferAttribute;
  const hasQuery = state.matchedIndices.size > 0;
  const hasBoxSelection = state.selectedIndices.size > 0;

  for (let i = 0; i < state.data.documents.length; i++) {
    let r = state.baseColors[i * 3];
    let g = state.baseColors[i * 3 + 1];
    let b = state.baseColors[i * 3 + 2];

    if (i === state.selectedIndex) {
      r = SELECTED_COLOR.r;
      g = SELECTED_COLOR.g;
      b = SELECTED_COLOR.b;
    } else if (hasBoxSelection) {
      if (state.selectedIndices.has(i)) {
        // Highlight selected points in bright cyan
        r = 0.0;
        g = 0.9;
        b = 1.0;
      } else {
        r *= DIM_FACTOR;
        g *= DIM_FACTOR;
        b *= DIM_FACTOR;
      }
    } else if (hasQuery) {
      if (state.matchedIndices.has(i)) {
        // Brightness based on similarity score
        const similarity = state.matchedIndices.get(i)!;
        const brightness = 0.5 + similarity * 0.5;
        r = HIGHLIGHT_COLOR.r * brightness;
        g = HIGHLIGHT_COLOR.g * brightness;
        b = HIGHLIGHT_COLOR.b * brightness;
      } else {
        r *= DIM_FACTOR;
        g *= DIM_FACTOR;
        b *= DIM_FACTOR;
      }
    }

    colors.setXYZ(i, r, g, b);
  }

  colors.needsUpdate = true;
}
