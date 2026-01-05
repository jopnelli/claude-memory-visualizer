// Box selection and raycasting

import * as THREE from 'three';
import { state, threeState } from '../state';

// Find all points within the 2D selection box
export function findPointsInSelection(
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  width: number,
  height: number
): void {
  if (!threeState.points || !state.data || !state.projectedPositions || !threeState.camera) {
    return;
  }

  state.selectedIndices.clear();

  // Convert screen bounds to normalized device coordinates
  const ndc = {
    minX: (minX / width) * 2 - 1,
    maxX: (maxX / width) * 2 - 1,
    minY: -(maxY / height) * 2 + 1, // Y is inverted
    maxY: -(minY / height) * 2 + 1,
  };

  // Check each point
  for (let i = 0; i < state.data.documents.length; i++) {
    const x = state.projectedPositions[i * 3];
    const y = state.projectedPositions[i * 3 + 1];
    const z = state.projectedPositions[i * 3 + 2];

    // Project 3D point to screen space
    const vector = new THREE.Vector3(x, y, z);
    vector.project(threeState.camera);

    // Check if within selection bounds
    if (
      vector.x >= ndc.minX &&
      vector.x <= ndc.maxX &&
      vector.y >= ndc.minY &&
      vector.y <= ndc.maxY &&
      vector.z < 1 // Only visible points (in front of camera)
    ) {
      state.selectedIndices.add(i);
    }
  }
}

// Raycast to find hovered point
export function raycastPoints(): number | null {
  if (!threeState.points || !threeState.raycaster || !threeState.camera) {
    return null;
  }

  threeState.raycaster.setFromCamera(threeState.mouse, threeState.camera);
  const intersects = threeState.raycaster.intersectObject(threeState.points);

  if (intersects.length > 0) {
    return intersects[0].index ?? null;
  }
  return null;
}

// Update selection box visual
export function updateSelectionBoxVisual(
  startX: number,
  startY: number,
  currentX: number,
  currentY: number
): void {
  if (!threeState.selectionBox) return;

  const left = Math.min(startX, currentX);
  const top = Math.min(startY, currentY);
  const width = Math.abs(currentX - startX);
  const height = Math.abs(currentY - startY);

  threeState.selectionBox.style.left = `${left}px`;
  threeState.selectionBox.style.top = `${top}px`;
  threeState.selectionBox.style.width = `${width}px`;
  threeState.selectionBox.style.height = `${height}px`;
  threeState.selectionBox.style.display = 'block';
}

// Hide selection box
export function hideSelectionBox(): void {
  if (threeState.selectionBox) {
    threeState.selectionBox.style.display = 'none';
  }
}

// Enable/disable orbit controls
export function setControlsEnabled(enabled: boolean): void {
  if (threeState.controls) {
    threeState.controls.enabled = enabled;
  }
}
