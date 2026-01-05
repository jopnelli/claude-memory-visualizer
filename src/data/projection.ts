// Dimensionality reduction and projection

import * as druid from '@saehrimnir/druidjs';
import { state } from '../state';
import { createPointCloud } from '../three/points';
import type { ProjectionAlgorithm } from '../types';

type LoadingCallback = (message: string) => void;

// Compute dimensionality reduction
export async function computeProjection(
  algorithm: ProjectionAlgorithm,
  onLoading?: LoadingCallback
): Promise<void> {
  if (!state.data) return;

  // Store previous positions for smooth transition (only if same document count)
  const previousDocCount = state.projectedPositions
    ? state.projectedPositions.length / 3
    : 0;
  const sameDocCount = previousDocCount === state.data.documents.length;
  const hadPreviousPoints =
    state.projectedPositions !== null && sameDocCount;

  if (hadPreviousPoints && state.projectedPositions) {
    state.previousPositions = new Float32Array(state.projectedPositions);
  }

  // Check for pre-computed projections
  const precomputed = state.data.projections?.[algorithm];

  if (precomputed) {
    onLoading?.(`Loading ${algorithm.toUpperCase()}...`);
    await new Promise((resolve) => setTimeout(resolve, 10)); // Brief delay for UI update

    const positions = new Float32Array(state.data.documents.length * 3);
    for (let i = 0; i < precomputed.length; i++) {
      const point = precomputed[i];
      if (point) {
        const [x, y, z] = point;
        positions[i * 3] = x ?? 0;
        positions[i * 3 + 1] = y ?? 0;
        positions[i * 3 + 2] = z ?? 0;
      }
    }

    state.projectedPositions = positions;

    if (hadPreviousPoints) {
      // Trigger smooth transition animation
      state.animationProgress = 0;
    } else {
      createPointCloud();
    }

    onLoading?.('');
    return;
  }

  // Fall back to browser computation (no t-SNE - too slow)
  if (algorithm === 'tsne') {
    onLoading?.(
      't-SNE requires pre-computed projections. Run: bun run export-data'
    );
    return;
  }

  onLoading?.(`Computing ${algorithm.toUpperCase()}...`);
  await new Promise((resolve) => setTimeout(resolve, 50));

  try {
    const embeddings = state.data.documents.map((d) => d.embedding);
    const matrix = druid.Matrix.from(embeddings);

    let result: druid.Matrix;

    switch (algorithm) {
      case 'umap':
        const umap = new druid.UMAP(matrix, {
          d: 3,
          n_neighbors: 15,
          min_dist: 0.1,
        });
        result = umap.transform();
        break;
      case 'pca':
      default:
        const pca = new druid.PCA(matrix, { d: 3 });
        result = pca.transform();
        break;
    }

    const positions = new Float32Array(state.data.documents.length * 3);
    const resultArray = result.to2dArray;

    let minX = Infinity,
      maxX = -Infinity;
    let minY = Infinity,
      maxY = -Infinity;
    let minZ = Infinity,
      maxZ = -Infinity;

    for (let i = 0; i < resultArray.length; i++) {
      const [x, y, z] = resultArray[i];
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
    }

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const rangeZ = maxZ - minZ || 1;
    const scale = 40;

    for (let i = 0; i < resultArray.length; i++) {
      const [x, y, z] = resultArray[i];
      positions[i * 3] = ((x - minX) / rangeX - 0.5) * scale;
      positions[i * 3 + 1] = ((y - minY) / rangeY - 0.5) * scale;
      positions[i * 3 + 2] = ((z - minZ) / rangeZ - 0.5) * scale;
    }

    state.projectedPositions = positions;

    if (hadPreviousPoints) {
      // Trigger smooth transition animation
      state.animationProgress = 0;
    } else {
      createPointCloud();
    }

    onLoading?.('');
  } catch (err) {
    console.error('Projection error:', err);
    onLoading?.('Error computing projection');
  }
}
