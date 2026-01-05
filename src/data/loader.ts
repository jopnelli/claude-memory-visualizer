// Data loading

import { state } from '../state';
import { initEmbedder } from '../embedding';
import { computeProjection } from './projection';
import type { DataSet, ProjectionAlgorithm } from '../types';

type LoadingCallback = (message: string) => void;
type StatusCallback = (message: string, showSpinner?: boolean) => void;
type StatsCallback = (data: DataSet) => void;

// Load data from URL
export async function loadData(
  url: string,
  algorithm: ProjectionAlgorithm,
  onLoading?: LoadingCallback,
  onStats?: StatsCallback,
  onSearchStatus?: StatusCallback
): Promise<void> {
  onLoading?.('Loading data...');

  try {
    const response = await fetch(url);
    const json = (await response.json()) as DataSet;

    state.data = json;

    // Initialize embedder (checks local server, falls back to browser)
    initEmbedder(onSearchStatus);

    // Compute projection (this sets timeRange for legend)
    await computeProjection(algorithm, onLoading);

    // Notify about loaded data AFTER projection (legend needs timeRange)
    onStats?.(json);
  } catch (err) {
    console.error('Error loading data:', err);
    onLoading?.('Error loading data');
  }
}

// Check if a data file exists
export async function checkDataExists(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    return response.ok;
  } catch {
    return false;
  }
}
