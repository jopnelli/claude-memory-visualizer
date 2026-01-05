// Browser-based embedding using Transformers.js (fallback)

import { pipeline, FeatureExtractionPipeline } from '@huggingface/transformers';
import { embeddingState } from '../state';

type StatusCallback = (message: string, showSpinner?: boolean) => void;

// Load browser embedding model
export async function loadBrowserEmbedder(
  onStatus?: StatusCallback
): Promise<FeatureExtractionPipeline | null> {
  // Return existing embedder
  if (embeddingState.embedder) return embeddingState.embedder;

  // Wait for existing load if in progress
  if (embeddingState.embedderPromise) return embeddingState.embedderPromise;

  // Start loading - all-mpnet-base-v2 is ~420MB
  onStatus?.('Loading embedding model in browser (first time only)...', true);

  embeddingState.embedderPromise = (async () => {
    try {
      // Use same model as claude-memory: all-mpnet-base-v2 (768 dims)
      embeddingState.embedder = await pipeline(
        'feature-extraction',
        'Xenova/all-mpnet-base-v2',
        { dtype: 'fp32' }
      );
      onStatus?.('Semantic search ready (browser)');
      return embeddingState.embedder;
    } catch (err) {
      console.error('Failed to load browser embedder:', err);
      onStatus?.('Semantic search unavailable - using text search');
      embeddingState.embedderPromise = null;
      return null;
    }
  })();

  return embeddingState.embedderPromise;
}

// Get embedding from browser model
export async function getBrowserEmbedding(
  text: string,
  onStatus?: StatusCallback
): Promise<number[] | null> {
  const emb = await loadBrowserEmbedder(onStatus);
  if (!emb) return null;

  try {
    const output = await emb(text, { pooling: 'mean', normalize: true });

    // Handle Tensor output from transformers.js
    if (output && typeof output.tolist === 'function') {
      const list = output.tolist();
      const embedding = Array.isArray(list[0]) ? list[0] : list;
      return embedding;
    } else if (output.data) {
      return Array.from(output.data as Float32Array);
    } else if (Array.isArray(output)) {
      return output.flat();
    }
  } catch (err) {
    console.error('Browser embedding error:', err);
  }

  return null;
}
