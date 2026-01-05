// Semantic and text search

import { state } from './state';
import { getEmbedding, cosineSimilarity } from './embedding';
import { updatePointColors } from './three/points';
import type { SearchMatch } from './types';

type StatusCallback = (message: string, showSpinner?: boolean) => void;
type ResultsCallback = (
  matched: number,
  total: number,
  topMatches: SearchMatch[],
  bestScore?: number
) => void;

// Fallback to text search
export function fallbackTextSearch(
  query: string,
  onResults?: ResultsCallback
): void {
  if (!state.data) return;

  state.matchedIndices.clear();

  const lowerQuery = query.toLowerCase();
  const terms = lowerQuery.split(/\s+/).filter((t) => t.length > 0);

  const matches: SearchMatch[] = [];

  for (let i = 0; i < state.data.documents.length; i++) {
    const doc = state.data.documents[i];
    const text = doc.text.toLowerCase();

    const allTermsMatch = terms.every((term) => text.includes(term));
    if (allTermsMatch) {
      // Score based on how many times terms appear
      let score = 0;
      for (const term of terms) {
        const count = (text.match(new RegExp(term, 'g')) || []).length;
        score += count;
      }
      const normalizedScore = Math.min(1.0, score / 10);
      state.matchedIndices.set(i, normalizedScore);
      matches.push({ index: i, score: normalizedScore });
    }
  }

  // Sort by score
  matches.sort((a, b) => b.score - a.score);

  updatePointColors();
  onResults?.(matches.length, state.data.documents.length, matches.slice(0, 8), 1.0);
}

// Semantic search using embeddings
export async function semanticSearch(
  query: string,
  onStatus?: StatusCallback,
  onResults?: ResultsCallback
): Promise<void> {
  if (!state.data) return;

  state.matchedIndices.clear();

  if (query.trim() === '') {
    updatePointColors();
    onResults?.(0, 0, []);
    onStatus?.('');
    return;
  }

  // Check if data has real embeddings (not 3D demo coordinates)
  const embDim = state.data.documents[0]?.embedding?.length || 0;
  if (embDim < 100) {
    // Use text search for demo data
    onStatus?.('Text search (demo mode)');
    fallbackTextSearch(query, onResults);
    return;
  }

  onStatus?.('Searching...', true);

  // Get embedding for query
  const queryEmbedding = await getEmbedding(query, onStatus);

  if (queryEmbedding) {
    // Calculate similarities
    const similarities: SearchMatch[] = [];

    for (let i = 0; i < state.data.documents.length; i++) {
      const score = cosineSimilarity(queryEmbedding, state.data.documents[i].embedding);
      similarities.push({ index: i, score });
    }

    // Sort by similarity
    similarities.sort((a, b) => b.score - a.score);

    const bestScore = similarities[0]?.score || 0;
    const minScore = 0.3; // Absolute minimum
    const relativeThreshold = bestScore * 0.7; // Within 70% of best match
    const threshold = Math.max(minScore, relativeThreshold);

    // Take top matches that meet threshold, max 20 for visualization clarity
    const topK = 20;
    const matches = similarities.filter((s) => s.score >= threshold).slice(0, topK);

    for (const match of matches) {
      state.matchedIndices.set(match.index, match.score);
    }

    updatePointColors();
    onResults?.(
      matches.length,
      state.data.documents.length,
      matches.slice(0, 8),
      bestScore
    );

    if (matches.length > 0) {
      onStatus?.(`Top ${matches.length} matches (${(bestScore * 100).toFixed(0)}% best)`);
    } else {
      onStatus?.('No strong matches found');
    }
  } else {
    // Fall back to text search if embedding failed
    fallbackTextSearch(query, onResults);
  }
}
