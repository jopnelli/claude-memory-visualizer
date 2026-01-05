// Search UI components

import { state } from '../state';
import type { SearchMatch } from '../types';
import { escapeHtml } from './utils';

// Algorithm explanations
export const ALGORITHM_INFO: Record<string, string> = {
  umap: `<strong>UMAP</strong>
<br><br>
Best balance of speed and quality. Similar conversations cluster together while keeping the overall shape meaningful.`,

  tsne: `<strong>t-SNE</strong>
<br><br>
Creates tight, well-separated clusters. Great for finding distinct groups of conversations.`,

  pca: `<strong>PCA</strong>
<br><br>
Fastest option. Good for a quick overview, but clusters may overlap more than other methods.`,
};

// Search explanation
export const SEARCH_INFO = `<strong>How search works</strong>
<br><br>
Type anything and we'll find conversations with <strong>similar meaning</strong> — even if they use different words.
<br><br>
<details style="margin-top: 8px; font-size: 11px; color: #888;">
  <summary style="cursor: pointer; color: #666;">Learn more</summary>
  <div style="margin-top: 8px; padding: 8px; background: #111; border-radius: 4px;">
    <strong style="color: #ccc;">Embedding</strong><br>
    Text is converted to a 768-dimensional vector using all-mpnet-base-v2 via a local Python server (same model as claude-memory).
    <br><br>
    <strong style="color: #ccc;">Similarity</strong><br>
    We use <em>cosine similarity</em> — measuring the angle between vectors. Score of 100% = identical meaning, 0% = unrelated.
  </div>
</details>`;

// Update search status display
export function updateSearchStatus(
  statusEl: HTMLElement,
  message: string,
  showSpinner: boolean = false
): void {
  if (showSpinner) {
    statusEl.innerHTML = `<div class="spinner"></div><span>${message}</span>`;
  } else if (message) {
    statusEl.innerHTML = `<span>${message}</span>`;
  } else {
    statusEl.innerHTML = '';
  }
}

// Update search results display
export function updateSearchResults(
  resultsEl: HTMLElement,
  queryInput: HTMLInputElement,
  matched: number,
  total: number,
  topMatches: SearchMatch[],
  bestScore?: number
): void {
  if (matched > 0 && topMatches.length > 0 && state.data) {
    let html = '<div style="font-size: 11px;">';

    for (const match of topMatches) {
      const doc = state.data.documents[match.index];
      if (!doc) continue;
      const userMatch = doc.text.match(/User:\s*(.+?)(?:\n|$)/);
      const preview = userMatch?.[1]
        ? userMatch[1].slice(0, 80)
        : doc.text.slice(0, 80).replace(/\n/g, ' ');
      const score = (match.score * 100).toFixed(0);
      const barWidth = Math.round((match.score / (bestScore || 1)) * 100);

      html += `
        <div style="margin-top: 6px; padding: 6px; background: #1a1a1a; border-radius: 4px; cursor: pointer;"
             onclick="window.selectDocument && window.selectDocument(${match.index})">
          <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
            <div style="width: 40px; height: 4px; background: #333; border-radius: 2px; overflow: hidden;">
              <div style="width: ${barWidth}%; height: 100%; background: #3b82f6;"></div>
            </div>
            <span style="color: #3b82f6; font-size: 10px;">${score}%</span>
          </div>
          <div style="color: #ccc; font-size: 11px;">${escapeHtml(preview)}${preview.length >= 80 ? '...' : ''}</div>
        </div>
      `;
    }

    html += '</div>';
    resultsEl.innerHTML = html;
  } else if (queryInput.value.trim() !== '') {
    resultsEl.innerHTML =
      '<span style="color: #666; font-size: 11px;">No strong matches found. Try different words.</span>';
  } else {
    resultsEl.innerHTML = '';
  }
}

// Update algorithm info panel
export function updateAlgorithmInfo(
  infoEl: HTMLElement,
  algorithm: string
): void {
  infoEl.innerHTML = ALGORITHM_INFO[algorithm] || '';
}
