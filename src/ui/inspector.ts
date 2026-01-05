// Document inspector panel

import { state } from '../state';
import type { Document } from '../types';
import { escapeHtml } from './utils';

// Update inspector panel
export function updateInspector(
  inspectorEl: Element,
  doc: Document | null,
  isSelected: boolean
): void {
  if (!doc) {
    inspectorEl.innerHTML =
      '<div class="inspector-empty">Hover over a point to see details</div>';
    return;
  }

  const docIndex = state.data?.documents.indexOf(doc) ?? -1;
  const similarity = state.matchedIndices.get(docIndex);
  const similarityBadge = similarity
    ? `<span style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-right: 4px;">${(similarity * 100).toFixed(0)}% match</span>`
    : '';

  const metaHtml = doc.metadata
    ? `<div class="doc-meta">${Object.entries(doc.metadata)
        .map(([k, v]) => `<strong>${k}:</strong> ${v}`)
        .join('<br>')}</div>`
    : '';

  const statusBadge = isSelected
    ? '<span style="background: #10b981; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-bottom: 8px; display: inline-block;">SELECTED</span>'
    : '';

  inspectorEl.innerHTML = `
    <div style="margin-bottom: 8px;">${statusBadge}${similarityBadge}</div>
    <div class="doc-text">${escapeHtml(doc.text)}</div>
    ${metaHtml}
  `;
}
