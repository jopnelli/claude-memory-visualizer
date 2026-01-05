// Modal dialogs (settings, selection summary)

import { state, clearSelection } from '../state';
import { updatePointColors } from '../three/points';
import {
  checkOllama,
  resetOllamaCache,
  getSummaryModel,
  setSummaryModel,
  generateWithOllama,
} from '../embedding/ollama';
import type { Document, TopicCluster } from '../types';
import { escapeHtml } from './utils';

// Extract topic from a document
function extractTopic(doc: Document): string {
  const userMatch = doc.text.match(/User:\s*(.+?)(?:\n|$)/i);
  if (userMatch) {
    let topic = userMatch[1].trim();
    if (topic.length > 60) {
      topic = topic.slice(0, 57) + '...';
    }
    return topic;
  }
  const fallback = doc.text.slice(0, 60).replace(/\n/g, ' ').trim();
  return fallback + (doc.text.length > 60 ? '...' : '');
}

// Group documents by category or topic
function clusterTopics(docs: Document[]): TopicCluster[] {
  // Check if docs have category metadata
  const hasCategories = docs.some((d) => typeof d.metadata?.category === 'string');

  if (hasCategories) {
    const categoryGroups = new Map<string, number>();
    for (const doc of docs) {
      const category = (doc.metadata?.category as string) || 'uncategorized';
      categoryGroups.set(category, (categoryGroups.get(category) || 0) + 1);
    }

    return Array.from(categoryGroups.entries())
      .map(([category, count]) => ({
        topic: category.charAt(0).toUpperCase() + category.slice(1),
        count,
        example: '',
      }))
      .sort((a, b) => b.count - a.count);
  }

  // Topic-based clustering
  const topicGroups = new Map<string, { count: number; examples: string[] }>();

  for (const doc of docs) {
    const topic = extractTopic(doc);
    const lowerTopic = topic.toLowerCase();

    let foundGroup = false;
    for (const [key, group] of topicGroups.entries()) {
      const keyLower = key.toLowerCase();
      const topicWords = new Set(lowerTopic.split(/\s+/).filter((w) => w.length > 2));
      const keyWords = new Set(keyLower.split(/\s+/).filter((w) => w.length > 2));
      const overlap = [...topicWords].filter((w) => keyWords.has(w)).length;

      if (overlap >= 1 && (overlap >= 2 || overlap / Math.max(topicWords.size, 1) > 0.3)) {
        group.count++;
        if (group.examples.length < 3) {
          group.examples.push(topic);
        }
        foundGroup = true;
        break;
      }
    }

    if (!foundGroup) {
      topicGroups.set(topic, { count: 1, examples: [topic] });
    }
  }

  const clusters = Array.from(topicGroups.entries())
    .map(([topic, { count, examples }]) => ({
      topic: examples[0],
      count,
      example: examples[0],
    }))
    .sort((a, b) => b.count - a.count);

  const significantClusters = clusters.slice(0, 10);
  const otherCount = clusters.slice(10).reduce((sum, c) => sum + c.count, 0);

  if (otherCount > 0) {
    significantClusters.push({ topic: 'Other topics', count: otherCount, example: '' });
  }

  return significantClusters;
}

// Generate AI summary using Ollama
async function generateAISummary(docs: Document[]): Promise<string> {
  const ollamaReady = await checkOllama();
  if (!ollamaReady) {
    throw new Error('OLLAMA_NOT_AVAILABLE');
  }

  const maxChunks = 15;
  const chunksToSummarize = docs.slice(0, maxChunks);
  const chunksText = chunksToSummarize
    .map((doc, i) => `[${i + 1}] ${doc.text.slice(0, 500)}`)
    .join('\n\n---\n\n');

  const prompt = `Here are ${chunksToSummarize.length} conversation chunks from a Claude memory database. Please provide a concise summary (2-3 sentences) of the main themes and topics discussed across these conversations:\n\n${chunksText}`;

  return generateWithOllama(prompt);
}

// Close selection modal
function closeSelectionModal(): void {
  document.getElementById('selection-modal')?.remove();
  document.getElementById('selection-backdrop')?.remove();
  clearSelection();
  updatePointColors();
}

// Show summary modal for selected chunks
export function showSelectionSummary(): void {
  if (!state.data || state.selectedIndices.size === 0) return;

  const selectedDocs = Array.from(state.selectedIndices).map(
    (i) => state.data!.documents[i]
  );
  const clusters = clusterTopics(selectedDocs);

  // Remove existing modal
  document.getElementById('selection-modal')?.remove();

  const modal = document.createElement('div');
  modal.id = 'selection-modal';
  modal.style.cssText = `
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 24px;
    min-width: 500px;
    max-width: 600px;
    max-height: 80vh;
    overflow-y: auto;
    z-index: 1000;
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
  `;

  const clustersHtml = clusters
    .map(
      (c) => `
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 12px; background: #111; border-radius: 6px; margin-bottom: 6px;">
      <span style="color: #ccc; font-size: 13px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding-right: 12px;">${escapeHtml(c.topic)}</span>
      <span style="color: #3b82f6; font-size: 13px; font-weight: 600; white-space: nowrap;">${c.count}</span>
    </div>
  `
    )
    .join('');

  modal.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
      <h3 style="color: #fff; margin: 0; font-size: 18px; flex: 1;">Selection Summary</h3>
      <button id="close-selection-modal" style="background: none; border: none; color: #666; font-size: 24px; cursor: pointer; padding: 0; margin-left: 16px; line-height: 1; width: 24px; height: 24px; flex-shrink: 0;">&times;</button>
    </div>
    <div style="color: #3b82f6; font-size: 15px; margin-bottom: 20px;">
      <strong>${state.selectedIndices.size}</strong> conversations selected
    </div>
    <div style="margin-bottom: 20px;">
      <div style="color: #888; font-size: 12px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em;">AI Summary</div>
      <div id="ai-summary-result" style="font-size: 13px; color: #888;">
        <div class="spinner" style="margin: 8px 0;"></div> Generating summary...
      </div>
    </div>
    <div style="padding-top: 16px; border-top: 1px solid #333;">
      <div style="color: #888; font-size: 12px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em;">Topics</div>
      ${clustersHtml}
    </div>
    <div style="margin-top: 16px;">
      <button id="clear-selection" style="width: 100%; padding: 12px; background: #333; border: none; border-radius: 6px; color: #fff; cursor: pointer; font-size: 14px;">Clear Selection</button>
    </div>
  `;

  document.body.appendChild(modal);

  // Add backdrop
  const backdrop = document.createElement('div');
  backdrop.id = 'selection-backdrop';
  backdrop.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.5);
    z-index: 999;
  `;
  document.body.appendChild(backdrop);

  // Event listeners
  document
    .getElementById('close-selection-modal')
    ?.addEventListener('click', closeSelectionModal);
  document.getElementById('clear-selection')?.addEventListener('click', () => {
    clearSelection();
    updatePointColors();
    closeSelectionModal();
  });
  backdrop.addEventListener('click', closeSelectionModal);

  // Generate AI summary
  generateAISummary(selectedDocs)
    .then((summary) => {
      const resultDiv = document.getElementById('ai-summary-result');
      if (resultDiv) {
        resultDiv.innerHTML = `<div style="color: #e0e0e0; line-height: 1.5; padding: 12px; background: #111; border-radius: 6px;">${escapeHtml(summary)}</div>`;
      }
    })
    .catch((err) => {
      const resultDiv = document.getElementById('ai-summary-result');
      if (resultDiv) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        if (errorMessage === 'OLLAMA_NOT_AVAILABLE') {
          resultDiv.innerHTML =
            '<span style="color: #666;">Ollama not running. Start Ollama to enable AI summaries.</span>';
        } else {
          resultDiv.innerHTML = `<span style="color: #ef4444;">Error: ${escapeHtml(errorMessage)}</span>`;
        }
      }
    });
}

// Show settings modal
export function showSettingsModal(): void {
  document.getElementById('settings-modal')?.remove();

  const currentSummaryModel = getSummaryModel();

  const modal = document.createElement('div');
  modal.id = 'settings-modal';
  modal.style.cssText = `
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 24px;
    min-width: 450px;
    max-width: 550px;
    max-height: 85vh;
    overflow-y: auto;
    z-index: 1000;
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
  `;

  modal.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
      <h3 style="color: #fff; margin: 0; font-size: 18px;">Settings</h3>
      <button id="close-settings-modal" style="background: none; border: none; color: #666; font-size: 24px; cursor: pointer; padding: 0; line-height: 1; width: 24px; height: 24px;">&times;</button>
    </div>

    <!-- Ollama Status -->
    <div style="margin-bottom: 20px;">
      <div id="ollama-status" style="font-size: 12px; color: #666; padding: 10px; background: #111; border-radius: 6px;">Checking Ollama...</div>
    </div>

    <!-- Search Info -->
    <div style="margin-bottom: 20px;">
      <div style="color: #888; font-size: 12px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;">Semantic Search</div>
      <div style="font-size: 12px; color: #666; padding: 10px; background: #111; border-radius: 6px;">
        Uses <code style="background: #222; padding: 1px 4px; border-radius: 2px;">all-mpnet-base-v2</code> (same as claude-memory) via local Python server or browser fallback.
      </div>
    </div>

    <!-- Summary Model -->
    <div style="margin-bottom: 20px;">
      <div style="color: #888; font-size: 12px; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;">Summary Model</div>
      <input type="text" id="settings-summary-model" placeholder="qwen2.5:1.5b" value="${escapeHtml(currentSummaryModel)}" style="width: 100%; padding: 10px 12px; background: #111; border: 1px solid #333; border-radius: 6px; color: #fff; font-size: 13px; box-sizing: border-box;">
      <div style="font-size: 11px; color: #555; margin-top: 6px;">
        Used for AI summaries when selecting points (any Ollama text model).
      </div>
    </div>

    <button id="save-settings" style="width: 100%; padding: 12px; background: #2563eb; border: none; border-radius: 6px; color: #fff; cursor: pointer; font-size: 14px;">Save Settings</button>
  `;

  document.body.appendChild(modal);

  // Add backdrop
  const backdrop = document.createElement('div');
  backdrop.id = 'settings-backdrop';
  backdrop.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.5);
    z-index: 999;
  `;
  document.body.appendChild(backdrop);

  // Check Ollama status
  (async () => {
    const statusEl = document.getElementById('ollama-status');
    if (!statusEl) return;

    resetOllamaCache();
    const available = await checkOllama();
    if (available) {
      statusEl.innerHTML = '<span style="color: #10b981;">✓ Ollama is running</span>';
    } else {
      statusEl.innerHTML =
        '<span style="color: #f59e0b;">⚠ Ollama not detected</span>';
    }
  })();

  // Event listeners
  const closeSettings = () => {
    document.getElementById('settings-modal')?.remove();
    document.getElementById('settings-backdrop')?.remove();
  };

  document.getElementById('close-settings-modal')?.addEventListener('click', closeSettings);
  backdrop.addEventListener('click', closeSettings);

  document.getElementById('save-settings')?.addEventListener('click', () => {
    const summaryModelInput = document.getElementById(
      'settings-summary-model'
    ) as HTMLInputElement;
    const summaryModel = summaryModelInput?.value?.trim() || 'qwen2.5:1.5b';

    setSummaryModel(summaryModel);
    closeSettings();
  });
}
