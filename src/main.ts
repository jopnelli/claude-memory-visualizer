// Claude Memory Visualizer - Main Entry Point
// A 3D visualization tool for exploring conversation embeddings

import { state, threeState, clearSearch } from './state';
import type { ProjectionAlgorithm, DataSet } from './types';

// Three.js
import {
  initThree,
  render,
  updateControls,
  getRendererElement,
} from './three/scene';
import { createPointCloud, updatePointColors } from './three/points';
import {
  findPointsInSelection,
  raycastPoints,
  updateSelectionBoxVisual,
  hideSelectionBox,
  setControlsEnabled,
} from './three/selection';
import { animatePositionTransition } from './three/animation';

// Data
import { loadData, checkDataExists, computeProjection } from './data';

// Search
import { semanticSearch } from './search';

// UI
import {
  initSpinnerStyles,
  updateInspector,
  updateLegend,
  showSelectionSummary,
  showSettingsModal,
  updateSearchStatus,
  updateSearchResults,
  updateAlgorithmInfo,
  SEARCH_INFO,
} from './ui';

// DOM elements
const container = document.getElementById('canvas-container')!;
const loadingEl = document.getElementById('loading')!;
const statsEl = document.getElementById('stats')!;
const inspectorEl = document.querySelector('.inspector-content')!;
const legendEl = document.getElementById('legend')! as HTMLElement;
const sampleSelect = document.getElementById('sample-select') as HTMLSelectElement;
const algorithmSelect = document.getElementById('algorithm-select') as HTMLSelectElement;
const queryInput = document.getElementById('query-input') as HTMLInputElement;

// Search UI elements (created dynamically)
let searchStatusEl: HTMLElement;
let searchResultsEl: HTMLElement;

// Animation loop
function animate() {
  requestAnimationFrame(animate);
  updateControls();

  // Handle smooth position transitions
  animatePositionTransition();

  // Raycasting for hover (skip during selection)
  if (threeState.points && state.data && !state.isSelecting) {
    const hoveredIdx = raycastPoints();
    if (hoveredIdx !== null && hoveredIdx !== state.hoveredIndex) {
      state.hoveredIndex = hoveredIdx;
      if (state.selectedIndex === null && state.selectedIndices.size === 0) {
        updateInspector(inspectorEl, state.data.documents[state.hoveredIndex], false);
      }
    }
  }

  render();
}

// Mouse event handlers
function onMouseMove(event: MouseEvent) {
  const rendererEl = getRendererElement();
  if (!rendererEl) return;

  const rect = rendererEl.getBoundingClientRect();
  threeState.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  threeState.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  // Update selection box if selecting
  if (state.isSelecting && state.selectionStart) {
    const currentX = event.clientX - rect.left;
    const currentY = event.clientY - rect.top;
    updateSelectionBoxVisual(
      state.selectionStart.x,
      state.selectionStart.y,
      currentX,
      currentY
    );
  }
}

function onMouseDown(event: MouseEvent) {
  // Ctrl/Cmd+click starts selection mode
  if (event.ctrlKey || event.metaKey) {
    event.preventDefault();
    state.isSelecting = true;
    setControlsEnabled(false);

    const rendererEl = getRendererElement();
    if (!rendererEl) return;

    const rect = rendererEl.getBoundingClientRect();
    state.selectionStart = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  }
}

function onMouseUp(event: MouseEvent) {
  if (state.isSelecting && state.selectionStart) {
    const rendererEl = getRendererElement();
    if (!rendererEl) return;

    const rect = rendererEl.getBoundingClientRect();
    const endX = event.clientX - rect.left;
    const endY = event.clientY - rect.top;

    // Calculate selection bounds in screen space
    const minX = Math.min(state.selectionStart.x, endX);
    const maxX = Math.max(state.selectionStart.x, endX);
    const minY = Math.min(state.selectionStart.y, endY);
    const maxY = Math.max(state.selectionStart.y, endY);

    // Only process if selection is larger than 10px
    if (maxX - minX > 10 && maxY - minY > 10) {
      findPointsInSelection(minX, minY, maxX, maxY, rect.width, rect.height);
      if (state.selectedIndices.size > 0) {
        showSelectionSummary();
        updatePointColors();
      }
    }

    // Reset selection state
    state.isSelecting = false;
    state.selectionStart = null;
    hideSelectionBox();
    setControlsEnabled(true);
  }
}

function onMouseClick() {
  if (!threeState.points || !state.data) return;

  const clickedIndex = raycastPoints();

  if (clickedIndex !== null) {
    if (state.selectedIndex === clickedIndex) {
      state.selectedIndex = null;
      updateInspector(inspectorEl, null, false);
    } else {
      state.selectedIndex = clickedIndex;
      updateInspector(inspectorEl, state.data.documents[state.selectedIndex], true);
    }
    updatePointColors();
  } else {
    state.selectedIndex = null;
    updateInspector(inspectorEl, null, false);
    updatePointColors();
  }
}

// Setup search UI elements
function setupSearchUI() {
  const queryContainer = document.querySelector('.query-input');
  if (!queryContainer) return;

  // Status element
  searchStatusEl = document.createElement('div');
  searchStatusEl.id = 'search-status';
  searchStatusEl.style.cssText =
    'font-size: 12px; color: #888; margin-top: 8px; display: flex; align-items: center; gap: 8px;';
  queryContainer.appendChild(searchStatusEl);

  // Results element
  searchResultsEl = document.createElement('div');
  searchResultsEl.id = 'search-results';
  searchResultsEl.style.cssText =
    'font-size: 12px; margin-top: 8px; max-height: 200px; overflow-y: auto;';
  queryContainer.appendChild(searchResultsEl);

  // Search explanation
  const searchInfoEl = document.createElement('div');
  searchInfoEl.id = 'search-info';
  searchInfoEl.className = 'info-panel';
  searchInfoEl.style.marginTop = '16px';
  searchInfoEl.innerHTML = SEARCH_INFO;
  queryContainer.appendChild(searchInfoEl);
}

// Callbacks for data loading
function onLoading(message: string) {
  loadingEl.textContent = message;
}

function onStats(data: DataSet) {
  queryInput.disabled = false;
  statsEl.innerHTML = `
    <strong>${data.metadata.name}</strong><br>
    ${data.documents.length} documents<br>
    ${data.metadata.embedding_dim}D embeddings<br>
    Model: ${data.metadata.embedding_model}
  `;
  updateLegend(legendEl);
}

function onSearchStatus(message: string, showSpinner?: boolean) {
  updateSearchStatus(searchStatusEl, message, showSpinner);
}

// Global function for clicking search results
(window as unknown as { selectDocument: (index: number) => void }).selectDocument = (
  index: number
) => {
  if (state.data && index >= 0 && index < state.data.documents.length) {
    state.selectedIndex = index;
    updateInspector(inspectorEl, state.data.documents[index], true);
    updatePointColors();
  }
};

// Get current algorithm
function getAlgorithm(): ProjectionAlgorithm {
  return algorithmSelect.value as ProjectionAlgorithm;
}

// Initialize event listeners
function setupEventListeners() {
  const rendererEl = getRendererElement();
  if (rendererEl) {
    rendererEl.addEventListener('mousemove', onMouseMove);
    rendererEl.addEventListener('click', onMouseClick);
    rendererEl.addEventListener('mousedown', onMouseDown);
    rendererEl.addEventListener('mouseup', onMouseUp);
  }

  // Dataset selector
  sampleSelect.addEventListener('change', (e) => {
    const value = (e.target as HTMLSelectElement).value;

    // Clear search when switching datasets
    queryInput.value = '';
    clearSearch();
    searchResultsEl.innerHTML = '';
    searchStatusEl.innerHTML = '';

    if (value === 'claude-memory') {
      loadData(
        '/data/claude-memory.json',
        getAlgorithm(),
        onLoading,
        onStats,
        onSearchStatus
      );
    } else if (value === 'demo') {
      loadData('/data/demo.json', getAlgorithm(), onLoading, onStats, onSearchStatus);
    }

    // Remove placeholder option
    const defaultOption = sampleSelect.querySelector('option[value=""]');
    if (defaultOption) defaultOption.remove();
  });

  // Algorithm selector
  algorithmSelect.addEventListener('change', () => {
    const infoEl = document.getElementById('algorithm-info');
    if (infoEl) {
      updateAlgorithmInfo(infoEl, algorithmSelect.value);
    }
    if (state.data) {
      computeProjection(getAlgorithm(), onLoading);
    }
  });

  // Search input with debounce
  let searchTimeout: number | null = null;
  queryInput.addEventListener('input', (e) => {
    const query = (e.target as HTMLInputElement).value;

    if (searchTimeout) {
      clearTimeout(searchTimeout);
    }

    searchTimeout = window.setTimeout(() => {
      semanticSearch(
        query,
        onSearchStatus,
        (matched, total, topMatches, bestScore) => {
          updateSearchResults(
            searchResultsEl,
            queryInput,
            matched,
            total,
            topMatches,
            bestScore
          );
        }
      );
    }, 300);
  });

  // Settings button
  document.getElementById('settings-btn')?.addEventListener('click', showSettingsModal);
}

// Initialize the application
async function init() {
  // Setup UI
  initSpinnerStyles();
  setupSearchUI();

  // Initialize Three.js
  initThree(container);

  // Setup event listeners
  setupEventListeners();

  // Update algorithm info
  const infoEl = document.getElementById('algorithm-info');
  if (infoEl) {
    updateAlgorithmInfo(infoEl, algorithmSelect.value);
  }

  // Start animation loop
  animate();

  // Auto-load data
  const hasClaudeMemory = await checkDataExists('/data/claude-memory.json');
  if (hasClaudeMemory) {
    sampleSelect.value = 'claude-memory';
    loadData(
      '/data/claude-memory.json',
      getAlgorithm(),
      onLoading,
      onStats,
      onSearchStatus
    );
  } else {
    // Disable claude-memory option if data doesn't exist
    const claudeOption = sampleSelect.querySelector('option[value="claude-memory"]') as HTMLOptionElement;
    if (claudeOption) {
      claudeOption.disabled = true;
      claudeOption.textContent = 'Claude Memory (not exported)';
    }
    sampleSelect.value = 'demo';
    loadData('/data/demo.json', getAlgorithm(), onLoading, onStats, onSearchStatus);
  }

  // Remove placeholder
  const defaultOption = sampleSelect.querySelector('option[value=""]');
  if (defaultOption) defaultOption.remove();
}

// Start the app
init();
