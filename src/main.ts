import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import * as druid from '@saehrimnir/druidjs';
import { pipeline, FeatureExtractionPipeline } from '@huggingface/transformers';

interface Document {
  id: string;
  text: string;
  embedding: number[];
  metadata?: Record<string, unknown>;
}

interface DataSet {
  metadata: {
    name: string;
    embedding_model: string;
    embedding_dim: number;
    count?: number;
  };
  documents: Document[];
  projections?: {
    umap?: number[][];
    tsne?: number[][];
    pca?: number[][];
  };
}

// Global state
let data: DataSet | null = null;
let projectedPositions: Float32Array | null = null;
let previousPositions: Float32Array | null = null; // For smooth transitions
let animationProgress: number = 1; // 0 to 1, 1 = complete
let points: THREE.Points | null = null;
let hoveredIndex: number | null = null;
let selectedIndex: number | null = null;
let selectedIndices: Set<number> = new Set(); // For box selection
let matchedIndices: Map<number, number> = new Map(); // index -> similarity score
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let renderer: THREE.WebGLRenderer;
let controls: OrbitControls;
let raycaster: THREE.Raycaster;
let mouse: THREE.Vector2;
let baseColors: Float32Array | null = null;
let embedder: FeatureExtractionPipeline | null = null;
let embedderPromise: Promise<FeatureExtractionPipeline | null> | null = null;

// Selection state
let isSelecting: boolean = false;
let selectionStart: { x: number; y: number } | null = null;
let selectionBox: HTMLDivElement | null = null;

// Grid helpers
let gridHelper: THREE.GridHelper | null = null;
let axesGroup: THREE.Group | null = null;

// DOM elements
const container = document.getElementById('canvas-container')!;
const loadingEl = document.getElementById('loading')!;
const statsEl = document.getElementById('stats')!;
const inspectorEl = document.querySelector('.inspector-content')!;
const legendEl = document.getElementById('legend')!;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const sampleSelect = document.getElementById('sample-select') as HTMLSelectElement;
const algorithmSelect = document.getElementById('algorithm-select') as HTMLSelectElement;
const queryInput = document.getElementById('query-input') as HTMLInputElement;

// Color gradient for time-based coloring (old → new)
const TIME_GRADIENT = [
  new THREE.Color(0x3b82f6), // blue (oldest)
  new THREE.Color(0x06b6d4), // cyan
  new THREE.Color(0x10b981), // green
  new THREE.Color(0xfbbf24), // yellow
  new THREE.Color(0xf97316), // orange
  new THREE.Color(0xef4444), // red (newest)
];

// Category colors for demo dataset
const CATEGORY_COLORS: Record<string, THREE.Color> = {
  geography: new THREE.Color(0x3b82f6), // blue
  programming: new THREE.Color(0x10b981), // green
  ai: new THREE.Color(0xa855f7), // purple
  food: new THREE.Color(0xf97316), // orange
  music: new THREE.Color(0xec4899), // pink
  random: new THREE.Color(0x6b7280), // gray
  // Outlier categories - varied colors
  science: new THREE.Color(0x14b8a6), // teal
  trivia: new THREE.Color(0xeab308), // yellow
  art: new THREE.Color(0xf43f5e), // rose
  history: new THREE.Color(0x8b5cf6), // violet
  astronomy: new THREE.Color(0x0ea5e9), // sky blue
  physics: new THREE.Color(0x22c55e), // lime
  medicine: new THREE.Color(0xef4444), // red
  biology: new THREE.Color(0x84cc16), // lime green
  technology: new THREE.Color(0x06b6d4), // cyan
  economics: new THREE.Color(0xd946ef), // fuchsia
  engineering: new THREE.Color(0xf59e0b), // amber
  weather: new THREE.Color(0x64748b), // slate
};

// Interpolate through gradient
function getGradientColor(t: number): THREE.Color {
  t = Math.max(0, Math.min(1, t));
  const segments = TIME_GRADIENT.length - 1;
  const segment = Math.min(Math.floor(t * segments), segments - 1);
  const segmentT = (t * segments) - segment;

  const color = new THREE.Color();
  color.lerpColors(TIME_GRADIENT[segment], TIME_GRADIENT[segment + 1], segmentT);
  return color;
}

// Time range for legend
let timeRange: { min: Date; max: Date } | null = null;

const HIGHLIGHT_COLOR = new THREE.Color(0xffffff);
const SELECTED_COLOR = new THREE.Color(0x00ff00);
const DIM_FACTOR = 0.15;

// Simplified explanations for algorithms
const ALGORITHM_INFO: Record<string, string> = {
  umap: `<strong>UMAP</strong> <span class="tag">recommended</span>
<br><br>
Best balance of speed and quality. Similar conversations cluster together while keeping the overall shape meaningful.`,

  tsne: `<strong>t-SNE</strong> <span class="tag">pre-computed</span>
<br><br>
Creates tight, well-separated clusters. Great for finding distinct groups of conversations.`,

  pca: `<strong>PCA</strong> <span class="tag">fast</span>
<br><br>
Fastest option. Good for a quick overview, but clusters may overlap more than other methods.`,
};

const SEARCH_INFO = `<strong>How search works</strong>
<br><br>
Type anything and we'll find conversations with <strong>similar meaning</strong> — even if they use different words.
<br><br>
<details style="margin-top: 8px; font-size: 11px; color: #888;">
  <summary style="cursor: pointer; color: #666;">Learn more</summary>
  <div style="margin-top: 8px; padding: 8px; background: #111; border-radius: 4px;">
    <strong style="color: #ccc;">Embedding</strong><br>
    Text is converted to a 768-dimensional vector using an AI model (all-mpnet-base-v2, ~420MB, cached after first use).
    <br><br>
    <strong style="color: #ccc;">Similarity</strong><br>
    We use <em>cosine similarity</em> — measuring the angle between vectors. Score of 100% = identical meaning, 0% = unrelated.
  </div>
</details>`;

// Create circular point texture
function createCircleTexture(): THREE.Texture {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  // Draw circle with soft edge
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
  gradient.addColorStop(0.7, 'rgba(255, 255, 255, 1)');
  gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

// Easing function for smooth transitions
function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

// Create 3D grid and axes
function createGridAndAxes() {
  // Remove existing
  if (gridHelper) {
    scene.remove(gridHelper);
    gridHelper.dispose();
  }
  if (axesGroup) {
    scene.remove(axesGroup);
  }

  // Create subtle grid on the "floor" (XZ plane)
  gridHelper = new THREE.GridHelper(50, 20, 0x333333, 0x222222);
  gridHelper.position.y = -25;
  scene.add(gridHelper);

  // Create axes with labels
  axesGroup = new THREE.Group();

  const axisLength = 25;
  const axisColors = {
    x: 0xff4444, // red
    y: 0x44ff44, // green
    z: 0x4444ff, // blue
  };

  // Create axis lines
  const createAxis = (start: THREE.Vector3, end: THREE.Vector3, color: number) => {
    const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
    const material = new THREE.LineBasicMaterial({ color, opacity: 0.5, transparent: true });
    return new THREE.Line(geometry, material);
  };

  // X axis (red)
  axesGroup.add(createAxis(
    new THREE.Vector3(-axisLength, -25, 0),
    new THREE.Vector3(axisLength, -25, 0),
    axisColors.x
  ));

  // Y axis (green)
  axesGroup.add(createAxis(
    new THREE.Vector3(0, -25, 0),
    new THREE.Vector3(0, axisLength, 0),
    axisColors.y
  ));

  // Z axis (blue)
  axesGroup.add(createAxis(
    new THREE.Vector3(0, -25, -axisLength),
    new THREE.Vector3(0, -25, axisLength),
    axisColors.z
  ));

  scene.add(axesGroup);
}

// Create selection box overlay
function createSelectionBox() {
  selectionBox = document.createElement('div');
  selectionBox.style.cssText = `
    position: absolute;
    border: 2px solid #3b82f6;
    background: rgba(59, 130, 246, 0.1);
    pointer-events: none;
    display: none;
    z-index: 100;
  `;
  container.appendChild(selectionBox);
}

// Initialize Three.js
function initThree() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0a);

  camera = new THREE.PerspectiveCamera(
    60,
    container.clientWidth / container.clientHeight,
    0.1,
    1000
  );
  camera.position.z = 50;

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  raycaster = new THREE.Raycaster();
  raycaster.params.Points!.threshold = 0.5;
  mouse = new THREE.Vector2();

  // Add grid and axes
  createGridAndAxes();

  // Create selection box overlay
  createSelectionBox();

  window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  renderer.domElement.addEventListener('mousemove', onMouseMove);
  renderer.domElement.addEventListener('click', onMouseClick);
  renderer.domElement.addEventListener('mousedown', onMouseDown);
  renderer.domElement.addEventListener('mouseup', onMouseUp);

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();

  // Handle smooth position transitions
  if (points && data && animationProgress < 1 && previousPositions && projectedPositions) {
    animationProgress = Math.min(1, animationProgress + 0.02); // ~50 frames for full transition
    const easedProgress = easeOutCubic(animationProgress);

    const geometry = points.geometry;
    const positions = geometry.getAttribute('position') as THREE.BufferAttribute;

    for (let i = 0; i < data.documents.length; i++) {
      const px = previousPositions[i * 3] + (projectedPositions[i * 3] - previousPositions[i * 3]) * easedProgress;
      const py = previousPositions[i * 3 + 1] + (projectedPositions[i * 3 + 1] - previousPositions[i * 3 + 1]) * easedProgress;
      const pz = previousPositions[i * 3 + 2] + (projectedPositions[i * 3 + 2] - previousPositions[i * 3 + 2]) * easedProgress;

      positions.setXYZ(i, px, py, pz);
    }

    positions.needsUpdate = true;
  }

  // Raycasting for hover (skip during selection)
  if (points && data && !isSelecting) {
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObject(points);

    if (intersects.length > 0) {
      const newIndex = intersects[0].index!;
      if (newIndex !== hoveredIndex) {
        hoveredIndex = newIndex;
        if (selectedIndex === null && selectedIndices.size === 0) {
          updateInspector(data.documents[hoveredIndex], false);
        }
      }
    }
  }

  renderer.render(scene, camera);
}

function onMouseMove(event: MouseEvent) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  // Update selection box if selecting
  if (isSelecting && selectionStart && selectionBox) {
    const currentX = event.clientX - rect.left;
    const currentY = event.clientY - rect.top;

    const left = Math.min(selectionStart.x, currentX);
    const top = Math.min(selectionStart.y, currentY);
    const width = Math.abs(currentX - selectionStart.x);
    const height = Math.abs(currentY - selectionStart.y);

    selectionBox.style.left = `${left}px`;
    selectionBox.style.top = `${top}px`;
    selectionBox.style.width = `${width}px`;
    selectionBox.style.height = `${height}px`;
    selectionBox.style.display = 'block';
  }
}

function onMouseDown(event: MouseEvent) {
  // Ctrl+click starts selection mode
  if (event.ctrlKey || event.metaKey) {
    event.preventDefault();
    isSelecting = true;
    controls.enabled = false; // Disable orbit controls during selection

    const rect = renderer.domElement.getBoundingClientRect();
    selectionStart = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  }
}

function onMouseUp(event: MouseEvent) {
  if (isSelecting && selectionStart && selectionBox) {
    const rect = renderer.domElement.getBoundingClientRect();
    const endX = event.clientX - rect.left;
    const endY = event.clientY - rect.top;

    // Calculate selection bounds in screen space
    const minX = Math.min(selectionStart.x, endX);
    const maxX = Math.max(selectionStart.x, endX);
    const minY = Math.min(selectionStart.y, endY);
    const maxY = Math.max(selectionStart.y, endY);

    // Only process if selection is larger than 10px
    if (maxX - minX > 10 && maxY - minY > 10) {
      findPointsInSelection(minX, minY, maxX, maxY, rect.width, rect.height);
    }

    // Reset selection state
    isSelecting = false;
    selectionStart = null;
    selectionBox.style.display = 'none';
    controls.enabled = true;
  }
}

// Find all points within the 2D selection box
function findPointsInSelection(minX: number, minY: number, maxX: number, maxY: number, width: number, height: number) {
  if (!points || !data || !projectedPositions) return;

  selectedIndices.clear();

  // Convert screen bounds to normalized device coordinates
  const ndc = {
    minX: (minX / width) * 2 - 1,
    maxX: (maxX / width) * 2 - 1,
    minY: -(maxY / height) * 2 + 1, // Y is inverted
    maxY: -(minY / height) * 2 + 1,
  };

  // Check each point
  for (let i = 0; i < data.documents.length; i++) {
    const x = projectedPositions[i * 3];
    const y = projectedPositions[i * 3 + 1];
    const z = projectedPositions[i * 3 + 2];

    // Project 3D point to screen space
    const vector = new THREE.Vector3(x, y, z);
    vector.project(camera);

    // Check if within selection bounds
    if (vector.x >= ndc.minX && vector.x <= ndc.maxX &&
        vector.y >= ndc.minY && vector.y <= ndc.maxY &&
        vector.z < 1) { // Only visible points (in front of camera)
      selectedIndices.add(i);
    }
  }

  if (selectedIndices.size > 0) {
    showSelectionSummary();
    updatePointColors();
  }
}

// Extract topic from a document (first user message, cleaned up)
function extractTopic(doc: Document): string {
  const userMatch = doc.text.match(/User:\s*(.+?)(?:\n|$)/i);
  if (userMatch) {
    let topic = userMatch[1].trim();
    // Truncate long topics
    if (topic.length > 60) {
      topic = topic.slice(0, 57) + '...';
    }
    return topic;
  }
  // Fallback: first 60 chars
  const fallback = doc.text.slice(0, 60).replace(/\n/g, ' ').trim();
  return fallback + (doc.text.length > 60 ? '...' : '');
}

// Group documents by category or topic
function clusterTopics(docs: Document[]): Array<{ topic: string; count: number; example: string }> {
  // Check if docs have category metadata - use that for grouping
  const hasCategories = docs.some(d => typeof d.metadata?.category === 'string');

  if (hasCategories) {
    // Group by category
    const categoryGroups = new Map<string, number>();
    for (const doc of docs) {
      const category = (doc.metadata?.category as string) || 'uncategorized';
      categoryGroups.set(category, (categoryGroups.get(category) || 0) + 1);
    }

    return Array.from(categoryGroups.entries())
      .map(([category, count]) => ({
        topic: category.charAt(0).toUpperCase() + category.slice(1),
        count,
        example: ''
      }))
      .sort((a, b) => b.count - a.count);
  }

  // Fall back to topic-based clustering for real data
  const topicGroups = new Map<string, { count: number; examples: string[] }>();

  for (const doc of docs) {
    const topic = extractTopic(doc);
    const lowerTopic = topic.toLowerCase();

    // Try to find an existing similar group
    let foundGroup = false;
    for (const [key, group] of topicGroups.entries()) {
      const keyLower = key.toLowerCase();
      // Relaxed similarity: check for common words (3+ chars)
      const topicWords = new Set(lowerTopic.split(/\s+/).filter(w => w.length > 2));
      const keyWords = new Set(keyLower.split(/\s+/).filter(w => w.length > 2));
      const overlap = [...topicWords].filter(w => keyWords.has(w)).length;

      // Match if 1+ significant overlap or 30%+ word overlap
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

  // Convert to array and sort by count
  const clusters = Array.from(topicGroups.entries())
    .map(([topic, { count, examples }]) => ({
      topic: examples[0],
      count,
      example: examples[0]
    }))
    .sort((a, b) => b.count - a.count);

  // Take top clusters, merge rest into "Other"
  const significantClusters = clusters.slice(0, 10);
  const otherCount = clusters.slice(10).reduce((sum, c) => sum + c.count, 0);

  if (otherCount > 0) {
    significantClusters.push({ topic: 'Other topics', count: otherCount, example: '' });
  }

  return significantClusters;
}

// Generate AI summary using OpenRouter
async function generateAISummary(docs: Document[]): Promise<string> {
  const apiKey = getApiKey();
  if (!apiKey) {
    throw new Error('NO_API_KEY');
  }

  // Prepare the chunks text (limit to avoid token limits)
  const maxChunks = 20;
  const chunksToSummarize = docs.slice(0, maxChunks);
  const chunksText = chunksToSummarize.map((doc, i) => `[${i + 1}] ${doc.text}`).join('\n\n---\n\n');

  const prompt = `Here are ${chunksToSummarize.length} conversation chunks from a Claude memory database. Please provide a concise summary (2-3 sentences) of the main themes and topics discussed across these conversations:\n\n${chunksText}`;

  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: 'google/gemini-2.0-flash-001',
      max_tokens: 300,
      messages: [{ role: 'user', content: prompt }],
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error?.message || `API error: ${response.status}`);
  }

  const result = await response.json();
  return result.choices?.[0]?.message?.content || 'No summary generated';
}

// Show summary modal for selected chunks
function showSelectionSummary() {
  if (!data || selectedIndices.size === 0) return;

  // Get selected documents
  const selectedDocs = Array.from(selectedIndices).map(i => data!.documents[i]);

  // Cluster topics
  const clusters = clusterTopics(selectedDocs);

  // Create modal
  const existingModal = document.getElementById('selection-modal');
  if (existingModal) existingModal.remove();

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

  // Build clusters HTML
  const clustersHtml = clusters.map(c => `
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 12px; background: #111; border-radius: 6px; margin-bottom: 6px;">
      <span style="color: #ccc; font-size: 13px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding-right: 12px;">${escapeHtml(c.topic)}</span>
      <span style="color: #3b82f6; font-size: 13px; font-weight: 600; white-space: nowrap;">${c.count}</span>
    </div>
  `).join('');

  const hasApiKey = getApiKey() !== null;

  modal.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
      <h3 style="color: #fff; margin: 0; font-size: 18px; flex: 1;">Selection Summary</h3>
      <button id="close-selection-modal" style="background: none; border: none; color: #666; font-size: 24px; cursor: pointer; padding: 0; margin-left: 16px; line-height: 1; width: 24px; height: 24px; flex-shrink: 0;">&times;</button>
    </div>
    <div style="color: #3b82f6; font-size: 15px; margin-bottom: 20px;">
      <strong>${selectedIndices.size}</strong> conversations selected
    </div>
    <div style="margin-bottom: 20px;">
      <div style="color: #888; font-size: 12px; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em;">AI Summary</div>
      <div id="ai-summary-result" style="font-size: 13px; color: #888;">
        ${hasApiKey ? '<div class="spinner" style="margin: 8px 0;"></div> Generating summary...' : '<span style="color: #666;">Set OpenRouter API key in ⚙️ Settings to enable AI summaries</span>'}
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
  document.getElementById('close-selection-modal')?.addEventListener('click', closeSelectionModal);
  document.getElementById('clear-selection')?.addEventListener('click', () => {
    selectedIndices.clear();
    updatePointColors();
    closeSelectionModal();
  });
  backdrop.addEventListener('click', closeSelectionModal);

  // Auto-generate AI summary if API key is set
  if (hasApiKey) {
    generateAISummary(selectedDocs)
      .then(summary => {
        const resultDiv = document.getElementById('ai-summary-result');
        if (resultDiv) {
          resultDiv.innerHTML = `<div style="color: #e0e0e0; line-height: 1.5; padding: 12px; background: #111; border-radius: 6px;">${escapeHtml(summary)}</div>`;
        }
      })
      .catch(err => {
        const resultDiv = document.getElementById('ai-summary-result');
        if (resultDiv) {
          const errorMessage = err instanceof Error ? err.message : 'Unknown error';
          if (errorMessage === 'NO_API_KEY') {
            resultDiv.innerHTML = '<span style="color: #666;">Set OpenRouter API key in ⚙️ Settings to enable AI summaries</span>';
          } else {
            resultDiv.innerHTML = `<span style="color: #ef4444;">Error: ${escapeHtml(errorMessage)}</span>`;
          }
        }
      });
  }
}

function closeSelectionModal() {
  document.getElementById('selection-modal')?.remove();
  document.getElementById('selection-backdrop')?.remove();
  // Clear selection when modal closes
  selectedIndices.clear();
  updatePointColors();
}

function onMouseClick() {
  if (!points || !data) return;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObject(points);

  if (intersects.length > 0) {
    const clickedIndex = intersects[0].index!;

    if (selectedIndex === clickedIndex) {
      selectedIndex = null;
      updateInspector(null, false);
    } else {
      selectedIndex = clickedIndex;
      updateInspector(data.documents[selectedIndex], true);
    }

    updatePointColors();
  } else {
    selectedIndex = null;
    updateInspector(null, false);
    updatePointColors();
  }
}

// Embedding configuration
const LOCAL_EMBED_URL = 'http://localhost:5001';
const OLLAMA_URL = 'http://localhost:11434';
const OLLAMA_MODEL_STORAGE = 'ollama-embedding-model';
const DEFAULT_OLLAMA_MODEL = 'nomic-embed-text';

function getOllamaModel(): string {
  return localStorage.getItem(OLLAMA_MODEL_STORAGE) || DEFAULT_OLLAMA_MODEL;
}

// Check if local embedding server is available (uses same model as claude-memory)
let localEmbedAvailable: boolean | null = null;
async function checkLocalEmbed(): Promise<boolean> {
  if (localEmbedAvailable !== null) return localEmbedAvailable;

  try {
    const response = await fetch(`${LOCAL_EMBED_URL}`, {
      method: 'OPTIONS',
      signal: AbortSignal.timeout(1000),
    });
    localEmbedAvailable = response.ok;
    return localEmbedAvailable;
  } catch {
    localEmbedAvailable = false;
    return false;
  }
}

// Get embedding from local Python server (same model as claude-memory)
async function getLocalEmbedding(text: string): Promise<number[] | null> {
  try {
    const response = await fetch(`${LOCAL_EMBED_URL}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      console.error('Local embed error:', response.status);
      return null;
    }

    const result = await response.json();
    return result.embedding;
  } catch (err) {
    console.error('Local embed failed:', err);
    return null;
  }
}

// Check if Ollama is available
let ollamaAvailable: boolean | null = null;
async function checkOllama(): Promise<boolean> {
  if (ollamaAvailable !== null) return ollamaAvailable;

  try {
    const response = await fetch(`${OLLAMA_URL}/api/tags`, {
      method: 'GET',
      signal: AbortSignal.timeout(2000),
    });
    ollamaAvailable = response.ok;
    return ollamaAvailable;
  } catch {
    ollamaAvailable = false;
    return false;
  }
}

// Get embedding from Ollama (only for nomic-embed-text data)
async function getOllamaEmbedding(text: string): Promise<number[] | null> {
  try {
    const response = await fetch(`${OLLAMA_URL}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: getOllamaModel(),
        prompt: text,
      }),
    });

    if (!response.ok) {
      console.error('Ollama embedding error:', response.status);
      return null;
    }

    const result = await response.json();
    return result.embedding;
  } catch (err) {
    console.error('Ollama embedding failed:', err);
    return null;
  }
}

// Load browser embedding model (fallback)
async function loadBrowserEmbedder(): Promise<FeatureExtractionPipeline | null> {
  // Return existing embedder
  if (embedder) return embedder;

  // Wait for existing load if in progress
  if (embedderPromise) return embedderPromise;

  // Start loading - all-mpnet-base-v2 is ~420MB
  updateSearchStatus('Loading embedding model in browser (first time only)...', true);

  embedderPromise = (async () => {
    try {
      // Use same model as claude-memory: all-mpnet-base-v2 (768 dims)
      embedder = await pipeline('feature-extraction', 'Xenova/all-mpnet-base-v2', {
        dtype: 'fp32',
      });
      updateSearchStatus('Semantic search ready (browser)');
      return embedder;
    } catch (err) {
      console.error('Failed to load browser embedder:', err);
      updateSearchStatus('Semantic search unavailable - using text search');
      embedderPromise = null;
      return null;
    }
  })();

  return embedderPromise;
}

// Get embedding - uses model that matches document embeddings
// Priority: Local Python server > Ollama (for nomic data) > Browser fallback
async function getEmbedding(text: string): Promise<number[] | null> {
  const docModel = data?.metadata?.embedding_model || '';
  const isNomicData = docModel.includes('nomic');

  // For all-mpnet-base-v2 data (claude-memory): try local Python server first
  if (!isNomicData && await checkLocalEmbed()) {
    const embedding = await getLocalEmbedding(text);
    if (embedding) {
      console.log('Using local embedding server (all-mpnet-base-v2)');
      return embedding;
    }
  }

  // For nomic data: try Ollama
  if (isNomicData && await checkOllama()) {
    const embedding = await getOllamaEmbedding(text);
    if (embedding) {
      console.log('Using Ollama (nomic-embed-text)');
      return embedding;
    }
  }

  // Fallback to browser embedding (downloads ~420MB model)
  console.log('Falling back to browser embedding...');
  const emb = await loadBrowserEmbedder();
  if (emb) {
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
  }

  return null;
}

// Initialize embedder - determine which model to use based on data
async function initEmbedder() {
  const docModel = data?.metadata?.embedding_model || '';
  const isNomicData = docModel.includes('nomic');

  // Check available embedding sources
  const hasLocalEmbed = await checkLocalEmbed();
  const hasOllama = await checkOllama();

  if (!isNomicData && hasLocalEmbed) {
    updateSearchStatus('Semantic search ready (local server)');
  } else if (isNomicData && hasOllama) {
    updateSearchStatus(`Semantic search ready (Ollama: ${getOllamaModel()})`);
  } else {
    // Fallback to browser model
    updateSearchStatus('Loading embedding model (~420MB, first time only)...', true);
    loadBrowserEmbedder();
  }
}

// Cosine similarity
function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Load data from JSON
async function loadData(source: File | string) {
  loadingEl.textContent = 'Loading data...';

  try {
    let json: DataSet;

    if (typeof source === 'string') {
      const response = await fetch(source);
      json = await response.json();
    } else {
      const text = await source.text();
      json = JSON.parse(text);
    }

    data = json;
    queryInput.disabled = false;

    statsEl.innerHTML = `
      <strong>${data.metadata.name}</strong><br>
      ${data.documents.length} documents<br>
      ${data.metadata.embedding_dim}D embeddings<br>
      Model: ${data.metadata.embedding_model}
    `;

    updateLegend();

    // Initialize embedder (checks Ollama, falls back to browser)
    initEmbedder();

    // Auto-compute projection
    await computeProjection();
  } catch (err) {
    console.error('Error loading data:', err);
    loadingEl.textContent = 'Error loading data';
  }
}

// Compute dimensionality reduction
async function computeProjection() {
  if (!data) return;

  const algorithm = algorithmSelect.value as 'umap' | 'tsne' | 'pca';

  // Store previous positions for smooth transition
  const hadPreviousPoints = points !== null && projectedPositions !== null;
  if (hadPreviousPoints) {
    previousPositions = new Float32Array(projectedPositions!);
  }

  // Check for pre-computed projections
  const precomputed = data.projections?.[algorithm];

  if (precomputed) {
    loadingEl.textContent = `Loading ${algorithm.toUpperCase()}...`;
    await new Promise((resolve) => setTimeout(resolve, 10)); // Brief delay for UI update

    const positions = new Float32Array(data.documents.length * 3);
    for (let i = 0; i < precomputed.length; i++) {
      const [x, y, z] = precomputed[i];
      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;
    }

    projectedPositions = positions;

    if (hadPreviousPoints) {
      // Trigger smooth transition animation
      animationProgress = 0;
    } else {
      createPointCloud();
    }

    loadingEl.textContent = '';
    return;
  }

  // Fall back to browser computation (no t-SNE - too slow)
  if (algorithm === 'tsne') {
    loadingEl.textContent = 't-SNE requires pre-computed projections. Run: bun run export-data';
    return;
  }

  loadingEl.textContent = `Computing ${algorithm.toUpperCase()}...`;
  await new Promise((resolve) => setTimeout(resolve, 50));

  try {
    const embeddings = data.documents.map((d) => d.embedding);
    const matrix = druid.Matrix.from(embeddings);

    let result: druid.Matrix;

    switch (algorithm) {
      case 'umap':
        const umap = new druid.UMAP(matrix, { d: 3, n_neighbors: 15, min_dist: 0.1 });
        result = umap.transform();
        break;
      case 'pca':
      default:
        const pca = new druid.PCA(matrix, { d: 3 });
        result = pca.transform();
        break;
    }

    const positions = new Float32Array(data.documents.length * 3);
    const resultArray = result.to2dArray;

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    for (let i = 0; i < resultArray.length; i++) {
      const [x, y, z] = resultArray[i];
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
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

    projectedPositions = positions;

    if (hadPreviousPoints) {
      // Trigger smooth transition animation
      animationProgress = 0;
    } else {
      createPointCloud();
    }

    loadingEl.textContent = '';
  } catch (err) {
    console.error('Projection error:', err);
    loadingEl.textContent = 'Error computing projection';
  }
}

// Check if dataset uses category-based coloring
function hasCategoryData(): boolean {
  if (!data || data.documents.length === 0) return false;
  // Check if first document has category metadata
  return typeof data.documents[0].metadata?.category === 'string';
}

// Create Three.js point cloud with circular points
function createPointCloud() {
  if (!data || !projectedPositions) return;

  if (points) {
    scene.remove(points);
    points.geometry.dispose();
    (points.material as THREE.PointsMaterial).dispose();
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(projectedPositions, 3));

  const colors = new Float32Array(data.documents.length * 3);
  const useCategories = hasCategoryData();

  if (useCategories) {
    // Category-based coloring
    timeRange = null;
    for (let i = 0; i < data.documents.length; i++) {
      const doc = data.documents[i];
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

    for (let i = 0; i < data.documents.length; i++) {
      const doc = data.documents[i];
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
      timeRange = { min: new Date(minTime), max: new Date(maxTime) };
    } else {
      timeRange = null;
    }

    // Assign colors based on time (or default gray if no timestamp)
    const timeRangeMs = maxTime - minTime || 1;
    const defaultColor = new THREE.Color(0x666666);

    for (let i = 0; i < data.documents.length; i++) {
      const time = timestamps[i];
      let color: THREE.Color;

      if (time !== null && timeRange) {
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

  updateLegend();

  baseColors = new Float32Array(colors);
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: 0.6,
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    alphaTest: 0.5,
    map: createCircleTexture(),
  });

  points = new THREE.Points(geometry, material);
  scene.add(points);

  selectedIndex = null;
  matchedIndices.clear();

  camera.position.set(0, 0, 50);
  controls.reset();
}

// Update point colors based on selection and search
function updatePointColors() {
  if (!points || !data || !baseColors) return;

  const geometry = points.geometry;
  const colors = geometry.getAttribute('color') as THREE.BufferAttribute;
  const hasQuery = matchedIndices.size > 0;
  const hasBoxSelection = selectedIndices.size > 0;

  for (let i = 0; i < data.documents.length; i++) {
    let r = baseColors[i * 3];
    let g = baseColors[i * 3 + 1];
    let b = baseColors[i * 3 + 2];

    if (i === selectedIndex) {
      r = SELECTED_COLOR.r;
      g = SELECTED_COLOR.g;
      b = SELECTED_COLOR.b;
    } else if (hasBoxSelection) {
      if (selectedIndices.has(i)) {
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
      if (matchedIndices.has(i)) {
        // Brightness based on similarity score
        const similarity = matchedIndices.get(i)!;
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

// Semantic search using embeddings
async function semanticSearch(query: string) {
  if (!data) return;

  matchedIndices.clear();

  if (query.trim() === '') {
    updatePointColors();
    updateSearchResults(0, 0, []);
    updateSearchStatus('');
    return;
  }

  // Check if data has real embeddings (not 3D demo coordinates)
  const embDim = data.documents[0]?.embedding?.length || 0;
  if (embDim < 100) {
    updateSearchStatus('Demo data - search disabled');
    const resultsEl = document.getElementById('search-results');
    if (resultsEl) {
      resultsEl.innerHTML = '<span style="color: #666; font-size: 11px;">Demo dataset uses placeholder embeddings. Load real data for semantic search.</span>';
    }
    return;
  }

  // Show loading status
  const resultsEl = document.getElementById('search-results');
  if (resultsEl) {
    resultsEl.innerHTML = '<span style="color: #666;">Searching...</span>';
  }

  updateSearchStatus('Searching...', true);

  // Get embedding for query
  const queryEmbedding = await getEmbedding(query);

  if (queryEmbedding) {
    console.log('Query embedding dimensions:', queryEmbedding.length);
    console.log('Document embedding dimensions:', data.documents[0].embedding.length);
    console.log('Query embedding sample:', queryEmbedding.slice(0, 5));
    console.log('Doc embedding sample:', data.documents[0].embedding.slice(0, 5));

    // Calculate similarities
    const similarities: Array<{ index: number; score: number }> = [];

    for (let i = 0; i < data.documents.length; i++) {
      const score = cosineSimilarity(queryEmbedding, data.documents[i].embedding);
      similarities.push({ index: i, score });
    }

    // Sort by similarity
    similarities.sort((a, b) => b.score - a.score);

    // Debug: log top scores
    console.log('Top 5 similarities:', similarities.slice(0, 5).map(s => ({
      score: s.score.toFixed(4),
      preview: data!.documents[s.index].text.slice(0, 50)
    })));

    const bestScore = similarities[0]?.score || 0;
    const minScore = 0.3; // Absolute minimum
    const relativeThreshold = bestScore * 0.7; // Within 70% of best match
    const threshold = Math.max(minScore, relativeThreshold);

    // Take top matches that meet threshold, max 20 for visualization clarity
    const topK = 20;
    const matches = similarities
      .filter((s) => s.score >= threshold)
      .slice(0, topK);

    for (const match of matches) {
      matchedIndices.set(match.index, match.score);
    }

    updatePointColors();
    updateSearchResults(matches.length, data.documents.length, matches.slice(0, 8), bestScore);

    if (matches.length > 0) {
      updateSearchStatus(`Top ${matches.length} matches (${(bestScore * 100).toFixed(0)}% best)`);
    } else {
      updateSearchStatus('No strong matches found');
    }
  } else {
    // Fall back to text search if embedding failed
    fallbackTextSearch(query);
  }
}

// Fallback to text search
function fallbackTextSearch(query: string) {
  if (!data) return;

  const lowerQuery = query.toLowerCase();
  const terms = lowerQuery.split(/\s+/).filter((t) => t.length > 0);

  for (let i = 0; i < data.documents.length; i++) {
    const doc = data.documents[i];
    const text = doc.text.toLowerCase();

    const matches = terms.every((term) => text.includes(term));
    if (matches) {
      matchedIndices.set(i, 1.0);
    }
  }

  updatePointColors();
  updateSearchResults(matchedIndices.size, data.documents.length, []);
}

// CSS spinner keyframes (add once)
const spinnerStyle = document.createElement('style');
spinnerStyle.textContent = `
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  .spinner {
    width: 14px;
    height: 14px;
    border: 2px solid #333;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
`;
document.head.appendChild(spinnerStyle);

function updateSearchStatus(message: string, showSpinner: boolean = false) {
  const statusEl = document.getElementById('search-status');
  if (statusEl) {
    if (showSpinner) {
      statusEl.innerHTML = `<div class="spinner"></div><span>${message}</span>`;
    } else if (message) {
      statusEl.innerHTML = `<span>${message}</span>`;
    } else {
      statusEl.innerHTML = '';
    }
  }
}

function updateSearchResults(matched: number, total: number, topMatches: Array<{ index: number; score: number }>, bestScore?: number) {
  const resultsEl = document.getElementById('search-results');
  if (resultsEl) {
    if (matched > 0 && topMatches.length > 0 && data) {
      let html = '<div style="font-size: 11px;">';

      for (const match of topMatches) {
        const doc = data.documents[match.index];
        // Show first line of user message
        const userMatch = doc.text.match(/User:\s*(.+?)(?:\n|$)/);
        const preview = userMatch
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
      resultsEl.innerHTML = '<span style="color: #666; font-size: 11px;">No strong matches found. Try different words.</span>';
    } else {
      resultsEl.innerHTML = '';
    }
  }
}

// Update inspector panel
function updateInspector(doc: Document | null, isSelected: boolean) {
  if (!doc) {
    inspectorEl.innerHTML = '<div class="inspector-empty">Hover over a point to see details</div>';
    return;
  }

  const similarity = matchedIndices.get(data?.documents.indexOf(doc) ?? -1);
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

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function updateLegend() {
  // Check for category-based coloring
  if (hasCategoryData()) {
    // Collect unique categories from data
    const categories = new Set<string>();
    for (const doc of data!.documents) {
      const cat = doc.metadata?.category as string;
      if (cat) categories.add(cat);
    }

    const categoryItems = Array.from(categories).map(cat => {
      const color = CATEGORY_COLORS[cat] || new THREE.Color(0x666666);
      return `
        <div class="legend-item">
          <div class="legend-dot" style="background: #${color.getHexString()};"></div>
          <span>${cat}</span>
        </div>
      `;
    }).join('');

    legendEl.innerHTML = `
      <div style="font-size: 11px; color: #888; margin-bottom: 8px;">
        <strong>Categories</strong>
      </div>
      ${categoryItems}
    `;
    return;
  }

  if (!timeRange) {
    legendEl.innerHTML = '<div style="color: #666; font-size: 11px;">No timestamp data</div>';
    return;
  }

  const formatDate = (d: Date) => d.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });

  // Create gradient bar
  const gradientColors = TIME_GRADIENT.map(c => '#' + c.getHexString()).join(', ');

  legendEl.innerHTML = `
    <div style="font-size: 11px; color: #888; margin-bottom: 8px;">
      <strong>Time Range</strong>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
      <span style="font-size: 10px; color: #666;">${formatDate(timeRange.min)}</span>
      <div style="flex: 1; height: 8px; border-radius: 4px; background: linear-gradient(to right, ${gradientColors});"></div>
      <span style="font-size: 10px; color: #666;">${formatDate(timeRange.max)}</span>
    </div>
    <div style="font-size: 10px; color: #555; margin-top: 4px; text-align: center;">
      older → newer
    </div>
  `;
}

// Update algorithm explanation panel
function updateAlgorithmInfo() {
  const infoEl = document.getElementById('algorithm-info');
  if (infoEl) {
    const algorithm = algorithmSelect.value;
    infoEl.innerHTML = ALGORITHM_INFO[algorithm] || '';
  }
}

// Event listeners
const dropZone = document.getElementById('drop-zone');

// File input change
fileInput.addEventListener('change', (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file) loadData(file);
});

// Click to browse
dropZone?.addEventListener('click', () => {
  fileInput.click();
});

// Drag and drop
dropZone?.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.style.borderColor = '#3b82f6';
  dropZone.style.background = 'rgba(59, 130, 246, 0.1)';
});

dropZone?.addEventListener('dragleave', (e) => {
  e.preventDefault();
  dropZone.style.borderColor = '#333';
  dropZone.style.background = 'transparent';
});

dropZone?.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.style.borderColor = '#333';
  dropZone.style.background = 'transparent';

  const file = e.dataTransfer?.files?.[0];
  if (file && file.name.endsWith('.json')) {
    loadData(file);
  }
});

sampleSelect.addEventListener('change', (e) => {
  const value = (e.target as HTMLSelectElement).value;
  if (value === 'claude-memory') {
    loadData('/data/claude-memory.json');
    // Remove "Select sample..." option after selection
    const defaultOption = sampleSelect.querySelector('option[value=""]');
    if (defaultOption) defaultOption.remove();
  } else if (value === 'demo') {
    loadData('/data/demo.json');
    // Remove "Select sample..." option after selection
    const defaultOption = sampleSelect.querySelector('option[value=""]');
    if (defaultOption) defaultOption.remove();
  }
});

// Algorithm dropdown - update explanation and auto-compute
algorithmSelect.addEventListener('change', () => {
  updateAlgorithmInfo();
  // Auto-compute if data is loaded
  if (data) {
    computeProjection();
  }
});

// Query input with debounce
let searchTimeout: number | null = null;
queryInput.addEventListener('input', (e) => {
  const query = (e.target as HTMLInputElement).value;

  if (searchTimeout) {
    clearTimeout(searchTimeout);
  }

  searchTimeout = window.setTimeout(() => {
    semanticSearch(query);
  }, 300);
});

// Add search UI elements
const queryContainer = document.querySelector('.query-input');
if (queryContainer) {
  // Status with spinner (directly below input)
  const statusEl = document.createElement('div');
  statusEl.id = 'search-status';
  statusEl.style.cssText = 'font-size: 12px; color: #888; margin-top: 8px; display: flex; align-items: center; gap: 8px;';
  queryContainer.appendChild(statusEl);

  // Results (below status)
  const resultsEl = document.createElement('div');
  resultsEl.id = 'search-results';
  resultsEl.style.cssText = 'font-size: 12px; margin-top: 8px; max-height: 200px; overflow-y: auto;';
  queryContainer.appendChild(resultsEl);

  // Add simplified search explanation
  const searchInfoEl = document.createElement('div');
  searchInfoEl.id = 'search-info';
  searchInfoEl.className = 'info-panel';
  searchInfoEl.style.marginTop = '16px';
  searchInfoEl.innerHTML = SEARCH_INFO;
  queryContainer.appendChild(searchInfoEl);
}

// Global function for clicking search results
(window as unknown as { selectDocument: (index: number) => void }).selectDocument = (index: number) => {
  if (data && index >= 0 && index < data.documents.length) {
    selectedIndex = index;
    updateInspector(data.documents[index], true);
    updatePointColors();
  }
};

// Settings management
const OPENROUTER_API_KEY_STORAGE = 'openrouter-api-key';

function getApiKey(): string | null {
  return localStorage.getItem(OPENROUTER_API_KEY_STORAGE);
}

function setApiKey(key: string) {
  localStorage.setItem(OPENROUTER_API_KEY_STORAGE, key);
}

function showSettingsModal() {
  const existingModal = document.getElementById('settings-modal');
  if (existingModal) existingModal.remove();

  const currentKey = getApiKey() || '';
  const hasKey = currentKey.length > 0;
  const currentOllamaModel = getOllamaModel();

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

    <!-- Ollama Settings -->
    <div style="margin-bottom: 24px;">
      <div style="color: #888; font-size: 12px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em;">Embeddings</div>
      <div id="ollama-status" style="font-size: 12px; color: #666; margin-bottom: 12px; padding: 10px; background: #111; border-radius: 6px;">Checking Ollama...</div>
      <div style="margin-bottom: 12px;">
        <label style="display: block; color: #666; font-size: 11px; margin-bottom: 4px;">Ollama Model</label>
        <input type="text" id="settings-ollama-model" placeholder="nomic-embed-text" value="${escapeHtml(currentOllamaModel)}" style="width: 100%; padding: 10px 12px; background: #111; border: 1px solid #333; border-radius: 6px; color: #fff; font-size: 13px; box-sizing: border-box;">
      </div>
      <div style="font-size: 11px; color: #555;">
        Ollama auto-starts with dev server. Falls back to browser if unavailable.
      </div>
    </div>

    <!-- OpenRouter Settings -->
    <div style="margin-bottom: 20px; padding-top: 16px; border-top: 1px solid #333;">
      <div style="color: #888; font-size: 12px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em;">AI Summaries (OpenRouter)</div>
      <div style="display: flex; gap: 8px; margin-bottom: 8px;">
        <input type="password" id="settings-api-key" placeholder="sk-or-..." value="${escapeHtml(currentKey)}" style="flex: 1; padding: 10px 12px; background: #111; border: 1px solid #333; border-radius: 6px; color: #fff; font-size: 13px;">
      </div>
      <div style="font-size: 11px; color: #666;">
        ${hasKey ? '✓ API key is set' : 'Required for AI summaries when selecting points'}
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

    ollamaAvailable = null; // Reset cache to recheck
    const available = await checkOllama();
    if (available) {
      statusEl.innerHTML = '<span style="color: #10b981;">✓ Ollama is running</span>';
    } else {
      statusEl.innerHTML = '<span style="color: #f59e0b;">⚠ Ollama not detected</span>';
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
    const apiKeyInput = document.getElementById('settings-api-key') as HTMLInputElement;
    const ollamaModelInput = document.getElementById('settings-ollama-model') as HTMLInputElement;

    const apiKey = apiKeyInput?.value?.trim() || '';
    const ollamaModel = ollamaModelInput?.value?.trim() || DEFAULT_OLLAMA_MODEL;

    // Save all settings
    if (apiKey) {
      setApiKey(apiKey);
    }
    localStorage.setItem(OLLAMA_MODEL_STORAGE, ollamaModel);

    // Reset Ollama cache so next search uses new settings
    ollamaAvailable = null;

    closeSettings();

    // Re-init embedder with new settings
    initEmbedder();
  });
}

// Settings button listener
document.getElementById('settings-btn')?.addEventListener('click', showSettingsModal);

// Initialize
initThree();

// Show default algorithm explanation
updateAlgorithmInfo();
