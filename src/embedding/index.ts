// Embedding module - unified interface for all embedding providers

import { state } from '../state';
import { checkLocalEmbed, getLocalEmbedding } from './local';
import { checkOllama, getOllamaEmbedding, getOllamaModel } from './ollama';
import { loadBrowserEmbedder, getBrowserEmbedding } from './browser';

export * from './local';
export * from './ollama';
export * from './browser';

type StatusCallback = (message: string, showSpinner?: boolean) => void;

// Get embedding - uses model that matches document embeddings
// Priority: Local Python server > Ollama (for nomic data) > Browser fallback
export async function getEmbedding(
  text: string,
  onStatus?: StatusCallback
): Promise<number[] | null> {
  const docModel = state.data?.metadata?.embedding_model || '';
  const isNomicData = docModel.includes('nomic');

  // For all-mpnet-base-v2 data (claude-memory): try local Python server first
  if (!isNomicData && (await checkLocalEmbed())) {
    const embedding = await getLocalEmbedding(text);
    if (embedding) {
      console.log('Using local embedding server (all-mpnet-base-v2)');
      return embedding;
    }
  }

  // For nomic data: try Ollama
  if (isNomicData && (await checkOllama())) {
    const embedding = await getOllamaEmbedding(text);
    if (embedding) {
      console.log('Using Ollama (nomic-embed-text)');
      return embedding;
    }
  }

  // Fallback to browser embedding (downloads ~420MB model)
  console.log('Falling back to browser embedding...');
  return getBrowserEmbedding(text, onStatus);
}

// Initialize embedder - determine which model to use based on data
export async function initEmbedder(onStatus?: StatusCallback): Promise<void> {
  const docModel = state.data?.metadata?.embedding_model || '';
  const isNomicData = docModel.includes('nomic');

  // Check available embedding sources
  const hasLocalEmbed = await checkLocalEmbed();
  const hasOllama = await checkOllama();

  if (!isNomicData && hasLocalEmbed) {
    onStatus?.('Semantic search ready (local server)');
  } else if (isNomicData && hasOllama) {
    onStatus?.(`Semantic search ready (Ollama: ${getOllamaModel()})`);
  } else {
    // Fallback to browser model
    onStatus?.('Loading embedding model (~420MB, first time only)...', true);
    loadBrowserEmbedder(onStatus);
  }
}

// Cosine similarity between two vectors
export function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const aVal = a[i] ?? 0;
    const bVal = b[i] ?? 0;
    dotProduct += aVal * bVal;
    normA += aVal * aVal;
    normB += bVal * bVal;
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
