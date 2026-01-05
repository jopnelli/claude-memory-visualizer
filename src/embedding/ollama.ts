// Ollama embedding and summary generation

import { embeddingState } from '../state';

export const OLLAMA_URL = 'http://localhost:11434';

// Storage keys
const OLLAMA_MODEL_STORAGE = 'ollama-embedding-model';
const OLLAMA_SUMMARY_MODEL_STORAGE = 'ollama-summary-model';

// Defaults
const DEFAULT_OLLAMA_MODEL = 'nomic-embed-text';
const DEFAULT_SUMMARY_MODEL = 'qwen2.5:1.5b';

export function getOllamaModel(): string {
  return localStorage.getItem(OLLAMA_MODEL_STORAGE) || DEFAULT_OLLAMA_MODEL;
}

export function getSummaryModel(): string {
  return localStorage.getItem(OLLAMA_SUMMARY_MODEL_STORAGE) || DEFAULT_SUMMARY_MODEL;
}

export function setSummaryModel(model: string): void {
  localStorage.setItem(OLLAMA_SUMMARY_MODEL_STORAGE, model);
}

// Check if Ollama is available
export async function checkOllama(): Promise<boolean> {
  if (embeddingState.ollamaAvailable !== null) {
    return embeddingState.ollamaAvailable;
  }

  try {
    const response = await fetch(`${OLLAMA_URL}/api/tags`, {
      method: 'GET',
      signal: AbortSignal.timeout(2000),
    });
    embeddingState.ollamaAvailable = response.ok;
    return embeddingState.ollamaAvailable;
  } catch {
    embeddingState.ollamaAvailable = false;
    return false;
  }
}

// Reset Ollama availability cache (for re-checking)
export function resetOllamaCache(): void {
  embeddingState.ollamaAvailable = null;
}

// Get embedding from Ollama
export async function getOllamaEmbedding(text: string): Promise<number[] | null> {
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

    const result = (await response.json()) as { embedding: number[] };
    return result.embedding;
  } catch (err) {
    console.error('Ollama embedding failed:', err);
    return null;
  }
}

// Generate text using Ollama (for summaries)
export async function generateWithOllama(prompt: string): Promise<string> {
  const response = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: getSummaryModel(),
      prompt,
      stream: false,
      options: {
        num_predict: 200,
      },
    }),
  });

  if (!response.ok) {
    throw new Error(`Ollama error: ${response.status}`);
  }

  const result = (await response.json()) as { response?: string };
  return result.response || 'No summary generated';
}
