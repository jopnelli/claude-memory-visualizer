// Local Python embedding server (uses same model as claude-memory)

import { embeddingState } from '../state';

export const LOCAL_EMBED_URL = 'http://localhost:5001';

// Check if local embedding server is available
export async function checkLocalEmbed(): Promise<boolean> {
  if (embeddingState.localEmbedAvailable !== null) {
    return embeddingState.localEmbedAvailable;
  }

  try {
    const response = await fetch(LOCAL_EMBED_URL, {
      method: 'OPTIONS',
      signal: AbortSignal.timeout(1000),
    });
    embeddingState.localEmbedAvailable = response.ok;
    return embeddingState.localEmbedAvailable;
  } catch {
    embeddingState.localEmbedAvailable = false;
    return false;
  }
}

// Get embedding from local Python server
export async function getLocalEmbedding(text: string): Promise<number[] | null> {
  try {
    const response = await fetch(LOCAL_EMBED_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      console.error('Local embed error:', response.status);
      return null;
    }

    const result = (await response.json()) as { embedding: number[] };
    return result.embedding;
  } catch (err) {
    console.error('Local embed failed:', err);
    return null;
  }
}
