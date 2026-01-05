# Claude Memory Visualizer

3D visualization of embedding spaces from [claude-memory](https://github.com/jopnelli/claude-memory) - explore your Claude Code conversations in semantic space.

## Quick Start

```bash
# Install dependencies
bun install

# Export claude-memory data with pre-computed projections
python scripts/export-chromadb.py

# Start dev server
bun run dev
```

Open http://localhost:5173

## Features

- **3D Point Cloud** - Visualize thousands of conversation embeddings in 3D space
- **Pre-computed Projections** - UMAP, t-SNE, PCA computed server-side for instant switching
- **Semantic Search** - Find conversations by meaning using in-browser embeddings (Transformers.js)
- **Time-based Coloring** - Older conversations in blue, newer in red (gradient legend)
- **Educational Explanations** - Learn how each algorithm works with simplified explanations
- **Inspector Panel** - Hover/click to see full document text and metadata

## Export Script

The Python export script (`scripts/export-chromadb.py`) reads from ChromaDB and pre-computes all three projections:

```bash
# Basic usage (reads from ~/.claude-memory/chroma)
python scripts/export-chromadb.py

# Custom options
python scripts/export-chromadb.py \
  --chroma-path /path/to/chroma \
  --output public/data/my-data.json \
  --limit 1000
```

**Requirements:**
```bash
pip install chromadb umap-learn scikit-learn numpy
```

**Output format:**
```json
{
  "metadata": {
    "name": "Claude Memory",
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "embedding_dim": 768,
    "count": 3486
  },
  "documents": [...],
  "projections": {
    "umap": [[x, y, z], ...],
    "tsne": [[x, y, z], ...],
    "pca": [[x, y, z], ...]
  }
}
```

## Data Sources

1. **Claude Memory** - Select "Claude Memory (real)" from the dropdown to load pre-exported data
2. **Custom JSON** - Upload any JSON file with `documents` array containing `id`, `text`, `embedding`, and optional `metadata`

## Tech Stack

- **Three.js** - 3D rendering with point clouds and OrbitControls
- **DRUIDJS** - Browser-side dimensionality reduction (fallback if no pre-computed projections)
- **Transformers.js** - In-browser semantic search with all-mpnet-base-v2 (768D, ~420MB cached)
- **Vite** - Dev server and bundling
- **TypeScript** - Type-safe codebase

## Algorithms

| Algorithm | Speed | Quality | Best For |
|-----------|-------|---------|----------|
| **UMAP** | Fast | High | Default choice, good balance |
| **t-SNE** | Slow | High | Tight, well-separated clusters |
| **PCA** | Instant | Medium | Quick overview, may have overlap |

## Known Limitations

- **Turn-level chunking**: claude-memory chunks at individual turns (1 user + 1 assistant message), not full conversations. Search may miss context that spans multiple turns.
- **First search delay**: Downloads ~420MB embedding model on first search (cached after)
- **Best for <10k docs**: Larger datasets may be slow to render

## Development

```bash
# Start dev server
bun run dev

# Re-export data after claude-memory changes
python scripts/export-chromadb.py
```

## License

MIT
