# Claude Memory Visualizer - Project Context

3D visualization tool for exploring conversation embeddings from claude-memory.

## Quick Commands

```bash
bun install          # Install dependencies
bun run dev          # Start dev server (http://localhost:5173)
python scripts/export-chromadb.py  # Re-export data after claude-memory changes
```

## Architecture

```
src/main.ts          # All visualization logic (Three.js, search, UI)
index.html           # Single-page app shell
scripts/
  export-chromadb.py # Exports ChromaDB → JSON with pre-computed projections
public/data/
  claude-memory.json # Exported data (~100MB, gitignored)
  demo.json          # Small demo dataset (committed)
```

## Key Technical Details

- **Rendering**: Three.js point cloud with OrbitControls
- **Projections**: UMAP, t-SNE, PCA pre-computed in Python (export script)
- **Search**: Transformers.js in-browser embeddings (all-mpnet-base-v2, ~420MB cached)
- **AI Summaries**: Ollama local LLM for box-selection summaries

## Data Flow

1. `claude-memory` stores conversations in ChromaDB with embeddings
2. `export-chromadb.py` reads ChromaDB, computes 3D projections, outputs JSON
3. Browser loads JSON, renders point cloud, enables search/interaction

## Common Tasks

**Add new feature**: Edit `src/main.ts` - it's a single-file app

**Update after claude-memory changes**: Run `python scripts/export-chromadb.py`

**Change projection algorithms**: Modify `export-chromadb.py` (Python) for computation, `main.ts` for UI

## Dependencies

- Uses **Bun** (not npm/node) - see `bun install`, `bun run dev`
- Export script needs Python with: chromadb, umap-learn, scikit-learn, numpy
- Optional: Ollama for AI summaries
