#!/usr/bin/env python3
"""
Export ChromaDB embeddings to JSON for Claude Memory Visualizer.

Includes pre-computed 3D projections (UMAP, t-SNE, PCA) for instant visualization.

Usage:
    python export-chromadb.py [--output FILE] [--limit N]

Examples:
    python export-chromadb.py
    python export-chromadb.py --output my-data.json --limit 1000
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    import chromadb
except ImportError:
    print("Error: chromadb not installed. Run: pip install chromadb")
    exit(1)

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError:
    print("Error: scikit-learn not installed. Run: pip install scikit-learn")
    exit(1)

try:
    import umap
except ImportError:
    print("Error: umap-learn not installed. Run: pip install umap-learn")
    exit(1)


def compute_projections(embeddings: np.ndarray) -> dict:
    """Compute 3D projections using UMAP, t-SNE, and PCA."""

    n_samples = len(embeddings)
    print(f"Computing projections for {n_samples} documents...")

    projections = {}

    # PCA (fastest)
    print("  Computing PCA...")
    pca = PCA(n_components=3, random_state=42)
    pca_result = pca.fit_transform(embeddings)
    projections["pca"] = normalize_projection(pca_result)
    print("  PCA done.")

    # UMAP (good balance)
    print("  Computing UMAP...")
    umap_model = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    umap_result = umap_model.fit_transform(embeddings)
    projections["umap"] = normalize_projection(umap_result)
    print("  UMAP done.")

    # t-SNE (slowest but good clusters)
    print("  Computing t-SNE (this may take a while)...")
    # Adjust perplexity for smaller datasets
    perplexity = min(30, max(5, n_samples // 5))
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
    )
    tsne_result = tsne.fit_transform(embeddings)
    projections["tsne"] = normalize_projection(tsne_result)
    print("  t-SNE done.")

    return projections


def normalize_projection(points: np.ndarray, scale: float = 40.0) -> list:
    """Normalize projection to [-scale/2, scale/2] range and convert to list."""
    # Center and scale
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  # Avoid division by zero

    normalized = (points - min_vals) / ranges - 0.5  # Center at 0
    normalized *= scale

    return normalized.tolist()


def export_chromadb(
    chroma_path: str = os.path.expanduser("~/.claude-memory/chroma"),
    collection_name: str = "conversations",
    output_file: str = "public/data/claude-memory.json",
    limit: int | None = None,
    skip_projections: bool = False,
) -> None:
    """Export ChromaDB collection to JSON for visualization."""

    print(f"Connecting to ChromaDB at {chroma_path}...")
    client = chromadb.PersistentClient(path=chroma_path)

    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        print(f"Error: Could not get collection '{collection_name}': {e}")
        print("Available collections:", [c.name for c in client.list_collections()])
        exit(1)

    print(f"Found collection '{collection_name}' with {collection.count()} documents")

    # Get all data with embeddings
    if limit:
        data = collection.get(
            include=["embeddings", "documents", "metadatas"],
            limit=limit,
        )
    else:
        data = collection.get(include=["embeddings", "documents", "metadatas"])

    print(f"Exporting {len(data['ids'])} documents...")

    # Get embedding dimension from first embedding (handle numpy arrays)
    embeddings = data.get("embeddings")
    if embeddings is None:
        embeddings = []
    documents = data.get("documents")
    if documents is None:
        documents = []
    metadatas = data.get("metadatas")
    if metadatas is None:
        metadatas = []

    embedding_dim = len(embeddings[0]) if len(embeddings) > 0 else 0

    # Convert to numpy array for projection computation
    embeddings_array = np.array(embeddings)

    # Build output structure
    output = {
        "metadata": {
            "name": "Claude Memory",
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "embedding_dim": embedding_dim,
            "count": len(data["ids"]),
        },
        "documents": [
            {
                "id": data["ids"][i],
                "text": documents[i] if i < len(documents) else "",
                "embedding": [float(x) for x in embeddings[i]] if i < len(embeddings) else [],
                "metadata": metadatas[i] if i < len(metadatas) else {},
            }
            for i in range(len(data["ids"]))
        ],
    }

    # Compute projections
    if not skip_projections and len(embeddings) > 0:
        output["projections"] = compute_projections(embeddings_array)
    else:
        print("Skipping projection computation.")

    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported to {output_file} ({file_size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export ChromaDB to JSON for Claude Memory Visualizer")
    parser.add_argument(
        "--chroma-path",
        default=os.path.expanduser("~/.claude-memory/chroma"),
        help="Path to ChromaDB directory",
    )
    parser.add_argument(
        "--collection",
        default="conversations",
        help="Collection name to export",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="public/data/claude-memory.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Limit number of documents to export",
    )
    parser.add_argument(
        "--skip-projections",
        action="store_true",
        help="Skip computing projections (faster export)",
    )

    args = parser.parse_args()

    export_chromadb(
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        output_file=args.output,
        limit=args.limit,
        skip_projections=args.skip_projections,
    )


if __name__ == "__main__":
    main()
