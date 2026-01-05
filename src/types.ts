// Core data types for Claude Memory Visualizer

export interface Document {
  id: string;
  text: string;
  embedding: number[];
  metadata?: Record<string, unknown>;
}

export interface DataSet {
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

export type ProjectionAlgorithm = 'umap' | 'tsne' | 'pca';

export interface TimeRange {
  min: Date;
  max: Date;
}

export interface SelectionBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface SearchMatch {
  index: number;
  score: number;
}

export interface TopicCluster {
  topic: string;
  count: number;
  example: string;
}
