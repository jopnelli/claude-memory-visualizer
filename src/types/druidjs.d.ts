// Type declarations for @saehrimnir/druidjs
declare module '@saehrimnir/druidjs' {
  export class Matrix {
    static from(data: number[][]): Matrix;
    get to2dArray(): number[][];
  }

  export class UMAP {
    constructor(
      matrix: Matrix,
      options?: { d?: number; n_neighbors?: number; min_dist?: number }
    );
    transform(): Matrix;
  }

  export class PCA {
    constructor(matrix: Matrix, options?: { d?: number });
    transform(): Matrix;
  }
}
