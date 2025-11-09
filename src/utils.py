from typing import Any, Dict, List
import numpy as np
def make_psd(S: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Ensure Sigma is symmetric PSD. If tiny negative eigenvalues appear, add ֿ\epsilon I.
    Args:
        S (np.ndarray): The input matrix to be projected.
        eps (float, optional): A small value to ensure numerical stability. Defaults to 1e-8.
    Returns:
        np.ndarray: The projected positive semi-definite matrix.
    """
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, eps)
    return (V * w) @ V.T

def pretty_print_buffer():
    print("\n" + "-"*32 + "\n" )

def iter_grid(grid: Dict[str, List[Any]]):
    from itertools import product
    keys = list(grid.keys())
    for values in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))