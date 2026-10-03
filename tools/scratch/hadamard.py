"""Hadamard-matrix construction: Sylvester (powers of 2) + Paley type I
(order q+1 for prime power q = 3 mod 4). Only need small orders here
(4, 8, 12, 16, 20, 24) for month-layer channel patterns.
"""

from __future__ import annotations

import numpy as np


def _sylvester(n: int) -> np.ndarray:
    assert n & (n - 1) == 0, "sylvester only for powers of 2"
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in range(2, int(n**0.5) + 1):
        if n % p == 0:
            return False
    return True


def _legendre_table(q: int) -> np.ndarray:
    # chi[a] for a in 0..q-1, chi[0]=0
    residues = {(x * x) % q for x in range(1, q)}
    chi = np.zeros(q)
    for a in range(1, q):
        chi[a] = 1.0 if a in residues else -1.0
    return chi


def _paley1(q: int) -> np.ndarray:
    """Hadamard matrix of order q+1, q prime, q % 4 == 3."""
    assert _is_prime(q) and q % 4 == 3
    chi = _legendre_table(q)
    Q = np.zeros((q, q))
    for i in range(q):
        for j in range(q):
            Q[i, j] = chi[(j - i) % q]
    H = np.zeros((q + 1, q + 1))
    H[0, 0] = 1.0
    H[0, 1:] = 1.0
    H[1:, 0] = 1.0
    H[1:, 1:] = Q - np.eye(q)
    return H


def hadamard(n: int) -> np.ndarray:
    if n & (n - 1) == 0 and n >= 1:
        return _sylvester(n)
    if (n - 1) > 0 and _is_prime(n - 1) and (n - 1) % 4 == 3:
        return _paley1(n - 1)
    raise ValueError(f"no construction implemented for order {n}")


if __name__ == "__main__":
    for n in [4, 8, 12, 16, 20]:
        H = hadamard(n)
        HHt = H @ H.T
        ok = np.allclose(HHt, n * np.eye(n))
        print(n, "ok" if ok else "FAIL", HHt.diagonal()[:3], HHt[0, 1:4])
