import numpy as np


def qr_numpy(A: np.ndarray):
    """
    Decomposição QR de uma matriz A usando a função QR do NumPy.
    
    Parâmetros:
    A (np.ndarray): Matriz a ser decomposta.
    
    Retorna:
    Q (np.ndarray): Matriz ortonormal.
    R (np.ndarray): Matriz triangular superior.
    """
    
    Q, R = np.linalg.qr(A)
    
    return Q, R

def qr_classic(A: np.array):
    """
    Decomposição QR clássica de uma matriz A usando o método de Gram-Schmidt.
    Decomposição reduzida e instável para matrizes retangulares.
    
    Parâmetros:
    A (np.ndarray): Matriz a ser decomposta.
    
    Retorna:
    Q (np.ndarray): Matriz ortonormal.
    R (np.ndarray): Matriz triangular superior.
    """
    
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j] if R[j, j] != 0 else 0
    
    return Q, R

def qr_modified(A: np.array):
    """
    Decomposição QR modificada de uma matriz A usando o método de Gram-Schmidt.
    Decomposição reduzida e estável (mas não inversamente estável) para matrizes retangulares.
    
    Parâmetros:
    A (np.ndarray): Matriz a ser decomposta.
    
    Retorna:
    Q (np.ndarray): Matriz ortonormal.
    R (np.ndarray): Matriz triangular superior.
    """
    
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))    
    V = [A[:, i].copy() for i in range(n)]
    
    for i in range(n):
        v_i = V[i]
        R[i, i] = np.linalg.norm(v_i)
        Q[:, i] = v_i / R[i, i] if R[i, i] != 0 else 0
        for j in range(i+1, n):
            R[i, j] = np.dot(Q[:, i], V[j])
            V[j] = V[j] - R[i, j] * Q[:, i]
    
    return Q, R

def householder_reflection(a: np.array):
    """
    Cria a matriz de reflexão de Householder que anula os elementos abaixo do primeiro de a.
    """
    n = a.shape[0]
    v = a.copy()
    v[0] = v[0] + np.sign(a[0]) * np.linalg.norm(a)
    P = np.identity(n) - 2 * np.outer(v,v) / np.dot(v,v)
    return P

def qr_householder(A: np.array):
    """
    Decomposição QR de uma matriz A usando reflexões de Householder.
    Decomposição reduzida e inversamente estável para matrizes retangulares.
    
    Parâmetros:
    A (np.ndarray): Matriz a ser decomposta.
    
    Retorna:
    Q (np.ndarray): Matriz ortonormal.
    R (np.ndarray): Matriz triangular superior.
    """
    
    m, n = A.shape
    R = A.copy()
    Q = np.identity(m)

    for i in range(n):
        P_i = householder_reflection(R[i:, i])
        H = np.identity(m)
        H[i:, i:] = P_i
        R = H @ R
        Q = Q @ H

    return Q[:, :n], R[:n, :]
