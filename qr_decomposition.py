import numpy as np

# # Initializing a random integer matrix
# A = np.random.randint(0, 100, (5,4))
# print("Matrix A: \n", A)
# print("\n")

# # QR decomposition with numpy function
# q_A, r_A = np.linalg.qr(A)
# print("Matrix Q (função do numpy): \n", q_A)
# print("Matrix R (função do numpy): \n", r_A)


def qr_decomp(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    # Pegando as dimensões da matriz
    m, n = A.shape
    
    # Inicializando Q e R
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v_j = A[:, j].copy()  # Copiando a coluna de A para evitar modificações indesejadas
        
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])  # Produto interno com vetores ortonormalizados
            v_j = v_j - R[i, j] * Q[:, i]  # Ortogonalização
        
        R[j, j] = np.linalg.norm(v_j)  # Norma do vetor resultante
        Q[:, j] = v_j / R[j, j]  # Normalizando o vetor
    
    return Q, R
