import numpy as np

def jacobi(A: np.array, b: np.array, max_iter: int = 10, tol: float = 1e-8):
    """
    Método de Jacobi para resolver um sistema linear Ax = b.
    Matriz de iteração: D^-1 * (L + U)

    Parâmetros:
    - A: Matriz de coeficientes (numpy array de dimensão nxn)
    - b: Vetor de termos independentes (numpy array de dimensão n)
    - max_iter: Número máximo de iterações (padrão: 10)
    - tol: Tolerância para critério de convergência (padrão: 1e-8)

    Retorna:
    - x_new: Aproximação da solução do sistema
    - erro_hist: Histórico de erros ao longo das iterações
    """
    
    n = b.shape[0]  # Obtém o número de variáveis
    x = np.zeros(n)  # Vetor de solução inicial zerado
    x_new = np.zeros(n)  # Vetor auxiliar para armazenar os novos valores de x
    erro_hist = []  # Lista para armazenar os erros a cada iteração

    for iter in range(max_iter):  # Loop principal das iterações
        for i in range(n):  # Itera sobre cada equação do sistema
            # Calcula a soma dos elementos da linha, ignorando a diagonal principal
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            # Atualiza o valor da variável i usando a fórmula de Jacobi
            x_new[i] = (b[i] - s) / A[i, i]

        # Calcula o erro como a norma da diferença entre as soluções sucessivas (caso geral seria melhor a norma infinito)
        erro = np.linalg.norm(A@x_new - b)  
        erro_hist.append(erro)  # Armazena o erro no histórico

        # Se o erro for menor que a tolerância, interrompe a execução
        if erro < tol:
            return x_new, erro_hist

        x = x_new.copy()  # Atualiza o vetor de solução para a próxima iteração

    return (x_new, erro_hist)  # Retorna a solução final e o histórico de erros
