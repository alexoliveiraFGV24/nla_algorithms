import numpy as np

def gauss_seidel(A: np.array, b: np.array, max_iter: int = 10, tol: float = 1e-8):
    """
    Método de Gauss-Seidel para resolver um sistema linear Ax = b.
    Matriz de iteração: (D+L)^-1 * U

    Parâmetros:
    - A: Matriz de coeficientes (numpy array de dimensão nxn)
    - b: Vetor de termos independentes (numpy array de dimensão n)
    - max_iter: Número máximo de iterações (padrão: 10)
    - tol: Tolerância para critério de convergência (padrão: 1e-8)

    Retorna:
    - x: Aproximação da solução do sistema
    - erro_hist: Histórico de erros ao longo das iterações
    """
    
    n = b.shape[0]  # Obtém o número de variáveis
    x = np.zeros(n)  # Vetor de solução inicial zerado
    erro_hist = []  # Lista para armazenar os erros a cada iteração

    for iter in range(max_iter):  # Loop principal das iterações
        x_old = x.copy()  # Armazena a solução anterior para verificar a convergência

        for i in range(n):  # Itera sobre cada equação do sistema
            # Calcula a soma dos elementos já atualizados na linha
            s_new = sum(A[i, j] * x[j] for j in range(i))
            # Calcula a soma dos elementos ainda não atualizados na linha
            s_old = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            # Atualiza o valor da variável i usando a fórmula de Gauss-Seidel
            x[i] = (b[i] - s_new - s_old) / A[i, i]

        # Calcula o erro como a norma da diferença entre as soluções sucessivas (caso geral seria melhor a norma infinito)
        erro = np.linalg.norm(A@x - b)
        erro_hist.append(erro)  # Armazena o erro no histórico

        # Se o erro for menor que a tolerância, interrompe a execução
        if erro < tol:
            return (x, erro_hist)
    
    return (x, erro_hist)
