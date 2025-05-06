import numpy as np

def check_representability(n, dtype=np.float32):
    for offset in range(-3, 4):
        x = n + offset
        fx = dtype(x)
        is_exact = (int(fx) == x)
        print(f"{x}: Representável? {is_exact} (convertido: {fx})")

print("Single Precision (float32):")
n_single = 2**24 + 1
check_representability(n_single, np.float32)

print("\nDouble Precision (float64):")
n_double = 2**53 + 1
check_representability(n_double, np.float64)
