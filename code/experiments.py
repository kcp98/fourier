import numpy as np
from evaluation import fdiv, fmul, mpe

def e0(n):
    x = np.linspace(4, 10, n)
    y = np.linspace(1, 2, n)
    xy_ref = np.polynomial.polynomial.polyval(y,x)
    xy_fft = mpe(x,y,n)
    return np.isclose(xy_ref, xy_fft).all(), np.std(abs(xy_fft - xy_ref))

print(
    e0(16),
    e0(32)
)