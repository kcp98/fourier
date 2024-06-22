import numpy as np
from division import fdiv, fmul

def meval(p : np.ndarray, alphas : np.ndarray) -> np.ndarray:
    evaluations = np.zeros(len(alphas))
    for i, alpha in enumerate(alphas):
        for j, a in enumerate(p):
            evaluations[i] += a * alpha**j
    return np.array(evaluations)

class FourierTree:
    def __init__(self, ary) -> None:
        n,n2  = len(ary), int(len(ary)/2)
        if n == 1:
            self.prod = np.array([-ary[0], 1])
            self.left, self.right = None, None
        else:
            self.left  = FourierTree(ary[:n2])
            self.right = FourierTree(ary[n2:])
            self.prod  = fmul(self.left.prod, self.right.prod)

def feval(f : np.ndarray, alphas : np.ndarray, n : int):
    prods = FourierTree(alphas)
    def __feval(f : np.ndarray, pt : FourierTree, n : int):
        if n == 1:
            return f
        n2 = int(n/2)
        _q, f1 = fdiv(f, pt.left.prod) 
        _q, f2 = fdiv(f, pt.right.prod)
        return np.append(
            __feval(f1, pt.left,  n2),
            __feval(f2, pt.right, n2)
        )
    return __feval(f, prods, n)

def __compare(n):
    x = np.linspace(4, 10, n)
    y = np.linspace(1, 2, n)
    #xy_ref = np.polynomial.polynomial.polyval(y,x)
    xy_ref = meval(x,y)
    xy_fft = feval(x,y,n)
    print(np.std(abs(xy_fft - xy_ref)))
    assert np.isclose(xy_ref, xy_fft).all()

def __compares():
    for i in range(1, 7):
        n = 2**i
        print(f"{i:2d}: {n:5d}")
        __compare(n)

if __name__ == "__main__":
    __compares()
