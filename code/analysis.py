# https://docs.python.org/3/library/timeit.html
import numpy as np
import timeit

def timestmt(n : int, stmt : str):
    SETUP_CODE = (
        f"import numpy as np; n = {n};"
        + "from multiplication import lmul, fmul;"
        + "x = np.linspace(-64, 64, n);"
        + "y = np.linspace(32, -16, n);"
    )
    print(f"n = {n}")
    times = timeit.repeat(setup=SETUP_CODE,stmt=stmt,number=10)
    return min(times) * 1000 # miliseconds

def benchmark(func, fname, limit):
    """ Save a table of running time (in ms) for range of inputs
    """
    ns = 1 << np.arange(6,limit)
    ts = np.vectorize(func)(ns)
    r  = np.vstack((ns,ts))
    np.savetxt(f"benchmark/data/{fname}.txt",  r.T)

fmul = lambda n : timestmt(n, "fmul(x,y)")
lmul = lambda n : timestmt(n, "lmul(x,y)")
nmul = lambda n : timestmt(n, "np.polynomial.polynomial.polymul(x,y)")

if __name__ == "__main__":
    benchmark(fmul, "fmul", limit = 16)
    benchmark(lmul, "lmul", limit = 13)
    benchmark(nmul, "nmul", limit = 16)
