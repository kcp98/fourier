# https://docs.python.org/3/library/timeit.html
import numpy as np
import timeit

def timestmt(n : int, stmt : str):
    SETUP_CODE = (
        f"import numpy as np; n = {n};"
        + "from multiplication import lmul, fmul;"
        + "from division import ldiv, fdiv;"
        + "x = np.linspace(-64, 64, n);"
        + "y = np.linspace(32, -16, n);"
        + "g = np.linspace(32, 16, int(n/3));"
    )
    print(f"n = {n}")
    times = timeit.repeat(setup=SETUP_CODE,stmt=stmt,number=10)
    return min(times)

def benchmark(func, fname, limit):
    """ Save a table of running time (in seconds) for range of inputs
    """
    ns = 1 << np.arange(limit-3,limit)
    ts = np.vectorize(func)(ns)
    r  = np.vstack((ns,ts))
    np.savetxt(f"benchmark/data/{fname}.txt",  r.T)

lambda_fmul = lambda n : timestmt(n, "fmul(x,y)")
lambda_lmul = lambda n : timestmt(n, "lmul(x,y)")
lambda_nmul = lambda n : timestmt(n, "np.polynomial.polynomial.polymul(x,y)")

lambda_fdiv = lambda n : timestmt(n, "fdiv(x,g)")
lambda_ldiv = lambda n : timestmt(n, "ldiv(x,g)")
lambda_ndiv = lambda n : timestmt(n, "np.polynomial.polynomial.polydiv(x,g)")

if __name__ == "__main__":
    benchmark(lambda_fmul, "fmul", limit = 20)
    benchmark(lambda_lmul, "lmul", limit = 14)
    benchmark(lambda_nmul, "nmul", limit = 18)
    
    benchmark(lambda_fdiv, "fdiv", limit = 20) #  1:04:05.84 total, never again wtf
    benchmark(lambda_ldiv, "ldiv", limit = 20) #  2:47:18.58 total, oops i did it again
    benchmark(lambda_ndiv, "ndiv", limit = 20)
