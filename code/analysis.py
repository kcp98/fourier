import numpy as np
import timeit
import resource
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--algo", help="the algorithm to benchmark", required=True)
args = parser.parse_args()
assert args.algo in ["fmul", "nmul", "fdiv", "ndiv"]

def get_mem():
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru_maxrss / (1024 * 1024)

def timestmt(n : int, stmt : str):
    SETUP_CODE = (
        f"import numpy as np; n = {n};"
        + "from multiplication import fmul;"
        + "from division import fdiv;"
        + "x = np.linspace(-64, 64, n);"
        + "y = np.linspace(32, -16, n);"
        + "g = np.linspace(32, 16, int(n/3));"
    )
    print(f"n = {n}")
    times = timeit.repeat(setup=SETUP_CODE,stmt=stmt,number=2)
    return min(times), get_mem()

def benchmark(func, fname, limit = 10):
    # Running time (seconds), memory usage (MB)
    ns = np.array([1 << i for i in range(5, limit)])
    res = np.array([func(1 << i) for i in range(5, limit)])
    r  = np.vstack((ns,res[:,0], res[:,1]))
    np.savetxt(f"benchmark/data/{fname}.txt", r.T, delimiter=',', header="n,seconds,MB")

options = {
    "fmul": lambda n : timestmt(n, "fmul(x,y)"),
    "nmul": lambda n : timestmt(n, "np.polynomial.polynomial.polymul(x,y)"),
    "fdiv": lambda n : timestmt(n, "fdiv(x,g)"),
    "ndiv": lambda n : timestmt(n, "np.polynomial.polynomial.polydiv(x,g)")
}

if __name__ == "__main__":
    #benchmark(options[args.algo], args.algo, limit=19) # mul
    #benchmark(options[args.algo], args.algo, limit=21) # div
    benchmark(options[args.algo], args.algo, limit=15)
