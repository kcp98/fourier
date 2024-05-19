from matplotlib import pyplot as plt
import numpy as np

def regression(x : np.array, y : np.array):
    """ Perform linear regression for y = m * x + b
        Returns an arrays of predicted y's, and an R^2 value for the fit.
    """
    p, ssr, *rem = np.polyfit(x, y, 1, full=True)
    ybar = np.sum(y) / len(y)
    sst  = np.sum((y - ybar)**2)
    r2   = 1 - ssr / sst
    return p[0] * x + p[1], r2

def make_plot(fname : str, title : str):
    array = np.loadtxt(f"benchmark/data/{fname}.txt")
    x, y = array[:,0], array[:,1] / 1000 # back to seconds...
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.suptitle(f"Running time for {title}")
    fig.set_size_inches(15, 5)

    if True:
        fnlgn, r2 = regression(x * np.log(x), y)
        ax1.plot(x, fnlgn, '-', color="orchid")
        ax1.plot(x,     y, '.', color="black")
        ax1.annotate(f"$R^2 = {r2[0]:0.6f}$", (min(x), max(y)))
        ax1.set_title(r"$O(n \lg n)$")
        ax1.set_ylabel('time (seconds)')
        ax1.set_xlabel('input size')
        ax1.set_xticks(x[-4::])
        ax1.set_yticks(y[-4::])

    if True:
        fnn, r2 = regression(x * x, y)
        ax2.plot(x, fnn, '-', color="orchid")
        ax2.plot(x,   y, '.', color="black")
        ax2.annotate(f"$R^2 = {r2[0]:0.6f}$", (min(x), max(y)))
        ax2.set_title(r"$O(n^2)$")
        ax2.set_xticks([])
        ax2.set_yticks([])
 
    if True:
        fn, r2 = regression(x, y)
        ax3.plot(x, fn, '-', color="orchid")
        ax3.plot(x,  y, '.', color="black")
        ax3.annotate(f"$R^2 = {r2[0]:0.6f}$", (min(x), max(y)))
        ax3.set_title(r"$O(n)$")
        ax3.set_xticks([])
        ax3.set_yticks([])

    fig.savefig(f"benchmark/images/{fname}.png")
    fig.clear()

if __name__ == "__main__":
    make_plot("fmul", "fft multiplication")
    make_plot("rmul", "long multiplication")
    make_plot("nmul", "numpys polymul")
