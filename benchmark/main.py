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
    x, y = array[:,0], array[:,1]
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.suptitle(f"Running time for {title}", fontsize=16)
    fig.set_size_inches(18, 6)

    if True: # O(nlgn)
        fnlgn, r2 = regression(x * np.log(x), y)
        ax1.plot(x, fnlgn, '-', color="orchid")
        ax1.plot(x,     y, '.', color="gray")
        ax1.annotate(f"$R^2 = {r2[0]:0.5f}$", (max(x)/2, 0), fontsize=14)
        ax1.set_title(r"$O(n \lg n)$", fontsize=16)
        ax1.set_ylabel('time (seconds)', fontsize=16)
        ax1.set_xlabel('input size', fontsize=16)
        ax1.set_xticks(x[-4::])
        ax1.set_yticks(y[-4::])
        ax1.set_xticklabels([f"2^{j:.0f}" for j in np.log2(x[-4::])])

    if True: # O(n^2)
        fnn, r2 = regression(x * x, y)
        ax2.plot(x, fnn, '-', color="orchid")
        ax2.plot(x,   y, '.', color="gray")
        ax2.annotate(f"$R^2 = {r2[0]:0.5f}$", (max(x)/2, 0), fontsize=14)
        ax2.set_title(r"$O(n^2)$", fontsize=16)
        ax2.set_xticks([])
        ax2.set_yticks([])
 
    if True: # O(n)
        fn, r2 = regression(x, y)
        ax3.plot(x, fn, '-', color="orchid")
        ax3.plot(x,  y, '.', color="gray")
        ax3.annotate(f"$R^2 = {r2[0]:0.6f}$", (max(x)/2, 0), fontsize=14)
        ax3.set_title(r"$O(n)$", fontsize=16)
        ax3.set_xticks([])
        ax3.set_yticks([])

    fig.savefig(f"benchmark/images/{fname}.png")
    fig.clear()

def make_reference_plots():
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.suptitle(f"Reference frame", fontsize=16)
    fig.set_size_inches(18, 6)

    if True: # long mul
        array = np.loadtxt(f"benchmark/data/lmul.txt")
        x, y = array[:,0], array[:,1]
        f, r2 = regression(x * x, y)
        ax1.plot(x, f, '-', color="darkorchid")
        ax1.plot(x, y, '.', color="black")
        ax1.annotate(f"$R^2 = {r2[0]:0.5f}$", (max(x)/2, 0), fontsize=14)
        ax1.set_title(r"$O(n^2)$ for long multiplication", fontsize=14)
        ax1.set_ylabel('time (seconds)', fontsize=14)
        ax1.set_xlabel('input size', fontsize=14)
        ax1.set_xticks(x[-4::])
        ax1.set_yticks(y[-4::])
        ax1.set_xticklabels([f"2^{j:.0f}" for j in np.log2(x[-4::])])

    if True: # numpy
        array = np.loadtxt(f"benchmark/data/nmul.txt")
        x, y = array[:,0], array[:,1]
        f, r2 = regression(x * x, y)
        ax2.plot(x, f, '-', color="deepskyblue")
        ax2.plot(x,  y, '.', color="black")
        ax2.annotate(f"$R^2 = {r2[0]:0.6f}$", (max(x)/2, 0), fontsize=14)
        ax2.set_title(r"$O(n^2)$ for numpys polymul", fontsize=14)
        ax2.set_xticks(x[-4::])
        ax2.set_yticks(y[-4::])
        ax2.set_xticklabels([f"2^{j:.0f}" for j in np.log2(x[-4::])])
    
    if True: # numpy vs. fmul
        array = np.loadtxt(f"benchmark/data/nmul.txt")
        x, y = array[:,0][2:-1], array[:,1][2:-1]
        ax3.plot(x, y, '-', color="deepskyblue")
        ax3.plot(x, y, '.', color="black")
        
        array = np.loadtxt(f"benchmark/data/fmul.txt")
        x, y = array[:,0][:-3], array[:,1][:-3]
        ax3.plot(x, y, '-', color="orchid")
        ax3.plot(x, y, '.', color="gray")

        ax3.set_title(r"numpy vs. fmul", fontsize=14)
        ax3.set_xticks(x[-4::])
        ax3.set_yticks(y[-4::])

        ax3.set_xticklabels([f"2^{j:.0f}" for j in np.log2(x[-4::])])

    fig.savefig(f"benchmark/images/ref.png")
    fig.clear()


if __name__ == "__main__":
    make_plot("fmul", "fft multiplication")
    make_reference_plots()
