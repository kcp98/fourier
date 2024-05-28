from matplotlib import pyplot as plt
import numpy as np
import argparse

"""parser = argparse.ArgumentParser()
parser.add_argument("--algo", help="the algorithm to benchmark", required=True)
args = parser.parse_args()
assert args.algo in ["mul", "div"]"""
class args:
    algo = "mul"

def regression(x : np.array, y : np.array):
    # linear for for y = m * x + b and r^2
    p, ssr, *rem = np.polyfit(x, y, 1, full=True)
    ybar = np.sum(y) / len(y)
    sst  = np.sum((y - ybar)**2)
    r2   = 1 - ssr / sst
    return p[0] * x + p[1], r2

def load(fname):
    array = np.loadtxt(f"benchmark/data/{fname}.txt", delimiter=",")
    n, t, m = array[:,0], array[:,1], array[:,2]
    return n, t, m

def plotsub(ax, x, fx, y, **kwargs):
    fy, r2 = regression(fx, y)
    ax.plot(x, fy, '-', **kwargs)
    kwargs.pop("label")
    ax.plot([],[], '-', **kwargs, label=f"$R^2 = {r2[0]:0.5f}$")
    ax.plot(x,  y, '.', color="black")
    ax.set_xticks(x[-4::])
    ax.set_yticks(y[-4::])
    ax.set_xticklabels([f"2^{j:.0f}" for j in np.log2(x[-4::])])

titles = {
    "div": "Polynomial division",
    "mul": "Polynomial multiplication"
}
subtitles = {
    "div": "division",
    "mul": "multiplication"
}

def plotmul():
    nx,ny,nz = load(f"n{args.algo}")
    x,y,z    = load(f"f{args.algo}")

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.suptitle(f"{titles[args.algo]}", fontsize=16)
    fig.set_size_inches(18, 6)

    if True: # time, fast fourier O(nlgn)
        plotsub(ax1, x, x * np.log(x), y, color="cyan", linewidth=3, label='FFT')
        ax1.set_title(f"fft {subtitles[args.algo]}: " + r"$O(n \lg n)$", fontsize=16)
        ax1.set_ylabel('time (seconds)', fontsize=16)
        ax1.set_xlabel('input size', fontsize=16)
        ax1.legend(prop={'size': 14})

    if True: # time, numpy O(n2) vs fast fouirer
        ax2.plot(x,    y, '-', color="cyan", linewidth=3, label='FFT')
        ax2.plot(x,    y, '.', color="black")
        plotsub(ax2, nx, nx * nx, ny, color="crimson", linewidth=3, label='NumPy')
        ax2.set_title(f"NumPy {subtitles[args.algo]}: " + r"$O(n^2)$ (vs fft)", fontsize=16)
        ax2.legend(prop={'size': 14})

    if True: # memory, numpy vs fast fourier
        ax3.plot(x,   z, '-', color="cyan",     linewidth=3, label='FFT')
        ax3.plot(x,   z, '.', color="black")
        ax3.plot(nx, nz, '-', color="crimson", linewidth=3, label='NumPy')
        ax3.plot(nx,   nz, '.', color="black")

        #ax3.plot([], [], color="orchid", label='Inline label 2')
        ax3.legend(prop={'size': 14})
        ax3.set_title(f"NumPy vs fft", fontsize=14)
        ax3.set_xticks(x[-4::])
        ax3.set_yticks(z[-4::])
        ax3.set_ylabel('memory (MB)', fontsize=16)
        ax3.set_xticklabels([f"2^{j:.0f}" for j in np.log2(x[-4::])])

    fig.savefig(f"benchmark/images/{args.algo}.png")
    fig.clear()

plotmul()
