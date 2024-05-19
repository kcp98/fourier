import numpy as np

def __fft(p : np.ndarray, n : int, omegas : np.ndarray) -> np.ndarray:
    if (n==1):
        return p
    
    f0 = __fft(p[0::2], int(n/2), omegas[::2])
    f1 = __fft(p[1::2], int(n/2), omegas[::2])
    
    f = np.concatenate([f0,f0]) + omegas * np.concatenate([f1,f1])
    return f

def fft(p : np.ndarray, n : int) -> np.ndarray:
    """ Calculate the DFT of the polynomial p.
        - p should be coeffiecients in ascending order degree.
        - n must be a power of 2.
    """
    omega  = 2 * np.pi * 1j / n
    omegas = np.exp(omega * np.arange(n))
    
    return __fft(p,n,omegas)

def ifft(f: np.ndarray, n: int) -> np.ndarray:
    """ Recover a polynomial from its DFT.
        - n must be a power of 2.
    """
    p = fft(f, n)
    p[1:] = p[n:0:-1]
    return p / n

def __compare(n):
    p      = np.linspace(-16, 64, n)
    p_fft  = fft(p, n)
    p_ifft = ifft(p_fft, n)

    assert np.isclose(p, p_ifft).all()

def __compares():
    for i in range(6, 16):
        print(f"n = {2**i:5d}")
        __compare(2**i)

if __name__ == "__main__":
    __compares()
