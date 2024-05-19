import numpy as np
from fourier import fft, ifft

def __getn(target, n = 1):
    while n < target:
        n = n<<1
    return n

def lmul(p : np.ndarray, q : np.ndarray) -> np.ndarray:
    """ Long multiplication.
    """
    length = len(p) + len(q) - 1
    result = np.zeros(length)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            result[i+j] += a * b
    return result

def fmul(p : np.ndarray, q : np.ndarray) -> np.ndarray:
    """ Multiplication using fft
    """
    length = len(p) + len(q) - 1
    n = __getn(length)

    p_pad = np.pad(p, (0, n - len(p)))
    q_pad = np.pad(q, (0, n - len(q)))

    p_fft = fft(p_pad, n)
    q_fft = fft(q_pad, n)
    
    result = ifft(p_fft * q_fft, n)

    return result[:length]

def __compare(n):
    x = np.linspace(-16, 64, n)
    y = np.linspace(64, -16, n)

    xy_ref = lmul(x,y)
    xy_fft = fmul(x,y)

    assert np.isclose(xy_ref, xy_fft).all()

def __compares():
    for i in range(6, 9):
        print(f"n = {2**i:5d}")
        __compare(2**i)

if __name__ == "__main__":
    __compares()
