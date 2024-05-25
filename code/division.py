import numpy as np
from multiplication import fmul

def ldiv(f, g):
    """ Long division.
    """
    f,  g = f[::-1], g[::-1]

    deg_q = len(f) - len(g) + 1
    q     = np.zeros(deg_q)
    while len(f) >= len(g):
        deg_q -= 1
        fg = f[0] / g[0]
        q[deg_q] = fg
        g_pad    = np.pad(g, (0, len(f) - len(g))) * fg
        f = np.trim_zeros(f - g_pad, "f")

    return q, f[::-1]

def __getn(target, n = 1):
    while n < target:
        n = n<<1
    return n

def __inverse(h : np.ndarray, n : int) -> np.ndarray:
    if n == 1:
        return 1 / h
    
    n2 = int(n/2)

    a  = __inverse(h[:n2], n2)
    h0 = h[:n2]
    h1 = h[n2:n]

    c     = fmul(a, h0)
    c_pad = np.pad(c, (0, n - len(c)))[n2:n]

    b = fmul(-a, fmul(h1, a)[:n2] + c_pad)[:n2]
    
    return np.concatenate((a, b))

def inverse(h : np.ndarray, ell : int) -> np.ndarray:
    n  = __getn(ell)
    if n > len(h):
        h  = np.pad(h, (0, -len(h) + n))
    hi = __inverse(h, n)
    return hi[:ell]

def __fdiv(f : np.ndarray, g : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frev, grev = f[::-1], g[::-1]
    ell, nr = len(f) - len(g) + 1, len(g) - 1
    h = inverse(grev, ell)
    qrev = fmul(frev, h)

    q  = qrev[:ell][::-1]
    r  = f - fmul(g, q)
    return q, r[:nr]

def fdiv(f : np.ndarray, g : np.ndarray) -> np.ndarray:
    """ Division with hensel lifting inverses, using fft multiplication
    """
    if g[-1] < 0:
        q, r = __fdiv(f, -g)
        return -q, r
    
    return __fdiv(f, g)

def __compare(n, m):
    x = np.linspace( 16, -64, n)
    y = np.linspace(2,  4, m)
    if m % 3 == 1:
        y = -y
    q_ref, r_ref = ldiv(x,y)
    q_fft, r_fft = fdiv(x,y)

    if not np.isclose(q_ref, q_fft).all():
        print(np.std(q_ref - q_fft))
        exit()
    assert np.isclose(r_ref, r_fft).all()

def __compares():
    for i in range(6, 10):
        n = 2**i
        print(f"{i:2d}: {n:5d}")
        for j in range(2,n-1,2):
            if j % (n/5) == 0:
                print(f"    deg(g): {int(j)}")
            __compare(n, j)

if __name__ == "__main__":
    __compares()
    #bg = 2**19
    #x = np.linspace(16, -64, bg)
    #g = np.linspace(32, 16, int(bg/3))
    #nq,nr = np.polynomial.polynomial.polydiv(x,g)
    #fq,fr = fdiv(x,g)
    #assert np.isclose(nq, fq).all()
    #assert np.isclose(nr, fr).all()
