import numpy as np
from evaluation import fdiv, fmul, mpe

def div_0(x,y):
    fq,fr = fdiv(x,y)
    nq,nr = np.polynomial.polynomial.polydiv(x,y)

    qclose, rclose = np.isclose(fq,nq).all(), np.isclose(fr,nr).all()
    if (not qclose) or (not rclose):
        print("-----------------------------------------")
        print(f"n = {len(x)}, m = {len(y)}")

    if not qclose and False:
        print(
            f"q.close {np.isclose(fq,nq).all()}"
            
        )
        dq = np.absolute((fq - nq).real)
        print(
            f"q.mean {np.mean(dq)} - q.std {np.std(dq)} - nq.mean {np.mean(nq)}"
        )
    if not rclose:
        dr = np.absolute((fr - nr).real)
        print(
            f"r.close {np.isclose(fr,nr).all()}"
        )
        print(
            f"r.mean {np.mean(dr)} - r.std {np.std(dr)} - nr.mean {np.mean(nr)}"
        )
    if (not qclose) or (not rclose):
        print("-----------------------------------------")

def div_1():
    n = 2**6
    for m in range(1,n,3):
        div_0(
            x = np.linspace(-100, 100, n),
            y = np.linspace(-100,  20, m)
        )

def div_2():
    n = 2**7
    for m in range(1,n,4):
        div_0(
            x = np.linspace(20, 100, n),
            y = np.linspace(50, 10,  m)
        )

def div_3():
    n = 2**15
    for m in range(1,int(n/3),27):
        div_0(
            x = np.linspace(20, 100, n),
            y = np.linspace(50, 10,  m)
        )
div_3()

def mul_0(x,y):
    fr = fmul(x,y)
    nr = np.polynomial.polynomial.polymul(x,y)
    rclose = np.isclose(fr,nr).all()
    if not rclose:
        print("-----------------------------------------")
        print(f"n = {len(x)}, m = {len(y)}")
        dr = np.absolute((fr - nr).real)
        print(
            f"r.close {np.isclose(fr,nr).all()}"
        )
        print(
            f"r.mean {np.mean(dr)} - r.std {np.std(dr)} - nr.mean {np.mean(nr)}"
        )
        print("-----------------------------------------")

def mul_1():
    n = 2**18
    mul_0(
        x = np.linspace(121, 202, n),
        y = np.linspace(-100, 220, n)
    )

def eval_0(x,y):
    fr = mpe(x,y,len(x))
    nr = np.polynomial.polynomial.polyval(y,x)
    rclose = np.isclose(fr,nr).all()
    if not rclose:
        print("-----------------------------------------")
        print(f"n = {len(x)}, m = {len(y)}")
        dr = np.absolute((fr - nr).real)
        print(
            f"r.close {np.isclose(fr,nr).all()}"
        )
        print(
            f"r.mean {np.mean(dr)} - r.std {np.std(dr)} - nr.mean {np.mean(nr)}"
        )
        print("-----------------------------------------")

def eval_1():
    for i in range(1,7):
        n = 2**i
        eval_0(
            x = np.linspace(-30, 30, n),
            y = np.linspace(-10, 10, n)
        )

def eval_2():
    for i in range(1,7):
        n = 2**i
        eval_0(
            x = np.linspace(-3, 3, n),
            y = np.linspace(-2, 2, n)
        )
