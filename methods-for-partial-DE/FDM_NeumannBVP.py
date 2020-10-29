# Finite differences for a Neumann bvp

# Suppose y(x)=cos(2πx)/(1+4π^2) the solution of the boundary-value problem: -y''(x)+y(x)=cos(2πx) forall x in [0,1] and
# with y'(0)=0, y'(1)=0. Solve the bvp using Ν=50 and plot the approximate and the actual solutions.
# Furthermore, for N = [100,200,400] find the approach errors.

import numpy as np
import matplotlib.pyplot as plt

def y(x):
    return np.cos(2*np.pi*x)/(1+4*(np.pi**2))

def q(x):
    return 1

def f(x):
    return np.cos(2*np.pi*x)


yA = 0
yB = 0
N = 50

U = fdm(N, yA, yB)

def fdm(N, yA, yB):
    h = np.pi/N
    t = np.linspace(0, np.pi, N+1)

    # definition of array F, (AU=F)
    F = f(t[1:-1])*(h**2)
    F[0] += yA
    F[-1] += yB

    Q = np.ones(N-1)*q(t[1:-2])*(2+h**2)
    Z = np.ones(N-2)*(-1)

    # AU = F => (Z+h(^2)Q)U = F(h^2)
    A = np.diag(Q,0) + np.diag(Z, 1) + np.diag(Z, -1)

    U = np.linalg.solve(A,F)
    U = np.insert(U, 0, yA)
    U = np.append(U, yB)
    
    return U