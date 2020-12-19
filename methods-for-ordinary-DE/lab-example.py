import numpy as np
import matplotlib.pyplot as plt

def f(t,y):
    return -30*y + 31*np.cos(t) + 29*np.sin(t)

def y_exact(t):
    return np.cos(t) + np.sin(t)

def Euler(t, N):
    h = t[1] - t[0]

    y = np.zeros(N+1)
    y[0] = 0

    for i in range(N):
        y[i+1] = y[i] + h*f(t[i],y[i])

    return y

# ερώτηση 1
NN = 100
t = np.linspace(0, 4, NN+1)
y_Euler = Euler(t, NN)
print(f"y[4]: {y_Euler[-1]}")

# ερώτηση 2
N = list(range(60,121,20))
errors = []

for n in N:
    t = np.linspace(0, 4, n+1)
    err = max(abs(y_exact(t) - Euler(t,n)))
    errors.append(err)
    print(f"error for N={n} is {err}")

print("Πειραματική ταξη σύγκλισης")
print(f"{np.log(errors[-1]/errors[-2])/np.log(100/120)}")
