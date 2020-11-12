import numpy as np
# import matplotlib.pyplot as plt

# Άσκηση 2
def y_exact(t):
    return np.arctan(t)

def f(t,y):
    return y**2

def g(t, yn, x):
    return yn+h*f(t,x)

def Euler_implicit(t,h,N):
    y = np.zeros(N+1)
    y[0] = 0

    tol = 1.e-5
    Nmax = 100
    
    for i in range(N):
        x0 = y[i]
        k = 0 
        err = 1.
    
        while (err > tol) and (k <= Nmax):
            x = g(t[i+1],y[i],x0)
            e = abs(x-x0)
            k = k+1
            x0 = x
        
        y[i+1] = y[i]+h*f(t[i],y[i])

    return y

# 1.
NN = 200
t = np.linspace(0,1,NN+1)
h = 1/NN

y = Euler_implicit(t,h,NN)

k = t[0]
for i in range(len(t)):
    if t[i] == 0.5:
        print(f"y[{t[i]}]={y[i]}")

print(y)

# 2.
N=[100,200,400,800]

errors = []
for n in N:
    t = np.linspace(0,1,n+1)
    h = 1/n

    y = Euler_implicit(t,h,n)
    errors.append(max(abs(y_exact(t) - y)))

# 3.
print(f"p = {np.log(errors[-1]/errors[-2])/np.log(100/120)}")
