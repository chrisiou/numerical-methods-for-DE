import numpy as np
# import matplotlib.pyplot as plt

#Άσκηση 1
def f1(t,x,y):
    return y

def f2(t,x,y):
    return -4*x

def x_exact(t):
    return np.cos(2*t)

def y_exact(t):
    return -2*np.sin(t)

def Euler(t,h,N):
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    x[0] = 1
    y[0] = 0

    for i in range(N):
        x[i+1] = x[i] + h*f1(t,x[i],y[i])
        y[i+1] = y[i] + h*f2(t,x[i],y[i])

    return x,y

N = 100
t = np.linspace(0,4,N+1)
h = 4/N

x, y = Euler(t, h, N)
# 1.
print(x[-1])
print(y[-1])

# 2.
N = list(range(60,121,20))

err_x = []
err_y = []

for n in N:
    t = np.linspace(0,4,n+1)
    h = 4/n
    x, y = Euler(t, h, n)
    err_x.append(max(abs(x_exact(t) - x)))
    err_y.append(max(abs(y_exact(t) - y)))

for i in range(len(err_x)):
    print(f"x= {err_x[i]}, y = {err_y[i]}")

# 3.
print(f"p_x = {np.log(err_x[-1]/err_x[-2])/np.log(100/120)}")
print(f"p_y = {np.log(err_y[-1]/err_y[-2])/np.log(100/120)}")
