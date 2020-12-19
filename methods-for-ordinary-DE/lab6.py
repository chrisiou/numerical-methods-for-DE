import numpy as np
import matplotlib.pyplot as plt

def makeGraph(y, y_approx, t, title):
    plt.plot(t, y_approx, linewidth = 3, label = "approx sol")
    plt.plot(t, y, marker='o', markersize = 1, label = "actual sol")

    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title(title)
    plt.legend()
    plt.show()

# Άσκηση 1
def y_exact(t):
    return np.exp(0.25 - (t-0.5)**2)

def f(t,y):
    return (1-2*t)*y

N = 500
t = np.linspace(0, 4, N+1)
h = t[1] - t[0]

y_mid = np.zeros(N+1)
y_ab = np.zeros(N+1)

y_mid[0] = 1
y_ab[0] = 1

y_mid[1] = y_exact(h) # o bro leei y_mid[1]=y_exact(t[1])
y_ab[1] = y_exact(h)

# WARNING: Προσοχή στον αριθμό των επαναλήψεων. Στη διβηματική μέθοδο κάνουμε 1 λιγότερη επανάληψη
for i in range(N-1):
    # midpoint method
    y_mid[i+2] = y_mid[i] + 2*h*f(t[i+1], y_mid[i+1])

    # AB(2)
    y_ab[i+2] = y_ab[i+1] + h*((3./2)*f(t[i+1],y_ab[i+1]) - (1./2)*f(t[i],y_ab[i]))

makeGraph(y_exact(t), y_mid, t, "midpoint method")
makeGraph(y_exact(t), y_ab, t, "AB(2)")

print(f"max_error for midpoint: {max(abs(y_exact(t)-y_mid))}")
print(f"max_error for AB(2): {max(abs(y_exact(t)-y_ab))}")

#λείπει να βρω την πειραματική τάξη σύγκλισης

# Άσκηση 2
