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
    return (50/2501)*(np.sin(t) + 50 * np.cos(t)) - (2500/2501)*np.exp(-50*t)

def f(t, y):
    return -50*(y - np.cos(t))

def g(t):
    return np.cos(t)

def Euler(N):
    t = np.linspace(0, 1, N+1)
    h = t[1] - t[0]

    y = np.zeros(N+1)
    y[0] = 0

    for i in range(N):
        y[i+1] = y[i] + h*f(t[i],y[i])

    makeGraph(y_exact(t), y, t, "Euler's method for N: " + str(N))

    # - apo Chazip
    # plt.plot(t,y_exact(t),t,y)
    # plt.show()
    plt.plot(t,abs(y_exact(t)-y))
    plt.show()
    # - //

    return y

def implicit_Euler(N):
    t = np.linspace(0, 1, N+1)
    h = t[1] - t[0]

    y = np.zeros(N+1)
    y[0] = 0

    for i in range(N):
        y[i+1] = y[i] / (1 - h*g(t[i+1])) 
        # Chatzip - y[i+1] = (y[i] + h*g(t[i+1])) / (1 - (-50)*h)

    makeGraph(y_exact(t), y, t, "Implicit Euler's method for N: " + str(N))

    # - apo Chazip
    # plt.plot(t,y_exact(t),t,y)
    # plt.show()
    plt.plot(t,abs(y_exact(t)-y))
    plt.show()

    return y

def Trapezoidal(N):
    t = np.linspace(0, 1, N+1)
    h = t[1] - t[0]

    y = np.zeros(N+1)
    y[0] = 0

    for i in range(N):
        # y[i+1] = y[i]*(1 + h * (g(i)/2))/(1 - h*g(i+1)/2)
        # Chatzi - 
        y[i+1]=(y[i]+(h/2)*f(t[i],y[i])+(h/2)*g(t[i+1]))/(1-(-50)*h/2)

    makeGraph(y_exact(t), y, t, "Trapezoidal's method for N: " + str(N))

    # - apo Chazip
    # plt.plot(t,y_exact(t),t,y)
    # plt.show()
    plt.plot(t,abs(y_exact(t)-y))
    plt.show()

    return y

# N = list(range(20,61,10))
# for n in N:
#     Euler(n)
#     implicit_Euler(n)
#     Trapezoidal(n)


# Άσκηση 2
# θα επέλεγα 
def g_2(t):
    return np.cos(t) # an kai me vash ton chatzip tha einai 100*cos(t)


# Συστήματα Διαφορικών Εξισώσεων
def f1(t, x, y):
    return x + 2*y + 1

def f2(t, x, y):
    return -x +y + t


N = 250
t = np.linspace(0,5,N+1)
h = t[1] - t[0]

# Άμεση Euler
x = np.zeros(N+1)
y = np.zeros(N+1)

# αρχικές τιμες
x[0] = 2
y[0] = -1

for i in range(N):
    x[i+1] = x[i] + h*f1(t[i],x[i],y[i])
    y[i+1] = y[i] + h*f2(t[i],x[i],y[i])
    
# plt.plot(x,y)
# plt.show()
plt.plot(t,x,t,y)
plt.show()
# print(x[N],y[N]) # Τιμές των προσεγγίσεων στο τέλος

### Πεπλεγμένη Euler
xbe = np.zeros(N+1)
ybe = np.zeros(N+1)

# αρχικές τιμες
xbe[0] = 2
ybe[0] = -1

B = np.eye(2)-h*np.array([[1,2],[-1,1]])### ο πίνακας : I-hA (ειναι ο ίδιος σε καθεβημα)

for i in range(N):
    b = np.array([xbe[i],ybe[i]])+h*np.array([1,t[i+1]]) # δεξιό μέλος για να λύσουμεσε κάθε βημα
    Y = np.linalg.solve(B,b) # λύση γραμμικου συστηματος σε καθε βημα
    
    # Προσεγγίσειςστοεπόμενοσημείο (t_{n+1})
    xbe[i+1] = Y[0]
    ybe[i+1] = Y[1]

# plt.plot(xbe,ybe)
# plt.show()
plt.plot(t,xbe,t,ybe)
plt.show()
print(xbe[N],ybe[N]) # Τιμές των προσεγγίσεων στο τέλος

### Τραπεζίου
xtr = np.zeros(N+1)
ytr = np.zeros(N+1)

xtr[0] = 2
ytr[0] = -1