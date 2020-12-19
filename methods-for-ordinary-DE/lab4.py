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
def y(t):
    return np.exp(0.25-(t-0.5)**2)

def f(t,y):
    return (1-2*t)*y

# για την εύρεση σταθερού σημείου - Πεπλεγμένη
# π.χ. f(x) = lnx − x + 2,x >0. Γράφουμε g(x) = lnx + 2 = x.
# f(x) = x^3 + 4x^2 − 10 = 0 έχει μία ρίζα στο[1,1.5]. Η 
# μέθοδος x=g(x) έχει διαφορετική ταχύτητα σύγκλισης ανάλογα με την επιλογή
# της g(x), π.χ. g(x) = x−x3−4x2+ 10,
def g(t): 
    return 1-2*t


def Euler(N):
    t = np.linspace(0,4,N+1)
    h = t[1]-t[0]

    y_approx = np.zeros(N+1)
    y_approx[0] = 1

    for i in range(N):
        y_approx[i+1] = y_approx[i] + h*f(t[i],y_approx[i])

    # makeGraph(y(t), y_approx, t, "Euler's method for N: " + str(N))
    e1_Euler = max(abs(y(t)-y_approx))
    print(f"Euler's method error maximum for N={N}:          {e1_Euler}")

    return e1_Euler

def Euler_implicit(N):
    t = np.linspace(0,4,N+1)
    h = t[1]-t[0]

    y_approx = np.zeros(N+1)
    y_approx[0] = 1

    for i in range(N):
        y_approx[i+1] = y_approx[i] / (1 - h*g(t[i+1]))

    # makeGraph(y(t), y_approx, t, "Euler's implicit method for N: " + str(N))
    e2_Euler_implicit = max(abs(y(t)-y_approx))
    print(f"Euler's implicit method maximum error for N={N}: {e2_Euler_implicit}")

    return e2_Euler_implicit

def Trapezoidal(N):
    t = np.linspace(0,4,N+1)
    h = t[1]-t[0]

    y_approx = np.zeros(N+1)
    y_approx[0] = 1

    for i in range(N):
        y_approx[i+1] = y_approx[i]*(1 + h*g(t[i]) / 2) / (1 - h*g(t[i+1]) / 2)

    # makeGraph(y(t), y_approx, t, "Trapezoidal method for N: " + str(N)")
    e3_trapezoidal = max(abs(y(t)-y_approx))
    print(f"Trapezoidal method maximum error for N={N}:      {e3_trapezoidal}")

    return e3_trapezoidal

# N = 50
Euler(50)
Euler_implicit(50)
Trapezoidal(50)
print()

# N = 100, 200, 300, 400, 500
N  = list(range(100,501,100))

err_Euler = []
err_Euler_impl = []
err_Trapez = []
for n in N:
    err_Euler.append(Euler(n))
    err_Euler_impl.append(Euler_implicit(n))
    err_Trapez.append(Trapezoidal(n))
    print()

errors = [err_Euler, err_Euler_impl, err_Trapez]
print("Πειραματική ταξη σύγκλισης")
for err in errors:
    for i in range(len(N)-1):
        print(f"{np.log(err[i+1]/err[i])/np.log(N[i]/N[i+1])}")
    print()


# Άσκηση 2 - Μέθοδος σταθερού σημείου
def g_2(t):
    return np.cos(t)

x0 = 1 # αρχική προσέγγιση
tol = 1.e-8
Nmax = 500
k = 0

err = 1. # Θετουμε αρχικά το σφάλμα ίσον με 1 για να ξεκινήσει η διαδικασία
while (err > tol) and (k <= Nmax):
    x = g_2(x0) # επόμενη προσέγγιση
    err = abs(x-x0) # σφάλμα

    k = k+1 # αυξάνουμε τον μετρητή βημάτων
    x0 = x # θέτουμε τη νέα προσέγγιση ως την παλιά για το επόμενο βήμα

print('Η προσέγγιση του σταθερού σημείου είναι:',x)
print('Αριθμός βημάτων:',k)
print()


# Άσκηση 3
def y_3(t):
    return t/(1+t**2)
    
def f_3(t,y):
    return 1/(1+t**2)-2*y**2
    
def g_3(t,yn,x):
    return yn+h*f_3(t,x)

# Πεπλεγμένη Euler (Μη γραμμική f ως προς y)
N = 50
t = np.linspace(0,1,N+1)
h = t[1]-t[0]

y_approx = np.zeros(N+1)
y_approx[0] = 0

tol = 1.e-8
Nmax = N

for i in range(N): # εύρεση σταθερού σημείου
    x0 = y_approx[i] #αρχική προσέγγιση στο i-βημα
    k = 0 
    err = 1. # Θέτουμε αρχικά το σφάλμα ίσον με 1 για να ξεκινήσει η διαδικασία 
    
    while (err > tol) and (k <= Nmax):
        x = g_3(t[i+1],y_approx[i],x0) # επόμενη προσέγγιση
        e = abs(x-x0) # σφάλμα
        k = k+1 # αυξάνουμε τον μετρητή βημάτων
        x0 = x
        
    y_approx[i+1] = x # τελειώνει η επανάληψη σταθερού σημείου και θέτουμε τη προσεγγιση που βρήκαμε ως την προσέγγιση της λύσης στο σημείο t[i+1]

makeGraph(y_3(t), y_approx, t, "Euler's implicit method - Άσκηση 3")

# για Ν = 100, 200, 300, 400, 500
N  = list(range(100,501,100))
err_y = np.zeros(len(N))

tol = 1.e-8
Nmax = N[-1]

for j in range(len(N)):
    y_approx = np.zeros(N[j]+1)
    y_approx[0] = 0

    t =np.linspace(0,1,N[j]+1)
    h = t[1] - t[0]

    for i in range(N[j]):
        x0 = y_approx[i] # αρχική προσέγγιση στο i-βήμα
        k = 0

        err = 1 # θέτουμε αρχικά το σφάλμαίσον με 1 για να ξεκινήσει η διαδικασία

        while (err > tol) and (k <= Nmax):
            x = g_3(t[i+1],y_approx[i],x0) # επόμενη προσέγγιση
            e = abs(x-x0)
            k = k+1 # αυξάνουμε τον μετρητή βημάτων
        x0 = x
        
    y_approx[i+1] = x # τελειώνει η επανάληψη σταθερού σημείου και θέτουμε τη προσεγγιση που βρήκαμε ως την προσέγγιση της λύσης στο σημείο t[i+1]

    err_y[j] = max(abs(y_3(t) - y_approx))
    print(f"Μέγιστο σφάλμα για Ν={N[j]}: {err_y[j]}")

print('\nΠειραματική τάξη σύγκλισης μεθόδου Πεπλεγμένη Euler - Άσκησης 3')
for i in range(len(N)-1):
    print(np.log(err_y[i+1]/err_y[i])/np.log(N[i]/N[i+1]))

# Άσκηση 4
# Επαναλάβετε το πρόβλημα για τη μέθοδο του τραπεζίου.

# Τραπεζιου (Μη γραμμική f ως προς y)
# Η αλλαγή σε σχέση με την προηγούμενη άσκηση είναι ο ορισμός της συνάρτησης που δίνει
# τη μέθοδο. Ο τύπος της μεθόδου του τραπεζίου είναι
#               y_{n+1} = y_n + (h/2)*(f(t_{n+1}, y_{n+1}) + f(t_{n}, y_{n}))

def g_4(t, yn, x):
    return yn + (h/2)*(f_3(t-h, yn) + f_3(t, x))

# ουσιαστικά έχω επανάληψη προηγούμενης διαδκασίας με διαφορετικό g
N = 50
t = np.linspace(0,1,N+1)
h = t[1]-t[0]

y_approx = np.zeros(N+1)
y_approx[0] = 0

tol = 1.e-8
Nmax = N

for i in range(N): # εύρεση σταθερού σημείου
    x0 = y_approx[i] #αρχική προσέγγιση στο i-βημα
    k = 0 
    err = 1. # Θέτουμε αρχικά το σφάλμα ίσον με 1 για να ξεκινήσει η διαδικασία 
    
    while (err > tol) and (k <= Nmax):
        x = g_4(t[i+1],y_approx[i],x0) # επόμενη προσέγγιση
        err = abs(x-x0) # σφάλμα
        k = k+1 # αυξάνουμε τον μετρητή βημάτων
        x0 = x
        
    y_approx[i+1] = x # τελειώνει η επανάληψη σταθερού σημείου και θέτουμε τη προσεγγιση που βρήκαμε ως την προσέγγιση της λύσης στο σημείο t[i+1]

makeGraph(y_3(t), y_approx, t, "Trapezoidal method - Άσκηση 4")

# για Ν = 100, 200, 300, 400, 500
N  = list(range(100,501,100))
err_y = np.zeros(len(N))

tol = 1.e-8
Nmax = N[-1]

for j in range(len(N)):
    y_approx = np.zeros(N[j]+1)
    y_approx[0] = 0

    t =np.linspace(0,1,N[j]+1)
    h = t[1] - t[0]

    for i in range(N[j]):
        x0 = y_approx[i] # αρχική προσέγγιση στο i-βήμα
        k = 0

        err = 1 # θέτουμε αρχικά το σφάλμαίσον με 1 για να ξεκινήσει η διαδικασία

        while (err > tol) and (k <= Nmax):
            x = g_4(t[i+1],y_approx[i],x0) # επόμενη προσέγγιση
            e = abs(x-x0)
            k = k+1 # αυξάνουμε τον μετρητή βημάτων
        x0 = x
        
    y_approx[i+1] = x # τελειώνει η επανάληψη σταθερού σημείου και θέτουμε τη προσεγγιση που βρήκαμε ως την προσέγγιση της λύσης στο σημείο t[i+1]

    err_y[j] = max(abs(y_3(t) - y_approx))
    print(f"Μέγιστο σφάλμα για Ν={N[j]}: {err_y[j]}")

print('\nΠειραματική τάξη σύγκλισης μεθόδου Τραπεζίου - Άσκηση 4')
for i in range(len(N)-1):
    print(np.log(err_y[i+1]/err_y[i])/np.log(N[i]/N[i+1]))