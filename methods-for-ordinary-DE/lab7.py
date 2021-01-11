# Για επίλυση ΠΑΤ με μη γραμμική εξίσωφη f() με χρήση πολυβηματικών μεθόδων χρησιμοποιούμε τη μέθοδο του σταθερού σημείου
# για τον υπολογίσμο της προσεγγισης (εδώ συμβολζουμε με x) τη χρονική στιγμή t
# απ το μεθοδο σταθερού παίρνω x για το i+2, αρα το y(i+2)  το παίρνω με t και τις 
# υπόλοιπες προσεγγίσεις y(i+1) με t-h και y(i) για t-2h

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

def y_exact(t):
    return ((t**2)+1)**2

def f(t,y):
    return 4*t*(y**(1/2))

def g(t,h,x,y1,y0, method):
    # Οι μεταβλητές y1, y0 είναι οι προσεγγίσεις στα 2 προηγούμενα βήματα
    # Η μεταβλητή x είναι η προσέγγιση που έχουμε ήδη βρει στον αλγόριθμο σταθερου σημείου

    if method == 0:
        s = y1 + h*((5/12)*f(t,x) + (2/3)*f(t-h,y1) - (1/12)* f(t-2*h,y0)) # AM2
    else:
        s = (1/3)*(4*y1-y0)+(2*h/3)*f(t,x) # BDF2
    return s

def AM2(t, N):
    h = t[1]-t[0]

    y = np.zeros(N+1)
    y[0] = 1
    y[1] = y[0] + h*f(t[0], y[0]) # πρώτο βήμα με άμεση Euler

    tol = 1.e-3 # ακρίβεια σφάλματος 
    Nmax = 100 # μέγιστος αριθμός επαναλήψεων για τον υπολογισμό της λύσης

    for i in range(N-1):
        x0 = y[i] # αρχική προσέγγιση στο i-βήμα
        k = 0 # μετρητή βημάτων
        err = 1. # θέτουμε αρχικά το σφάλμα ίσον με 1 για να ξεκινήσει η διαδιασία

        while (err > tol) and (k <= Nmax):
            x = g(t[i+2], h, x0, y[i+1],y[i], 1)
            err = abs(x-x0)
            k += 1
            x0 = x

        y[i+2] = x # τελειώνει η επανάληψη σταθερού σημείου και θετούμε τη προσέγγιση της y ως προσέγγιση της λύσης στο σημείο t[i+1]
    
    # makeGraph(y_exact(t), y, t, "AM2 method for N: " + str(N))
    return y

def BFD(t, N):
    h = t[1]-t[0]

    y = np.zeros(N+1)
    y[0] = 1
    y[1] = y[0] + h*f(t[0], y[0]) # πρώτο βήμα με άμεση Euler

    tol = 1.e-3 # ακρίβεια σφάλματος 
    Nmax = 100 # μέγιστος αριθμός επαναλήψεων για τον υπολογισμό της λύσης

    for i in range(N-1):
        x0 = y[i] # αρχική προσέγγιση στο i-βήμα
        k = 0 # μετρητή βημάτων
        err = 1. # θέτουμε αρχικά το σφάλμα ίσον με 1 για να ξεκινήσει η διαδιασία

        while (err > tol) and (k <= Nmax):
            x = g(t[i+2], h, x0, y[i+1],y[i], 1)
            err = abs(x-x0)
            k += 1
            x0 = x

        y[i+2] = x # τελειώνει η επανάληψη σταθερού σημείου και θετούμε τη προσέγγιση της y ως προσέγγιση της λύσης στο σημείο t[i+1]
    
    # makeGraph(y_exact(t), y, t, "BDF method for N: " + str(N))
    return y
    

# TODO: χρησιμοποιώ generator για να φτιάξω N αλλά αυτός το θέλει απλά με το χέρι
def stepDown(n):
    while n < 161:
        yield n
        n = n*2

n = 20 # αρχικό
N = []
for i in stepDown(n):
    N.append(i)

for i in range(len(N)):
    t = np.linspace(0,2,N[i]+1)
    print(f"for N: {N[i]} max_error of AM2 is {max(abs(y_exact(t)-AM2(t, N[i])))} and for BFD is {max(abs(y_exact(t) - BFD(t, N[i])))}")


# TODO: with comprehension
tt = [x / 10.0 for x in range(5, 21, 5)]
# TODO: or use lambda / map: not working
# t = map(lambda x: x/10.0, range(5, 21, 5))
# print(t)
# οπως το έχω υλοποιήσει δεν μπορώ άμεσα να βρω τις τιμές για αυτά τα t
# ουσιαστικά χωρίζεις το Ν/4 και παιρνεις το κατω ακεραιο μερος ιντ() και μετα για καθε σημειο τη μ
# μέθοδο και για αυτα τα 4-5 είναι αυτα που ψάχνω

# Άσκηση 2
# ιδιο ΠΑΤ αλλα για τη μέθοδο

def g1(t, h, x,y1,y0): # μη συνεπής/ θα έχει διαφορετικό g()
    return y1 + (h/12) * (4*f(t,x) + 8*f(t-h,y1) - f(t-2*h,y0)) # gia t (~i+2) y[i+2] = x, gia t-h (i+1) y[i+1] = 1,  gia t-2h (i) y[i] = y0

def method1(t, N):
    h = t[1]-t[0]

    y = np.zeros(N+1)
    y[0] = 1
    y[1] = y[0] + h*f(t[0], y[0]) # πρώτο βήμα με άμεση Euler

    tol = 1.e-3 # ακρίβεια σφάλματος 
    Nmax = 100 # μέγιστος αριθμός επαναλήψεων για τον υπολογισμό της λύσης

    for i in range(N-1):
        x0 = y[i] # αρχική προσέγγιση στο i-βήμα
        k = 0 # μετρητή βημάτων
        err = 1. # θέτουμε αρχικά το σφάλμα ίσον με 1 για να ξεκινήσει η διαδιασία

        while (err > tol) and (k <= Nmax):
            x = g1(t[i+2], h, x0, y[i+1],y[i])
            err = abs(x-x0)
            k += 1
            x0 = x

        y[i+2] = x # τελειώνει η επανάληψη σταθερού σημείου και θετούμε τη προσέγγιση της y ως προσέγγιση της λύσης στο σημείο t[i+1]
    
    # makeGraph(y_exact(t), y, t, "method1 for N: " + str(N))
    return y

n = 20 # αρχικό
N = []
for i in stepDown(n):
    N.append(i)

for i in range(len(N)):
    t = np.linspace(0,2,N[i]+1)
    print(f"for N: {N[i]} max_error of new method 1 is {max(abs(y_exact(t)-method1(t, N[i])))}")

# Άσκηση 3
# Παρατηρώ ότι η τελευταία μέθοδος τα σφάλματα της τείνουν στο 0 σε σχεση με την περιεργη.
# Η μέθοδος που θεωρήσαμε στην προηγούμενη άσκηση είναι ευσταθής αλλά δεν είναι συνεπής. Επομένως δεν
# είναι συγκλίνουσα και άρα οι προσεγγίσεις yn για το ΠΑΤ της 2 δεν θα συγκλίνουν στην πραγματική λύση
# Για την ακριβή η περιέργη ειναι λύση του ΠΑΤ y'(t)=(11/12)*4*t*(y(t))**(1/2), y(0)=1
# WARNIG: ΔΗΛΑΔΗ ΟΙ ΠΡΟΣΕΓΓΙΣΕΙΣ ΠΟΥ ΠΑΙΡΝΟΥΜΕ ΑΠΟ ΜΙΑ ΜΗ ΣΥΝΕΠΗ ΛΥΣΗ ΟΔΗΓΟΥΝ ΣΤΗ ΛΥΣΗ ΕΝΟΣ ΑΛΛΟΥ ΠΡΟΒΛΗΜΑΤΟΣ

def y_perierh(t):
    return ((11/12)*(t**2)+1)**2
    
for i in range(len(N)):
    t = np.linspace(0,2,N[i]+1)
    print(f"for N: {N[i]} max_error of new method + periergis is {max(abs(y_perierh(t)-method1(t, N[i])))}")

# Άσκηση 4
def y2_exact(t):
    return np.exp(-10*t)

def f(t,y):
    return -10*y

def g2(t, h, x,y1,y0):
    return -y1 + 2*y0 + (h/4)*(f(t,x) + 8*f(t-h, y1) + 3*f(t-2*h, y0))

def method2(t, N):
    h = t[1]-t[0]

    y = np.zeros(N+1)
    y[0] = 1
    y[1] = y[0] + h*f(t[0], y[0]) # πρώτο βήμα με άμεση Euler

    tol = 1.e-3 # ακρίβεια σφάλματος 
    Nmax = 100 # μέγιστος αριθμός επαναλήψεων για τον υπολογισμό της λύσης

    for i in range(N-1):
        x0 = y[i] # αρχική προσέγγιση στο i-βήμα
        k = 0 # μετρητή βημάτων
        err = 1. # θέτουμε αρχικά το σφάλμα ίσον με 1 για να ξεκινήσει η διαδιασία

        while (err > tol) and (k <= Nmax):
            x = g2(t[i+2], h, x0, y[i+1],y[i])
            err = abs(x-x0)
            k += 1
            x0 = x

        y[i+2] = x # τελειώνει η επανάληψη σταθερού σημείου και θετούμε τη προσέγγιση της y ως προσέγγιση της λύσης στο σημείο t[i+1]
    
    # makeGraph(y_exact(t), y, t, "method2 for N: " + str(N))
    return y

N = [20,40,80,160]
for i in range(len(N)):
    t = np.linspace(0,2,N[i]+1)
    print(f"for N: {N[i]} max_error of method 2 is {max(abs(y_exact(t)-method2(t, N[i])))}")

# Παρατήρηση: η μέθοδος που θεωρήσαμε δεν είναι ευσταθής αλλά είναι συνεπής. Επομένως δεν είναι συγκλίνουσα
# άρα οι προσεγγίσεις που βρήκα για το ΠΑΤ δεν θα συγκλίνουν στη λύση του ΠΑΤ

# _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ #
# Συστήματα Διαφορικών Εξισώσεων

def f1(t,x,y):
    return (0.05)*x*(1-0.01*y)
    
def f2(t,x,y):
    return (0.1)*y*(0.005*x-2)

NN = 2500
t = np.linspace(0,150,NN+1)
h = t[1] - t[0]

x = np.zeros(NN+1)
y = np.zeros(NN+1)

x[0] = 500
y[0] = 100

### Αμεση Euler
for i in range(NN):
    x[i+1] = x[i] + h*f1(t,x[i],y[i])
    y[i+1] = y[i] + h*f2(t,x[i],y[i])

### Γραφική παράσταση των 2 συναρτήσεων - Οι ακριβείς λύσεις είναι περιοδικές συναρτήσεις
plt.plot(t,x,t,y)
plt.show()

### Γραφική παράσταση xy-επιπεδο
plt.plot(x,y)
plt.show()

## Πληθυσμός στο τελος
print(x[N],y[N])