# Euler's method
# Για την αριθμητική επίλυση ενός προβλήματος αρχικών τιμών (Π.Α.Τ.)
#               y'(t) = f(t,y(t)), tε[a,b]
#               y(0) = y0
# Έστω ένας ομοιόμορφος διαμερισμός του [a,b], στα σημεία tn = a + nh, n=0,1,..,N
# με βήμα h = (b-a)/N. Θεωρούμε τη μέθοδο Euler. Υπολογίζουμε τις τιμές yn που
# αποτελούν προσεγγίσεις στις τιμές y(tn), n = 0,.., N
#               yn+1 = yn + hf(tn,yn), n = 0,.. ,N-1

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


def err_Euler(l,NN):
    t = np.linspace(0,5,NN+1)
    h = t[1]-t[0]
    y_approx = np.zeros(NN+1)

    y_approx[0] = 1 # Αρχική τιμή
    for i in range(NN):
        y_approx[i+1] = y_approx[i] + h*f(t[i],l, y_approx[i])

    
    # makeGraph(y(t,l), y_approx, t, "for N=" + str(NN))
    return max(abs(y(t,l) - y_approx))

def err_Taylor(l,NN):
    t = np.linspace(0,5,NN+1)# Σημεία διαμερισμουστο [0,5], Ν+1
    h = t[1]-t[0]
    y_approx = np.zeros(NN+1)

    y_approx[0] = 1 # Αρχική τιμή
    for i in range(NN):
        y_approx[i+1] = y_approx[i] + h*f(t[i],l,y_approx[i]) + g(t[i],l,y_approx[i])*((h**2)/2) # Μέθοδος TS(2)

    # makeGraph(y(t,l), y_approx, t, "for N=" + str(NN))
    return max(abs(y(t,l) - y_approx))

# Άσκηση 1:
# Έστω y(t) = exp(-λt) + sin(2t) στο [0,5] η λύση του προβλήματος αρχικών τιμών (Π.Α.Τ.)
#               y'(t) = -λy(t) + 2cos(2t) + λsin(2t), tε[0,5]
#               y(0) = 1
# Θεωρείστε ότι λ = 50. Για Ν = 50, κατασκευάστε τις προσεγγίσεις που δίνει η μέθοδος
# του Euler, δημιουργείστε τη γραφική παράσταση της προσεγγιστικής λύσης και της
# ακριβούς στο διάστημα [0,5]. 
# Στη συνέχεια βρείτε το σφάλμα max(0<=n<=N) |yn - y(tn)|.
# Επαναλάβετε για Ν = 500

def y(t,l): # ακριβής λύση
    return np.exp(-l*t) + np.sin(2*t)

def f(t,l,y):
    return -l*y + 2*np.cos(2*t) + l*np.sin(2*t)

print("Μέθοδος Euler")
N = [50, 500]
l = 50

for i in N:
    print(f"for N: {i} error is: {err_Euler(l,i)}")

print()

# Πειραματική εκτίμηση της τάξης σύγκλισης
# Γνωρίζουμε ότι για τη μέθοδο Euler το σφάλμα της μεθόδου ικανοποιεί
#               max(0<=n<=N) |yn - y(tn)| <= C(h^p), με p=1
# Υπολογίζοντας το σφάλμα
#               errN = max(0<=n<=N) |yn - y(tn)|
# για δύο διαφορεικές διαμερίσεις με Ν1<Ν2, 
# η πειραματική τάξη σύγκλισης ορίζεται ως
#               p = ln(errN2/errN1)/ln(N1/N2)

# Άσκηση 2:
# Θεωρείστε τις διαμερίσεις του [0,5] με Ν =10,20,30,..100.
# Υπολογίστε τα σφάλματα errN και βρείτε τους λόγους που χρησιμοποιούμε για
# την πειραματική τάξη σύγκλισης της μέθοδου του Euler. Είναι p ~= 1;

N = list(range(10,101,10))

errors_Euler100 = []
for i in range(len(N)):
    errors_Euler100.append(err_Euler(l,N[i]))
    print("Μέγιστο σφάλμα για N=", N[i],':', errors_Euler100[i])

print("\nΠειραματική τάξη σύγκλισης")
for i in range(len(N)-1):
    print(f"Για Ν1={N[i]} και Ν2={N[i+1]}, το p: {np.log(errors_Euler100[i+1]/errors_Euler100[i])/np.log(N[i]/N[i+1])}")

print()

# Άσκηση 3:
# Επαναλάβετε την προηγούμενη άσκηση αλλά τώρα θεωρείστε τις διαμερίσεις του
# [0,5] με Ν = 200,210,220,230,.., 300. Υπολογίστε τα σφάλματα errN και βρείτε τους λόγους
# που χρησιμοποιούμε για την πειραματική τάξη σύγκλισης της μεθόδου του Euler.
# Είναι p ~= 1;

N = list(range(200,301,10))

errors_Euler300 = []
for i in range(len(N)):
    errors_Euler300.append(err_Euler(l,N[i]))
    print("Μέγιστο σφάλμα για N=", N[i],':', errors_Euler300[i])

print("\nΠειραματική τάξη σύγκλισης")
for i in range(len(N)-1):
    print(f"Για Ν1={N[i]} και Ν2={N[i+1]}, το p: {np.log(errors_Euler300[i+1]/errors_Euler300[i])/np.log(N[i]/N[i+1])}")
print()

# Μέθοδος Taylor(2)
# Για την αριθμητική επίλυση ενός προβλήματος αρχικών τιμών (Π.Α.Τ.)
#               y'(t) = f(t,y(t)), tε[a,b]
#               y(0) = y0
# Έστω ένας ομοιόμορφος διαμερισμός του [a,b], στα σημεία tn = a + nh, n=0,1,..,N
# με βήμα h = (b-a)/N. Θεωρούμε τη μέθοδο Taylor(2) όπου υπολογίζουμε τις τιμές yn
# που αποτελούν προσεγγίσεις στις τιμές y(tn), n=0,1,...N σύμφωνα με
#               y[n+1] = y[n] + h*f(tn,yn) + (h^2)/2 * g(tn,yn), n=0,1,..N
#                                           όπου g(t,y(t)) = d/dt(f(t,y(t)))

# Άσκηση 4:
# Έστω y(t) = exp(-λt) + sin(2t) στο [0,5] η λύση του προβλήματος αρχικών τιμών (Π.Α.Τ.)
#               y'(t) = -λy(t) + 2cos(2t) + λsin(2t), tε[0,5]
#               y(0) = 1
# Θεωρείστε ότι λ = 50.
# 1. Για Ν = 50 κατασκευάστε τις προσεγγίσεις που δίνει η μέθοδος του Taylor(2),
# δημιουργήστε τη γραφική παράσταση της προσεγγιστικής λύσης και της ακριβής στο
# διάστημα [0,5]. Στη συνέχεια βρείτε το σφάλμα max(0<=n<=N) |yn - y(tn)| και 
# συγκρίνεται με το αντίστοιχο σφάλμα για τη μέθοδο του Euler.
# 2. Θεωρείστε τις διαμερίσεις του [0,5] με Ν =10,20,30,..100.
# Υπολογίστε τα σφάλματα errN και βρείτε τους λόγους που χρησιμοποιούμε για
# την πειραματική τάξη σύγκλισης της μέθοδου του Taylor. Είναι p ~= 2;
# Συγκρίνετε τα σφάλματα με τα αντίστοιχα για τη μέθοδο του Euler
# 3. Επαναλάβετε για Ν = 200,210,220,230,.., 300.

def g(t,l,y):
    return -l*f(t,l,y)-4*np.sin(2*t)+l*np.cos(2*t)

print("Μέθοδος Taylor 2")
# 4.1
N = 50
print(f"for N:{N}\nTaylor's error is: {err_Taylor(l,N)}\nEuler's error is: {err_Euler(l,N)}\n")

# 4.2
N = list(range(10,101,10))

errors_Taylor100 = []
for i in range(len(N)):
    errors_Taylor100.append(err_Euler(l,N[i]))
    print("Μέγιστο σφάλμα για N=", N[i],':', errors_Taylor100[i])

for i in range(len(N)):
    print(f"for N:{N[i]}\nTaylor's error is: {errors_Taylor100[i]}\nEuler's error is: {errors_Euler100[i]}\n")

print("Πειραματική τάξη σύγκλισης")
for i in range(len(N)-1):
    print(f"Για Ν1={N[i]} και Ν2={N[i+1]}, το p: {np.log(errors_Taylor100[i+1]/errors_Taylor100[i])/np.log(N[i]/N[i+1])}")

print()

# 4.3
N = list(range(200,301,10))

errors_Taylor300 = []
for i in range(len(N)):
    errors_Taylor300.append(err_Taylor(l,N[i]))
    print("Μέγιστο σφάλμα για N=", N[i],':', errors_Taylor300[i])

for i in range(len(N)):
    print(f"for N:{N[i]}\nTaylor's error is: {errors_Taylor300[i]}\nEuler's error is: {errors_Euler300[i]}\n")

print("Πειραματική τάξη σύγκλισης")
for i in range(len(N)-1):
    print(f"Για Ν1={N[i]} και Ν2={N[i+1]}, το p: {np.log(errors_Taylor300[i+1]/errors_Taylor300[i])/np.log(N[i]/N[i+1])}")

print()

# Συστήματα Διαφορικών Εξισώσεων
# Θεωρούμε τώρα το ακόλουθο σύστημα διαφορικών εξισώσεων
#               x'(t) = -y(t)
#               y'(t) = x(t)
# για tε[0,2π], x(0)=1, y(0)=0
# και ακριβή λύση x(t) = cos(t), y(t) = sin(t)

# Άσκηση 5
# Θεωρείστε μια διαμέριση του [0,2π] με Ν = 100 και εφαρμόστε τη μέθοδο του Euler για
# συστήματα για να βρείτε τις προσεγγίσεις (xn,yn) των (x(tn), y(tn)), n = 0,1,...N
# 1. Δημιουργείστε τη γραφική παράσταση των λύσεων ως προς το χρόνο t
# 2. Δημιουργείστε τη γραφική παράσταση ανάμεσα στις (x(t),y(t))
# 3. Βρείτε το max(0<=n<=N) (xn**2(t) + yn**2(t)).
# Ισχύει ότι x**2(t) + y**2(t) = 1; Στη συνέχεια σημιουργείστε τη γραφική παράσταση 
# των 2 προσεγγίσεων (xn,yn) στο πεδίο xy. Δημιουργείται κύκλος;

def x_exact(t):
    return np.cos(t)

def y_exact(t):
    return np.sin(t) 
    
# 5.1
N = 100
t = np.linspace(0,2*np.pi,N+1)

makeGraph(x_exact(t),y_exact(t), t, "")

# 5.2
plt.title("x_exact-y_exact")
plt.plot(x_exact(t),y_exact(t))
plt.show()

# 5.3
def f1(t,x,y):
    return -y

def f2(t,x,y):
    return x 
    
N = 100
t = np.linspace(0,2*np.pi,N+1)
h = t[1] - t[0]

# Θεσεις για να αποθηκευσω τις προσεγγισεις
x = np.zeros(N+1)
y = np.zeros(N+1)
x[0]=1
y[0]=0

# Μεθοδος Euler για συστήματα
for i in range(N):
    x[i+1] = x[i] + h*f1(t,x[i],y[i])
    y[i+1] = y[i] + h*f2(t,x[i],y[i])

# Στην Numpy η παρακάτω πράξη γίνεται σε κάθε στοιχείο των διανυσμάτων x,y
s = max(x**2+y**2)
print(s)

makeGraph(x,x_exact(t), t, "x(t)")
makeGraph(y,y_exact(t), t, "y(t)")

plt.title("x_approx-y_approx")
plt.plot(x,y)
plt.show()