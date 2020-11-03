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

# Άσκηση 1:
# Έστω y(t) = exp(-t) + sin(t) στο [0,10]. Δημιουργήστε μια διαμέριση του [0,10] με 51
# σημεία tn, n=0,1,..,50 και χρησιμοποιήστε τη βιβιοθήκη matplotlib για να σχηματίσετε
# το γράφημα της y(t).

def y(t):
    return np.exp(-t) + np.sin(t)

# t = np.linspace(0,10,51)
# plt.plot(t,y(t))
# plt.xlabel('t')
# plt.ylabel('y')
# plt.legend(['exp(-t)+sin(t)'])
# plt.show()

# Άσκηση 2:
# Έστω y(t) = exp(-t) + sin(t) στο [0,10] η λύση της
#               y'(t) = -y(t)+cos(t)+sin(t), tε[0,10]
#               y(0) = y0
# Ορίστε την συνάρτηση δύο μεταβλητών:  f(t,y) = -y + cos(t) + sint(t)
# 1. για βήμα h= 0.5 και y0 = 1, υπολογίστε με την μέθοδο του Euler τη προσέγγιση y10
# 2. για Ν = 50 κατασκευάστε τις προσεγγίσεις yn με Euler και δημιουργήστε γραφική
# παράσταση της προσεγγιστικής λύσης.

def f(t,y):
    return -y + np.cos(t) + np.sin(t)

N = 50 #21 βηματα, h=0.5
t = np.linspace(0,10,N+1)
h = t[1]-t[0]

y_approx = np.zeros(N+1)

y_approx[0]=1 # Αρχική τιμή
for i in range(N):
    y_approx[i+1] = y_approx[i] + h*f(t[i],y_approx[i])

print(y_approx[10])
print(t[10])
plt.plot(t,y_approx)
plt.show()

# Άσκηση 3:
# Για το παραπάνω Π.Α.Τ. υπολογίστε το σφάλμα ανάμεσα στην ακριβή λύση και την
# προσεγγιστική στο σημείο t = 10, όταν N = 50,100,200,400. Αν τα σφάλματα είναι
# αντίστοιχα err1, err2, err3, err4. Διαπιστώστε αν ο λόγος erri/erri+1 είναι
# περίπου ίδιος για i = 1,2,3

N =[50,100,200,400]

err = np.zeros(len(N))

for j in range(len(N)):
    t = np.linspace(0,10,N[j]+1)
    h = t[1] - t[0]

    y_approx = np.zeros(N[j]+1)
    y_approx[0] = 1

    for i in range(N[j]):
        y_approx[i+1] = y_approx[i] + h*f(t[i],y_approx[i])

    err[j] = abs(y_approx[-1]-y(10)) # σφαλματα για αντιστοιχη διαμεριση

for j in range(3):
    print(err[j+1]/err[j])
print()

# Άσκηση 4:
# Στην προηγούμενη άσκηση3 θεωρούμε τώρα ως
#               erri = max(0<=n<=N) |yn_approx - y(tn)|, N = 50,100,200,400
# και διαπιστώστε αν ο λόγος erri/erri+1 είναι περίπου ίδιος για i = 1,2,3

err = np.zeros(len(N))

for j in range(len(N)):
    t = np.linspace(0,10,N[j]+1)
    h = t[1] - t[0]

    y_approx = np.zeros(N[j]+1)
    y_approx[0] = 1

    for i in range(N[j]):
        y_approx[i+1] = y_approx[i] + h*f(t[i],y_approx[i])

    err[j] = max(abs(y_approx - y(t))) # σφαλματα για αντιστοιχη διαμεριση.

for j in range(3):
    print(err[j+1]/err[j])
print()