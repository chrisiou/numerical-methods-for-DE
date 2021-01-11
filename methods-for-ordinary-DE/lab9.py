# Άμεσες μέθοδοι Runge-Kutta
import numpy as np
import matplotlib.pyplot as plt

def f(t,y):
    return 4*t*y**(1/2)
    
def y_exact(t):
    return (t**2+1)**2

def runge(t,y0):
    ## βελτιωμένη Euler
    a = 1/2
    N = len(t)-1
    y = np.zeros(N+1)

    h = t[1]-t[0]
    y[0] = y0

    for i in range(N):
        t2 = a # το βήμα για το 2ο ενδιάμεσο σημείο

        ### Υπολογισμός 1ου ενδιάμεσου βήματος και της αντίστοιχης τιμής της f
        ky1 = f(t[i],y[i])
        y1 = y[i]

        ### Υπολογισμός 2ου ενδιάμεσου βηματος και της αντίστοιχης τιμής της f
        y2 = y[i] + h*a*ky1
        ky2 = f(t[i]+t2*h,y2)
        
        #### Υπολογισμός της επόμενης προσέγγισης (χρησιμοποιούμε τις ενδιάμεσες προσεγγισεις π 
        y[i+1] = y[i] + h*ky2

    return y


N = [20,40,80,160,320]

#### Υπολογισμός τάξης ακρίβειας για την βελτιωμένη Euler
print("βελτιωμένη Euler")
err = np.zeros(len(N))
for j in range(len(N)):
    t = np.linspace(0,2,N[j]+1)
    h = t[1]-t[0]
    y0 = 1
    y = runge(t,y0)
    
    err[j] = max(y_exact(t)-y)

for i in range(len(N)-1):
    print(f"p for N1= {N[i]} and N2= {N[i+1]} is:{np.log(err[i+1]/err[i])/np.log(N[i]/N[i+1])}")

# Άσκηση 2
def three_steps(t, y0):
    N = len(t)-1
    y = np.zeros(N+1)

    h = t[1] - t[0]
    y[0] = y0

    for i in range(N):
        # πρώτο ενδιάμεσο σημείο
        ky1 = f(t[i],y[i])
        y1 = y[i]

        # δεύτερο ενδιάμεσο σημείο
        y2 = y[i] + h*(1/3)*ky1
        ky2 = f(t[i] + (1/3)*h, y2)

        # τρίτο ενδιάμεσο σημείο
        y3 = y[i] + h*(2/3)*ky2
        ky3 = f(t[i] + (2/3)*h, y3)

        # επόμενη προσέγγιση με χρήση των ενδιάμεσων
        y[i+1] = y[i] + h*((1/4)*ky1 + 0*ky2 + (3/4)*ky3)

    return y

#### Υπολογισμός τάξης ακρίβειας για τη 3 σταδίων
print("Runge-Kutta 3 steps")
err = np.zeros(len(N))
for j in range(len(N)):
    t = np.linspace(0,2,N[j]+1)
    h = t[1]-t[0]
    y0 = 1
    y = three_steps(t,y0)
    
    err[j] = max(y_exact(t)-y)

for i in range(len(N)-1):
    print(f"p for N1= {N[i]} and N2= {N[i+1]} is:{np.log(err[i+1]/err[i])/np.log(N[i]/N[i+1])}")


# Άσκηση 3
def four_steps(t, y0):
    N = len(t)-1
    y = np.zeros(N+1)

    h = t[1] - t[0]
    y[0] = y0

    for i in range(N):
        # πρώτο ενδιάμεσο σημείο
        ky1 = f(t[i],y[i])
        y1 = y[i]

        # δεύτερο ενδιάμεσο σημείο
        y2 = y[i] + h*(1/2)*ky1
        ky2 = f(t[i] + (1/2)*h, y2)

        # τρίτο ενδιάμεσο σημείο
        y3 = y[i] + h*(1/2)*ky2
        ky3 = f(t[i] + (1/2)*h, y3)

        # τέταρτο ενδιάμεσο σημείο
        y4 = y[i] + h*1*ky3
        ky4 = f(t[i] + 1*h, y4)

        # επόμενη προσέγγιση με χρήση των ενδιάμεσων
        y[i+1] = y[i] + h*((1/6)*ky1 + (1/3)*ky2 + (1/3)*ky3 + (1/6*ky4))

    return y

#### Υπολογισμός τάξης ακρίβειας για τη 4 σταδίων
print("Runge-Kutta 4 steps")
err = np.zeros(len(N))
for j in range(len(N)):
    t = np.linspace(0,2,N[j]+1)
    h = t[1]-t[0]
    y0 = 1
    y = four_steps(t,y0)
    
    err[j] = max(y_exact(t)-y)

for i in range(len(N)-1):
    print(f"p for N1= {N[i]} and N2= {N[i+1]} is:{np.log(err[i+1]/err[i])/np.log(N[i]/N[i+1])}")

# _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ #
# Πεπλεγμένες μέθοδοι Runge-Kutta

# Ημιπεπλεγμένες μέθοδοι
print("\nΗμιπεπλεγμένες μέθοδοι")
# Άσκηση 4 #TODO: kati paei lathos gt bgazw taksi akrivias ~= 0.005 enw thelw 3 th tuxi mou
def impl_2_steps(t, y0):
    ## 2 σταδιων
    m = 1/2 + np.sqrt(3)/6
    a_11 = m
    # a_12 = 0
    a_21 = 1-2*m
    a_22 = m 
    b1 = 1/2
    b2 = 1/2 
    t1 = m
    t2 = 1-m

    # Εφαρμόζουμε μια διαδικασία σταθερού σημείου όπως κάναμε για τις πεπλεγμένες
    # μεθόδους π.χ. Euler, Τραπεζιου, BDF
    # Για κάθε ένα ενδιάμεσο βήμα ορίζουμε την κατάλληλη συνάρτηση για να βρούμε το σταθερό σημείο
    
    def g1(t1,yn,x):
        return yn + h*a_11*f(t1,x)

    def g2(t2,yn,k1,x):
        return yn + h*(a_21*k1 + a_22*f(t2,x))

    N = len(t)-1
    y = np.zeros(N+1)
    h = t[1]-t[0]
    y[0] = y0
    tol = 1.e-5
    Nmax = 100
    
    for i in range(N):
        ### υπολογισμός 1-σταδιου
        tn1 = t[i]+t1*h
        tn2 = t[i]+t2*h

        x0 = y[i] #αρχική προσέγγιση στο i-βημα
        k = 0
        err = 1. # Θέτουμε αρχικά το σφάλμα ίσον με 1 για να ξεκινήσει η διαδικασία
        
        while (err > tol) and (k <= Nmax):
            x = g1(tn1, y[i], x0) # επόμενη προσέγγιση
            err = abs(x-x0) # σφάλμα
            k = k+1 # αυξάνουμε τον μετρητή βημάτων
            x0 = x

        y1 = x

    for i in range(N):
        ### υπολογισμός 2-σταδιου
        tn1 = t[i]+t1*h
        tn2 = t[i]+t2*h

        x0 = y1 #αρχική προσέγγιση στο i-βημα
        k = 0
        err = 1. # Θέτουμε αρχικά το σφάλμα ίσον με 1 για να ξεκινήσει η διαδικασία
        
        while (err > tol) and (k <= Nmax):
            x = g2(tn2, y[i], y1, x0) # επόμενη προσέγγιση
            err = abs(x-x0) # σφάλμα
            k = k+1 # αυξάνουμε τον μετρητή βημάτων
            x0 = x

        y[i+1] = x

    return y

print("half impl Runge-Kutta 2 steps")
err = np.zeros(len(N))
for j in range(len(N)):
    t = np.linspace(0,2,N[j]+1)
    h = t[1]-t[0]
    y0 = 1
    y = impl_2_steps(t,y0)
    
    err[j] = max(y_exact(t)-y)

for i in range(len(N)-1):
    print(f"p for N1= {N[i]} and N2= {N[i+1]} is:{np.log(err[i+1]/err[i])/np.log(N[i]/N[i+1])}")


# Πλήρως πεπλεγμένες μεθόδοι
#### Κάνετε ανάλογα όπως με όλες τις πεπλεγμένες μεθόδους
#### Οι ενδιάμεσες προσεγγίσεις είναι ένα σταθερό σημείο (διάνυσμα τώρα) μια διανυσματικής
#### Εφαρμόζετε τη διαδικασία σταθερού σημείου για αυτή τη διανυσματική συνάρτηση, η
#### οποια σύμφωνα με το μητρώο που περιγράφει τη μέθοδο Runge Kutta
def skata(t, y0):
    # 2 σταδιων
    m = np.sqrt(3)/6
    a_11 = 1/4
    a_12 = (1/4) - m
    a_21 = (1/4) + m
    a_22 = 1/4
    b1 = 1/2
    b2 = 1/2 
    t1 = (1/2) - m
    t2 = (1/2) + m

    # Εφαρμόζουμε μια διαδικασία σταθερού σημείου όπως κάναμε για τις πεπλεγμένες
    # μεθόδους π.χ. Euler, Τραπεζιου, BDF
    # Για κάθε ένα ενδιάμεσο βήμα ορίζουμε την κατάλληλη συνάρτηση για να βρούμε το σταθερό σημείο
    
    def g1(t1,t2,yn,x):
        return yn + h*a_11*f(t1,x) + h*a_12*f(t2, ?) 

    def g2(t2,yn,k1,x):
        return yn + h*(a_21*k1 + a_22*f(t2,x))

    N = len(t)-1
    y = np.zeros(N+1)
    h = t[1]-t[0]
    y[0] = y0
    tol = 1.e-5
    Nmax = 100
    
    for i in range(N):
        ### υπολογισμός 1-σταδιου
        tn1 = t[i]+t1*h
        tn2 = t[i]+t2*h

        x0 = y[i] #αρχική προσέγγιση στο i-βημα
        k = 0
        err = 1. # Θέτουμε αρχικά το σφάλμα ίσον με 1 για να ξεκινήσει η διαδικασία
        
        while (err > tol) and (k <= Nmax):
            x = g1(tn1, tn2, y[i], x0) # επόμενη προσέγγιση
            err = abs(x-x0) # σφάλμα
            k = k+1 # αυξάνουμε τον μετρητή βημάτων
            x0 = x

        y1 = x

    for i in range(N):
        ### υπολογισμός 2-σταδιου
        tn1 = t[i]+t1*h
        tn2 = t[i]+t2*h

        x0 = y1 #αρχική προσέγγιση στο i-βημα
        k = 0
        err = 1. # Θέτουμε αρχικά το σφάλμα ίσον με 1 για να ξεκινήσει η διαδικασία
        
        while (err > tol) and (k <= Nmax):
            x = g2(tn2, y[i], y1, x0) # επόμενη προσέγγιση
            err = abs(x-x0) # σφάλμα
            k = k+1 # αυξάνουμε τον μετρητή βημάτων
            x0 = x

        y[i+1] = x

    return y

print("full impl Runge-Kutta 2 steps")
err = np.zeros(len(N))
for j in range(len(N)):
    t = np.linspace(0,2,N[j]+1)
    h = t[1]-t[0]
    y0 = 1
    y = skata(t,y0)
    
    err[j] = max(y_exact(t)-y)

for i in range(len(N)-1):
    print(f"p for N1= {N[i]} and N2= {N[i+1]} is:{np.log(err[i+1]/err[i])/np.log(N[i]/N[i+1])}")

