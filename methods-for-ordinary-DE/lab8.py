# Μέθοδοι πρόβλεψης διόρθωσης
import numpy as np
import matplotlib.pyplot as plt

def y_exact(t):
    return (t**2+1)**2

def f(t,y):
    return 4*t*(y**(1/2))


N = [20,40,80,160,320]
err = np.zeros(len(N))
# Παρατήρηση: Μπορούμε να συνδυάσουμε μια μονοβηματική μέθοδο προβλεψης (Euler) με μια διβηματική
# μέθοδο διόρθωσης. Προσοχή το βήμα διόρθωσης θα πρέπει να είναι tn και tn+1 στο tn+2, ενώ
# το βήμα πρόβλεψης tn+1 στο tn+2.

print("Pred-Corr/ Euler - Trapezointal")
for i in range(len(N)):
    t = np.linspace(0, 2, N[i]+1)
    h = t[1] - t[0]

    y = np.zeros(N[i]+1)
    y[0] = 1

    for j in range(N[i]):
        # Προβλεψη Άμεση Euler
        y_pred = y[j] + h*f(t[j],y[j])

        # Διόρθωση Τραπεζίου
        y[j+1] = y[j]+(h/2)*(f(t[j],y[j]) + f(t[j+1],y_pred))

    err[i] = max(abs(y_exact(t) - y))
    print(f"error for N={N[i]} is: {err[i]}")

for i in range(len(N)-1):
    print(f"p for N1= {N[i]} and N2= {N[i+1]} is:{np.log(err[i+1]/err[i])/np.log(N[i]/N[i+1])}")

# Άσκηση 2
print("Pred-Corr/ Euler - ΑΜ(2)")
err = np.zeros(len(N))
for i in range(len(N)):
    t = np.linspace(0, 2, N[i]+1)
    h = t[1] - t[0]

    y = np.zeros(N[i]+1)
    y[0] = 1
    y[1] = y[0] + h*f(t[0], y[0]) # πρώτο βήμα με άμεση Euler

    for j in range(N[i]-1):
        # Προβλεψη Άμεση Euler
        y_pred = y[j+1] + h*f(t[j+1],y[j+1])

        # Διόρθωση AM2
        y[j+2] = y[j+1] +  h*((5/12)*f(t[j+2], y_pred) + (2/3)*f(t[j+1],y[j+1]) - (1/12)* f(t[j],y[j]))

     
    err[i] = max(abs(y_exact(t) - y))
    print(f"error for N={N[i]} is: {err[i]}")

for i in range(len(N)-1):
    print(f"p for N1= {N[i]} and N2= {N[i+1]} is:{np.log(err[i+1]/err[i])/np.log(N[i]/N[i+1])}")

print("Pred-Corr/ Euler - BDF")
err = np.zeros(len(N))
for i in range(len(N)):
    t = np.linspace(0, 2, N[i]+1)
    h = t[1] - t[0]

    y = np.zeros(N[i]+1)
    y[0] = 1
    y[1] = y[0] + h*f(t[0], y[0]) # πρώτο βήμα με άμεση Euler

    for j in range(N[i]-1):
        # Προβλεψη Άμεση Euler
        y_pred = y[j+1] + h*f(t[j+1],y[j+1])

        # Διόρθωση BDF
        y[j+2] = (1/3)*(4*y[j+1]-y[j])+(2*h/3)*f(t[j+2],y_pred)
    
    err[i] = max(abs(y_exact(t) - y))
    print(f"error for N={N[i]} is: {err[i]}")

for i in range(len(N)-1):
    print(f"p for N1= {N[i]} and N2= {N[i+1]} is:{np.log(err[i+1]/err[i])/np.log(N[i]/N[i+1])}")


# Άσκηση 3
print("Simple method of exercise 3")
err = np.zeros(len(N))
for i in range(len(N)):
    t = np.linspace(0, 2, N[i]+1)
    h = t[1] - t[0]

    y = np.zeros(N[i]+1)
    y[0] = 1
    y[1] = (h**2 + 1)**2

    for j in range(N[i]-1):
        y[j+2] = -4*y[j+1] + 5*y[j] +h*(4*f(t[j+1], y[j+1]) + 2*f(t[j], y[j]))

    err[i] = max(abs(y_exact(t) - y))
    print(f"error for N={N[i]} is: {err[i]}")
for i in range(len(N)-1):
    print(f"p for N1= {N[i]} and N2= {N[i+1]} is:{np.log(err[i+1]/err[i])/np.log(N[i]/N[i+1])}")

print("Pred-Corr/ exercise 3")
for i in range(len(N)):
    t = np.linspace(0, 2, N[i]+1)
    h = t[1] - t[0]

    y = np.zeros(N[i]+1)
    y[0] = 1
    y[1] = (h**2 + 1)**2

    for j in range(N[i]-1):
        # Προβλεψη 
        y_pred = -4*y[j+1] + 5*y[j] + h*(4*f(t[j+1], y[j+1]) + 2*f(t[j], y[j]))

        # Διόρθωση 
        y[j+2] = (4/3)*y[j+1] - (1/3)*y[j] + (2/3)*h*f(t[j+2], y_pred)

    err[i] = max(abs(y_exact(t) - y))
    print(f"error for N={N[i]} is: {err[i]}")

for i in range(len(N)-1):
    print(f"p for N1= {N[i]} and N2= {N[i+1]} is:{np.log(err[i+1]/err[i])/np.log(N[i]/N[i+1])}")

# TODO: άσκηση 4 (to be done) και στην άσκηση 3 βγάζει ένα RuntimeWarning: invalid value encountered 
# in double_scalars - το οποιο μάλλον οφείέται σε κάποιο μη επτιρεπτό υπολογισμό π.χ. διαίρεση με 0 αλλα
# δεν έχω βρει τι

# Συστήματα Διαφορικών εξισώσεων
def f1(t,x,y):
    return (0.05)*x*(1-0.01*y)
    
def f2(t,x,y):
    return (0.1)*y*(0.005*x-2)

NN = 600

t = np.linspace(0,150,NN+1)
h = t[1]-t[0]
x =np.zeros(NN+1)
y = np.zeros(NN+1)

x[0] = 500
y[0] = 100
x[1] = x[0] + h*f1(t[0],x[0],y[0])
y[1] = y[0] + h*f2(t[0],x[0],y[0])

for i in range(NN-1):
    # Άμεση Euler
    x_pred = x[i+1] + h*f1(t[i+1], x[i+1], y[i+1])
    y_pred = y[i+1] + h*f2(t[i+1], x[i+1], y[i+1])

    # Διόρθωση ΑΜ2
    y[i+2] = y[i+1] + (h)*((5/12)*f2(t[i+2],x_pred,y_pred) + (2/3)*f2(t[i+1],x[i+1],y[i+1]) - (1/12)*f2(t[i],x[i],y[i]))
    x[i+2] = x[i+1] + (h)*((5/12)*f1(t[i+2],x_pred,y_pred) + (2/3)*f1(t[i+1],x[i+1],y[i+1]) - (1/12)*f1(t[i],x[i],y[i]))
    
## Γραφική παράσταση των 2 συναρτήσεων
plt.plot(t,x,t,y)
plt.show()

### Γραφική παράσταση xy-επιπεδο
plt.plot(x,y)
plt.show()

## Πληθυσμός στο τέλος
print(x[NN],y[NN])

# Οι ακριβείς λύσεις είναι περιοδικές συναρτήσεις. 
# Όσο μεγαλώνει ο αριθμός των σημείων N, το παρατηρουμε καλυτερα.