# Finite element method (FEM)
# The FEM is a particular numerical method for solving partial differential equations in two or three space variables 
# (i.e., some boundary value problems). To solve a problem, the FEM subdivides a large system into smaller, simpler parts that 
# are called finite elements. This is achieved by a particular space discretization in the space dimensions, which is 
# implemented by the construction of a mesh of the object: the numerical domain for the solution, which has a finite number
# of points. The finite element method formulation of a boundary value problem finally results in a system of algebraic equations. 
# The method approximates the unknown function over the domain. The simple equations that model these finite elements are then 
# assembled into a larger system of equations that models the entire problem. The FEM then uses variational methods from the calculus of 
# variations to approximate a solution by minimizing an associated error function. 

# Άσκηση 1: Θεωρούμε την συνάρτηση f(x)=lnx και το σημείο x0 =1.1. Υπολογίστε τις πεπερασμένες διαφορές στο σημείο x0 για h=[0.5,0.1,0.05,0.01].

import numpy as np

def f(x):
    return np.log(x)

def df(x):
    return 1//x

def approx(f, x0, h):
    dh_plus = []
    dh_minus = []
    dh_c = []
    for i in h:
	    dh_plus.append((f(x0 + i) - f(x0))/i)
	    dh_minus.append((f(x0) - f(x0-i))/i)
	    dh_c.append((f(x0+i) - f(x0-i))/(2*i))

    return dh_plus, dh_minus, dh_c

x0 = 1.1
h = [0.5, 0.1, 0.05, 0.01]
dh_plus, dh_minus, dh_c = approx(f, x0, h)
print("Approximations")
print("Forward Difference = ", dh_plus)
print("Backward Difference = ", dh_minus)
print("Central Difference = ", dh_c)


# Θεωρήστε τα δεδομένα της άσκησης 1. Βρείτε την "πειραματική" τάξη σύγκλισης των πεπερασμένων διαφορών. Ποιο είναι το p στις παραπάνω τρεις περιπτώσεις;

print("\nErrors")
print("Forward Difference = ", [abs(df(x0)-i) for i in dh_plus])
print("Backward Difference = ", [abs(df(x0)-i) for i in dh_minus])
print("Central Difference = ", [abs(df(x0)-i) for i in dh_c])

def find_p(e, h):
	p = []
	for i in range(0,2):
		p.append((np.log(e[i] / e[i+1])) / (np.log(h[i] / h[i+1])))
	return p

print("\nAccuracy p")
print("p1 = ", find_p([abs(df(x0)-i) for i in dh_plus], h))
print("p2 = ", find_p([abs(df(x0)-i) for i in dh_minus], h))
print("p3 = ", find_p([abs(df(x0)-i) for i in dh_c], h))

# Άσκηση 3: Επαναλάβετε τις ασκήσεις 1 και 2 για τις πεπερασμένες διαφορές για την προσέγγισης της δεύτερης παραγώγου
ddh_plus, ddh_minus, ddh_c = approx(df, x0, h)
print("\nApproximations using approximations of first derivative")
print("Forward Difference = ", ddh_plus)
print("Backward Difference = ", ddh_minus)
print("Central Difference = ", ddh_c)
print("wtf?!")

print("\nApproximations using central difference approx of second derivative")
print("Central Difference = ", [(f(x0+i) - 2*f(x0) + f(x0-i))/(i**2) for i in h])

def ddf(x):
    return (-1)*(1/(x**2))
print("\nErrors")
print("Central Difference = ", [abs(ddf(x0)-i) for i in [(f(x0+i) - 2*f(x0) + f(x0-i))/(i**2) for i in h]])
