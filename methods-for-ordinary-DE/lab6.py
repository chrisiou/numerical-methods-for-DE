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
    return np.exp(0.25 - (t-0.5)**2)

def f(t,y):
    return (1-2*t)*y


def Euler(N, ex_num): # ex_num is exercise number because I need both methods in all exercise
    t = np.linspace(0, 4, N+1)
    h = t[1] - t[0]

    y = np.zeros(N+1)
    y[0] = 1

    for i in range(N):
        y[i+1] = y[i] + h*f(t[i],y[i])
    
    error = max(abs(y_exact(t)-y))

    # makeGraph(y_exact(t), y, t, "ex." + str(ex_num) + " Euler's method for N: " + str(N))
    # print(f"ex.{str(ex_num)} max_error for Euler: {error}")
    return error

def midpoint(N, ex_num):
    t = np.linspace(0, 4, N+1)
    h = t[1] - t[0]

    y_mid = np.zeros(N+1)
    y_mid[0] = 1

    if (ex_num == 1):
        y_mid[1] = y_exact(h) # o bro leei y_mid[1]=y_exact(t[1])
    else:
        y_mid[1] = y_mid[0] + h*f(t[0], y_mid[0]) # άσκηση 2

    # WARNING: Προσοχή στον αριθμό των επαναλήψεων. Στη διβηματική μέθοδο κάνουμε 1 λιγότερη επανάληψη
    for i in range(N-1):
        y_mid[i+2] = y_mid[i] + 2*h*f(t[i+1], y_mid[i+1])

    # makeGraph(y_exact(t), y_mid, t, "ex." + str(ex_num) + "Midpoint's method for N: " + str(N))
    # print(f"ex.{str(ex_num)} max_error for midpoint: {max(abs(y_exact(t)-y_mid))}")
    return max(abs(y_exact(t)-y_mid))

def AB2(N, ex_num):
    t = np.linspace(0, 4, N+1)
    h = t[1] - t[0]

    y_ab = np.zeros(N+1)
    y_ab[0] = 1

    if (ex_num == 1):
        y_ab[1] = y_exact(h) # o bro leei y_ab[1]=y_exact(t[1])
    else:
        y_ab[1] = y_ab[0] + h*f(t[0], y_ab[0]) # άσκηση 2

    # WARNING: Προσοχή στον αριθμό των επαναλήψεων. Στη διβηματική μέθοδο κάνουμε 1 λιγότερη επανάληψη
    for i in range(N-1):
        y_ab[i+2] = y_ab[i+1] + h*((3./2)*f(t[i+1],y_ab[i+1]) - (1./2)*f(t[i],y_ab[i]))

    # makeGraph(y_exact(t), y_ab, t, "ex." + str(ex_num) + "AB2's method for N: " + str(N))
    # print(f"ex.{str(ex_num)} max_error for AB2: {max(abs(y_exact(t)-y_ab))}")
    return max(abs(y_exact(t)-y_ab))

# Άσκηση 1
# για Ν = 100, 200, 300, 400, 500
N  = list(range(100,501,100))
err_Euler = np.zeros(len(N))
err_mid = np.zeros(len(N))
err_ab2 = np.zeros(len(N))

for i in range(len(N)):
    err_Euler[i] = Euler(N[i], 1)
    err_mid[i] = midpoint(N[i], 1)
    err_ab2[i] = AB2(N[i], 1)

# Πειραματική τάξη σύγκλισης
# p_Euler , p_mid, p_ab2 = [], [], []

# for i in range(len(N)-1):
#     p_Euler.append(np.log(err_Euler[i+1]/err_Euler[i])/np.log(N[i]/N[i+1]))
#     p_mid.append(np.log(err_mid[i+1]/err_mid[i])/np.log(N[i]/N[i+1]))
#     p_Euler.append(np.log(err_Euler[i+1]/err_Euler[i])/np.log(N[i]/N[i+1]))


# Άσκηση 2 - ίδια άσκηση με 1 μόνο που για y1 παίρνω τη προσέγγιση με χρήση της άμεσης Euler
# για Ν = 100, 200, 300, 400, 500
N  = list(range(100,501,100))
err_Euler = np.zeros(len(N))
err_mid = np.zeros(len(N))
err_ab2 = np.zeros(len(N))

for i in range(len(N)):
    err_Euler[i] = Euler(N[i], 2)
    err_mid[i] = midpoint(N[i], 2)
    err_ab2[i] = AB2(N[i], 2)

# Πειραματική τάξη σύγκλισης
p_Euler , p_mid, p_ab2 = [], [], []

for i in range(len(N)-1):
    p_Euler.append(np.log(err_Euler[i+1]/err_Euler[i])/np.log(N[i]/N[i+1]))
    p_mid.append(np.log(err_mid[i+1]/err_mid[i])/np.log(N[i]/N[i+1]))
    p_ab2.append(np.log(err_ab2[i+1]/err_ab2[i])/np.log(N[i]/N[i+1]))

# Άσκηση 3
print(f"N   Euler                p                     midpoint              p                     AB2                  p")
print(f"{N[0]} {err_Euler[0]}                         {err_mid[0]}                          {err_ab2[0]}")
for i in range(1,len(N)):
    print(f"{N[i]} {err_Euler[i]} {p_Euler[i-1]} {err_mid[i]}  {p_mid[i-1]}    {err_ab2[i]}  {p_ab2[i-1]}")

# _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ #
# Πολυβηματικές μέθοδοι

# Άσκηση 4

def g(t):
    return 1-2*t

def implEuler(N):
    t = np.linspace(0, 4, N+1)
    h = t[1] - t[0]

    y = np.zeros(N+1)
    y[0] = 1

    for i in range(N):
        y[i+1] = y[i] / (1 - h*g(t[i+1])) 

    # makeGraph(y_exact(t), y, t, "ex3. Implicit Euler's method for N: " + str(N))
    return max(abs(y_exact(t)-y))


def simpson(N):
    t = np.linspace(0,4,N+1)
    h = t[1]-t[0]
    
    y_simp = np.zeros(N+1)
    y_simp[0] = 1
    
    # Εισάγουμε την 1η προσέγγιση για να ξεκινήσει η διβηματική μεθοδος, π.χ. y1 προκύπτει με την πεπλεγμένη Euler
    y_simp[1] = y_simp[0]/(1-h*g(t[1]))

    # Προσοχή στη διβηματική μέθοδο κάνουμε 1 λιγότερη επανάληψη
    for i in range(N-1):
        y_simp[i+2] = (y_simp[i] + (h/3)*(4*f(t[i+1],y_simp[i+1]) + f(t[i],y_simp[i])))/(1-(h/3)*g(t[i+2]))

    # makeGraph(y_exact(t), y_simp, t, "ex3. Simpson's method for N: " + str(N))
    return max(abs(y_exact(t)-y_simp))

def AdamsMulton2(N):
    t = np.linspace(0,4,N+1)
    h = t[1]-t[0]

    y_am2 = np.zeros(N+1)
    y_am2[0] = 1

    # Εισάγουμε την 1η προσέγγιση για να ξεκινήσει η διβηματική μεθοδος, π.χ. y1 προκύπτει με την πεπλεγμένη Euler
    y_am2[1] = y_am2[0]/(1-h*g(t[1]))

    # Προσοχή στη διβηματική μέθοδο κάνουμε 1 λιγότερη επανάληψη
    for i in range(N-1):
        y_am2[i+2] = (1/ (1-((5/12)*h)*(1-2*t[i+2]))) * (y_am2[i+1] + h*((2/3)*(1-2*t[i+1])*y_am2[i+1] - (1/12)*(1-2*t[i])*y_am2[i]))

    # makeGraph(y_exact(t), y_am2, t, "ex3. AdamsMulton's method for N: " + str(N))
    return max(abs(y_exact(t)-y_am2))

def BDF(N):
    t = np.linspace(0,4,N+1)
    h = t[1]-t[0]

    y_bdf = np.zeros(N+1)
    y_bdf[0] = 1

    # Εισάγουμε την 1η προσέγγιση για να ξεκινήσει η διβηματική μεθοδος, π.χ. y1 προκύπτει με την πεπλεγμένη Euler
    y_bdf[1] = y_bdf[0]/(1-h*g(t[1]))

    # Προσοχή στη διβηματική μέθοδο κάνουμε 1 λιγότερη επανάληψη
    for i in range(N-1):
        y_bdf[i+2] = ((-4/3)*y_bdf[i+1] + (1/3)*y_bdf[i])/((2/3)*g(t[i+2]) - 1)

    # makeGraph(y_exact(t), y_bdf, t, "ex3. BDF's method for N: " + str(N))
    return max(abs(y_exact(t)-y_bdf))

err_implEuler = np.zeros(len(N))
err_simpson = np.zeros(len(N))
err_am2 = np.zeros(len(N))
err_bdf = np.zeros(len(N))

for i in range(len(N)):
    err_implEuler[i] = implEuler(N[i])
    err_simpson[i] = simpson(N[i])
    err_am2[i] = AdamsMulton2(N[i])
    err_bdf[i] = BDF(N[i])

# Πειραματική τάξη σύγκλισης
p_implEuler , p_simpson, p_am2, p_bdf = [], [], [], []

for i in range(len(N)-1):
    p_implEuler.append(np.log(err_implEuler[i+1]/err_implEuler[i])/np.log(N[i]/N[i+1]))
    p_simpson.append(np.log(err_simpson[i+1]/err_simpson[i])/np.log(N[i]/N[i+1]))
    p_am2.append(np.log(err_am2[i+1]/err_am2[i])/np.log(N[i]/N[i+1]))
    p_bdf.append(np.log(err_bdf[i+1]/err_bdf[i])/np.log(N[i]/N[i+1]))

print(f"N   impl.Euler            p                     simpson               p                     AM2                  p                     BDF                  p")
print(f"{N[0]} {err_implEuler[0]}                         {err_simpson[0]}                          {err_am2[0]}                          {err_bdf[0]}")
for i in range(1,len(N)):
    print(f"{N[i]} {err_implEuler[i]}  {p_implEuler[i-1]}  {err_simpson[i]}   {p_simpson[i-1]}     {err_am2[i]}   {p_am2[i-1]}     {err_bdf[i]}   {p_bdf[i-1]}")


##TODO: Gia kapoio logo to y_exact fen bgainei swsta sto bdf