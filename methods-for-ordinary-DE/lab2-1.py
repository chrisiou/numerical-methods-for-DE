import matplotlib.pyplot as p
import numpy as np

# Γραφικές Παραστάσεις
def f(t):
    return t**2*np.exp(-t**2)
    
t = np.linspace(0,3,51)
y = f(t)
p.plot(t,y,'r-')
p.xlabel('t')
p.ylabel('y')
p.legend(['t^2*exp(-t^2)'])
p.show()

# Δυο γραφικές παραστάσεις
def f2(t):
    return t**4*np.exp(-t**2)
    
p.subplot(2,1,1)
p.plot(t,y,'r-')
p.xlabel('t')
p.ylabel('y')
p.legend(['t^2*exp(-t^2)'])
y2 = f2(t)
p.subplot(2,1,2)
p.plot(t,y2,'bs')
p.xlabel('t')
p.ylabel('y')
p.legend(['t^4*exp(-t^2)'])
p.show()

# Αλλαγή της διάταξης των γραφικών παραστάσεων
p.subplot(1,2,1)
p.plot(t,y,'r-')
p.xlabel('t')
p.ylabel('y')
p.legend(['t^2*exp(-t^2)'],loc='lower center')
y2 = f2(t)
p.subplot(1,2,2)
p.plot(t,y2,'bs')
p.xlabel('t')
p.ylabel('y')
p.legend(['t^4*exp(-t^2)'],loc='lower right')
p.show()

# 3-D γραφικές παραστάσεις
from mpl_toolkits.mplot3d.axes3d import Axes3D
# Δημιουργούμε τα σημεία του επιπέδου xy που θα βρίσκετε το γράφημα:
x = np.linspace(-5,5,100)
y = x.copy()
X,Y = np.meshgrid(x,y)

# Δημιουργούμε το αντικείμενο που θα “ζωγραφίσουμε” το γράφημα
fig=p.figure()
axes=fig.add_axes([0.1,0.1,0.8,0.8],projection='3d')

# Δημιουργούμε το γράφημα 
Z=X**2-Y**2
axes.plot_surface(X,Y,Z,rstride=5,cstride=5,linewidth=1)
p.show()