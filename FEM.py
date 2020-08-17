# Finite element method (FEM)
# The FEM is a particular numerical method for solving partial 
# differential equations in two or three space variables 
# (i.e., some boundary value problems). To solve a problem, the 
# FEM subdivides a large system into smaller, simpler parts that 
# are called finite elements. This is achieved by a particular 
# space discretization in the space dimensions, which is 
# implemented by the construction of a mesh of the object: 
# the numerical domain for the solution, which has a finite number
# of points. The finite element method formulation of a boundary 
# value problem finally results in a system of algebraic equations. 
# The method approximates the unknown function over the domain. 
# The simple equations that model these finite elements are then 
# assembled into a larger system of equations that models the entire 
# problem. The FEM then uses variational methods from the calculus of 
# variations to approximate a solution by minimizing an associated 
# error function. 