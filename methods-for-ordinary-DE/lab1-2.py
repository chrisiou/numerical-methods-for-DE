import numpy as np

A = np.array([[1,2],[3,4]])
x = np.array([1,0])

print(np.dot(A,x)) # Πολλαπλασιασμός πίνακα 2x2 με διάνυσμα 2x1

# Άσκηση 1:
# Φτιάξτε μια συνάρτηση η οποία να δέχεται ως όρισμα 2 διανύσματα να ελέγχει αν αυτά
# είναι ορθογώνια και να τυπώνει ένα σχετικό μήνυμα. Με τη χρήση της dot τρόπο ορίζεται 
# και η πράξη πολλαπλασιασμός πίνακα με διάνυσμα και πίνακα με πίνακα.
def areOthogonal(x,y):
    if ((np.dot(x, y)) == 0):
        print(f"{x} and {y} are orhogonal vectors!")
        return
    print(f"{x} and {y} are not orhogonal vectors!")

x = np.array([2,4])
y = np.array([7,5])
z = np.array([-2,1])

areOthogonal(x,y)
areOthogonal(x,z)

x=np.eye(4)
print(f"x = np.eye(4):\n{x}")
x=np.array([[1,2],[2,3],[3,4]])
y=np.array([1,2])# Πολλαπλασιασμός πίνακα 3x2 με πίνακα 2x1
print(f"{x}*{y} = {np.dot(x,y)}")
print()
# print(np.dot(y,x)) # Πολλαπλασιασμός πίνακα 2x1 με πίνακα 3x2 δεν μπορεί να γίνει

# Άσκηση 2:
# Φτιάξτε μια συνάρτηση η οποία να δέχεται ως όρισμα 2 πίνακες A και B και να ελέγχει
# αν ο A ειναι ο αντίστροφοςτου B

def areSame(A,B):
    if (A.shape != B.shape):
        return False
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if (A[i][j] != B[i][j]):
                return False
    return True

def isBinverseOfA(A,B):
    if (areSame(np.dot(A,B), np.dot(B,A))):
        print(f"B:\n{B}\nis inverse of\nA:\n{A}\n")
    else:
        print(f"B:\n{B}\nis not inverse of\nA:\n{A}\n")


A = np.array([[2,3],[2,2]])
B = np.array([[-1,3/2],[1,-1]])
C = np.array([[1,2],[3,4]])

isBinverseOfA(A,B)
isBinverseOfA(A,C)