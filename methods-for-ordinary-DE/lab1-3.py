# Επίλυση γραμμικών συστημάτων με την Numpy
import numpy as np

A = np.array([[3,1],[1,2]])
b = np.array([[9],[8]])

# Επίλυση
x = np.linalg.solve(A,b)
print(f"A:{A} * x =\nb:{b}, gives\nx ={x}")
# Επαλήθευση
print(np.dot(A,x))
print()

# Άσκηση 1:
# Μεταβάλετε το στοιχείο (2,2) του πίνακα Α, προσθέτοντας δεκαδικά ψηφία με το 3. Έτσι
# ξεκινήστε με το .33 και λύστε το γραμμικό σύστημα. Συνεχίστε, προσθέτοντας "3-άρια" 
# ως δεκαδικά ψηφία ώστε να πλησιάσετε το 1/3. Σε κάποιο σημείο η λύση σας δεν θα είναι
# υπολογίσιμη.

A = np.array([[1./3,1./3],[1./3,.33]])
# A = np.array([[1./3,1./3],[1./3,.33333333333333333333333333333333]]) # den linetai gia ayto
b = np.array([[1],[0]])
# Επίλυση
x=np.linalg.solve(A,b)
print(f"x: {x}\n")
# Επαλήθευση
# print(np.dot(A,x))

# Νόρμες διανύσματος - ανάλογα και για πίνακες
s = np.linalg.norm(x,1)# νόρμα 1
print(s)
s=np.linalg.norm(x,np.inf)# νόρμα απείρου
print(s)
s=np.linalg.norm(x,2)# νόρμα 2 
print(s)

np.set_printoptions(precision=8) # τυπώνουμε έως 8 δεκαδικά ψηφία τον πίνακα Α
print(A)# πίνακας A
print()

# Αντίστροφος πίνακα - δείκτης κατάστασης κ(Α):= κ(Α)=|Α||Α^(-1)|
# Yπάρχει η εντολή inv για την εύρεση του αντιστρόφου, η οποία όμως λόγω σφαλμάτων
# πράξεων δίνει μια προσέγγιση του αντρίστροφου.
A = np.array([[3,1],[1,2]])
B = np.linalg.inv(A)
print(B)
print(np.dot(A,B))# Επαλήθευση οτι B είναι ο αντίστροφος
print(np.dot(B,A))# Επαλήθευση οτι B είναι ο αντίστροφος
print(np.linalg.inv(B))
print()

# Για τον υπολογισμό του δείκτη κατάστασης μπορούμε να χρησιμοποιήσουμε την εντολή
# cond.
s = np.linalg.norm(A,1)*np.linalg.norm(B,1)# γινόμενο των νορμών
print(s)
print(np.linalg.cond(A,1))# εντολή cond
print()

# Αν ο δείκτης κατάστασης κ(Α) είναι μεγάλος, τότε στην εντολή inv δημιουργούνται
# σφάλματα i. [Α^(-1)]*A != A*[Α^(-1)] και ii. [Α^(-1)]^(-1) != A
np.set_printoptions(precision=5) #θέτουμε 5 δεκαδικά ψηφια για την εκτύπωση των στοιχείων ενός πίνακα
A = np.array([[100,100],[100,100.0001]]) #πίνακας A
print(np.linalg.cond(A,1))# δείκτης κατάστασης είναι μεγάλος
B = np.linalg.inv(A)# υπολογισμός αντιστρόφου
print(B)
print(np.dot(A,B)) # Επαλήθευση οτι B είναι ο αντίστροφος
print()

# Άσκηση 2:
# Δοκιμάστε να τροποποιήστε το στοιχείο στη θέση (2,2), ώστε ο πίνακας να είναι πιο
# "κοντά" ή πιο μακρυά σε έναν μη αντιστρέψιμο και βρείτε το δείκτη κατάστασης, τον
# αντίστροφο και παρατηρήστε αν όντως είναι ο αντίστροφος. 

# WTF?!?!?!?!

# Ένας άλλος τρόπος να ελένξουμε αν έχουμε τον σωστό αντρίστροφο είναι να υπολογίσουμε
# τη νόρμα |Α*Α^(-1) - Ι|
A = np.array([[100,100],[100,100.001]])
print(np.linalg.cond(A,1)) # μεγάλος δείκτης κατάστασης

B = np.linalg.inv(A)
print(B)

C=np.dot(A,B)-np.eye(2)# AB-I
print(C)

# Επαλήθευση οτι B είναι ο αντίστροφος
print(np.linalg.norm(C,1))# Επαλήθευση οτι B είναι ο αντίστροφος
C = np.dot(B,A)-np.eye(2) # BA-I
print(C)
print(np.linalg.norm(C,1))

# Επαλήθευση οτι ο αντίστροφος του B είναι ο A
C=np.linalg.inv(B)-A# B-A
print(C)
print(np.linalg.norm(C,1))

# Άσκηση 3:
# Φτιάξτε μια συνάρτηση my_cond η οποία να υπολογίζει τον δείκτη κάταστασης ενός πίνακα.
# (χρησιμοποιήστε όποια νόρμα θέλετε)

def my_cond(A):
    B = np.linalg.inv(A)

    return np.linalg.norm(A,1)*np.linalg.norm(B,1)


print("Error my_k(A) and k(A): ", my_cond(A) - np.linalg.cond(A,1))
print()

# Επίλυση γραμμικών συστημάτων Ax=b
# Η επίλυση γραμμικών συστημάτων με πίνακες με μεγάλο δείκτη κατάστασης οδηγεί
# σε σφάλματα.
# Συγκρίνουμε τη λύση που βρήκαμε με την ακριβή, και υπολογίζουμε τη νόρμα της διαφοράς.

# Ένας άλλος τρόπος αν δεν γνωρίζουμε την ακριβή λύση είναι να υπολογίσουμε το υπόλοιπο.
# Αν η λύση μας είναι (προσεγγιστικά) καλή τότε το υπόλοιπο θα είναι 0.

A = np.array([[100,100],[100,100.01]])
b = np.array([[1],[2]])
x = np.linalg.solve(A,b)

r = np.dot(A,x)-b # υπόλοιπο Ax-b
print(r)
print(np.linalg.norm(r,1))
print()

# Άσκηση 4:
# Φτιάξτε μι ασυνάρτηση my_residual η οποία να υπολογίζει τη νόρμα του υπολοίπου της
# λύσης ενός γραμμικού συστήματος με πίνακα A και δεξιό μέλος b. (Xρησιμοποιήστεόποια
# νόρμα θέλετε)

def my_residual(A,b): #xrisimopoiw oti ekana akrivws prin apla evala tis entoles 
    # kateutheian x = np.linalg.solve(A,b), r = np.dot(A,x)-b, np.linalg.norm(r,1)
    return np.linalg.norm(np.dot(A,np.linalg.solve(A,b))-b,1)

print(my_residual(A,b))
print()

# Πίνακες Hilbert
# είναι τετραγωνικοί πίνακες με στοιχεία aij = 1/ (i+j-1)
# Άσκηση 5: Aλλάξτε τη διάσταση του πίνακα Hilbert και παρατηρήστε την αύξηση του
# δείκτη κατάστασης.
import scipy.linalg as sp # H βιβλιοθήκη linalg της Scipy 

def hilbertMatr_test(k, p): #k: διάσταση πίνακα, p MUST BE: 1 or 2 or np.inf
    # print(f"for k:{k}")
    A = sp.hilbert(k)
    np.set_printoptions(precision=5)
    # print(f"Hilbert martix A: {A}")
    b = np.ones((k,1))
    x = np.linalg.solve(A,b)
    # print(f"Epalitheusi A*x = {np.dot(A,x)}")
    r = np.dot(A,x)-b
    # print(f"|r| = {np.linalg.norm(r,p)}")
    # print(f"k(A): {np.linalg.cond(A)}\n")
    return np.linalg.norm(r,p)

for p in [1,2,np.inf]:
    r = []
    for i in [5,10,20,30]:
        r.append(hilbertMatr_test(i,p))
    print(f"for norm-{p} and dim: {i}\nr: {r}")
print()



# Υπολογισμός αντιστρόφου
# Άσκηση 7: Δημιουργήστε μια συνάρτηση η οποία να κατασκευάζει τον αντίστροφο ενός 
# πίνακα, και φτιάξτε ένα πρόγραμμα ωστε να ελέγξτε το αποτέλεσμα σας
# TODO: not finished
def my_inv(A):
    e = np.zeros(A.shape[1])
    B = np.array([]) #empty array

    for i in range(A.ndim):
        e[i] = 1
        x = np.linalg.solve(A,e)
        np.append(B,x)
        e[i] = 0
        print(x.T)

    return B

A = sp.hilbert(5)
print(my_inv(A))