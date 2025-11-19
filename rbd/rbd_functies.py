# rigid body dynamics numerieke functies
# 2025-10-10
# A. Stienen, P. de Jong

# er wordt gewerkt in 3D, dus beperken vectoren en matrices zich tot dimensies 
# 3x1 en 3x3 respectievelijk.

# de functies dienen niet aangepast te worden om goede werking te garanderen,
# hierbij is ook de volgorde waarin ze zijn gegeven van belang

# machtsverheffen doe je met een dubbele asterisk: x**n (x tot de macht n).
# numerieke waardes worden een "float" door een decimaal toe te voegen, 
# dus ook bij een geheel getal: 0.0 in plaats van 0

# aanpassing 10/10/2025: eigenvec en eigenwrd resulteren nu in matrices welke 
# gebruikt kunnen worden voor verdere berekeningen. 

###############################################################################

import numpy as np

from numpy import pi as pi
from numpy import sqrt as sqrt

from numpy import cos as cos
from numpy import sin as sin
from numpy import tan as tan

from numpy import arccos as acos
from numpy import arcsin as asin
from numpy import arctan as atan

def vector(a1,a2,a3):
    #vector met 3 rijen en 1 kolom (3x1)
    return np.matrix([[a1],
                      [a2],
                      [a3]])

def matrix(a11,a12,a13,a21,a22,a23,a31,a32,a33):
    #matrix met 3 rijen en 3 kolommen (3x3)
    return np.matrix([[a11,a12,a13],
                      [a21,a22,a23],
                      [a31,a32,a33]])

def diagmatrix(a11,a22,a33):
    #matrix met 3 rijen en 3 kolommen (3x3)
    return np.matrix([[a11,0,0],
                      [0,a22,0],
                      [0,0,a33]])

def inv(A):
    #inverse van matrix
    return np.linalg.inv(A)

def transpose(A):
    #getransponeerde vector van matrix
    return A.transpose()

def tilde(a):
    #tilde matrix van een 3x1 vector
    a1 = a[0,0]; a2 = a[1,0]; a3 = a[2,0]
    return matrix( 0.0, -a3,  a2,
                    a3,   0, -a1,
                   -a2,  a1,   0)

def dot(a,b):
    #inproduct van twee vectoren
    return (transpose(a)*b)[0,0]

def cross(a,b):
    #kruisproduct van twee 3x1 vectoren
    return tilde(a)*b

def norm(a):
    #norm van vector
    return float(np.sqrt(dot(a,a)))

def evec(a):
    #genormaliseerde eenheidsvector
    return a/norm(a)

def eigenwrd(A):
    #eigenwaardes van inertia matrix A
    # op volgorde van groot naar klein
    # en volgorde correspondeert met volgorde eigenvectoren
    EigenWaarden = np.linalg.eig(A)[0]
    IndexAflopend = EigenWaarden.argsort()[::-1] # sorteervolgorde bepalen 
    EigenWaarden = EigenWaarden[IndexAflopend]   # sorteren
    return np.matrix(np.diag(EigenWaarden))      # diag.matrix met eigenwaarden

def eigenvec(A):
    #eigenvectoren (kolommen) van matrix A
    # kolommen met eigenvectoren op volgorde eigenwaarden
    # met eigenwaarden van groot naar klein 
    EigenWaarden, EigenVectoren = np.linalg.eig(A)
    IndexAflopend = EigenWaarden.argsort()[::-1]   # sorteervolgorde bepalen
    EigenVectoren = EigenVectoren[:,IndexAflopend] # sorteren
    if np.linalg.det(EigenVectoren) < 0:
        EigenVectoren[:, 2] *= -1 # soms nodig om rechtsdraaiend stelsel te krijgen
    return EigenVectoren

def FCN_x(hoek):
    # rotatie matrix rond x voor het roteren van triad N naar F ([hoek] = rad)
    return matrix(1.0,        0.0,       0.0,
                  0.0,  cos(hoek), sin(hoek),
                  0.0, -sin(hoek), cos(hoek))

def FCN_y(hoek):
    # rotatie matrix rond y voor het roteren van triad N naar F ([hoek] = rad)
    return matrix(cos(hoek), 0.0, -sin(hoek),
                        0.0, 1.0,        0.0,
                  sin(hoek), 0.0,  cos(hoek))

def FCN_z(hoek):
    # rotatie matrix rond z voor het roteren van triad N naar F ([hoek] = rad)
    return matrix( cos(hoek), sin(hoek), 0.0,
                  -sin(hoek), cos(hoek), 0.0,
                         0.0,       0.0, 1.0)

###############################################################################

# onderstaande voorbeelden kan je zelf wijzigen en zullen bij het tentamen NIET 
# beschikbaar zijn

# voorbeeld rotatiematrices
print('### Voorbeelden met rotatiematrices\n')

alpha = pi/3
beta  = pi/4
gamma = pi/6

GCN = FCN_x(alpha)
HCG = FCN_z(beta)
BCH = FCN_y(gamma)

BCN = BCH*HCG*GCN
NCB = transpose(BCN)

print('BCN:\n', BCN)
print('NCB=transpose(BCN):\n', NCB)

# voorbeeld van N naar B frame vertalen
Nr = vector(1,0,0)
Br = BCN*Nr
print('Br:\n', Br)

# voorbeeld van B naar N frame vertalen
Br = vector(1,0,0)
Nr = transpose(BCN)*Br
print('Nr:\n', Nr)

# vector bewerkingen
BrBA = vector(1,3,2)
GrCA = vector(4,6,5)

NrBA = NCB*BrBA
NrCA = transpose(GCN)*GrCA
 
print('inproduct:\n', dot(NrBA,NrCA))
print('kruisproduct:\n', cross(NrBA,NrCA))
print('norm:\n', norm(NrBA))
print('eenheidsvector:\n', evec(NrBA))

# matrix bewerkingen
NIC = NCB*matrix(2.0, 0.0, 0.0, 
                 0.0, 4.0, 0.0, 
                 0.0, 0.0, 5.0)*BCN

print('inverse:\n', inv(NIC))
print('eigenwaardes:\n', eigenwrd(NIC))
print('eigenvectoren:\n', eigenvec(NIC))

# vinden principele rotatieas uit rotatiematrix
BCN_ewrd = eigenwrd(BCN)     # de eigenwaarde welke gelijk is aan 1, geeft bij de eigenvectoren de principele rotatieas aan
BCN_evec = eigenvec(BCN)     # de principiele rotatieas is de vector (kolom) die bij de eigenwaarde van 1 hoort
print('eigenwaarden rotatiematrix BCN:\n', BCN_ewrd)
print('eigenvectoren rotatiematrix BCN:\n', BCN_evec)

# voorbeeld inertiamatrices
print('### Inertiamatrices \n')

BI  = diagmatrix(3,1,2)     # voorbeeld diagonaal inertiamatrix in lokaal frame B
NI  = transpose(BCN)*BI*BCN # willekeurige rotatie naar het N frame (nu met kruisproducten!)
FI  = eigenwrd(NI)          # optimale I in optimaal lokaal frame F (niet perse gelijk aan B)
NCF = eigenvec(NI)          # bijbehorende rotatiematrix van F naar N
print('FI:\n', FI)
print('NCF:\n', NCF)

# kan ook direct gevonden worden uit BI
FI  = eigenwrd(BI)
BCF = eigenvec(BI)
print('FI:\n', FI)
print('BCF:\n', BCF)        # bijbehorende rotatiematrix van F naar B 
