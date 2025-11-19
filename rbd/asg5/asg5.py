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

import sympy as sm

from re import L
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

####

Ixx, Iyy, Izz, Ixy, Iyz, Ixz = (sm.symbols(
    'I_xx, I_yy, I_zz, I_xy, I_yz, I_xz', real=True))  
wx, wy, wz = sm.symbols(r'w_x, w_y, w_z', real=True)
vx, vy, vz = sm.symbols(r'v_x, v_y, v_z', real=True)
m = sm.symbols(r'm', positive=True)

vx_val = 0 #m/s
vy_val = 4*sm.sqrt(3) #m/s
vz_val = 4 #m/s
wx_val = 1 #rad/s
wy_val = 0 #rad/s
wz_val = 2 #rad/s
Ixx_val = 46*m/256 #kgm**2
Iyy_val = 125*m/2/256 #kgm**2
Izz_val = 103*m/2/256 #kgm**2
Ixy_val = 3*sm.sqrt(3)*m/256 #kgm**2
Ixz_val = -15*m/256 #kgm**2
Iyz_val = 5*sm.sqrt(3)*m/2/256 #kgm**2

sub_val = {vx:vx_val, vy:vy_val, vz:vz_val, 
           wx:wx_val, wy:wy_val, wz:wz_val,
           Ixx:Ixx_val, Iyy:Iyy_val, Izz:Izz_val,
           Ixy:Ixy_val, Ixz:Ixz_val, Iyz:Iyz_val}

FIS = sm.Matrix([[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]])

print(FIS.subs(sub_val).eigenvects())