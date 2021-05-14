#!/usr/bin/env python
# coding: utf-8

# In[22]:

import sympy
from sympy import *
from sympy.geometry import point
from sympy import Derivative as dv
from numpy import sin as nsin
from numpy import cos as ncos
import numpy as np 
import matplotlib.pyplot as plt 
import math 


# In[23]:


q1 , q2, q3, q4, q5, q6, q7,l1,l2,l3,l4,th1,th2,l5 = symbols('q1 q2 q3 q4 q5 q6 q7 self.l1 self.l2 self.l3 self.l4 self.theta1 self.theta2 self.l5')

l1 = 0.267
l2 = 0.293
l3 = 0.0525
l4 = 0.3512
l5 = 0.1232

theta1= 0.2225 #(rad) (=12.75deg)
theta2 = 0.6646 #(rad) (=38.08deg)




def A01_tf():
    q1 = symbols('q1')
    tf =Matrix([[cos(q1), -sin(q1), 0, 0],                [sin(q1),cos(q1),0,0],                [0,0,1,l1],                [0,0,0,1]])
    return tf

def A02_tf():
    q2 = symbols('q2')
    A12=Matrix([[cos(q2),-sin(q2),0,0],                [0,0,1,0],                [-sin(q2),-cos(q2),0,0],                [0,0,0,1]])
    tf = A01_tf()*A12 
    return tf

def A03_tf():
    q3 = symbols('q3')
    A23=Matrix([[cos(q3),-sin(q3),0,0],                [0,0,-1,-l2],                [sin(q3),cos(q3),0,0],                [0,0,0,1]])

    tf =A02_tf()*A23
    return tf

def A04_tf():
    q4 = symbols('q4')
    A34=Matrix([[cos(q4),-sin(q4),0,l3],                [0,0,-1,0],                [sin(q4),cos(q4),0,0],                [0,0,0,1]])

    tf = A03_tf()*A34
    return tf

def A05_tf():
    q5 = symbols('q5')
    A45=Matrix([[cos(q5),-sin(q5),0,l4*nsin(theta1)],                [0,0,-1,-l4*ncos(theta1)],                [sin(q5),cos(q5),0,0],                [0,0,0,1]])

    tf =A04_tf()*A45
    return tf

def A06_tf():
    q6 = symbols('q6')
    A56=Matrix([[cos(q6),-sin(q6),0,0],                [0,0,-1,0],                [sin(q6),cos(q6),0,0],                [0,0,0,1]])

    tf =A05_tf()*A56
    return tf

def A07_tf():
    q7 = symbols('q7')
    A67=Matrix([[cos(q7),-sin(q7),0,l5*nsin(theta2)],                [0,0,1,l5*ncos(theta2)],                [-sin(q7),-cos(q7),0,0],                [0,0,0,1]])
    tf =A06_tf()* A67
    return tf

def A04A_tf():
    s44A_tf= Matrix([[1,0,0,0],                        [0,1,0,0],                        [0,0,1,-0.069],                        [0,0,0,1]])
    tf = A04_tf()*s44A_tf
    return tf

def A04B_tf():
    s44B_tf= Matrix([[ncos(np.pi/4),-nsin(np.pi/4),0,0],                        [nsin(np.pi/4),ncos(np.pi/4),0,0],                        [0,0,1,0.10688],                        [0,0,0,1]])
    tf = A04_tf()*s44B_tf
    return tf

def A04C_tf():
    s4B4C_tf = Matrix([[1,0,0,0.126/2],                        [0,1,0,0],                        [0,0,1,0],                        [0,0,0,1]])
    tf = A04B_tf()*s4B4C_tf
    return tf

def A04D_tf():
    s4C4D_tf = Matrix([[1,0,0,0.126/2],                        [0,1,0,0],                        [0,0,1,0],                        [0,0,0,1]])
    tf = A04C_tf()*s4C4D_tf
    return tf

def A04E_tf():
    s4D4E_tf = Matrix([[1,0,0,0],                        [0,1,0,0],                        [0,0,1,-0.10688-0.042],                        [0,0,0,1]])
    tf = A04D_tf()*s4D4E_tf
    return tf

def A04F_tf():
        tf_4E4F = Matrix([[np.cos(-np.pi/4),-np.sin(-np.pi/4),0,np.cos(-np.pi/4)*0.126/2],                            [np.sin(-np.pi/4),np.cos(-np.pi/4),0,np.sin(-np.pi/4)*0.126/2],                            [0,0,1,0],                            [0,0,0,1]])
        tf = (A04E_tf()*tf_4E4F)
        return tf


# In[25]:

A07 = A07_tf()
J_11 = diff(A07[0,3],q1)
J_12 = diff(A07[0,3],q2)
J_13 = diff(A07[0,3],q3)
J_14 = diff(A07[0,3],q4)
J_15 = diff(A07[0,3],q5)
J_16 = diff(A07[0,3],q6)
J_17 = diff(A07[0,3],q7)

J_21 = diff(A07[1,3],q1)
J_22 = diff(A07[1,3],q2)
J_23 = diff(A07[1,3],q3)
J_24 = diff(A07[1,3],q4)
J_25 = diff(A07[1,3],q5)
J_26 = diff(A07[1,3],q6)
J_27 = diff(A07[1,3],q7)

J_31 = diff(A07[2,3],q1)
J_32 = diff(A07[2,3],q2)
J_33 = diff(A07[2,3],q3)
J_34 = diff(A07[2,3],q4)
J_35 = diff(A07[2,3],q5)
J_36 = diff(A07[2,3],q6)
J_37 = diff(A07[2,3],q7)


# ## Jacobian Calculation

# In[28]:


print(sympy.Matrix([ [ J_11 , J_12 , J_13 , J_14 , J_15 , J_16 , J_17 ],                        [ J_21 , J_22 , J_23 , J_24 , J_25 , J_26 , J_27 ],                        [ J_31 , J_32 , J_33 , J_34 , J_35 , J_36 , J_37 ]]))

# In[5]:


r_x ,r_y, r_z, g_x,g_y,g_z= symbols ('obs_x obs_y g_z obs_x obs_y r_z')
red_obs = Array([r_x,r_y,r_z])
green_obs = Array([g_x,g_y,g_z])


# In[6]:


pointF=(A04F_tf()[0:2,3])
pointE=(A04E_tf()[0:2,3])
pointD=(A04D_tf()[0:2,3])
pointC=(A04C_tf()[0:2,3])
pointB=(A04B_tf()[0:2,3])
pointA=(A04A_tf()[0:2,3])
P = Array([[pointA,pointB,pointC,pointD,pointE,pointF]])
# pointA=A04A_tf()[0:3,3].T


# ## Calculation of derivative of distance criterion according to the combination of points 

# In[10]:


d_safe ,kc = symbols('d_safe kc')
# Find the distances between each point and the perimeter of the Red obstacle
distAR = sqrt((pointA[0]-r_x)**2 + (pointA[1]-r_y)**2)-0.05
distBR = sqrt((pointB[0]-r_x)**2 + (pointB[1]-r_y)**2)-0.05
distCR = sqrt((pointC[0]-r_x)**2 + (pointC[1]-r_y)**2)-0.05
distDR = sqrt((pointD[0]-r_x)**2 + (pointD[1]-r_y)**2)-0.05
distER = sqrt((pointE[0]-r_x)**2 + (pointE[1]-r_y)**2)-0.05
distFR = sqrt((pointF[0]-r_x)**2 + (pointF[1]-r_y)**2)-0.05
###########################################################

# Find the distances between each point and the perimeter of the Green obstacle
distAG = sqrt((pointA[0]-g_x)**2 + (pointA[1]-g_y)**2)-0.05
distBG = sqrt((pointB[0]-g_x)**2 + (pointB[1]-g_y)**2)-0.05
distCG = sqrt((pointC[0]-g_x)**2 + (pointC[1]-g_y)**2)-0.05
distDG = sqrt((pointD[0]-g_x)**2 + (pointD[1]-g_y)**2)-0.05
distEG = sqrt((pointE[0]-g_x)**2 + (pointE[1]-g_y)**2)-0.05
distFG = sqrt((pointF[0]-g_x)**2 + (pointF[1]-g_y)**2)-0.05
    


# In[11]:


print("Grads for AR \n")
print(diff(distAR,q1),"\n")
print(diff(distAR,q2),"\n")
print(diff(distAR,q3),"\n")
print(diff(distAR,q4),"\n")
print(diff(distAR,q5),"\n")
print(diff(distAR,q6),"\n")
print(diff(distAR,q7),"\n")
print("Grads for BR \n")
print(diff(distBR,q1),"\n")
print(diff(distBR,q2),"\n")
print(diff(distBR,q3),"\n")
print(diff(distBR,q4),"\n")
print(diff(distBR,q5),"\n")
print(diff(distBR,q6),"\n")
print(diff(distBR,q7),"\n")
print("Grads for CR \n")
print(diff(distCR,q1),"\n")
print(diff(distCR,q2),"\n")
print(diff(distCR,q3),"\n")
print(diff(distCR,q4),"\n")
print(diff(distCR,q5),"\n")
print(diff(distCR,q6),"\n")
print(diff(distCR,q7),"\n")
print("Grads for DR \n")
print(diff(distDR,q1),"\n")
print(diff(distDR,q2),"\n")
print(diff(distDR,q3),"\n")
print(diff(distDR,q4),"\n")
print(diff(distDR,q5),"\n")
print(diff(distDR,q6),"\n")
print(diff(distDR,q7),"\n")
print("Grads for ER \n")
print(diff(distER,q1),"\n")
print(diff(distER,q2),"\n")
print(diff(distER,q3),"\n")
print(diff(distER,q4),"\n")
print(diff(distER,q5),"\n")
print(diff(distER,q6),"\n")
print(diff(distER,q7),"\n")
print("Grads for FR \n")
print(diff(distFR,q1),"\n")
print(diff(distFR,q2),"\n")
print(diff(distFR,q3),"\n")
print(diff(distFR,q4),"\n")
print(diff(distFR,q5),"\n")
print(diff(distFR,q6),"\n")
print(diff(distFR,q7),"\n")
print("Grads for AG \n")
print(diff(distAG,q1),"\n")
print(diff(distAG,q2),"\n")
print(diff(distAG,q3),"\n")
print(diff(distAG,q4),"\n")
print(diff(distAG,q5),"\n")
print(diff(distAG,q6),"\n")
print(diff(distAG,q7),"\n")
print("Grads for BG \n")
print(diff(distBG,q1),"\n")
print(diff(distBG,q2),"\n")
print(diff(distBG,q3),"\n")
print(diff(distBG,q4),"\n")
print(diff(distBG,q5),"\n")
print(diff(distBG,q6),"\n")
print(diff(distBG,q7),"\n")
print("Grads for CG \n")
print(diff(distCG,q1),"\n")
print(diff(distCG,q2),"\n")
print(diff(distCG,q3),"\n")
print(diff(distCG,q4),"\n")
print(diff(distCG,q5),"\n")
print(diff(distCG,q6),"\n")
print(diff(distCG,q7),"\n")
print("Grads for DG \n")
print(diff(distDG,q1),"\n")
print(diff(distDG,q2),"\n")
print(diff(distDG,q3),"\n")
print(diff(distDG,q4),"\n")
print(diff(distDG,q5),"\n")
print(diff(distDG,q6),"\n")
print(diff(distDG,q7),"\n")
print("Grads for EG \n")
print(diff(distEG,q1),"\n")
print(diff(distEG,q2),"\n")
print(diff(distEG,q3),"\n")
print(diff(distEG,q4),"\n")
print(diff(distEG,q5),"\n")
print(diff(distEG,q6),"\n")
print(diff(distEG,q7),"\n")
print("Grads for FG \n")
print(diff(distFG,q1),"\n")
print(diff(distFG,q2),"\n")
print(diff(distFG,q3),"\n")
print(diff(distFG,q4),"\n")
print(diff(distFG,q5),"\n")
print(diff(distFG,q6),"\n")
print(diff(distFG,q7),"\n")

