# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 12:29:02 2020

@author: Chaitanya Sampara
"""
import matplotlib.pyplot as plt
from numpy import zeros,cos,sin
nsteps = 100000 #no. of iterations

# constants of system
g = 9.8
m = 0.6*(10**-3)
M = 0.054
rcm = 0.08
I = m*(rcm**2)
freq = [10.9, 10.9]
eps = 0.1106
beta = [0.011, 0.011]
theta0 = 0.392
phi0 = 0.392
w = (m*rcm*g/I)**(1/2)
delta = (freq[0] - freq[1])/w
mu = eps/w

#initialization of theta and phi array
theta  = zeros(nsteps)
phi = zeros(nsteps)
x = zeros(nsteps)
y = zeros(nsteps)
t = zeros(nsteps)
tau = zeros(nsteps)

#setting intial values of theta and phi
theta[0] = 0.9
phi[0] = -0.9
t[0] = 0
x[0] = 0
y[0] = 0
dt = 0.01 #time period between each iteration

#functions to help set up the dfferential equation
def A1(th):
    return 1 - beta[0]*((cos(th))**2)
def A2(ph):
    return 1 - beta[1]*((cos(ph))**2)
def K1(th):
    return mu*((th/theta0)**2 - 1)
def K2(ph):
    return mu*((ph/phi0)**2 - 1)
def C1(th,ph):
    temp1 = beta[0]*cos(th)*cos(ph)/(A1(th)*A2(ph))
    return temp1
def C2(th,ph):
    temp2 = beta[1]*cos(th)*cos(ph)/(A1(th)*A2(ph))
    return temp2
def f1(x,y,th,ph):# D.E for theta'
    temp3 = 1 - beta[1]*cos(th)*cos(ph)*C1(th,ph)
    temp4 = C1(th,ph)*beta[1]*cos(ph)*sin(th) + beta[0]*cos(th)*sin(ph)
    temp5 = C1(th,ph)*beta[1]*cos(ph)*sin(ph) + beta[0]*cos(th)*sin(th)
    dxdt = (- (K1(th)/A1(th))*x - K2(ph)*C1(th,ph)*y - C1(th,ph)*(1 - delta)*sin(ph) - 
           (1 + delta)*sin(th)/A1(th) - temp4*(x**2) - temp5*(y**2))/temp3
    return dxdt
def f2(x,y,th,ph):# D.E for phi'
    temp6 = 1 - beta[0]*cos(ph)*cos(th)*C2(th,ph)
    temp7 = C2(th,ph)*beta[0]*cos(th)*sin(th) + beta[1]*cos(ph)*sin(th)
    temp8 = C2(th,ph)*beta[0]*cos(th)*sin(ph) + beta[1]*cos(ph)*sin(ph)
    dydt = (- (K2(ph)/A2(ph))*y - K1(th)*C2(th,ph)*x - C2(th,ph)*(1 + delta)*sin(th) - 
           (1 - delta)*sin(ph)/A2(ph) - temp7*(x**2) - temp8*(y**2))/temp6
    return dydt
def f3(x):# D.E for theta
    dthdt = x
    return dthdt
def f4(y):# D.E for phi
    dphdt = y
    return dphdt

#Runge Kutta 4th Order method
for i in range(0,nsteps-1):
    k11 = dt*f1(x[i],y[i],theta[i],phi[i])
    k21 = dt*f2(x[i],y[i],theta[i],phi[i])
    k31 = dt*f3(x[i])
    k41 = dt*f4(y[i])
    
    k12 = dt*f1(x[i]+k11*0.5,y[i]+k21*0.5,theta[i]+k31*0.5,phi[i]+k41*0.5)
    k22 = dt*f2(x[i]+k11*0.5,y[i]+k21*0.5,theta[i]+k31*0.5,phi[i]+k41*0.5)
    k32 = dt*f3(x[i]+k11*0.5)
    k42 = dt*f4(y[i]+k21*0.5)
    
    k13 = dt*f1(x[i]+k12*0.5,y[i]+k22*0.5,theta[i]+k32*0.5,phi[i]+k42*0.5)
    k23 = dt*f2(x[i]+k12*0.5,y[i]+k22*0.5,theta[i]+k32*0.5,phi[i]+k42*0.5)
    k33 = dt*f3(x[i]+k12*0.5)
    k43 = dt*f4(y[i]+k22*0.5)
    
    k14 = dt*f1(x[i]+k13,y[i]+k23,theta[i]+k33,phi[i]+k43)
    k24 = dt*f2(x[i]+k13,y[i]+k23,theta[i]+k33,phi[i]+k43)
    k34 = dt*f3(x[i]+k13)
    k44 = dt*f4(y[i]+k23)
    
    x[i+1] = x[i] + 1/6*(k11 + k12*2 + k13*2 + k14)
    y[i+1] = y[i] + 1/6*(k21 + k22*2 + k23*2 + k24)
    theta[i+1] = theta[i] + 1/6*(k31 + k32*2 + k33*2 + k34)
    phi[i+1] = phi[i] + 1/6*(k41 + k42*2 + k43*2 + k44)
    t[i+1] = t[i] + dt
    tau[i+1] = t[i+1]*w

plt.plot(tau,theta,"-r")
plt.plot(tau,phi,"-b")
plt.show