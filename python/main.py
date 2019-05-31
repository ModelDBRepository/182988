# -*- coding: utf-8 -*-

"""
Model of TRPM8-dependent dynamic response accompanying the publication:
TRPM8-DEPENDENT DYNAMIC RESPONSE IN A MATHEMATICAL MODEL OF COLD THERMORECEPTOR
Olivares, E. et al.

Submitted to PLoS One, July 2015
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import time as time_comp

#fixed (common) parameters
Ed = 50. ; Er = -90.; Em8 = 0.; El = -70.
tsd = 10.; tsr = 24.; tr = 1.5
ssd = 0.1; sd = 0.25; sr = 0.25
Vhsd = -40.; Vhd = -25.; Vhr = -25.
zm8 = 0.65; C = 67.; DE = 9000
Kcam8 = 0.0005; d = 1.
D = 0.5; twn = 1.
eta = 0.012; kappa = 0.18; F = 96500; R = 8.314

#parameters Sets (Table 1 of paper)
Params=np.loadtxt('parameters.txt')
Param_indx=[7,28,54,92,103,134,157,158,168,185,212,215,227,272,275,289,293,311,323,339]

#Temperature protocols
TempTrace1=np.loadtxt('short_pulses.txt',skiprows=1)
TempTrace2=np.loadtxt('temp_steps.txt',skiprows=1)

[gm8, gl, tca, tdv, pca, gsd, gsr, gd, gr, dvmin, dvmax] = [0,0,0,0,0,0,0,0,0,0,0]

rho = lambda x: 1.3**((x-25)/10)
phi = lambda x: 3.**((x-25)/10)

def integ_step(Var, dt, temp, accel):
    r = rho(temp)
    f = phi(temp)
    [v,ar,asd,asr,ca, dv, iwn]=Var 
    
    ad = 1./(1+np.exp(-sd*(v-Vhd)))                                                                         
    isd = r*gsd*asd*(v-Ed)
    
    dvinf = dvmin + (dvmax-dvmin)*ca/(ca + Kcam8)
    vhm8 = (C*R*(temp) -DE)/(zm8*F)*1000.
    
    am8 = 1./(1+np.exp(-zm8*F /(R*(temp+273.15)) *(v - vhm8 - dv)/1000.))
    im8 = gm8*am8*(v-Em8)
    
    Imemb=isd + r*gd*ad*(v - Ed) + r*(gr*ar + gsr*(asr**2)/(asr**2+0.4**2))*(v-Er) \
                + im8 + gl*(v - El)  + iwn 
    
    arinf = 1./(1+np.exp(-sr*(v-Vhr)));  
    asdinf = 1./(1+np.exp(-ssd*(v-Vhsd)));

    Delta=np.array([-Imemb,
            f*(arinf - ar)/tr, 
            f*(asdinf - asd)/tsd,
            f*(-eta*isd - kappa*asr)/tsr,
            -accel*(pca*10.0*im8/(2*F*d) + ca/tca),  # 10.0 is a correction factor for units
            (dvinf-dv)/tdv,
            -iwn + D*np.random.normal(0)/np.sqrt(dt)])
    return Delta
            
#Implementacion de Metodo de Euler Maruyama
def EulerV(integ_step,X,N,dt,temp):  #Voltage trace at a fixed temperature
    datos = np.zeros((7,N))
        
    #First an adaptation run
    for i in np.arange(15000/dt):
        X += + dt * integ_step(X, dt, temp, 100)

    #Then the real run    
    datos[:,0] = X
    i=1       
#    while i<N/4:
#        #Metodo de Euler. Otros metodos mas sofisticados se pueden implementar sobre la base de HyB()
#        datos[:,i]=datos[:,i-1] + dt * integ_step(datos[:,i-1], dt, temp, 100)
#        i+=1        
    while i<N:
        #Metodo de Euler. Otros metodos mas sofisticados se pueden implementar sobre la base de HyB()
        datos[:,i]=datos[:,i-1] + dt * integ_step(datos[:,i-1], dt, temp, 1)
        i+=1
    return datos 
    
def EulerS(integ_step,X,N,dt,temp): #No voltage trace, only spikes
    datos = X
    i=1 
    firing = 0
    thresh=-30      
    spikes=[]
    t0=time_comp.time()
    
    #First an adaptation run
    while i*dt<20000:
        if i%100000==0:
            print "*ADAPTATION* t=",i*dt,"real t=",time_comp.time()-t0,"remaining=",(N-i)*(time_comp.time()-t0)/i
        datos += + dt * integ_step(datos, dt, temp[i], 100)
        if (firing==0)*(datos[0]>thresh):
            spikes.append(i*dt)
            firing=1
        if firing*(datos[0]<thresh):
            firing=0
        i+=1        
        
    while i<N:
        if i%100000==0:
            print "t=",i*dt,"real t=",time_comp.time()-t0,"remaining=",(N-i)*(time_comp.time()-t0)/i
        datos += + dt * integ_step(datos, dt, temp[i], 1)
        if (firing==0)*(datos[0]>thresh):
            spikes.append(i*dt)
            firing=1
        if firing*(datos[0]<thresh):
            firing=0
        i+=1
    return np.array(spikes) 
    
#Simulacion  
    
def simulation(ParamSet=185,Temp=33.5):
    dt=0.05    
    if isinstance(Temp,(float,int)):
        tEnd=4000
        time= np.arange(0, tEnd, dt)
        Temp_t=Temp*np.ones_like(time)
    elif type(Temp)==np.ndarray:
        tEnd=Temp[-1,0]*1000
        time= np.arange(0, tEnd, dt)
        Temp_t=np.interp(time,Temp[:,0]*1000,Temp[:,1])
    else:
        return 0

    P_ind=Param_indx.index(ParamSet)
    
    global gm8, gl, tca, tdv, pca, gsd, gsr, gd, gr, dvmin, dvmax
    [gm8, gl, tca, tdv, pca, gsd, gsr, gd, gr, dvmin, dvmax]=Params[P_ind]
    
    #In Neuron conductances are S/cm2, so we need to correct to mS/cm2
    gm8*=1000;gl*=1000;gsd*=1000;gsr*=1000;gd*=1000;gr*=1000  
    
    N = time.size

    #initial conditions
    v=-65 
    
    r = rho(Temp_t[0])
    f = phi(Temp_t[0])
    
    ad = 1./(1+np.exp(-sd*(v-Vhd)));                          
    ar = 1./(1+np.exp(-sr*(v-Vhr)));
    asd = 1./(1+np.exp(-ssd*(v-Vhsd)));
    asr = -eta*r*gsd*asd*(v - Ed)/kappa; 
    
    dv = (37-Temp_t[0])*5.
    vhm8 = (C*R*(Temp_t[0]) -DE)/(zm8*F)*1000.
    am8 = 1./(1+np.exp(-zm8*F /(R*(Temp_t[0]+273.15)) *(v - vhm8 - dv)/1000))
    im8 = gm8*am8*(v-Em8)
    ca = -pca*10*im8/(2*F*d)*tca
    dv = dvmin + (dvmax-dvmin)*ca/(ca + Kcam8)
#    print vhm8,am8,ca,dv
    
    iwn = 0

    X=np.array([v,ar,asd,asr,ca, dv,iwn]) 
    if isinstance(Temp,(float,int)):
        return time,EulerV(integ_step,X,N,dt,Temp)
    else:
        return time,Temp_t,EulerS(integ_step,X,N,dt,Temp_t)


if __name__=='__main__':

        
    plt.figure(1,figsize=(8,12))    
    plt.clf()
    p=1
    for T in (34,30,26.,22.):
        print "Simulating temp=",T
        time,Vars = simulation(ParamSet=293,Temp=T)
        plt.subplot(4,1,p)
        plt.plot(time,Vars[0,:],'k')
        plt.ylabel('Voltage (mV)')
        plt.text(3500,0,u'%.2g °C'%T,bbox=dict(facecolor='white'))
        if p==4:
            plt.xlabel('Time (ms)')      
        p+=1
    
    time,temp,spikes=simulation(ParamSet=158,Temp=TempTrace1)
    
    plt.figure(2,figsize=(12,8))
    plt.clf()
    plt.subplot(311)
    plt.plot(time,temp,'k')
    plt.ylabel(u'Temp (°C)')
    plt.subplot(312)
    plt.hist(spikes,bins=np.arange(0,time[-1],1000))
    plt.ylim((0,70))
    plt.ylabel('Rate (/s)')
    plt.subplot(313)
    plt.plot(spikes[1:],np.diff(spikes),'k.')
    plt.yscale('log')
    plt.ylim((10,3000))
    plt.ylabel('ISIs (ms)')
    plt.xlabel('Time (ms)')
    
    plt.show()
    
       
    
    
 