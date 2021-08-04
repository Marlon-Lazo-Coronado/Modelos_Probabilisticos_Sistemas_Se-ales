# -*- coding: utf-8 -*-
"""
Created on Mon May 25 00:23:51 2020

@author: Alicia
"""
import csv
from math import *
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import scipy as sp
from scipy.stats import rayleigh
from scipy.stats import kurtosis
from scipy.stats import skew
from matplotlib import pyplot as plt

def main():

    
##########################################################################
    #Para convertir csv a lista
    with open('Tarea_Paython2.csv', 'r') as file:
       temp = csv.reader(file)
       for i in temp:
           P=list(temp)
    
    lista=[P[i][0] for i in range(4999)]
    N=[float(j) for j in lista]
    #N.insert(0, 67.5142)
    
    #print(N)
################################################################################   
    
    
   
    #Leemos el csv
    datos=pd.read_csv('Tarea_Paython2.csv', header=None, delim_whitespace=True)
   
    
    '''
    def f3(x):
        y = (x/896.763)*np.exp((-x**2)/1793.526)
        return y
    x = np.linspace(0, 30*pi, 2000)
    p3 = plt.plot(x, f3(x), label='Modelo Rayleigh')
    sns.distplot(datos, fit=stats.rayleigh)
    plt.legend(('Funcion 3'), prop = {'size': 10}, loc='upper right')
    plt.xlabel('Variable Aleatoria')
    plt.ylabel('Probabilidad')
    plt.title('Comparacion de las Tres Curvas')
    plt.legend(loc=1)
    plt.show()
    '''
    
    
########################################################################### 
    
    '''
    #Graficamos
    C=datos.hist(bins=100, density=True, edgecolor='black')
    A=datos.hist(bins=100)
    
    plt.title('Histograma')
    plt.xlabel('Variable Aleatoria')
    plt.ylabel('Probabilidad')
    plt.show(C)
    '''
    
    '''
    sns.distplot(datos, bins=100) #Curva de ajuste
    sns.distplot(datos, fit=stats.rayleigh, label='Ajuste_Raylegh') #Ajuste con Rayleigh
    plt.title('Mejor Curva de Ajuste Rayleigh del Histograma')
    plt.xlabel('Valores de los Datos de la Variable Aleatoria')
    plt.ylabel('Probabilidad')
    #plt.show(A)
    #print(float(np.mean(datos)))#CALCULO LA MEDIA
    '''

##############################################################################333  
    #Calculamos la probabilidad
    
    #a=rayleigh.cdf(100, loc=0, scale=29.946) #Calcula P(X<10)
    #print(a)
    
    #a = rayleigh.cdf(17, loc=0, scale=29.946) #Calcula P(X< 30)y lo guarda en a
    #b = rayleigh.cdf(37, loc=0, scale=29.946)  #Calcula P(X<60)y la almacena en b
    #print('Probabilidad en el intervalo [b-a] es: ', b-a) 
    
    
    #Calculamosla probabilidad usando el histograma. Convertimos csv a vector
    with open('Tarea_Paython2.csv', 'r') as file:
       temp = csv.reader(file)
       for i in temp:
           P=list(temp)
    
    lista=[P[i][0] for i in range(4999)]
    #lista=[float(j) for j in tem2]
    #print(N)

    
    #N=sorted(lista)
    subN=[i for i in N if i>=17 and i<= 37]
    print(len(subN))
    print(str(len(subN)/5000))
    
    '''
    #Esto no me funciona, averiguar por que!
    N=sorted(lista)
    for i in N:
        if (i>=17 and i<= 37):
            subN=i
    print(len(subN))
    print(str(len(subN)/5000))
    '''

############################################################################
    
    '''
    #Ese scale es el sigma de la funcion yque se despeja de obtener la media
    m=rayleigh.mean(loc=0, scale=29.946)
    print('La media es: ', m)
    v=rayleigh.var(loc=0, scale=29.946)
    print('La varianzaes: ', v)
    d=rayleigh.std(loc=0, scale=29.946)
    print('La desviacion estandar es: ', d)
    k=kurtosis(datos)               
    print('La curtosis es: ', float(k))
    s=skew(datos)                   
    print('La inclinacion es: ', float(s))

                     
    print('\n')
    print(float(np.mean(datos)))#CALCULO LA MEDIA
    print(float(np.var(datos)))
    print(float(np.std(datos)))
    print(float(datos.mean()))#CALCULO LA MEDIA
    print(float(datos.var()))
    print(float(datos.std()))
    '''
###############################################################################
    
    '''
    #Tranformacion de x
    A=[]
    for i in range(0,5001):
        A=datos**(1/2)
    print(A)
    B=A.hist(bins=100)
    
    sns.distplot(A, fit=stats.norm) #Bien!
    
    plt.title('Histograma')
    plt.xlabel('Variable Aleatoria')
    plt.ylabel('Probabilidad')
    
    
    #C=A.hist(bins=100, density=True, edgecolor='black')
    #plt.title('Histograma')
    #plt.xlabel('Trasformacion de la Variable Aleatoria')
    #plt.ylabel('Probabilidad')
    #plt.show(C)
    #print(float(np.mean(A)))
    #print(float(np.std(A)))
    '''
###############################################################################
    '''
    def f2(x):
        y = 0.2452*np.exp(-0.5*((x-5.9063)**2)/(2.6461))
        return y
    x = np.linspace(0, 4*pi, 2000)
    p3 = plt.plot(x, f2(x), label='Modelo Gaussiano')
    sns.distplot(A)
    plt.legend(('Funcion 3'), prop = {'size': 10}, loc='upper right')
    plt.xlabel('Variable Aleatoria')
    plt.ylabel('Probabilidad')
    plt.title('Comparacion entre las dos curvas')
    plt.legend(loc=1)
    plt.show()
    '''
    
    
main()






















#import numpy as np
#from scipy import stats
#from scipy.stats import expon
#expon.cdf(10, loc=0, scale=22) #Calcula P(X<10)



#a = expon.cdf(30, loc=0, scale=22)
#Calcula P(X< 30)y lo guarda en a
#b = expon.cdf(60, loc=0, scale=22)
#Calcula P(X<60)y la almacena en b
#b-a #Calcula P(X<60)-P(X< 30)=P(30<X<60)








    #ax = plt.subplots(1)
    #x = np.linspace(rayleigh.ppf(0.01), rayleigh.ppf(0.99), 100)
    #ax.plot(x, rayleigh.pdf(x), 'r-', lw=5, alpha=14.413000306364738, label='rayleigh pdf')
    #hist(r, density=True, histtype='stepfilled', alpha=0.2)
    #legend(loc='best', frameon=False)
    #plt.show()
    #rv = rayleigh()
    #ax.plot(datos, rv.pdf(x), 'k-', lw=2, label='frozen pdf')





    #x = np.linspace(rayleigh.ppf(0.01),    #Esto funciona
                #rayleigh.ppf(0.99), 100)
    #plt.plot(x, rayleigh.pdf(x))
    #plt.title(‘Distribución Exponencial’)
    #plt.ylabel(‘f(x)’)
    #plt.xlabel(‘X’)
    #plt.show()












