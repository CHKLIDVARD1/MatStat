#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp


# In[176]:


def Z_Test_description(sample, alpha, M0, std, type_, M1 = None,W = False): 
    n = sample.size
    sample_mean = sample.mean()
    z_stat = np.sqrt(len(sample))*(sample_mean-M0) / std
    
    
    if type_ == 'both sides':
        cvalue1 = st.norm.isf(alpha/2)
        cvalue2 = -st.norm.isf(alpha/2)
        print(f'Z наблюдаемая = {z_stat}, критическая область : (-inf, {cvalue2}),({cvalue1}, +inf)')
        pvalue = 2*min(st.norm().cdf(z_stat), st.norm().sf(z_stat))
        if pvalue > alpha:
            print(f'Гипотеза H0 не отклоняется, p-value = {pvalue}, alpha = {alpha}')
        else:
            print(f'Гипотеза H0 отклоняется, p-value = {pvalue}, alpha = {alpha}')
        if W == True:
            B = st.norm.cdf(st.norm.isf(alpha/2) - (n**0.5)/std * (M0-M1)) + st.norm.cdf(st.norm.isf(alpha/2) + (n**0.5)/std * (M0-M1)) - 1 
            W = 1-B
            print(f'B = {B}, W = {W}')
            
        x = np.linspace(-3,3, 200)
        c1space = np.linspace(cvalue1, 3,  200)
        c2space = np.linspace(-3, cvalue2, 200)
        plt.subplots(figsize = (8, 6))
        plt.plot(x, st.norm.pdf(x))
        plt.plot(cvalue2, st.norm.pdf(cvalue2), 'ro', label = 'Левая критическая точка')
        plt.plot(cvalue1, st.norm.pdf(cvalue1), 'ro', label = 'Правая критическая точка')
        plt.plot(z_stat, st.norm.pdf(z_stat), 'go', label = 'Значение Z наблюдаемой')
        plt.fill_between(c1space, st.norm.pdf(c1space), y2 = 0, alpha = 0.3, color = 'red')
        plt.fill_between(c2space, st.norm.pdf(c2space), y2 = 0, alpha = 0.3, color = 'red')
        plt.legend(loc = 'upper left');
        
        
    elif type_ == 'left side':
        cvalue = st.norm.ppf(alpha)
        print(f'Z наблюдаемая = {z_stat}, критическая область : (-inf, {cvalue})')
        pvalue = st.norm().cdf(z_stat)
        if pvalue > alpha:
            print(f'Гипотеза H0 не отклоняется, p-value = {pvalue}, alpha = {alpha}')
        else:
            print(f'Гипотеза H0 отклоняется, p-value = {pvalue}, alpha = {alpha}')
        if W == True:
            B = st.norm.cdf(st.norm.ppf(alpha) - (n**0.5)/std * (M0-M1)) 
            W = 1-B
            print(f'B = {B}, W = {W}')
            
        x = np.linspace(-3,3, 200)
        cspace = np.linspace(-3, cvalue, 200)
        plt.subplots(figsize = (8, 6))
        plt.plot(x, st.norm.pdf(x))
        plt.plot(cvalue, st.norm.pdf(cvalue), 'ro', label = 'Критическая точка')
        plt.plot(z_stat, st.norm.pdf(z_stat), 'go', label = 'Значение Z наблюдаемой')
        plt.fill_between(cspace, st.norm.pdf(cspace), y2 = 0, alpha = 0.3, color = 'red')
        plt.legend(loc = 'upper left');
    
    
    elif type_ == 'right side':
        cvalue = st.norm.isf(alpha)
        print(f'Z наблюдаемая = {z_stat}, критическая область : (-inf, {cvalue})')
        pvalue = st.norm().sf(z_stat)
        if pvalue > alpha:
            print(f'Гипотеза H0 не отклоняется, p-value = {pvalue}, alpha = {alpha}')
        else:
            print(f'Гипотеза H0 отклоняется, p-value = {pvalue}, alpha = {alpha}')
        if W == True:
            B = st.norm.cdf(st.norm.isf(alpha) - (n**0.5)/std * (M0-M1)) 
            W = 1-B
            print(f'B = {B}, W = {W}')
              
        x = np.linspace(-3,3, 200)
        cspace = np.linspace(cvalue, 3, 200)
        plt.subplots(figsize = (8, 6))
        plt.plot(x, st.norm.pdf(x))
        plt.plot(cvalue, st.norm.pdf(cvalue), 'ro', label = 'Критическая точка')
        plt.plot(z_stat, st.norm.pdf(z_stat), 'go', label = 'Значение Z наблюдаемой')
        plt.fill_between(cspace, st.norm.pdf(cspace), y2 = 0, alpha = 0.3, color = 'red')
        plt.legend(loc = 'upper left');
    
    
    else:
        print('Вы неверно указали тип H1, попробуйте both sides, right side или left side')


# In[78]:


def Z_Test(sample, alpha, M0, std, type_, M1 = None, W = False): 
    n = sample.size
    sample_mean = sample.mean()
    z_stat = np.sqrt(len(sample))*(sample_mean-M0) / std
    
    
    if type_ == 'both sides':
        cvalue1 = st.norm.isf(alpha/2)
        cvalue2 = -st.norm.isf(alpha/2)
        pvalue = 2*min(st.norm().cdf(z_stat), st.norm().sf(z_stat))
        if W == True:
            B = st.norm.cdf(st.norm.isf(alpha/2) - (n**0.5)/std * (M0-M1)) + st.norm.cdf(st.norm.isf(alpha/2) + (n**0.5)/std * (M0-M1)) - 1 
            W = 1-B
            return {'p-value': pvalue, 'alpha' : alpha, 'Z-stat': z_stat, 'c-value1' : cvalue1,                    'c-value2' : cvalue2, 'B' : B, 'W' : W}
        elif W == False:
            return {'p-value': pvalue, 'alpha' : alpha, 'Z-stat': z_stat, 'c-value1' : cvalue1,                   'c-value2' : cvalue2} 
        
        
    elif type_ == 'left side':
        cvalue = st.norm.ppf(alpha)
        pvalue = st.norm().cdf(z_stat)
        if W == True:
            B = st.norm.cdf(st.norm.ppf(alpha) - (n**0.5)/std * (M0-M1))
            W = 1-B
            return {'p-value': pvalue, 'alpha' : alpha, 'Z-stat': z_stat, 'c-value' : cvalue, 'B' : B, 'W' : W}
        elif W == False:
            return {'p-value': pvalue, 'alpha' : alpha, 'Z-stat': z_stat, 'c-value' : cvalue}
    
    
    elif type_ == 'right side':
        cvalue = st.norm.isf(alpha)
        pvalue = st.norm().sf(z_stat)
        if W == True:
            B = st.norm.cdf(st.norm.isf(alpha) - (n**0.5)/std * (M0-M1))
            W = 1-B
            return {'p-value': pvalue, 'alpha' : alpha, 'Z-stat': z_stat, 'c-value' : cvalue, 'B' : B, 'W' : W}
        elif W == False:
            return {'p-value': pvalue, 'alpha' : alpha, 'Z-stat': z_stat, 'c-value' : cvalue}
    
    
    else:
        print('Вы неверно указали тип H1, попробуйте both sides, right side или left side')


# In[105]:


def T_Test(sample, alpha, M0, type_, M1 = None, W = False): 
    sample_mean = sample.mean()
    n = sample.size
    t_stat = n**(1/2) * (sample_mean - M0) / sample.var(ddof = 1)**(1/2)
    if type_ == 'both sides':
        cvalue1 = st.t(n-1).isf(alpha/2)
        cvalue2 = -st.t(n-1).isf(alpha/2)
        pvalue = 2*min(st.t(n-1).cdf(t_stat), st.t(n-1).sf(t_stat))
        if W == True:
            delta = np.sqrt(n) * (M1 - M0) / sample.var(ddof = 1)**(1/2)
            B = st.nct(nc = delta, df = n-1).cdf(st.t(n-1).isf(alpha/2)) - st.nct(nc = delta, df = n-1).cdf(-st.t(n-1).isf(alpha/2))
            W = 1-B
            return {'p-value': pvalue, 'alpha' : alpha, 'T-stat': t_stat, 'c-value1' : cvalue1,                    'c-value2' : cvalue2, 'B' : B, 'W' : W}
        elif W == False:
            return {'p-value': pvalue, 'alpha' : alpha, 'T-stat': t_stat, 'c-value1' : cvalue1,                   'c-value2' : cvalue2} 
        
        
    elif type_ == 'left side':
        cvalue = st.t(n-1).ppf(alpha)
        pvalue = st.t(n-1).sf(t_stat)
        if W == True:
            delta = np.sqrt(n) * (M1 - M0) / sample.var(ddof = 1)**(1/2)
            B = st.nct(nc = delta, df = n-1).sf(st.t(n-1).ppf(alpha))
            W = 1-B
            return {'p-value': pvalue, 'alpha' : alpha, 'T-stat': t_stat, 'c-value' : cvalue,                    'B' : B, 'W' : W}
        elif W == False:
            return {'p-value': pvalue, 'alpha' : alpha, 'T-stat': t_stat, 'c-value' : cvalue,} 
        
        
    elif type_ == 'right side':
        cvalue = st.t(n-1).ppf(alpha)
        pvalue = st.t(n-1).cdf(t_stat)
        if W == True:
            delta = np.sqrt(n) * (M1 - M0) / sample.var(ddof = 1)**(1/2)
            B = st.nct(nc = delta, df = n-1).cdf(st.t(n-1).isf(alpha))
            W = 1-B
            return {'p-value': pvalue, 'alpha' : alpha, 'T-stat': t_stat, 'c-value' : cvalue,                    'B' : B, 'W' : W}
        elif W == False:
            return {'p-value': pvalue, 'alpha' : alpha, 'T-stat': t_stat, 'c-value' : cvalue,} 
        
        
    else:
        print('Вы неверно указали тип H1, попробуйте both sides, right side или left side')


# In[169]:


def T_Test_description(sample, alpha, M0, type_, M1 = None,W = False): 
    sample_mean = sample.mean()
    n = sample.size
    t_stat = n**(1/2) * (sample_mean - M0) / sample.var(ddof = 1)**(1/2)
    
    
    if type_ == 'both sides':
        cvalue1 = st.t(n-1).isf(alpha/2)
        cvalue2 = -st.t(n-1).isf(alpha/2)
        pvalue = 2*min(st.t(n-1).cdf(t_stat), st.t(n-1).sf(t_stat))
        print(f'T наблюдаемая = {t_stat}, критическая область : (-inf, {cvalue2}),({cvalue1}, +inf)')
        if pvalue > alpha:
            print(f'Гипотеза H0 не отклоняется, p-value = {pvalue}, alpha = {alpha}')
        else:
            print(f'Гипотеза H0 отклоняется, p-value = {pvalue}, alpha = {alpha}')
        if W == True:
            delta = np.sqrt(n) * (M1 - M0) / sample.var(ddof = 1)**(1/2)
            B = st.nct(nc = delta, df = n-1).cdf(st.t(n-1).isf(alpha/2)) - st.nct(nc = delta, df = n-1).cdf(-st.t(n-1).isf(alpha/2))
            W = 1-B
            print(f'B = {B}, W = {W}')
            
        x = np.linspace(-3,3, 200)
        c1space = np.linspace(cvalue1, 3, 200) 
        c2space = np.linspace(-3, cvalue2, 200) 
        plt.subplots(figsize = (8, 6))
        plt.plot(x, st.t(n-1).pdf(x))
        plt.plot(cvalue2, st.t(n-1).pdf(cvalue2), 'ro', label = 'Левая критическая точка')
        plt.plot(cvalue1, st.t(n-1).pdf(cvalue1), 'ro', label = 'Правая критическая точка')
        plt.plot(t_stat, st.t(n-1).pdf(t_stat), 'go', label = 'Значение T наблюдаемой')
        plt.fill_between(c1space, st.t(n-1).pdf(c1space), y2 = 0, alpha = 0.3, color = 'red')
        plt.fill_between(c2space, st.t(n-1).pdf(c2space), y2 = 0, alpha = 0.3, color = 'red')
        plt.legend(loc = 'upper left');
        
        
    elif type_ == 'left side':
        cvalue = st.t(n-1).ppf(alpha)
        pvalue = st.t(n-1).sf(t_stat)
        print(f'T наблюдаемая = {t_stat}, критическая область : (-inf, {cvalue})')
        if pvalue > alpha:
            print(f'Гипотеза H0 не отклоняется, p-value = {pvalue}, alpha = {alpha}')
        else:
            print(f'Гипотеза H0 отклоняется, p-value = {pvalue}, alpha = {alpha}')
        if W == True:
            delta = np.sqrt(n) * (M1 - M0) / sample.var(ddof = 1)**(1/2)
            B = st.nct(nc = delta, df = n-1).sf(st.t(n-1).ppf(alpha))
            W = 1-B
            print(f'B = {B}, W = {W}')
            
        x = np.linspace(-3,3, 200)
        cspace = np.linspace(-3, cvalue, 200)
        plt.subplots(figsize = (8, 6))
        plt.plot(x, st.t(n-1).pdf(x))
        plt.plot(cvalue, st.t(n-1).pdf(cvalue), 'ro', label = 'Критическая точка')
        plt.plot(t_stat, st.t(n-1).pdf(t_stat), 'go', label = 'Значение T наблюдаемой')
        plt.fill_between(cspace, st.t(n-1).pdf(cspace), y2 = 0, alpha = 0.3, color = 'red')
        plt.legend(loc = 'upper left');
    
    
    elif type_ == 'right side':
        cvalue = st.t(n-1).isf(alpha)
        pvalue = st.t(n-1).cdf(t_stat)
        print(f'T наблюдаемая = {t_stat}, критическая область : (-inf, {cvalue})')
        if pvalue > alpha:
            print(f'Гипотеза H0 не отклоняется, p-value = {pvalue}, alpha = {alpha}')
        else:
            print(f'Гипотеза H0 отклоняется, p-value = {pvalue}, alpha = {alpha}')
        if W == True:
            delta = np.sqrt(n) * (M1 - M0) / sample.var(ddof = 1)**(1/2)
            B = st.nct(nc = delta, df = n-1).cdf(st.t(n-1).isf(alpha))
            W = 1-B
            print(f'B = {B}, W = {W}')
              
        x = np.linspace(-3,3, 200)
        cspace = np.linspace(cvalue, 3, 200)
        plt.subplots(figsize = (8, 6))
        plt.plot(x, st.t(n-1).pdf(x))
        plt.plot(cvalue, st.t(n-1).pdf(cvalue), 'ro', label = 'Критическая точка')
        plt.plot(t_stat, st.t(n-1).pdf(t_stat), 'go', label = 'Значение T наблюдаемой')
        plt.fill_between(cspace, st.t(n-1).pdf(cspace), y2 = 0, alpha = 0.3, color = 'red')
        plt.legend(loc = 'upper left');
    
    
    else:
        print('Вы неверно указали тип H1, попробуйте both sides, right side или left side')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




