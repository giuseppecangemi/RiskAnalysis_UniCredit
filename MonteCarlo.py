#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 18:35:06 2022

@author: giuseppecangemi
"""
import numpy as np
import matplotlib.pyplot as plt
#df e ulteriori librerie dallo script sul calcolo del VaR classico


days = 365   # considero un anno
dt = 1/float(days)
sigma = 0.0293 # volatilità
mu = 0.05  # drift positivo


def randomwalk(startprice):
    price = np.zeros(days)
    shock = np.zeros(days)
    price[0] = startprice
    for t in range(1, days):
        shock[t] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        price[t] = max(0, price[t-1] + shock[t] * price[t-1])
    return price

plt.figure(figsize=(9,4))    
for simul in range(40):
    plt.plot(randomwalk(12.536)) #ultimo prezzo di chiusura adj.
    plt.xlabel("Time")
    plt.ylabel("Price")

runs = 10000
simulations = np.zeros(runs)

for run in range(runs):
    simulations[run] = randomwalk(12.536)[days-1]
q = np.percentile(simulations, 5)
plt.hist(simulations, density=True, bins=30, alpha=0.5)
plt.axvline(x=q, linewidth=3, color="r")
plt.title("VaR(0.95):{:.3}euro".format(12.536-q))

var = 12.536-q
max_loss = 10000*var
max_loss
#5691.19 euro



