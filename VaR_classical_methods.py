#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 14:47:29 2022

@author: giuseppecangemi
"""

import numpy as np
import pandas as pd
import scipy.stats as sp
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from yahoofinancials import YahooFinancials as yf


#stabilisco asset e range data:
ticker = "UCG.MI" #unicredit
start = "2020-01-01"
end = "2022-12-03"

#ottengo i dati di UniCredit in forma di dictionary:
unicredit_data = yf(ticker).get_historical_price_data(start, end, "daily")
#creo dataframe:
df = pd.DataFrame(columns=["open","close","adjclose"])
#ottengo prezzo di apertura, chiusura e adj.close e li inserisco nel dataframe:
for row in unicredit_data[ticker]["prices"]:
    data = dt.fromisoformat(row["formatted_date"])
    df.loc[data] = [row["open"], row["close"], row["adjclose"]]
    df.index.name = "data"

#creo grafico per l'adj.close:
plt.plot(df["adjclose"])
plt.title("UniCredit stock price 2020-2022")
#creo grafico per il prezzo di chiusura:
plt.plot(df["close"])
plt.title("UniCredit stock price 2020-2022")  
#confronto i due prezzi di chiusura (close-adj.close):
plt.plot(df["close"], label = "Close Price")
plt.plot(df["adjclose"], label = "Adj.close price")
plt.title("UniCredit stock price 2020-2022")  
plt.legend()  
    
#calcolo differenze prime (ritorni giornalieri):
df["return_adjclose"] = (df["adjclose"] - df["adjclose"].shift(1))/df["adjclose"].shift(1) 
# o usare il comando:  df["return_adjclose"].pct_change()  
df["return_close"] = (df["close"] - df["close"].shift(1))/df["close"].shift(1)
# o usare ul comando di cui sopra

#plotto le differenze prime dei prezzi di chiusura:
plt.plot(df["return_adjclose"], label = "Adj.close") 
plt.title("Ritorni giornalieri UniCredit")
plt.legend()

plt.plot(df["return_close"], label = "Close") 
plt.title("Ritorni giornalieri UniCredit")   
plt.legend()

plt.plot(df["return_adjclose"], label = "Adj.close", color="orange") 
plt.plot(df["return_close"], label = "Close", alpha = 0.4, color = "blue", linestyle = "--") 
plt.title("Ritorni giornalieri UniCredit")   
plt.legend()

#plotto la distribuzione dei ritorni giornalieri:
#plt.hist(df["return_close"], bins=60)    #matplotlib library
sns.histplot(df["return_close"], kde=True, label = "Close" )
sns.histplot(df["return_adjclose"], kde=True, label = "Adj.close")

#calcolo la deviazione standard dei ritorni:
std_adjclose = df["return_adjclose"].std()
std_close = df["return_close"].std()
mean = df["return_adjclose"].mean() 
 
#plotto grafico per capire se la distribuzione normale fitta bene con i dati
s = df["return_adjclose"].dropna()
sp.probplot(s, dist=sp.norm, plot=plt.figure().add_subplot(111))
#noto che ci sono delle code grasse, pertanto provo con una distribuzione t-student

#ottengo parametri gradi di libertà, media e std.deviation:
tdf, tmean, tsigma = sp.t.fit(s)    
sp.probplot(s, dist=sp.t, sparams=(tdf, tmean, tsigma), plot=plt.figure().add_subplot(111))

#plotto differenze tra le due distribuzioni (normale e t-student)
fig, (ax1, ax2) = plt.subplots(2)
sp.probplot(s, dist=sp.norm, plot=ax1)
sp.probplot(s, dist=sp.t, sparams=(tdf, tmean, tsigma), plot=ax2)
fig.suptitle("Differenze distribuzioni  sopra:Normale - sotto:t-Student")
#la distribuzione t-student fitta meglio i dati

#Dal seguente grafico possiamo osservare la differenza nella capacità di fit delle due distribuzioni:
support = np.linspace(df["return_adjclose"].min(), df["return_adjclose"].max(), 100)
plt.hist(df["return_adjclose"],bins=40, density=True, alpha=0.5)
plt.plot(support, sp.t.pdf(support, loc=tmean, scale=tsigma, df=tdf), "r-", label="t-Student")
plt.plot(support, sp.norm.pdf(support, loc=mean, scale=std_adjclose), "b-", label="Normal")
plt.title("Ritorni Giornalieri UniCredit")
plt.legend()
plt.grid()
#--------------------------------------------------------------------------------#
#CALCOLO VaR seguendo il Bootstrap method:
#calcolo cioè il VaR utilizzando la distribuzione dei ritorni giornalieri 
#attraverso i quali stimo i quantili.

loss_ptc = df["return_adjclose"].quantile(0.05) #utilizzo il 95% di confidenza
loss_ptc
#ipotizzo un investimento di 100k
# pertanto, al 95% di confidenza, posso dire che la perdita massima giornaliera
# sull'asset Unicredit sara:

max_loss = loss_ptc * 100000    
#4112.13 euro

#99 % confidence:
loss_ptc = df["return_adjclose"].quantile(0.01) #utilizzo il 99% di confidenza
loss_ptc

max_loss = loss_ptc * 100000       
max_loss
#7948.98 euro
#--------------------------------------------------------------------------------#
#CALCOLO VaR adjusted per la distribuzione t-student
#dato che abbiamo osservato come la distribuzione t-student fitti meglio i dati
#replichiamo il calcolo del VaR prendendo in considerazione i valori della t-student
#ottenuti in precedenza
loss_ptc_tstud = sp.t.ppf(0.05, tdf, tmean, tsigma)
loss_ptc_tstud

max_loss_tstud = loss_ptc_tstud*100000
max_loss_tstud
#4329.97 euro

#99 % confidence:
loss_ptc_tstud = sp.t.ppf(0.01, tdf, tmean, tsigma)
loss_ptc_tstud

max_loss_tstud = loss_ptc_tstud*100000
max_loss_tstud
#8149.71 euro


#Ovviamente, per evitare che l'analisi sia biased, dovremmo proporre i seguenti calcoli su un campione più grande.
