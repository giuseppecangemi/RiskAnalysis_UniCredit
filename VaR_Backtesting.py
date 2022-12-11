#dal precedente script ("VaR_classical_methods.py") ottengo ritorni giornalieri
dff = df["return_adjclose"].dropna()

#creo var ogni due ritorni giornalieri
i = 0
d = list()
var = list()
threshold = 1
while i <= threshold:
    d.append(dff[i])
    i += 1
    if i == threshold:
        tdf, tmean, tsigma = sp.t.fit(d)   
        var.append(sp.t.ppf(0.05, tdf, tmean, tsigma))
        threshold += 1

series = list()
for i in dff:
    series.append(i)

plt.plot(var, label="Stima del VaR t:t-1", color="red")
plt.plot(series, label="Rendimenti Giornalieri")
plt.title("Var and Backtesting")
plt.subtitle("A")
plt.legend()
plt.show()

#come possiamo vedere non performa in quanto spesso il rendimento va oltre il VaR stimato
#quindi provo a partire dall'anno t+1 cosi da aver un buon datasample (t).

len_a = len(dff)
len_b = (len(dff)/2) + 0.5 #normalizzo per arrivare all'unità (giorno)

d = list()
var2 = list()
for i in range(len_a):
    d.append(dff[i])
    if i >= len_b:
        tdf, tmean, tsigma = sp.t.fit(d)   
        var2.append(sp.t.ppf(0.05, tdf, tmean, tsigma))
        

plt.plot(var2)
plt.title("VaR per il 2021")
plt.show()
#questo var fa riferimento al var dal gennaio 2021 -> to end

series2 = list()
for i in range(len_a):
    if i >= len_b:
        series2.append(dff[i])
        
plt.plot(series2) #ritorni giornalieri ultimo anno (t+1)
plt.plot(var2)
plt.show()    

#Non migliora!

#creo var con distribuzione normale.
i = 0
d = list()
var3 = list()
threshold = 1
while i <= threshold:
    d.append(dff[i])
    i += 1
    if i == threshold:
        mean, sigma = sp.norm.fit(d)   
        var3.append(sp.norm.ppf(0.05,  mean, sigma))
        #var2.append(np.percentile(dff[i], 100 * (1-alpha)))
        threshold += 1

series = list()
for i in dff:
    series.append(i)

plt.plot(var3, label="Stima del VaR t:t-1 (Normale)", color="red")
plt.plot(var, label="Stima del VaR t:t-1 (t-Student)", color="orange")
plt.plot(series, label="Rendimenti Giornalieri")
plt.title("Var and Backtesting")
plt.subtitle("A")
plt.legend()
plt.grid()
plt.show()

# Problema: il var del t-student è piu basso della Normale. Questo non può essere vero, \\
# perché nell'analisi precedente il var t-student era più grande.
# Probabilmente stimando media e std su t e t-1, la stima si avvicina più ad una normale.
