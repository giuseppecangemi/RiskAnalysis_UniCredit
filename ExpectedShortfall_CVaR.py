import numpy as np
import pandas as pd

#ottengo df dal precedente script ("VaR_classical_methods.py")
returns = df["return_adjclose"].dropna()
cap_invested = 100000

#ottengo var
var = returns.quantile(0.05)
max_loss_VaR = var * cap_invested

#ottengo CVaR or ExpectedShorfall(ES)
#stimo la media dei ritorni che hanno un valore inferiore al VaR e cioè al quantile della distribuzione:
cvar = np.mean(returns[returns<var])
expected_loss = cvar * cap_invested

#creo grafico per osservare VaR e CVaR 
plt.hist(returns[returns>var], bins=20, label="Distribuzione storica dei rendimenti giornalieri")
plt.hist(returns[returns<var], bins=20, label="Ritorni giornalieri quando < VaR")
plt.axvline(var, color="red", linewidth=2, label="VaR")
plt.axvline(cvar, color="green", linewidth=2, linestyle="--", label="CVaR or ES")
plt.legend()
plt.show()



