import numpy as np; from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(24325)

S0 = 61.98     #Initial price on June 1
mu = 0.03   #Annualized continuously compounded rate of return
sigma = 0.21#Annualized volatility
t_intervals = 92 #The number of time intervals (N), the number of daily returns
iterations = 100000  #The number of simulated paths
dt0=29/365 #Time from June 1 to June 30
dt=1/365   #Time increments after June 30

df = pd.read_csv( 'C:/Python/MSF568_2023cFall_AnalyticTermProject.csv', index_col=0)

rhoTL = 0.88
rhoTW = 0.63
rhoLW = 0.72

# Generate independent standard normal random variables
epsilon_T = norm.ppf(np.random.rand(t_intervals+1, iterations))
epsilon_L = norm.ppf(np.random.rand(t_intervals+1, iterations))
epsilon_W = norm.ppf(np.random.rand(t_intervals+1, iterations))

L21 = +rhoTL
L22 = np.sqrt(1-rhoTL**2)
L31 = +rhoTW
L32 = (rhoLW-rhoTL*rhoTW)/np.sqrt(1-rhoTL**2)
L33 = np.sqrt(1-L31**2-L32**2)

e_Td = epsilon_T
e_Ld = L21 * epsilon_T + L22 * epsilon_L
e_Wd = L31 * epsilon_T + L32 * epsilon_L + L33 * epsilon_W

print ("Correlation of normal RVs =\n" + \
       str(np.corrcoef([e_Td[-1],e_Ld[-1],e_Wd[-1]])))


rhoT = 0.74
rhoL = 0.83

Td_bar = df.iloc[:,0]
Ld_bar = df.iloc[:,2]
sigma_T = df.iloc[:,1]
sigma_L = df.iloc[:,3]

Td_simulated = np.zeros((t_intervals, iterations))
Ld_simulated = np.zeros((t_intervals, iterations))

Td_star_star = np.zeros((t_intervals+1, iterations))
Ld_star_star = np.zeros((t_intervals+1, iterations))
# Simulate temperature and load for each day within t_intervals
for t in range(1, t_intervals+1):
    # Simulate standardized differences from the long-term mean
    Td_star_star[t] = rhoT * Td_simulated[t-1] + np.sqrt(1 - rhoT**2) * e_Td[t]
    Ld_star_star[t] = rhoL * Ld_simulated[t-1] + np.sqrt(1 - rhoL**2) * e_Ld[t]

Td_star_star = Td_star_star[1:]
Ld_star_star = Ld_star_star[1:]

for t in range(0, t_intervals):
    # Simulate differences from the long-term mean
    Td_star = sigma_T[t] * Td_star_star[t]
    Ld_star = sigma_L[t] * Ld_star_star[t]

    # Update temperature and load for the next day
    Td_simulated[t] = Td_bar[t] + Td_star
    Ld_simulated[t] = Ld_bar[t] + Ld_star

#%% Simulate Gross Returns from June 1 to June 30.
Z0=norm.ppf(np.random.rand(1, iterations)) #Simulated std. normal RV
sim_gross_returns0=np.exp((mu-sigma**2/2)*dt0+sigma*np.sqrt(dt0)*Z0) #Return until June 30

#%% Simulate Prices on June 30.
sim_prices = np.zeros([t_intervals+1,iterations])
sim_prices[0] = S0*sim_gross_returns0 #Simulated prices on June 30

#%% Simulate Gross Returns each day after June 30.
sim_gross_returns=np.exp((mu-sigma**2/2)*dt+sigma*np.sqrt(dt)*e_Wd) #Daily Return after June 30

#%% Simulate Prices after June 30.
for t in range(0, t_intervals): #Simulated prices from July 1 to Sep 30
    sim_prices[t+1] = sim_prices[t] * sim_gross_returns[t]
sim_prices = sim_prices[1:]

#%% Time Series Plot
plt.figure(figsize=(10,6))
plt.plot(sim_prices [1:]); #Excluded June 30 prices (row 0)
plt.title('Simulated GBM Paths');
plt.xlabel('Steps in the future');plt.ylabel('Price ($/share)')

L_q=np.sum(Ld_simulated,axis=0)
pi_unhedged = L_q*(70.48-sim_prices)

#pi_unhedged = Ld_simulated*(70.48-sim_prices)
#pi_unhedged = np.sum(pi_unhedged,axis=0)

plt.title('Simulated GBM Paths')
plt.xlabel('Steps in the future')
plt.ylabel('Price ($/share)')

plt.figure(figsize=(12, 6))
for i in range(iterations):
    plt.plot(range(t_intervals), Ld_simulated[:, i], color='blue', alpha=0.1)  # Blue for load

plt.title('Simulated Load Paths')
plt.xlabel('Days')
plt.ylabel('Load')

plt.figure(figsize=(12, 6))
for i in range(iterations):
    plt.plot(range(t_intervals), Td_simulated[:, i], color='blue', alpha=0.1)  # Blue for load

plt.title('Simulated Temp Paths')
plt.xlabel('Days')
plt.ylabel('Temp')


print ('Unhedged Earning:')
print ("mean(pi)=","{:,.0f}".format(np.mean(pi_unhedged)))
print ("median(pi)=","{:,.0f}".format(np.median(pi_unhedged)))
print ("5 percentile (pi)=","{:,.0f}".format(np.percentile(pi_unhedged,5)))
print ("95% value at risk=", \
       "{:,.0f}".format(np.mean(pi_unhedged)-np.percentile(pi_unhedged,5)))
print ("1 percentile (pi)=","{:,.0f}".format(np.percentile(pi_unhedged,1)))
print ("99% value at risk=", \
       "{:,.0f}".format(np.mean(pi_unhedged)-np.percentile(pi_unhedged,1)))

Elec_fwd_payoff=92*24*10*(np.mean(sim_prices,axis=0)-67)

position_TempFwd=np.linspace(-10,+10,21) #Position of temperature hedge (# contracts)
pi_hedged_mean=np.zeros(21)
pi_hedged_median=np.zeros(21)
pi_hedged_5Perc=np.zeros(21)
pi_hedged_1Perc=np.zeros(21)

i=0
for p in position_TempFwd:
    pi_hedged=pi_unhedged+p*Elec_fwd_payoff
    pi_hedged_mean[i]=np.mean(pi_hedged)
    pi_hedged_median[i]=np.median(pi_hedged)
    pi_hedged_5Perc[i]=np.percentile(pi_hedged,5)
    pi_hedged_1Perc[i]=np.percentile(pi_hedged,1)
    i=i+1
#
plt.figure(figsize=(6,4))
plt.scatter(position_TempFwd,pi_hedged_mean)
plt.scatter(position_TempFwd,pi_hedged_median)
plt.scatter(position_TempFwd,pi_hedged_5Perc)
plt.scatter(position_TempFwd,pi_hedged_1Perc)
plt.legend(['Mean Hedged Earning','Median Hedged Earning',\
            '5Percentaile of Hedged Earning','1Percentile of Hedged Earning'])
plt.xlabel('Position of Electricity Forward (contracts)')
plt.ylabel('$')

position_TempFwd = position_TempFwd[np.argmax(pi_hedged_1Perc)]
pi_hedged = pi_unhedged + position_TempFwd * Elec_fwd_payoff


print ('\nElectricity Hedged Earning:')
print ("position=","{:,.0f}".format(position_TempFwd))
print ("mean(pi)=","{:,.0f}".format(np.mean(pi_hedged)))
print ("median(pi)=","{:,.0f}".format(np.median(pi_hedged)))
print ("5 percentile (pi)=","{:,.0f}".format(np.percentile(pi_hedged,5)))
print ("95% value at risk=", \
       "{:,.0f}".format(np.mean(pi_hedged)-np.percentile(pi_hedged,5)))
print ("1 percentile (pi)=","{:,.0f}".format(np.percentile(pi_hedged,1)))
print ("99% value at risk=", \
       "{:,.0f}".format(np.mean(pi_hedged)-np.percentile(pi_hedged,1)))


CDD_d=np.maximum(Td_simulated-65,0) #Daily CDD
CDD_q=np.sum(CDD_d,axis=0) #Quarterly CDD, summation by column
CDD_fwd_payoff=20*(CDD_q-684) #Contract amount= 20 & fWd price=684


position_TempFwd=np.linspace(-1000,+1000,2001) #Position of CDD hedge (# contracts)
pi_hedged_mean=np.zeros(2001)
pi_hedged_median=np.zeros(2001)
pi_hedged_5Perc=np.zeros(2001)
pi_hedged_1Perc=np.zeros(2001)

i=0
for p in position_TempFwd:
    pi_hedged=pi_unhedged+p*CDD_fwd_payoff
    pi_hedged_mean[i]=np.mean(pi_hedged)
    pi_hedged_median[i]=np.median(pi_hedged)
    pi_hedged_5Perc[i]=np.percentile(pi_hedged,5)
    pi_hedged_1Perc[i]=np.percentile(pi_hedged,1)
    i=i+1
#
plt.figure(figsize=(6,4))
plt.scatter(position_TempFwd,pi_hedged_mean)
plt.scatter(position_TempFwd,pi_hedged_median)
plt.scatter(position_TempFwd,pi_hedged_5Perc)
plt.scatter(position_TempFwd,pi_hedged_1Perc)
plt.legend(['Mean Hedged Earning','Median Hedged Earning',\
            '5Percentaile of Hedged Earning','1Percentile of Hedged Earning'])
plt.xlabel('Position of CDD (contracts)')
plt.ylabel('$')

position_TempFwd = position_TempFwd[np.argmax(pi_hedged_1Perc)]
pi_hedged = pi_unhedged + position_TempFwd * CDD_fwd_payoff

print ('\nCDD Hedged Earning:')
print ("position=","{:,.0f}".format(position_TempFwd))
print ("mean(pi)=","{:,.0f}".format(np.mean(pi_hedged)))
print ("median(pi)=","{:,.0f}".format(np.median(pi_hedged)))
print ("5 percentile (pi)=","{:,.0f}".format(np.percentile(pi_hedged,5)))
print ("95% value at risk=", \
       "{:,.0f}".format(np.mean(pi_hedged)-np.percentile(pi_hedged,5)))
print ("1 percentile (pi)=","{:,.0f}".format(np.percentile(pi_hedged,1)))
print ("99% value at risk=", \
       "{:,.0f}".format(np.mean(pi_hedged)-np.percentile(pi_hedged,1)))

position_TempFwd_electricity = np.linspace(-10, +10, 21)  # Position of temperature hedge for electricity contracts
position_TempFwd_cdd = np.linspace(-10, +500, 511)  # Position of temperature hedge for CDD contracts

# Initialize arrays to store results
combined_pi_hedged_mean = np.zeros((len(position_TempFwd_electricity), len(position_TempFwd_cdd)))
combined_pi_hedged_median = np.zeros((len(position_TempFwd_electricity), len(position_TempFwd_cdd)))
combined_pi_hedged_5Perc = np.zeros((len(position_TempFwd_electricity), len(position_TempFwd_cdd)))
combined_pi_hedged_1Perc = np.zeros((len(position_TempFwd_electricity), len(position_TempFwd_cdd)))

# Iterate over both sets of positions
for i, p_electricity in enumerate(position_TempFwd_electricity):
    for j, p_cdd in enumerate(position_TempFwd_cdd):
        # Calculate combined portfolio earnings for each combination of positions
        pi_hedged = pi_unhedged + p_electricity * Elec_fwd_payoff + p_cdd * CDD_fwd_payoff

        # Store performance metrics
        combined_pi_hedged_mean[i, j] = np.mean(pi_hedged)
        combined_pi_hedged_median[i, j] = np.median(pi_hedged)
        combined_pi_hedged_5Perc[i, j] = np.percentile(pi_hedged, 5)
        combined_pi_hedged_1Perc[i, j] = np.percentile(pi_hedged, 1)

best_index_flat = np.argmax(combined_pi_hedged_1Perc)
best_index_electricity, best_index_cdd = np.unravel_index(best_index_flat, combined_pi_hedged_1Perc.shape)

# Now you can access the best positions for both CDD and electricity
best_position_TempFwd_electricity = position_TempFwd_electricity[best_index_electricity]
best_position_TempFwd_cdd = position_TempFwd_cdd[best_index_cdd]

# Calculate combined portfolio earnings using the best positions
best_pi_hedged = pi_unhedged + best_position_TempFwd_electricity * Elec_fwd_payoff + best_position_TempFwd_cdd * CDD_fwd_payoff

# Display the results
print ('\nCombined Hedged Earning:')
print ("Best position for electricity contracts:", "{:,.0f}".format(best_position_TempFwd_electricity))
print ("Best position for CDD contracts:", "{:,.0f}".format(best_position_TempFwd_cdd))
print ("mean(pi):", "{:,.0f}".format(np.mean(best_pi_hedged)))
print ("median(pi):", "{:,.0f}".format(np.median(best_pi_hedged)))
print ("5 percentile (pi):", "{:,.0f}".format(np.percentile(best_pi_hedged, 5)))
print ("95% value at risk:", "{:,.0f}".format(np.mean(best_pi_hedged) - np.percentile(best_pi_hedged, 5)))
print ("1 percentile (pi):", "{:,.0f}".format(np.percentile(best_pi_hedged, 1)))
print ("99% value at risk:", "{:,.0f}".format(np.mean(best_pi_hedged) - np.percentile(best_pi_hedged, 1)))

plt.show()
