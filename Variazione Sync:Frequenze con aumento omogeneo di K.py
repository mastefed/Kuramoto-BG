import numpy
import matplotlib.pyplot as plt

coupconsts = [5., 7., 10., 12., 15., 17., 20., 22., 25., 27., 30., 32., 35., 37., 40., 45., 50., 60., 70.]

syncsub1 = [.23, .70, .92, .97, .89, .95, .72, .83, .96, .99, .95, .99, .94, .99, .96, .99, .99, .99, .99]
syncsub2 = [.28, .76, .78, .81, .90, .91, .96, .92, .92, .96, .97, .99, .97, .99, .99, .99, .98, .99, .99]
syncsub3 = [.56, .85, .78, .86, .76, .85, .97, .98, .99, .96, .96, .99, .97, .99, .99, .97, .99, .99, .99]

freqsub1 = [21.87, 20.98, 20.45, 18.95, 16.83, 17.10, 16.78, 16.96, 17.19, 16.58, 16.94, 17.00, 16.81, 16.65, 16.87] # DA 15 IN POI
freqsub2 = [13.16, 14.06, 15.05, 15.56, 16.88, 17.01, 16.67, 16.98, 17.20, 16.59, 16.96, 17.02, 16.81, 16.67, 16.89]
freqsub3 = [16.08, 15.73, 15.15, 15.61, 16.65, 17.01, 16.67, 16.98, 17.20, 16.59, 16.96, 17.02, 16.81, 16.67, 16.89]

alphaconsts = [.2, .4, .6, .8, 1.]

freqalpha1 = [15.75, 17.30, 19.70, 20.80, 22.44]
freqalpha2 = [15.62, 14.90, 12.85, 12.38, 11.37]
freqalpha3 = [15.63, 14.78, 12.90, 12.59, 16.57]


plt.figure(figsize=(13,6))
plt.subplot(121)
plt.title('Sync. vs K')
plt.plot(coupconsts, syncsub1, label='Pop. 1')
plt.plot(coupconsts, syncsub2, label='Pop. 2')
plt.plot(coupconsts, syncsub3, label='Pop. 3')
plt.xlabel('Coupling constant')
plt.ylabel('Synchronization')
plt.legend()
plt.grid()

plt.subplot(122)
plt.title('Sync. Frequencies vs K')
plt.plot(coupconsts[4:], freqsub1, label='Pop. 1')
plt.plot(coupconsts[4:], freqsub2, label='Pop. 2')
plt.plot(coupconsts[4:], freqsub3, label='Pop. 3')
plt.axvline(25., color='r', ls='--')
plt.axvline(70., color='r', ls='--', label='Significance Interval')
plt.yticks([13., 14., 15., 16., 17., 18., 19., 20., 21., 22.])
plt.xlabel('Coupling constant')
plt.ylabel('Frequency of oscillation [Hz]')
plt.legend()
plt.grid()

plt.figure('alphas')
plt.title('Sync. Frequencies vs Phase Delay [k=25]')
plt.plot(alphaconsts, freqalpha1, label='Pop. 1')
plt.plot(alphaconsts, freqalpha2, label='Pop. 2')
plt.plot(alphaconsts, freqalpha3, label='Pop. 3')
plt.xlabel('Phase Delay for Inter-Pop. Interaction')
plt.ylabel('Frequency of oscillation [Hz]')
plt.legend()
plt.grid()

plt.show()