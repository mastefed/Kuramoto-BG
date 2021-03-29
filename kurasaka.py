class kurasaka_oscillators:
    def __init__(self, num_subpop1, num_subpop2, num_subpop3):
        self.N1 = num_subpop1
        self.N2 = num_subpop2
        self.N3 = num_subpop3
        self.N = self.N1 + self.N2 + self.N3
        self.Narray = [self.N1, self.N2, self.N3]
        self.reproducible_rng = numpy.random.default_rng(42)
        self.notreproducible_rng = numpy.random.default_rng()

    def settimes(self, time_start, time_end, time_points):
        self.time_start = time_start
        self.time_end = time_end
        self.time_points = time_points
        self.times = numpy.linspace(self.time_start, self.time_end, self.time_points)
        return self.times

    def setinitialconditions(self, clustered):
        if clustered == False:
            self.initialvalues = 2*numpy.pi*self.reproducible_rng.random(self.N) # random conditions of phases between 0 and 2pi

        elif clustered == True:
            self.init_values_N1 = self.reproducible_rng.normal(loc=2*numpy.pi*self.reproducible_rng.random(), scale=.5, size=self.N1)
            self.init_values_N2 = self.reproducible_rng.normal(loc=2*numpy.pi*self.reproducible_rng.random(), scale=.5, size=self.N2)
            self.init_values_N3 = self.reproducible_rng.normal(loc=2*numpy.pi*self.reproducible_rng.random(), scale=.5, size=self.N3)

            self.initialvalues = numpy.hstack((numpy.hstack((self.init_values_N1, self.init_values_N2)), self.init_values_N3))

        return self.initialvalues

    def setmodelconstants(self, fixed):
        if fixed == False:
            print("Please, choose the intra-subpopulations' coupling constants:")
            self.k11 = float(input('Choose the coupling constant for subpopulation 1 <--> subpopulation 1 interaction: '))
            self.k22 = float(input('Choose the coupling constant for subpopulation 2 <--> subpopulation 2 interaction: '))
            self.k33 = float(input('Choose the coupling constant for subpopulation 3 <--> subpopulation 3 interaction: '))

            print("\nNow, choose the inter-subpopulations' coupling constants:")
            self.k12 = float(input('Choose the coupling constant for subpopulation 1 <--> subpopulation 2 interaction: '))
            self.k13 = float(input('Choose the coupling constant for subpopulation 1 <--> subpopulation 3 interaction: '))
            self.k23 = float(input('Choose the coupling constant for subpopulation 2 <--> subpopulation 3 interaction: '))
            self.k21 = self.k12
            self.k31 = self.k13
            self.k32 = self.k23

            print("\nThen, choose the intra-subpopulations' phase delay alpha:")
            self.alpha11 = float(input('Choose alpha for subpopulation 1 <--> subpopulation 1 interaction: '))
            self.alpha22 = float(input('Choose alpha for subpopulation 2 <--> subpopulation 2 interaction: '))
            self.alpha33 = float(input('Choose alpha for subpopulation 3 <--> subpopulation 3 interaction: '))

            print("\nFinally, choose the inter-subpopulations' phase delay alpha:")
            self.alpha12 = float(input('Choose alpha for subpopulation 1 <--> subpopulation 2 interaction: '))
            self.alpha13 = float(input('Choose alpha for subpopulation 1 <--> subpopulation 3 interaction: '))
            self.alpha21 = float(input('Choose alpha for subpopulation 2 <--> subpopulation 1 interaction: '))
            self.alpha23 = float(input('Choose alpha for subpopulation 2 <--> subpopulation 3 interaction: '))
            self.alpha31 = float(input('Choose alpha for subpopulation 3 <--> subpopulation 1 interaction: '))
            self.alpha32 = float(input('Choose alpha for subpopulation 3 <--> subpopulation 2 interaction: '))

        if fixed == True:
            self.k11 = 1.
            self.k22 = 10. 
            self.k33 = 15.

            self.k12 = .5
            self.k13 = .5
            self.k23 = 15.
            self.k21 = self.k12
            self.k31 = self.k13
            self.k32 = self.k23

            self.alpha11 = 0.05 
            self.alpha22 = 0.05 
            self.alpha33 = 0.05

            self.alpha12 = 0.2
            self.alpha13 = 0.5
            self.alpha21 = 1.2
            self.alpha23 = 0.5
            self.alpha31 = 0.5
            self.alpha32 = 0.5

        self.kmatrix = numpy.matrix([
            [self.k11, self.k12, self.k13],
            [self.k21, self.k22, self.k23],
            [self.k31, self.k32, self.k33]
        ])

        self.alphamatrix = numpy.matrix([
            [self.alpha11, self.alpha12, self.alpha13],
            [self.alpha21, self.alpha22, self.alpha23],
            [self.alpha31, self.alpha32, self.alpha33]
        ])

        self.omega1 = cauchy.rvs(loc=143., scale=.2, size=self.N1)
        self.omega2 = cauchy.rvs(loc=71., scale=.2, size=self.N2)
        self.omega3 = cauchy.rvs(loc=95., scale=.2, size=self.N3)

        self.omegamatrix = numpy.hstack((numpy.hstack((self.omega1, self.omega2)), self.omega3))

        return self.kmatrix, self.omegamatrix, self.alphamatrix

    def kurasaka_function(self, x, t):
        self.variables = {}
        for i in range(self.N1):
            self.variables[f'theta1{i}'] = x[i] # gets an array of phases' value at time t_k, odeint update the values everytime for evert t_j

        for i in range(self.N2):
            self.variables[f'theta2{i}'] = x[self.N1 + i]

        for i in range(self.N3):
            self.variables[f'theta3{i}'] = x[self.N1 + self.N2 + i]

        def interaction(k, z):
            interaction_terms = 0.
            for j in range(self.Narray[k-1]):
                interaction_terms += self.kmatrix[z-1,k-1]/self.Narray[k-1]*numpy.sin(self.variables[f'theta{k}{j}'] - self.variables[f'theta{z}{i}'] - self.alphamatrix[z-1,k-1])
            return interaction_terms

        self.dthetadt = [] # Creates and updates the values' array with the desired differential equations

        for i in range(self.N1):
            self.dthetadt.append(
                self.omega1[i] + interaction(1, 1) + interaction(2, 1) + interaction(3, 1)
            )

        for i in range(self.N2):
            self.dthetadt.append(
                self.omega2[i] + interaction(1, 2) + interaction(2, 2) + interaction(3, 2)
            )

        for i in range(self.N3):
            self.dthetadt.append(
                self.omega3[i] + interaction(1, 3) + interaction(2, 3) + interaction(3, 3)
            )

        return numpy.array(self.dthetadt) # returns the function to be put in .evolve()

    def evolve(self, function):
        self.kurasaka_evo = odeint(function, self.initialvalues, self.times)
        return self.kurasaka_evo
    
    def evolvewithnoise(self, function):
        def noise(x, t):
            self.sigma = []
            for i in range(self.N):
                self.sigma.append(0.8)
            self.sigma = numpy.diag(self.sigma)
            return self.sigma
        
        self.kurasaka_evo = itoint(function, noise, self.initialvalues, self.times)
        return self.kurasaka_evo

    def findorderparameter(self, phases):
        self.orderparameter_subpop1 = []
        self.orderparameter_subpop2 = []
        self.orderparameter_subpop3 = []

        for i in range(len(self.times)):
            self.orderparameter_subpop1.append(1/self.N1 * sum(numpy.exp(complex(0,phases[i][j]))  for j in range(self.N1)))
            self.orderparameter_subpop2.append(1/self.N2 * sum(numpy.exp(complex(0,phases[i][self.N1 + j]))  for j in range(self.N2)))
            self.orderparameter_subpop3.append(1/self.N3 * sum(numpy.exp(complex(0,phases[i][self.N1 + self.N2 + j]))  for j in range(self.N3)))

        self.sync_subpop1 = []
        self.sync_subpop2 = []
        self.sync_subpop3 = []

        for i in range(len(self.orderparameter_subpop1)):
            self.sync_subpop1.append( numpy.sqrt(self.orderparameter_subpop1[i].real**2 + self.orderparameter_subpop1[i].imag**2) )
        for i in range(len(self.orderparameter_subpop2)):
            self.sync_subpop2.append( numpy.sqrt(self.orderparameter_subpop2[i].real**2 + self.orderparameter_subpop2[i].imag**2) )
        for i in range(len(self.orderparameter_subpop3)):
            self.sync_subpop3.append( numpy.sqrt(self.orderparameter_subpop3[i].real**2 + self.orderparameter_subpop3[i].imag**2) )

        self.syncs = [self.sync_subpop1, self.sync_subpop2, self.sync_subpop3]
        self.orderparameters = [self.orderparameter_subpop1, self.orderparameter_subpop2, self.orderparameter_subpop3]

        return self.syncs, self.orderparameters # returns |Z| and Z, both can be useful

    def findglobalorderparameter(self):
        def mediationterm(sigma, tau):
            mediation = self.kmatrix[sigma,tau]/self.kmatrix.sum()*numpy.exp(complex(0,-self.alphamatrix[sigma,tau]))
            return mediation

        self.globalorderparameter = []

        for i in range(len(self.times)):
            partialglobalorderparam = 0.
            for sigma in range(3):
                for tau in range(3):
                    partialglobalorderparam += mediationterm(sigma,tau)*self.orderparameters[tau][i]
            self.globalorderparameter.append(partialglobalorderparam)

        self.sync_global = []
        self.phase_global = []
        for i in range(len(self.globalorderparameter)):
            self.sync_global.append( numpy.sqrt(self.globalorderparameter[i].real**2 + self.globalorderparameter[i].imag**2) )
            self.phase_global.append(numpy.angle(self.globalorderparameter[i]))

        return self.sync_global, self.globalorderparameter

    def ordparam_phase(self):
        self.real_ordparam_subpop1 = []
        self.real_ordparam_subpop2 = []
        self.real_ordparam_subpop3 = []

        for i in range(len(self.times)):
            self.real_ordparam_subpop1.append(
                self.orderparameter_subpop1[i].real
            )
            self.real_ordparam_subpop2.append(
                self.orderparameter_subpop2[i].real
            )
            self.real_ordparam_subpop3.append(
                self.orderparameter_subpop3[i].real
            )

        return self.real_ordparam_subpop1, self.real_ordparam_subpop2, self.real_ordparam_subpop3

    def psdofordparam(self, save):
        self.freq1, self.psd1 = welch(self.real_ordparam_subpop1, fs=1/((self.time_end - self.time_start)/self.time_points))
        self.freq2, self.psd2 = welch(self.real_ordparam_subpop2, fs=1/((self.time_end - self.time_start)/self.time_points))
        self.freq3, self.psd3 = welch(self.real_ordparam_subpop3, fs=1/((self.time_end - self.time_start)/self.time_points))

        plt.figure('PSD', figsize=(6,6))
        plt.title('PSD of Re[Z]')
        plt.xlabel('Frequencies [Hz]')
        plt.ylabel('PSD')
        plt.grid()
        plt.plot(self.freq1, self.psd1, label='Pop. 1')
        plt.plot(self.freq2, self.psd2, label='Pop. 2')
        plt.plot(self.freq3, self.psd3, label='Pop. 3')
        plt.legend()

        if save == True:
            plt.savefig('C:/Users/feder/Desktop/PSD.png')
        elif save == False:
            pass

    def findperiod(self):
        self.peaks_phase_subpop1,_ = find_peaks(self.real_ordparam_subpop1)
        self.peaks_phase_subpop1 = self.peaks_phase_subpop1*self.time_end/self.time_points
        self.peaks_phase_subpop2,_ = find_peaks(self.real_ordparam_subpop2)
        self.peaks_phase_subpop2 = self.peaks_phase_subpop2*self.time_end/self.time_points
        self.peaks_phase_subpop3,_ = find_peaks(self.real_ordparam_subpop3)
        self.peaks_phase_subpop3 = self.peaks_phase_subpop3*self.time_end/self.time_points

        self.periods_subpop1 = []
        self.periods_subpop2 = []
        self.periods_subpop3 = []

        for i in range(len(self.peaks_phase_subpop1) - 1):
            self.periods_subpop1.append(
                1/ (self.peaks_phase_subpop1[i+1] - self.peaks_phase_subpop1[i])
            )
        self.mean_frequency_subpop1 = numpy.mean(self.periods_subpop1)

        for i in range(len(self.peaks_phase_subpop2) - 1):
            self.periods_subpop2.append(
                1 / (self.peaks_phase_subpop2[i+1] - self.peaks_phase_subpop2[i])
            )
        self.mean_frequency_subpop2 = numpy.mean(self.periods_subpop2)

        for i in range(len(self.peaks_phase_subpop3) - 1):
            self.periods_subpop3.append(
                1 / (self.peaks_phase_subpop3[i+1] - self.peaks_phase_subpop3[i])
            )
        self.mean_frequency_subpop3 = numpy.mean(self.periods_subpop3)

        self.mean_frequencies = [self.mean_frequency_subpop1, self.mean_frequency_subpop2, self.mean_frequency_subpop3]

        return self.mean_frequencies

    def printsyncparam(self, save):
        plt.figure(f'{self.N} Oscillators Sync', figsize=(13,6))
        plt.title(f'{self.N} Oscillators Sync')
        plt.plot(self.times, self.sync_subpop1, label='SubPop 1')
        plt.plot(self.times, self.sync_subpop2, label='SubPop 2')
        plt.plot(self.times, self.sync_subpop3, label='SubPop 3')
        plt.plot(self.times, self.sync_global, label='Global')
        plt.xlabel('Time Steps')
        plt.ylabel('R')
        plt.ylim([0.,1.])
        plt.yticks(numpy.arange(0, 1.1, step=0.1))
        plt.legend(loc='lower left')
        
        plt.axes([.55, .20, .3, .3])
        plt.plot(self.times, self.sync_subpop1, label='SubPop 1')
        plt.plot(self.times, self.sync_subpop2, label='SubPop 2')
        plt.plot(self.times, self.sync_subpop3, label='SubPop 3')
        plt.plot(self.times, self.sync_global, label='Global')
        plt.xlim([1.,1.5])
        plt.xticks([1.1, 1.2, 1.3, 1.4])
        plt.yticks([.2, .4, .6, .8, 1.])
        plt.tick_params(axis='x', direction='in', pad=-15)
        plt.tick_params(axis='y', direction='in', pad=-22)
        
        if save == True:
            plt.savefig('/home/f_mastellone/Images/SyncTrial.png')
        elif save == False:
            pass

    def printcosineordparam(self, save):
        plt.figure("Subpops' Phase Evolution", figsize=(13,6))
        plt.title("Subpops' Phase Evolution")
        plt.plot(self.times, self.real_ordparam_subpop1, label='SubPop 1')
        plt.plot(self.times, self.real_ordparam_subpop2, label='SubPop 2')
        plt.plot(self.times, self.real_ordparam_subpop3, label='SubPop 3')
        plt.xlabel('Time Steps')
        plt.xlim([0.,2.])
        plt.legend(loc='lower left')
        
        plt.axes([.55, .55, .3, .3])
        plt.plot(self.times, self.real_ordparam_subpop1, label='SubPop 1')
        plt.plot(self.times, self.real_ordparam_subpop2, label='SubPop 2')
        plt.plot(self.times, self.real_ordparam_subpop3, label='SubPop 3')
        plt.xlim([1.,1.2])
        plt.ylim([-1.3,1.1])
        plt.xticks([1.025, 1.050, 1.075, 1.100, 1.125, 1.150, 1.175])
        plt.yticks([])
        plt.grid()
        plt.tick_params(axis='x', direction='in', pad=-15)
        plt.tick_params(axis='y', direction='in', pad=-22)
        
        if save == True:
            plt.savefig('/home/f_mastellone/Images/CosOrdParTrial.png')
        elif save == False:
            pass

    def animate_function(self, i):
        self.phases = self.kurasaka_evo[i:i+1]
        self.timestep = self.times[0:i]
        self.R1 = self.sync_subpop1[0:i]
        self.R2 = self.sync_subpop2[0:i]
        self.R3 = self.sync_subpop3[0:i]
        self.RGlob = self.sync_global[0:i]

        self.imphasedict = {}
        self.rephasedict = {}

        for k in range(self.N):
            self.imphasedict[f'im_x{k}'] = numpy.exp(complex(0, self.phases[0][k])).imag
            self.rephasedict[f're_x{k}'] = numpy.exp(complex(0, self.phases[0][k])).real

        self.imagpart_ordparam_subpop1 = self.orderparameter_subpop1[i].imag
        self.realpart_ordparam_subpop1 = self.orderparameter_subpop1[i].real

        self.imagpart_ordparam_subpop2 = self.orderparameter_subpop2[i].imag
        self.realpart_ordparam_subpop2 = self.orderparameter_subpop2[i].real

        self.imagpart_ordparam_subpop3 = self.orderparameter_subpop3[i].imag
        self.realpart_ordparam_subpop3 = self.orderparameter_subpop3[i].real

        self.imagpart_global_ordparam = self.globalorderparameter[i].imag
        self.realpart_global_ordparam = self.globalorderparameter[i].real

        ticks = [-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8]

        self.ax1.clear()
        self.circ = plt.Circle((0, 0), radius=1, lw=0.3, edgecolor='k', facecolor='None')
        self.ax1.add_patch(self.circ)
        self.ax1.set_xlim(-1.2, 1.2)
        self.ax1.set_ylim(-1.2, 1.2)
        self.ax1.spines['left'].set_position('center')
        self.ax1.spines['right'].set_color('none')
        self.ax1.spines['bottom'].set_position('center')
        self.ax1.spines['top'].set_color('none')
        self.ax1.yaxis.set_ticks(ticks)
        self.ax1.xaxis.set_ticks(ticks)
        self.ax1.set_xlabel('Re', loc='right')
        self.ax1.set_ylabel('Im', loc='top')

        self.ax1.arrow(0., 0., self.realpart_ordparam_subpop1, self.imagpart_ordparam_subpop1, head_width=0.02, head_length=0.05, fc='b', ec='b', lw=1., label='Z Pop. 1')
        self.ax1.arrow(0., 0., self.realpart_ordparam_subpop2, self.imagpart_ordparam_subpop2, head_width=0.02, head_length=0.05, fc='g', ec='g', lw=1., label='Z Pop. 2')
        self.ax1.arrow(0., 0., self.realpart_ordparam_subpop3, self.imagpart_ordparam_subpop3, head_width=0.02, head_length=0.05, fc='r', ec='r', lw=1., label='Z Pop. 3')
        self.ax1.arrow(0., 0., self.realpart_global_ordparam, self.imagpart_global_ordparam, head_width=0.02, head_length=0.05, fc='k', ec='k', lw=1.3, label='Z Global')
        for k in range(self.N1):
            self.ax1.plot(self.rephasedict[f're_x{k}'], self.imphasedict[f'im_x{k}'], 'bo', ms=7.)
        for k in range(self.N2):
            self.ax1.plot(self.rephasedict[f're_x{self.N1 + k}'], self.imphasedict[f'im_x{self.N1 + k}'], 'go', ms=7.)
        for k in range(self.N3):
            self.ax1.plot(self.rephasedict[f're_x{self.N1 + self.N2 + k}'], self.imphasedict[f'im_x{self.N1 + self.N2 + k}'], 'ro', ms=7.)
        self.ax1.legend()

        self.ax2.clear()
        self.ax2.set_ylim([0.,1.])
        self.ax2.set_xlim([self.time_start, self.time_end])
        self.ax2.set_xlabel('Time Steps')
        self.ax2.set_ylabel('R')

        self.ax2.plot(self.timestep, self.R1, label='Sync. Par. Pop. 1')
        self.ax2.plot(self.timestep, self.R2, label='Sync. Par. Pop. 2')
        self.ax2.plot(self.timestep, self.R3, label='Sync. Par. Pop. 3')
        self.ax2.plot(self.timestep, self.RGlob, 'k', label='Global Sync.')
        self.ax2.legend()

    def animateoscillators(self):
        self.fig = plt.figure(f'{self.N} Oscillators Animated', figsize=(13,6))
        plt.suptitle(f'{self.N} Oscillators')
        self.ax1 = plt.subplot(121)
        self.ax2 = plt.subplot(122)
        self.animated = animation.FuncAnimation(self.fig, self.animate_function, frames = len(self.kurasaka_evo), interval=0.1)

        return self.animated

    def saveanimation(self, myanimation, save_path):
        print('\nVideo Processing started!')
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=120, metadata=dict(artist='F. V. Mastellone'), bitrate=1800)
        myanimation.save(save_path, writer=writer)
        print('Task finished.')

    def showplots(self):
        plt.show()


import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

from scipy.integrate import odeint
from scipy.stats import cauchy
from scipy.signal import find_peaks, welch

from sdeint import itoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to simulate an arbitrary number of Kuramoto-Sakaguchi Oscillators.")
    parser.add_argument('--savepath', help='Where would you save the animated evolution?', type=str)
    args = parser.parse_args()

    save_path = args.savepath

    num_subpop1 = 1000
    num_subpop2 = 3000
    num_subpop3 = 60

    print(f'Pop. 1 number of phase oscillators: {num_subpop1}')
    print(f'Pop. 2 number of phase oscillators: {num_subpop2}')
    print(f'Pop. 3 number of phase oscillators: {num_subpop3}\n')

    kuramotosakaguchi = kurasaka_oscillators(num_subpop1, num_subpop2, num_subpop3)
    coupconsts, omegas, alphas = kuramotosakaguchi.setmodelconstants(fixed=True)

    print(f'Coupling constants are:\n{coupconsts}\n')
    print(f'Phase delay constants are:\n{alphas}\n')

    init_random = kuramotosakaguchi.setinitialconditions(clustered=False)
    times = kuramotosakaguchi.settimes(0., 20., 2000)

    equations = kuramotosakaguchi.kurasaka_function
    phasesevolution = kuramotosakaguchi.evolvewithnoise(equations)
    syncs, ordparams = kuramotosakaguchi.findorderparameter(phasesevolution)
    globsync, globordparam = kuramotosakaguchi.findglobalorderparameter()

    print(f'Sync for SuPop 1: {numpy.mean(syncs[0][300:])}')
    print(f'Sync for SuPop 2: {numpy.mean(syncs[1][300:])}')
    print(f'Sync for SuPop 3: {numpy.mean(syncs[2][300:])}')
    print(f'Global Sync: {numpy.mean(globsync[300:])}\n')

    kuramotosakaguchi.ordparam_phase()
    frequencies = kuramotosakaguchi.findperiod()
    print(f'SubPop 1 frequency: {frequencies[0]}')
    print(f'SubPop 2 frequency: {frequencies[1]}')
    print(f'SubPop 3 frequency: {frequencies[2]}\n')

    kuramotosakaguchi.printcosineordparam(save=True)
    kuramotosakaguchi.printsyncparam(save=True)
    kuramotosakaguchi.psdofordparam(save=True)
    kuramotosakaguchi.showplots()
    myanim = kuramotosakaguchi.animateoscillators()
    kuramotosakaguchi.saveanimation(myanim, save_path='/home/f_mastellone/Images/videosimulazione.mp4')
