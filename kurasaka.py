class kurasaka_oscillators:
    def __init__(self, num_subpop1, num_subpop2, num_subpop3):
        self.N1 = num_subpop1
        self.N2 = num_subpop2
        self.N3 = num_subpop3
        self.N = self.N1 + self.N2 + self.N3
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

    def setmodelconstants(self, choose):
        if choose == False:
            print("Please, choose the intra-subpopulations' coupling constants:")
            self.k11 = float(input('Choose the coupling constant for subpopulation 1 <--> subpopulation 1 interaction: '))
            self.k22 = float(input('Choose the coupling constant for subpopulation 2 <--> subpopulation 2 interaction: '))
            self.k33 = float(input('Choose the coupling constant for subpopulation 3 <--> subpopulation 3 interaction: '))

            print("\nNow, choose the inter-subpopulations' coupling constants:")
            self.k12 = float(input('Choose the coupling constant for subpopulation 1 <--> subpopulation 2 interaction: '))
            self.k13 = float(input('Choose the coupling constant for subpopulation 1 <--> subpopulation 3 interaction: '))
            self.k21 = float(input('Choose the coupling constant for subpopulation 2 <--> subpopulation 1 interaction: '))
            self.k23 = float(input('Choose the coupling constant for subpopulation 2 <--> subpopulation 3 interaction: '))
            self.k31 = float(input('Choose the coupling constant for subpopulation 3 <--> subpopulation 1 interaction: '))
            self.k32 = float(input('Choose the coupling constant for subpopulation 3 <--> subpopulation 2 interaction: '))

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

        if choose == True:
            self.k11 = 25. #numpy.abs(self.notreproducible_rng.normal(loc=mean_coupconst, scale=.05))
            self.k22 = 25. #numpy.abs(self.notreproducible_rng.normal(loc=mean_coupconst, scale=.05))
            self.k33 = 25. #numpy.abs(self.notreproducible_rng.normal(loc=mean_coupconst, scale=.05))
            
            self.k12 = 25. #numpy.abs(self.notreproducible_rng.normal(loc=mean_coupconst, scale=.05))
            self.k13 = 25. #numpy.abs(self.notreproducible_rng.normal(loc=mean_coupconst, scale=.05))
            self.k21 = 25. #numpy.abs(self.notreproducible_rng.normal(loc=mean_coupconst, scale=.05))
            self.k23 = 25. #numpy.abs(self.notreproducible_rng.normal(loc=mean_coupconst, scale=.05))
            self.k31 = 25. #numpy.abs(self.notreproducible_rng.normal(loc=mean_coupconst, scale=.05))
            self.k32 = 25. #numpy.abs(self.notreproducible_rng.normal(loc=mean_coupconst, scale=.05))

            self.alpha11 = 0. #self.notreproducible_rng.random()
            self.alpha22 = 0. #self.notreproducible_rng.random()
            self.alpha33 = 0. #self.notreproducible_rng.random()

            self.alpha12 = 0. #self.notreproducible_rng.random()
            self.alpha13 = 0. #self.notreproducible_rng.random()
            self.alpha21 = 0. #self.notreproducible_rng.random()
            self.alpha23 = 0. #self.notreproducible_rng.random()
            self.alpha31 = 0. #self.notreproducible_rng.random()
            self.alpha32 = 0. #self.notreproducible_rng.random()

        self.kmatrix = numpy.array([
            [self.k11, self.k12, self.k13],
            [self.k21, self.k22, self.k23],
            [self.k31, self.k32, self.k33]
        ])

        self.alphamatrix = numpy.array([
            [self.alpha11, self.alpha12, self.alpha13],
            [self.alpha21, self.alpha22, self.alpha23],
            [self.alpha31, self.alpha32, self.alpha33]
        ])

        self.omega1 = cauchy.rvs(loc=143., scale=2., size=self.N1)
        self.omega2 = cauchy.rvs(loc=71., scale=2., size=self.N2)
        self.omega3 = cauchy.rvs(loc=100., scale=2., size=self.N3)

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
            if k == 1:
                num = self.N1
                if z == 1:
                    coupconst = self.k11
                    delayterm = self.alpha11
                elif z == 2:
                    coupconst = self.k21
                    delayterm = self.alpha21
                elif z == 3:
                    coupconst = self.k31
                    delayterm = self.alpha31
                    
            elif k == 2:
                num = self.N2
                if z == 1:
                    coupconst = self.k12
                    delayterm = self.alpha12
                elif z == 2:
                    coupconst = self.k22
                    delayterm = self.alpha22
                elif z == 3:
                    coupconst = self.k32
                    delayterm = self.alpha32

            elif k == 3:
                num = self.N3
                if z == 1:
                    coupconst = self.k13
                    delayterm = self.alpha13
                elif z == 2:
                    coupconst = self.k23
                    delayterm = self.alpha23
                elif z == 3:
                    coupconst = self.k33
                    delayterm = self.alpha33

            interaction_terms = 0.
            for j in range(num):
                interaction_terms += coupconst/num*numpy.sin(self.variables[f'theta{k}{j}'] - self.variables[f'theta{z}{i}'] - delayterm)

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

        return self.dthetadt # returns the function to be put in .evolve()

    def evolve(self, function):
        self.kurasaka_evo = odeint(function, self.initialvalues, self.times)
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

    def printsyncparam(self, num_trial, save):
        plt.figure(f'{self.N} Oscillators Sync; Trial {num_trial}', figsize=(13,6))
        plt.title(f'{self.N} Oscillators Sync; Trial {num_trial}')
        plt.plot(self.times, self.sync_subpop1, label='SubPop 1')
        plt.plot(self.times, self.sync_subpop2, label='SubPop 2')
        plt.plot(self.times, self.sync_subpop3, label='SubPop 3')
        plt.xlabel('Time Steps')
        plt.ylabel('R')
        plt.ylim([0.,1.])
        plt.yticks(numpy.arange(0, 1.1, step=0.1))
        plt.legend()
        if save == True:
            plt.savefig(f'/home/f_mastellone/Images/SyncTrial{num_trial}.png')
        elif save == False:
            pass

    def printcosineordparam(self, num_trial, save):
        plt.figure(f"Subpops' Phase Evolution; Trial {num_trial}", figsize=(13,6))
        plt.title(f"Subpops' Phase Evolution; Trial {num_trial}")
        plt.plot(self.times, self.real_ordparam_subpop1, label='SubPop 1')
        plt.plot(self.times, self.real_ordparam_subpop2, label='SubPop 2')
        plt.plot(self.times, self.real_ordparam_subpop3, label='SubPop 3')
        plt.xlabel('Time Steps')
        plt.xlim([0.,2.3])
        plt.legend()
        if save == True:
            plt.savefig(f'/home/f_mastellone/Images/CosOrdParTrial{num_trial}.png')
        elif save == False:
            pass

    def animate_function(self, i): 
        self.phases = self.kurasaka_evo[i:i+1]
        self.timestep = self.times[0:i]
        self.R1 = self.sync_subpop1[0:i]
        self.R2 = self.sync_subpop2[0:i]
        self.R3 = self.sync_subpop3[0:i]
        
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
        for k in range(self.N):
            self.ax1.plot(self.rephasedict[f're_x{k}'], self.imphasedict[f'im_x{k}'], 'o', ms=7.)
        self.ax1.legend()

        self.ax2.clear()
        self.ax2.set_ylim([0.,1.])
        self.ax2.set_xlim([self.time_start, self.time_end])
        self.ax2.set_xlabel('Time Steps')
        self.ax2.set_ylabel('R')

        self.ax2.plot(self.timestep, self.R1, label='Sync. Par. Pop. 1')
        self.ax2.plot(self.timestep, self.R2, label='Sync. Par. Pop. 2')
        self.ax2.plot(self.timestep, self.R3, label='Sync. Par. Pop. 3')
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
        writer = Writer(fps=15, metadata=dict(artist='F. V. Mastellone'), bitrate=1800)
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
from scipy.signal import find_peaks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to simulate an arbitrary number of Kuramoto-Sakaguchi Oscillators.")
    parser.add_argument('--savepath', help='Where would you save the animated evolution?', type=str)
    args = parser.parse_args()

    save_path = args.savepath

    output_simulation_sub1_sync = []
    output_simulation_sub2_sync = []
    output_simulation_sub3_sync = []

    output_simulation_sub1_freq = []
    output_simulation_sub2_freq = []
    output_simulation_sub3_freq = []

    listalistosa = [1, 2]

    for i in listalistosa:
        num_subpop1 = i*5
        num_subpop2 = i*15
        num_subpop3 = i*10

        print(f'\nTrial Number {i}\n')

        print(f'Pop. 1 number of phase oscillators: {num_subpop1}')
        print(f'Pop. 2 number of phase oscillators: {num_subpop2}')
        print(f'Pop. 3 number of phase oscillators: {num_subpop3}\n')

        kuramotosakaguchi = kurasaka_oscillators(num_subpop1, num_subpop2, num_subpop3)
        coupconsts, omegas, alphas = kuramotosakaguchi.setmodelconstants(choose=True)

        print(f'Coupling constants are:\n{coupconsts}\n')
        print(f'Phase delay constants are:\n{alphas}\n')

        init_random = kuramotosakaguchi.setinitialconditions(clustered=False)
        times = kuramotosakaguchi.settimes(0., 10., 1000)

        equations = kuramotosakaguchi.kurasaka_function
        phasesevolution = kuramotosakaguchi.evolve(equations)
        syncs, ordparams = kuramotosakaguchi.findorderparameter(phasesevolution)

        print(f'Sync for SuPop 1: {numpy.mean(syncs[0][300:])}')
        print(f'Sync for SuPop 2: {numpy.mean(syncs[1][300:])}')
        print(f'Sync for SuPop 3: {numpy.mean(syncs[2][300:])}\n')

        kuramotosakaguchi.ordparam_phase()

        # kuramotosakaguchi.printcosineordparam(1, save=True)

        frequencies = kuramotosakaguchi.findperiod()
        print(f'SubPop 1 frequency: {frequencies[0]}')
        print(f'SubPop 2 frequency: {frequencies[1]}')
        print(f'SubPop 3 frequency: {frequencies[2]}\n\n')
        
        output_simulation_sub1_sync.append(numpy.mean(syncs[0][300:]))
        output_simulation_sub2_sync.append(numpy.mean(syncs[1][300:]))
        output_simulation_sub3_sync.append(numpy.mean(syncs[2][300:]))

        output_simulation_sub1_freq.append(frequencies[0])
        output_simulation_sub2_freq.append(frequencies[0])
        output_simulation_sub3_freq.append(frequencies[0])

    plt.figure(1, figsize=(13,6))
    plt.title('Sync. Param vs i')
    plt.plot(listalistosa, output_simulation_sub1_sync, label='Pop. 1')
    plt.plot(listalistosa, output_simulation_sub2_sync, label='Pop. 2')
    plt.plot(listalistosa, output_simulation_sub3_sync, label='Pop. 3')
    plt.xlabel('Index i')
    plt.ylabel('Sync.')
    plt.legend()
    plt.grid()
    plt.savefig(f'/home/f_mastellone/Images/syncvsi.png')


    plt.figure(2, figsize=(13,6))
    plt.title('Sync. Frequencies vs i')
    plt.plot(listalistosa, output_simulation_sub1_freq, label='Pop. 1')
    plt.plot(listalistosa, output_simulation_sub2_freq, label='Pop. 2')
    plt.plot(listalistosa, output_simulation_sub3_freq, label='Pop. 3')
    plt.xlabel('Index i')
    plt.ylabel('Sync. Frequencies')
    plt.legend()
    plt.grid()
    plt.savefig(f'/home/f_mastellone/Images/freqvsi.png')