import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.integrate import odeint, simps
from scipy.stats import cauchy
from scipy.signal import find_peaks, welch

from sdeint import itoint

class kurasaka_oscillators:
    def __init__(self, num_subpop1, num_subpop2, num_subpop3):
        """ Init Function for the class

        Args:
            num_subpop1 (int): the number of oscillators in the first population
            num_subpop2 (int): the number of oscillators in the second population
            num_subpop3 (int): the number of oscillators in the third population
        """
        self.N1 = num_subpop1
        self.N2 = num_subpop2
        self.N3 = num_subpop3
        self.N = self.N1 + self.N2 + self.N3
        self.Narray = [self.N1, self.N2, self.N3]
        
        self.reproducible_rng = numpy.random.default_rng(42)
        self.notreproducible_rng = numpy.random.default_rng()

    def settimes(self, time_start, time_end, time_points):
        """ Generates the array of time points to integrate
            the set of differential equations

        Args:
            time_start (float): the starting time value
            time_end (float): the stop time value
            time_points (int): how many time points do you want?

        Returns:
            Numpy Array: The array of time points
        """
        self.time_start = time_start
        self.time_end = time_end
        self.time_points = time_points
        self.times = numpy.linspace(self.time_start, self.time_end, self.time_points)
        return self.times

    def setinitialconditions(self, clustered=False):
        """Set the initial phase value for every oscillator.

        Args:
            clustered (bool, optional): Set to True if you want to start the simulation with clustered phases. Defaults to False.

        Returns:
            Numpy Array: The array containing the initial values for every phase of every oscillator.
        """
        if clustered == False:
            self.initialvalues = 2*numpy.pi*self.reproducible_rng.random(self.N) # random conditions of phases between 0 and 2pi

        elif clustered == True:
            self.init_values_N1 = self.reproducible_rng.normal(
                loc=2*numpy.pi*self.reproducible_rng.random(), scale=.5, size=self.N1
                )
            self.init_values_N2 = self.reproducible_rng.normal(
                loc=2*numpy.pi*self.reproducible_rng.random(), scale=.5, size=self.N2
                )
            self.init_values_N3 = self.reproducible_rng.normal(
                loc=2*numpy.pi*self.reproducible_rng.random(), scale=.5, size=self.N3
                )

            self.initialvalues = numpy.hstack((numpy.hstack((self.init_values_N1, self.init_values_N2)), self.init_values_N3))

        return self.initialvalues

    def setmodelconstants(self, list_of_k, fixed=True):
        """Set the model's coupling constants, phase delay values and natural frequencies values

        Args:
            list_of_values (list): List of values for K12, K13, K23 constants. Note that K12=K21, K13=K31, K23=K32.
            fixed (bool, optional): If False one will be asked to decide each value for the coupling constants. Defaults to True.

        Returns:
            List of 2D arrays: Returns the Matrices containing K, alpha and omega values.
        """
        if fixed == False:
            print("Please, choose the intra-subpopulations' coupling constants:")
            self.k11 = float(input('Choose the coupling constant for subpopulation 1 <--> subpopulation 1 interaction: '))
            self.k22 = float(input('Choose the coupling constant for subpopulation 2 <--> subpopulation 2 interaction: '))
            self.k33 = float(input('Choose the coupling constant for subpopulation 3 <--> subpopulation 3 interaction: '))

            print("\nNow, choose the inter-subpopulations' coupling constants:")
            self.k12 = float(input('Choose the coupling constant for subpopulation 1 <--> subpopulation 2 interaction: '))
            self.k13 = float(input('Choose the coupling constant for subpopulation 1 <--> subpopulation 3 interaction: '))
            self.k23 = float(input('Choose the coupling constant for subpopulation 2 <--> subpopulation 3 interaction: '))
            self.k21 = float(input('Choose the coupling constant for subpopulation 2 <--> subpopulation 1 interaction: '))
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

        if fixed == True:
            self.k11 = 0.5
            self.k22 = 0.5 
            self.k33 = 0.2

            self.k12 = list_of_k[0]
            self.k13 = list_of_k[1]
            self.k23 = list_of_k[2]
            
            self.k21 = list_of_k[3]
            self.k31 = list_of_k[4]
            self.k32 = list_of_k[5]

            self.alpha11 = 0. 
            self.alpha22 = 0. 
            self.alpha33 = 0.

            self.alpha12 = 0.
            self.alpha13 = 0.
            self.alpha21 = 0.
            self.alpha23 = 0.
            self.alpha31 = 0.
            self.alpha32 = 0.

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

        self.omega1 = cauchy.rvs(loc=143., scale=.2, size=self.N1) # 143
        self.omega2 = cauchy.rvs(loc=71., scale=.2, size=self.N2) # 71
        self.omega3 = cauchy.rvs(loc=95., scale=.2, size=self.N3) # 95

        self.omegamatrix = numpy.hstack((numpy.hstack((self.omega1, self.omega2)), self.omega3))

        return self.kmatrix, self.omegamatrix, self.alphamatrix

    def evolve(self, noisy=True):
        
        def kurasaka_function(x, t):
            variables = {}
            """ gets an array of phases' value at time t_k, odeint update the values everytime for every t_j
            """
            for i in range(self.N1):
                variables[f'theta1{i}'] = x[i] 

            for i in range(self.N2):
                variables[f'theta2{i}'] = x[self.N1 + i]

            for i in range(self.N3):
                variables[f'theta3{i}'] = x[self.N1 + self.N2 + i]

            def interaction(k, z):
                """ Creates the interaction terms of the Kuramoto model, these are 
                    proportional to the sine of the difference between the phases
                """
                interaction_terms = 0.
                for j in range(self.Narray[k-1]):
                    sine_term = numpy.sin(variables[f'theta{k}{j}'] - variables[f'theta{z}{i}'] - self.alphamatrix[z-1,k-1])
                    interaction_terms += self.kmatrix[z-1,k-1]/self.Narray[k-1] * sine_term
                return interaction_terms

            """ Creates and updates the values' array with the desired differential equations
            """
            dthetadt = [] 

            for i in range(self.N1):
                dthetadt.append(
                    self.omega1[i] + interaction(1, 1) + interaction(2, 1) + interaction(3, 1)
                )

            for i in range(self.N2):
                dthetadt.append(
                    self.omega2[i] + interaction(1, 2) + interaction(2, 2) + interaction(3, 2)
                )

            for i in range(self.N3):
                dthetadt.append(
                    self.omega3[i] + interaction(1, 3) + interaction(2, 3) + interaction(3, 3)
                )

            dthetadt = numpy.array(dthetadt)
            return dthetadt

        def noise(x, t):
            sigma = []
            for i in range(self.N):
                sigma.append(0.8)
            sigma = numpy.diag(sigma)
            return sigma
        
        if noisy is True:
            self.kurasaka_evo = itoint(kurasaka_function, noise, self.initialvalues, self.times)
        elif noisy is False:
            self.kurasaka_evo = odeint(kurasaka_function, self.initialvalues, self.times)    
        
        return self.kurasaka_evo

    def findorderparameter(self, phases):
        self.orderparameter_subpop1 = []
        self.orderparameter_subpop2 = []
        self.orderparameter_subpop3 = []

        for i in range(len(self.times)):
            self.orderparameter_subpop1.append(
                1/self.N1 * sum(numpy.exp(complex(0,phases[i][j]))  for j in range(self.N1))
                )
            self.orderparameter_subpop2.append(
                1/self.N2 * sum(numpy.exp(complex(0,phases[i][self.N1 + j]))  for j in range(self.N2))
                )
            self.orderparameter_subpop3.append(
                1/self.N3 * sum(numpy.exp(complex(0,phases[i][self.N1 + self.N2 + j]))  for j in range(self.N3))
                )

        self.sync_subpop1 = []
        self.sync_subpop2 = []
        self.sync_subpop3 = []

        for i in range(len(self.orderparameter_subpop1)):
            self.sync_subpop1.append(
                numpy.sqrt(self.orderparameter_subpop1[i].real**2 + self.orderparameter_subpop1[i].imag**2)
                )
        for i in range(len(self.orderparameter_subpop2)):
            self.sync_subpop2.append(
                numpy.sqrt(self.orderparameter_subpop2[i].real**2 + self.orderparameter_subpop2[i].imag**2)
                )
        for i in range(len(self.orderparameter_subpop3)):
            self.sync_subpop3.append(
                numpy.sqrt(self.orderparameter_subpop3[i].real**2 + self.orderparameter_subpop3[i].imag**2)
                )

        self.syncs = [self.sync_subpop1, self.sync_subpop2, self.sync_subpop3]
        self.orderparameters = [self.orderparameter_subpop1, self.orderparameter_subpop2, self.orderparameter_subpop3]

        return self.syncs, self.orderparameters # returns |Z| and Z, both can be useful

    def findglobalorderparameter(self, order_parameters):
        def mediationterm(sigma, tau):
            mediation = self.kmatrix[sigma,tau]/self.kmatrix.sum()*numpy.exp(complex(0,-self.alphamatrix[sigma,tau]))
            return mediation

        self.globalorderparameter = []

        for i in range(len(self.times)):
            partialglobalorderparam = 0.
            for sigma in range(3):
                for tau in range(3):
                    partialglobalorderparam += mediationterm(sigma,tau)*order_parameters[tau][i]
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

    def psdofordparam(self, save, savepath):
        self.freq1, self.psd1 = welch(self.real_ordparam_subpop1, 
                                      fs=1/((self.time_end - self.time_start)/self.time_points))
        self.freq2, self.psd2 = welch(self.real_ordparam_subpop2, 
                                      fs=1/((self.time_end - self.time_start)/self.time_points))
        self.freq3, self.psd3 = welch(self.real_ordparam_subpop3, 
                                      fs=1/((self.time_end - self.time_start)/self.time_points))

        plt.figure('PSD', figsize=(6,6))
        plt.title('PSD of Re[Z]')
        plt.xlabel('Frequencies [Hz]')
        plt.ylabel('PSD')
        plt.xlim(0., 100.)
        plt.grid()
        plt.plot(self.freq1, self.psd1, label='Pop. 1')
        plt.plot(self.freq2, self.psd2, label='Pop. 2')
        plt.plot(self.freq3, self.psd3, label='Pop. 3')
        plt.legend()

        if save == True:
            plt.savefig(savepath)
        elif save == False:
            pass

    def findperiod_phases(self, phases):
        self.oscillators_frequencies = []
        
        for j in range(self.time_points - 1):
            dummy_array = []
            for i in range(self.N):
                frequency = ((phases[j+1,i] - phases[j,i])/(self.times[j+1] - self.times[j]))/(2*numpy.pi)
                dummy_array.append(frequency)
            self.oscillators_frequencies.append(dummy_array)
            
        self.oscillators_frequencies = numpy.matrix(self.oscillators_frequencies)
        
        self.populations_mean_frequencies = []
        for i in range(self.time_points - 1):
            dummy_array = []
            dummy_array.append(
                numpy.mean(
                    self.oscillators_frequencies[i, :self.N1]
                )
            )
            dummy_array.append(
                numpy.mean(
                    self.oscillators_frequencies[i, self.N1:self.N1 + self.N2]
                )
            )
            dummy_array.append(
                numpy.mean(
                    self.oscillators_frequencies[i, self.N1 + self.N2:]
                )
            )
            dummy_array = numpy.array(dummy_array)
            self.populations_mean_frequencies.append(dummy_array)
        self.populations_mean_frequencies = numpy.matrix(self.populations_mean_frequencies)
        return self.oscillators_frequencies, self.populations_mean_frequencies

    def findperiod_orderparameter(self):
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

        self.mean_frequencies = [self.mean_frequency_subpop1, 
                                 self.mean_frequency_subpop2, 
                                 self.mean_frequency_subpop3]

        return self.mean_frequencies

    def printsyncparam(self, save, savepath):
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
        
        plt.axes([.69, .125, .2, .2])
        plt.plot(self.times, self.sync_subpop1, label='SubPop 1')
        plt.plot(self.times, self.sync_subpop2, label='SubPop 2')
        plt.plot(self.times, self.sync_subpop3, label='SubPop 3')
        plt.plot(self.times, self.sync_global, label='Global')
        plt.xlim([5.,5.5])
        plt.xticks([5.1, 5.2, 5.3, 5.4])
        plt.yticks([.2, .4, .6, .8, 1.])
        plt.tick_params(axis='x', direction='in', pad=-15)
        plt.tick_params(axis='y', direction='in', pad=-22)
        
        if save == True:
            plt.savefig(savepath)
        elif save == False:
            pass

    def printcosineordparam(self, save, savepath):
        plt.figure("Subpops' Phase Evolution", figsize=(13,6))
        plt.title("Subpops' Phase Evolution")
        plt.plot(self.times, self.real_ordparam_subpop1, label='SubPop 1')
        plt.plot(self.times, self.real_ordparam_subpop2, label='SubPop 2')
        plt.plot(self.times, self.real_ordparam_subpop3, label='SubPop 3')
        plt.xlabel('Time Steps')
        plt.xlim([5.,7.])
        plt.legend(loc='lower left')
        
        plt.axes([.69, .125, .2, .2])
        plt.plot(self.times, self.real_ordparam_subpop1, label='SubPop 1')
        plt.plot(self.times, self.real_ordparam_subpop2, label='SubPop 2')
        plt.plot(self.times, self.real_ordparam_subpop3, label='SubPop 3')
        plt.xlim([5.,5.2])
        plt.ylim([-1.3,1.1])
        plt.xticks([5.050, 5.100, 5.150])
        plt.yticks([])
        plt.grid()
        plt.tick_params(axis='x', direction='in', pad=-15)
        plt.tick_params(axis='y', direction='in', pad=-22)
        
        if save == True:
            plt.savefig(savepath)
        elif save == False:
            pass

    def animateoscillators(self):     
        def animate_function(i):
            phases = self.kurasaka_evo[i:i+1]
            timestep = self.times[0:i]
            R1 = self.sync_subpop1[0:i]
            R2 = self.sync_subpop2[0:i]
            R3 = self.sync_subpop3[0:i]
            RGlob = self.sync_global[0:i]
    
            imphasedict = {}
            rephasedict = {}
    
            for k in range(self.N):
                imphasedict[f'im_x{k}'] = numpy.exp(complex(0, phases[0][k])).imag
                rephasedict[f're_x{k}'] = numpy.exp(complex(0, phases[0][k])).real
    
            imagpart_ordparam_subpop1 = self.orderparameter_subpop1[i].imag
            realpart_ordparam_subpop1 = self.orderparameter_subpop1[i].real
    
            imagpart_ordparam_subpop2 = self.orderparameter_subpop2[i].imag
            realpart_ordparam_subpop2 = self.orderparameter_subpop2[i].real
    
            imagpart_ordparam_subpop3 = self.orderparameter_subpop3[i].imag
            realpart_ordparam_subpop3 = self.orderparameter_subpop3[i].real
    
            imagpart_global_ordparam = self.globalorderparameter[i].imag
            realpart_global_ordparam = self.globalorderparameter[i].real
    
            ticks = [-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8]
    
            ax1.clear()
            circ = plt.Circle((0, 0), radius=1, lw=0.3, edgecolor='k', facecolor='None')
            ax1.add_patch(circ)
            ax1.set_xlim(-1.2, 1.2)
            ax1.set_ylim(-1.2, 1.2)
            ax1.spines['left'].set_position('center')
            ax1.spines['right'].set_color('none')
            ax1.spines['bottom'].set_position('center')
            ax1.spines['top'].set_color('none')
            ax1.yaxis.set_ticks(ticks)
            ax1.xaxis.set_ticks(ticks)
            ax1.set_xlabel('Re', loc='right')
            ax1.set_ylabel('Im', loc='top')
    
            ax1.arrow(0., 0., realpart_ordparam_subpop1, imagpart_ordparam_subpop1, 
                      head_width=0.02, head_length=0.05, fc='b', ec='b', lw=1., label='Z Pop. 1')
            
            ax1.arrow(0., 0., realpart_ordparam_subpop2, imagpart_ordparam_subpop2, 
                      head_width=0.02, head_length=0.05, fc='g', ec='g', lw=1., label='Z Pop. 2')
            
            ax1.arrow(0., 0., realpart_ordparam_subpop3, imagpart_ordparam_subpop3, 
                      head_width=0.02, head_length=0.05, fc='r', ec='r', lw=1., label='Z Pop. 3')
            
            ax1.arrow(0., 0., realpart_global_ordparam, imagpart_global_ordparam, 
                      head_width=0.02, head_length=0.05, fc='k', ec='k', lw=1.3, label='Z Global')
            
            for k in range(self.N1):
                ax1.plot(rephasedict[f're_x{k}'], imphasedict[f'im_x{k}'], 'bo', ms=7.)
            for k in range(self.N2):
                ax1.plot(rephasedict[f're_x{self.N1 + k}'], imphasedict[f'im_x{self.N1 + k}'], 'go', ms=7.)
            for k in range(self.N3):
                ax1.plot(rephasedict[f're_x{self.N1 + self.N2 + k}'], imphasedict[f'im_x{self.N1 + self.N2 + k}'], 'ro', ms=7.)
            ax1.legend()
    
            ax2.clear()
            ax2.set_ylim([0.,1.])
            ax2.set_xlim([self.time_start, self.time_end])
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('R')
    
            ax2.plot(timestep, R1, label='Sync. Par. Pop. 1')
            ax2.plot(timestep, R2, label='Sync. Par. Pop. 2')
            ax2.plot(timestep, R3, label='Sync. Par. Pop. 3')
            ax2.plot(timestep, RGlob, label='Global Sync.')
            ax2.legend()
        
        
        fig = plt.figure(f'{self.N} Oscillators Animated', figsize=(13,6))
        plt.suptitle(f'{self.N} Oscillators')
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        self.animated = animation.FuncAnimation(fig, animate_function, frames = len(self.kurasaka_evo), interval=0.1)

        return self.animated

    def saveanimation(self, myanimation, save_path):
        print('\nVideo Processing started!')
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=120, metadata=dict(artist='F. V. Mastellone'), bitrate=1800)
        myanimation.save(save_path, writer=writer)
        print('Task finished.')

    def showplots(self):
        plt.show()