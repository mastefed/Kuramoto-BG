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
        N1 = num_subpop1
        N2 = num_subpop2
        N3 = num_subpop3
        
        self.N = N1 + N2 + N3
        self.Narray = [N1, N2, N3]
        
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
        times = numpy.linspace(time_start, time_end, time_points)
        return times

    def setinitialconditions(self, clustered=False):
        """Set the initial phase value for every oscillator.
        
        Args:
            clustered (bool, optional): Set to True if you want to start the simulation with clustered phases. Defaults to False.
            
        Returns:
            Numpy Array: The array containing the initial values for every phase of every oscillator.
        """
        if clustered == False:
            initialvalues = 2*numpy.pi*self.reproducible_rng.random(self.N) # random conditions of phases between 0 and 2pi
        elif clustered == True:
            init_values_N1 = self.reproducible_rng.normal(
                loc=2*numpy.pi*self.reproducible_rng.random(), scale=.5, size=self.Narray[0]
                )
            init_values_N2 = self.reproducible_rng.normal(
                loc=2*numpy.pi*self.reproducible_rng.random(), scale=.5, size=self.Narray[1]
                )
            init_values_N3 = self.reproducible_rng.normal(
                loc=2*numpy.pi*self.reproducible_rng.random(), scale=.5, size=self.Narray[2]
                )
                
            initialvalues = numpy.hstack((numpy.hstack((init_values_N1, init_values_N2)), init_values_N3))
            
        return initialvalues

    def setmodelconstants(self, list_of_k):
        """Set the model's coupling constants, phase delay values and natural frequencies values
        
        Args:
            list_of_values (list): List of values for K12, K13, K23 constants. Note that K12=K21, K13=K31, K23=K32.
            
        Returns:
            List of 2D arrays: Returns the Matrices containing K and alpha
        """
        k11 = 0.5
        k12 = list_of_k[0]
        k13 = list_of_k[1]
        
        k21 = list_of_k[2]
        k22 = 0.5
        k23 = list_of_k[3]
        
        k31 = list_of_k[4]
        k32 = list_of_k[5]
        k33 = 0.2
        
        alpha11 = 0. 
        alpha22 = 0. 
        alpha33 = 0.
        
        alpha12 = 0.
        alpha13 = 0.
        alpha21 = 0.
        alpha23 = 0.
        alpha31 = 0.
        alpha32 = 0.
        
        self.kmatrix = numpy.matrix([
            [k11, k12, k13],
            [k21, k22, k23],
            [k31, k32, k33]
        ])
        
        self.alphamatrix = numpy.matrix([
            [alpha11, alpha12, alpha13],
            [alpha21, alpha22, alpha23],
            [alpha31, alpha32, alpha33]
        ])
        
        self.omega1 = cauchy.rvs(loc=143., scale=.2, size=self.Narray[0]) # 143
        self.omega2 = cauchy.rvs(loc=71.  , scale=.2, size=self.Narray[1]) # 71
        self.omega3 = cauchy.rvs(loc=95.  , scale=.2, size=self.Narray[2]) # 95
        
        return self.kmatrix, self.alphamatrix

    def evolve(self, initial_values, times, noisy=True):
        """ Given the initial values for every phase oscillator, this function calculates
              the next set of values using the Kuramoto-Sakaguchi set of equations.
              One can choose to use the noiseless Kuramoto-Sakaguchi model or add some noise.
              When noise is provided the next set of points are calculated using
              the Rößler2010 order 1.0 strong Stochastic Runge-Kutta algorithm SRI2.
              When noise is not provided the next set of points are calculated using
              scipy.odeint().
        
        Args:
            noisy (bool, optional): Choose if you want noise or not. Defaults to True.
        """
        def kurasaka_function(x, t):
            variables = {}
            # gets an array of phases' value at time t_k, odeint update the values everytime for every t_j
            
            for i in range(self.Narray[0]):
                variables[f'theta1{i}'] = x[i]
                
            for i in range(self.Narray[1]):
                variables[f'theta2{i}'] = x[self.Narray[0] + i]
                
            for i in range(self.Narray[2]):
                variables[f'theta3{i}'] = x[self.Narray[0] + self.Narray[1] + i]
                
            def interaction(k, z):
                # Creates the interaction terms of the Kuramoto model, these are 
                # proportional to the sine of the difference between the phases
                interaction_terms = 0.
                for j in range(self.Narray[k-1]):
                    sine_term = numpy.sin(variables[f'theta{k}{j}'] - variables[f'theta{z}{i}'] - self.alphamatrix[z-1,k-1])
                    interaction_terms += self.kmatrix[z-1,k-1]/self.Narray[k-1] * sine_term
                return interaction_terms
                
            # Creates and updates the values' array with the desired differential equations
            dthetadt = [] 
            
            for i in range(self.Narray[0]):
                dthetadt.append(
                    self.omega1[i] + interaction(1, 1) + interaction(2, 1) + interaction(3, 1)
                )
                
            for i in range(self.Narray[1]):
                dthetadt.append(
                    self.omega2[i] + interaction(1, 2) + interaction(2, 2) + interaction(3, 2)
                )
                
            for i in range(self.Narray[2]):
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
            kurasaka_evo = itoint(kurasaka_function, noise, initial_values, times)
        elif noisy is False:
            kurasaka_evo = odeint(kurasaka_function, initial_values, times)    
        
        return kurasaka_evo

    def findorderparameter(self, times, phases): # Qua ho effettivamente bisogno solo di phases
        orderparameter_subpop1 = []
        orderparameter_subpop2 = []
        orderparameter_subpop3 = []
        
        for i in range(len(times)):
            orderparameter_subpop1.append(
                1/self.Narray[0] * sum(numpy.exp(complex(0,phases[i][j]))  for j in range(self.Narray[0]))
                )
            orderparameter_subpop2.append(
                1/self.Narray[1] * sum(numpy.exp(complex(0,phases[i][self.Narray[0] + j]))  for j in range(self.Narray[1]))
                )
            orderparameter_subpop3.append(
                1/self.Narray[2] * sum(numpy.exp(complex(0,phases[i][self.Narray[0] + self.Narray[1] + j]))  for j in range(self.Narray[2]))
                )
                
        sync_subpop1 = []
        sync_subpop2 = []
        sync_subpop3 = []
        
        for i in range(len(orderparameter_subpop1)):
            sync_subpop1.append(numpy.absolute(orderparameter_subpop1[i]))
            
        for i in range(len(orderparameter_subpop2)):
            sync_subpop2.append(numpy.absolute(orderparameter_subpop2[i]))
            
        for i in range(len(orderparameter_subpop3)):
            sync_subpop3.append(numpy.absolute(orderparameter_subpop3[i]))
            
        
        syncs = [sync_subpop1, sync_subpop2, sync_subpop3]
        orderparameters = [orderparameter_subpop1, orderparameter_subpop2, orderparameter_subpop3]
        
        return syncs, orderparameters # returns |Z| and Z, both can be useful

    def findglobalorderparameter(self, times, order_parameters):
        def mediationterm(sigma, tau):
            mediation = self.kmatrix[sigma,tau]/self.kmatrix.sum()*numpy.exp(complex(0,-self.alphamatrix[sigma,tau]))
            return mediation

        globalorderparameter = []

        for i in range(len(times)):
            partialglobalorderparam = 0.
            for sigma in range(3):
                for tau in range(3):
                    partialglobalorderparam += mediationterm(sigma,tau)*order_parameters[tau][i]
            globalorderparameter.append(partialglobalorderparam)

        sync_global = []
        phase_global = []
        for i in range(len(globalorderparameter)):
            sync_global.append(numpy.absolute(globalorderparameter[i]))
            phase_global.append(numpy.angle(globalorderparameter[i]))

        return sync_global, globalorderparameter
    
    def retrieveSyncSpeed(self, syncs):
        syncSpeeds = {}
        
        if numpy.any(numpy.array(syncs[0]) > 0.8) == True:
            minTimePointPop1 = numpy.argwhere(numpy.array(syncs[0]) > 0.8).min()
            syncSpeeds['Population 1 Sync Speed'] = minTimePointPop1
            print(f'Population 1 reaches |Z|>0.80 in {minTimePointPop1} integration points')
        else:
            minTimePointPop1 = None
            syncSpeeds['Population 1 Sync Speed'] = minTimePointPop1
            print('Population 1 didn\'t reach a |Z| value big enough to be considered synchronized')
            
        if numpy.any(numpy.array(syncs[1]) > 0.8) == True:
            minTimePointPop2 = numpy.argwhere(numpy.array(syncs[1]) > 0.8).min()
            syncSpeeds['Population 2 Sync Speed'] = minTimePointPop2
            print(f'Population 2 reaches |Z|>0.80 in {minTimePointPop2} integration points')
        else:
            minTimePointPop2 = None
            syncSpeeds['Population 2 Sync Speed'] = minTimePointPop2
            print('Population 2 didn\'t reach a |Z| value big enough to be considered synchronized')
        
        if numpy.any(numpy.array(syncs[2]) > 0.8) == True:
            minTimePointPop3 = numpy.argwhere(numpy.array(syncs[2]) > 0.8).min()
            syncSpeeds['Population 3 Sync Speed'] = minTimePointPop3
            print(f'Population 3 reaches |Z|>0.80 in {minTimePointPop3} integration points\n')
        else:
            minTimePointPop3 = None
            syncSpeeds['Population 3 Sync Speed'] = minTimePointPop3
            print('Population 3 didn\'t reach a |Z| value big enough to be considered synchronized\n')    
        
        return syncSpeeds

    def psdofordparam(self, time_start, time_end, time_points, real_part_orderparameters, save=False, savepath=None):
        
        my_fs = 1./((time_end - time_start)/time_points)
        
        freq1, psd1 = welch(real_part_orderparameters[0], fs = my_fs)
        freq2, psd2 = welch(real_part_orderparameters[1], fs = my_fs)
        freq3, psd3 = welch(real_part_orderparameters[2], fs = my_fs)
        
        plt.figure('PSD', figsize=(6,6))
        plt.title('PSD of Re[Z]')
        plt.xlabel('Frequencies [Hz]')
        plt.ylabel('PSD')
        plt.xlim(0., 100.)
        plt.grid()
        plt.plot(freq1, psd1, label='Pop. 1')
        plt.plot(freq2, psd2, label='Pop. 2')
        plt.plot(freq3, psd3, label='Pop. 3')
        plt.legend()
        
        if save == True:
            plt.savefig(savepath)
        elif save == False:
            pass

    def psdofsyncs(self, time_start, time_end, time_points, syncs, printPlot=False, save=False, savepath=None):
        
        my_fs = 1./((time_end - time_start)/time_points)
        
        first_sync = syncs[0] - numpy.mean(syncs[0])
        second_sync = syncs[1] - numpy.mean(syncs[1])
        third_sync = syncs[2] - numpy.mean(syncs[2])
        
        freq1, psd1 = welch(first_sync, fs = my_fs)
        freq2, psd2 = welch(second_sync, fs = my_fs)   
        freq3, psd3 = welch(third_sync, fs = my_fs)
        
        freq1max = freq1[numpy.argmax(psd1)]
        freq2max = freq2[numpy.argmax(psd2)]
        freq3max = freq3[numpy.argmax(psd3)]
        
        psd1max = psd1[numpy.argmax(psd1)]
        psd2max = psd2[numpy.argmax(psd2)]
        psd3max = psd3[numpy.argmax(psd3)]
        
        psds = [psd1, psd2, psd3]
        freqsmax = [freq1max, freq2max, freq3max]
        psdsmax = [psd1max, psd2max, psd3max]
        
        if printPlot == True:
            plt.figure('PSD', figsize=(6,6))
            plt.title('PSD of |Z|')
            plt.xlabel('Frequencies [Hz]')
            plt.ylabel('PSD')
            plt.xlim((0,10))
            plt.grid()
            plt.plot(freq1, psd1, label='Pop. 1')
            plt.plot(freq2, psd2, label='Pop. 2')
            plt.plot(freq3, psd3, label='Pop. 3')
            plt.axvspan(1, 4, color='silver', alpha=0.5)
            plt.legend()
            
            if save == True:
                plt.savefig(savepath)
            elif save == False:
                pass
        
        return psds, freqsmax, psdsmax
        
    def detectDeltas(self, psdOfSyncs):
        psd1 = psdOfSyncs[0]
        psd2 = psdOfSyncs[1]
        psd3 = psdOfSyncs[2]
        
        delta_pop1 = simps(psd1[3:6])
        
        psd1max_LowDelta = psd1[numpy.argmax(psd1[:4])]
        threshold_LowDelta_Pop1 = numpy.mean(psd1)+5*numpy.std(psd1)
        
        if (psd1max_LowDelta > threshold_LowDelta_Pop1):
            print("Population 1: Sub Delta oscillations detected.")
            lowDeltaDetection_Pop1 = True
        else:
            print("Population 1: Sub Delta oscillations not detected.")
            lowDeltaDetection_Pop1 = False
            
        delta_pop2 = simps(psd2[3:6])
        
        psd2max_LowDelta = psd2[numpy.argmax(psd2[:4])]
        threshold_LowDelta_Pop2 = numpy.mean(psd2)+5*numpy.std(psd2)
        
        if (psd2max_LowDelta > threshold_LowDelta_Pop2):
            print("Population 2: Sub Delta oscillations detected.")
            lowDeltaDetection_Pop2 = True
        else:
            print("Population 2: Sub Delta oscillations not detected.")
            lowDeltaDetection_Pop2 = False
            
        delta_pop3 = simps(psd3[3:6])
        
        psd3max_LowDelta = psd3[numpy.argmax(psd3[:4])]
        threshold_LowDelta_Pop3 = numpy.mean(psd3)+5*numpy.std(psd3)
        
        if (psd3max_LowDelta > threshold_LowDelta_Pop3):
            print("Population 3: Sub Delta oscillations detected.\n")
            lowDeltaDetection_Pop3 = True
        else:
            print("Population 3: Sub Delta oscillations not detected.\n")
            lowDeltaDetection_Pop3 = False
        
        lowDeltaDetection = [lowDeltaDetection_Pop1, lowDeltaDetection_Pop2, lowDeltaDetection_Pop3]
        deltaIntegration = [delta_pop1, delta_pop2, delta_pop3]
        
        return lowDeltaDetection, deltaIntegration
        

    def findFrequenciesMeanStd(self, time_points, times, phases, syncs):
        oscillators_frequencies = []
        
        for j in range(time_points - 1):
            dummy_array = []
            
            for i in range(self.N):
                frequency = ((phases[j+1,i] - phases[j,i])/(times[j+1] - times[j]))/(2*numpy.pi)
                dummy_array.append(frequency)
            
            oscillators_frequencies.append(dummy_array)
            """ ^^^
                For each time point t_j I get an array containing
                all the frequencies for every oscillator
            """
        oscillators_frequencies = numpy.matrix(oscillators_frequencies)
        
        populations_mean_frequencies = []
        populations_std_of_frequencies = []
        
        for i in range(time_points - 1):
            dummy_array_freq = []
            dummy_array_std = []
            
            dummy_array_freq.append(
                numpy.mean(oscillators_frequencies[i, :self.Narray[0]])
            )
            dummy_array_freq.append(
                numpy.mean(oscillators_frequencies[i, self.Narray[0]:self.Narray[0] + self.Narray[1]])
            )
            dummy_array_freq.append(
                numpy.mean(oscillators_frequencies[i, self.Narray[0] + self.Narray[1]:])
            )
            
            dummy_array_freq = numpy.array(dummy_array_freq)
            populations_mean_frequencies.append(dummy_array_freq)
            """ ^^^
                For each time point t_j I get an array containing
                3 values: mean frequency of pop1, pop2 and pop3 
            """
            
            dummy_array_std.append(
                numpy.std(oscillators_frequencies[i, :self.Narray[0]])
            )
            dummy_array_std.append(
                numpy.std(oscillators_frequencies[i, self.Narray[0]:self.Narray[0] + self.Narray[1]])
            )
            dummy_array_std.append(
                numpy.std(oscillators_frequencies[i, self.Narray[0] + self.Narray[1]:])
            )
            
            dummy_array_std = numpy.array(dummy_array_std)
            populations_std_of_frequencies.append(dummy_array_std)
            """ ^^^
                For each time point t_j I get an array containing
                3 values: std of frequency of pop1, pop2 and pop3 
            """
            
        populations_mean_frequencies = numpy.matrix(populations_mean_frequencies)
        populations_std_of_frequencies = numpy.matrix(populations_std_of_frequencies)
        
        if numpy.any(numpy.array(syncs[0]) > 0.8) == True:
            min_tp_1 = numpy.argwhere(numpy.array(syncs[0]) > 0.8).min()
            print(f'Population 1\'s mean frequency after synchronization: {numpy.mean(populations_mean_frequencies[:,0][min_tp_1:])}')
        else:
            print(f'Population 1 can\'t be considered synchronized, its mean frequency across the simulation is  {numpy.mean(populations_mean_frequencies[:,0][300:])}')
            
        if numpy.any(numpy.array(syncs[1]) > 0.8) == True:
            min_tp_2 = numpy.argwhere(numpy.array(syncs[1]) > 0.8).min()
            print(f'Population 2\'s mean frequency after synchronization: {numpy.mean(populations_mean_frequencies[:,1][min_tp_2:])}')
        else:
            print(f'Population 2 can\'t be considered synchronized, its mean frequency across the simulation is  {numpy.mean(populations_mean_frequencies[:,1][300:])}')
            
        if numpy.any(numpy.array(syncs[2]) > 0.8) == True:
            min_tp_3 = numpy.argwhere(numpy.array(syncs[2]) > 0.8).min()
            print(f'Population 3\'s mean frequency after synchronization: {numpy.mean(populations_mean_frequencies[:,2][min_tp_3:])}\n')
        else:
            print(f'Population 3 can\'t be considered synchronized, its mean frequency across the simulation is  {numpy.mean(populations_mean_frequencies[:,2][300:])}\n')
        
        return oscillators_frequencies, populations_mean_frequencies, populations_std_of_frequencies
        
    def printsyncparam(self, times, syncs, globsync, save=False, savepath=None):
        plt.figure(f'{self.N} Oscillators Sync', figsize=(13,6))
        plt.title(f'{self.N} Oscillators Sync')
        plt.plot(times, syncs[0], label='SubPop 1')
        plt.plot(times, syncs[1], label='SubPop 2')
        plt.plot(times, syncs[2], label='SubPop 3')
        plt.plot(times, globsync, label='Global')
        plt.xlabel('Time [s]')
        plt.ylabel('|Z|')
        plt.ylim([0.,1.])
        plt.yticks(numpy.arange(0, 1.1, step=0.1))
        plt.legend(loc='lower left')
        
        plt.axes([.69, .125, .2, .2])
        plt.plot(times, syncs[0], label='SubPop 1')
        plt.plot(times, syncs[1], label='SubPop 2')
        plt.plot(times, syncs[2], label='SubPop 3')
        plt.plot(times, globsync, label='Global')
        plt.xlim([5.,5.5])
        plt.xticks([5.1, 5.2, 5.3, 5.4])
        plt.yticks([.2, .4, .6, .8])
        plt.tick_params(axis='x', direction='in', pad=-15)
        plt.tick_params(axis='y', direction='in', pad=-22)
        
        if save == True:
            plt.savefig(savepath)
        elif save == False:
            pass

    def printcosineordparam(self, times, orderparameters, save=False, savepath=None):
        plt.figure("Subpops' Phase Evolution", figsize=(13,6))
        plt.title("Subpops' Phase Evolution")
        plt.plot(times, numpy.real(orderparameters[0]), label='SubPop 1')
        plt.plot(times, numpy.real(orderparameters[1]), label='SubPop 2')
        plt.plot(times, numpy.real(orderparameters[2]), label='SubPop 3')
        plt.xlabel('Time Steps')
        plt.xlim([5.,7.])
        plt.legend(loc='lower left')
        
        plt.axes([.69, .125, .2, .2])
        plt.plot(times, numpy.real(orderparameters[0]), label='SubPop 1')
        plt.plot(times, numpy.real(orderparameters[1]), label='SubPop 2')
        plt.plot(times, numpy.real(orderparameters[2]), label='SubPop 3')
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

    def printRealImagPartOrderParameter(self, orderparameters):
        plt.rc('text', usetex=True)
        plt.figure("RealImagPartOrderParameter")
        plt.xlabel(r"$\rho_1 \cos(\Psi_1)$")
        plt.ylabel(r"$\rho_1 \sin(\Psi_1)$")
        plt.xlim((-1,1))
        plt.ylim((-1,1))
        ordparam_pop1 = orderparameters[0]
        partreale_ordparam_pop1 = numpy.real(ordparam_pop1)
        partimag_ordparam_pop1 = numpy.imag(ordparam_pop1)
        plt.plot(partreale_ordparam_pop1, partimag_ordparam_pop1)
        plt.scatter(partreale_ordparam_pop1[0], partimag_ordparam_pop1[0], c='red', s=100, label='Initial point')
        plt.scatter(partreale_ordparam_pop1[-1], partimag_ordparam_pop1[-1], c='green', s=100, label='Final point')
        plt.legend()

    def animateoscillators(self, times, time_start, time_end, phaseevolution, syncs, globsync, orderparameter, globorderparameter):     
        def animate_function(i):
            phases = phaseevolution[i:i+1]
            timestep = times[0:i]
            R1 = syncs[0][0:i]
            R2 = syncs[1][0:i]
            R3 = syncs[2][0:i]
            RGlob = globsync[0:i]
    
            imphasedict = {}
            rephasedict = {}
    
            for k in range(self.N):
                imphasedict[f'im_x{k}'] = numpy.exp(complex(0, phases[0][k])).imag
                rephasedict[f're_x{k}'] = numpy.exp(complex(0, phases[0][k])).real
    
            imagpart_ordparam_subpop1 = orderparameter[0][i].imag
            realpart_ordparam_subpop1 = orderparameter[0][i].real
    
            imagpart_ordparam_subpop2 = orderparameter[1][i].imag
            realpart_ordparam_subpop2 = orderparameter[1][i].real
    
            imagpart_ordparam_subpop3 = orderparameter[2][i].imag
            realpart_ordparam_subpop3 = orderparameter[2][i].real
    
            imagpart_global_ordparam = globorderparameter[i].imag
            realpart_global_ordparam = globorderparameter[i].real
    
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
            
            for k in range(self.Narray[0]):
                ax1.plot(rephasedict[f're_x{k}'], imphasedict[f'im_x{k}'], 'bo', ms=7.)
            for k in range(self.Narray[1]):
                ax1.plot(rephasedict[f're_x{self.N1 + k}'], imphasedict[f'im_x{self.N1 + k}'], 'go', ms=7.)
            for k in range(self.Narray[2]):
                ax1.plot(rephasedict[f're_x{self.N1 + self.N2 + k}'], imphasedict[f'im_x{self.N1 + self.N2 + k}'], 'ro', ms=7.)
            ax1.legend()
    
            ax2.clear()
            ax2.set_ylim([0.,1.])
            ax2.set_xlim([time_start, time_end])
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
        self.animated = animation.FuncAnimation(fig, animate_function, frames = len(phaseevolution), interval=0.1)

        return self.animated

    def saveanimation(self, myanimation, save_path):
        print('\nVideo Processing started!')
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=120, metadata=dict(artist='F. V. Mastellone'), bitrate=1800)
        myanimation.save(save_path, writer=writer)
        print('Task finished.')

    def showplots(self):
        plt.show()