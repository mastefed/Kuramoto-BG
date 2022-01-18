from kurasaka import *
import time

if __name__ == "__main__":
    
    t0 = time.time()
    
    num_subpop1 = 40
    num_subpop2 = 120
    num_subpop3 = 20
    
    #                    k12 k13 k21 k23 k31 k32 
    couplingconstants = [35.5, 5., 5., 25., 5., 5.]
    
    t_start = 0.
    t_end = 10.
    t_points = 1000

    print(f'Pop. 1 number of phase oscillators: {num_subpop1}')
    print(f'Pop. 2 number of phase oscillators: {num_subpop2}')
    print(f'Pop. 3 number of phase oscillators: {num_subpop3}\n')

    kuramotosakaguchi = kurasaka_oscillators(num_subpop1, num_subpop2, num_subpop3)
    coupconsts, alphas = kuramotosakaguchi.setmodelconstants(couplingconstants)

    print(f'Coupling constants are:\n{coupconsts}\n')

    initial_values = kuramotosakaguchi.setinitialconditions(clustered=False)
    times = kuramotosakaguchi.settimes(t_start, t_end, t_points)

    phasesevolution = kuramotosakaguchi.evolve(initial_values, times, noisy=True)

    syncs, ordparams = kuramotosakaguchi.findorderparameter(times, phasesevolution)
    
    syncSpeeds = kuramotosakaguchi.retrieveSyncSpeed(syncs)
    
    psdssync, freqsmax, _ = kuramotosakaguchi.psdofsyncs(t_start, t_end, t_points, syncs, printPlot=True)
    
    lowDeltaDetection, deltaIntegration = kuramotosakaguchi.detectDeltas(psdssync)
    
    print("Regarding the oscillations of the Synchronization parameters:")
    print(f"Presence of sub deltas oscillations: {lowDeltaDetection}")
    print(f"Value of the integral in the Delta band: {deltaIntegration}\n")
    
    globsync, globordparam = kuramotosakaguchi.findglobalorderparameter(times, ordparams)
    
    frequencies_array, mean_frequencies, std_frequencies = kuramotosakaguchi.findFrequenciesMeanStd(t_points, times, phasesevolution, syncs)
    kuramotosakaguchi.printFrequenciesPlot(num_subpop1, num_subpop2, frequencies_array, mean_frequencies)
    
    print(f'Population 1: The frequency at which the PSD is maximal is {freqsmax[0]}')
    print(f'Population 2: The frequency at which the PSD is maximal is {freqsmax[1]}')
    print(f'Population 3: The frequency at which the PSD is maximal is {freqsmax[2]}\n')
    
    kuramotosakaguchi.printsyncparam(times, syncs, globsync)
    kuramotosakaguchi.printRealImagPartOrderParameter(ordparams)
    plt.show()

    print(f'Mean Sync for SuPop 1: {numpy.mean(syncs[0][300:])}')
    print(f'Mean Sync for SuPop 2: {numpy.mean(syncs[1][300:])}')
    print(f'Mean Sync for SuPop 3: {numpy.mean(syncs[2][300:])}')
    print(f'Global Sync: {numpy.mean(globsync[300:])}\n')
    
    t1 = time.time()
    print(f'Tempo di esecuzione del codice: {(t1-t0)/60} minuti!\n')