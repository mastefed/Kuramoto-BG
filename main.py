# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:33:47 2021

@author: Federico Vincenzo Mastellone
"""

from kurasaka import *
import time

if __name__ == "__main__":
    t0 = time.time()
    
    num_subpop1 = 40
    num_subpop2 = 120
    num_subpop3 = 20
    N = num_subpop1 + num_subpop2 + num_subpop3
    couplingconstants = [20., 20., 20., 20., 20., 20.]

    print(f'Pop. 1 number of phase oscillators: {num_subpop1}')
    print(f'Pop. 2 number of phase oscillators: {num_subpop2}')
    print(f'Pop. 3 number of phase oscillators: {num_subpop3}\n')

    kuramotosakaguchi = kurasaka_oscillators(num_subpop1, num_subpop2, num_subpop3)
    coupconsts, omegas, alphas = kuramotosakaguchi.setmodelconstants(couplingconstants)

    print(f'Coupling constants are:\n{coupconsts}\n')

    init_random = kuramotosakaguchi.setinitialconditions(clustered=False)
    times = kuramotosakaguchi.settimes(0., 10., 1000)

    phasesevolution = kuramotosakaguchi.evolve(noisy=False)
    frequencies_array, mean_frequencies = kuramotosakaguchi.findperiod_phases(phasesevolution)
    
    plt.figure()
    plt.xlabel('Integration Points')
    plt.ylabel('Frequencies')
    plt.plot(frequencies_array[:, : num_subpop1], 'bo', ms=0.3)
    plt.plot(frequencies_array[:, num_subpop1 : num_subpop1 + num_subpop2], 'gs', ms=0.3)
    plt.plot(frequencies_array[:, num_subpop1 + num_subpop2 :], 'r^', ms=0.3)
    plt.plot(mean_frequencies[:,0], 'b', lw=2, label='Mean Pop. 1 Frequency')
    plt.plot(mean_frequencies[:,1], 'g', lw=2, label='Mean Pop. 2 Frequency')
    plt.plot(mean_frequencies[:,2], 'r', lw=2, label='Mean Pop. 3 Frequency')
    plt.legend()
    plt.show()
    
    syncs, ordparams = kuramotosakaguchi.findorderparameter(phasesevolution)
    
    time_points_1 = numpy.argwhere(numpy.array(syncs[0]) > 0.8)
    print(f'Population 1 reaches |Z|>0.80 in {time_points_1.min()} integration points')
    time_points_2 = numpy.argwhere(numpy.array(syncs[1]) > 0.8)
    print(f'Population 2 reaches |Z|>0.80 in {time_points_2.min()} integration points')
    time_points_3 = numpy.argwhere(numpy.array(syncs[2]) > 0.8)
    print(f'Population 3 reaches |Z|>0.80 in {time_points_3.min()} integration points\n')
    
    globsync, globordparam = kuramotosakaguchi.findglobalorderparameter(ordparams)

    print(f'Sync for SuPop 1: {numpy.mean(syncs[0][300:])}')
    print(f'Sync for SuPop 2: {numpy.mean(syncs[1][300:])}')
    print(f'Sync for SuPop 3: {numpy.mean(syncs[2][300:])}')
    print(f'Global Sync: {numpy.mean(globsync[300:])}\n')
    
    print(f'SubPop 1 frequency: {numpy.mean(mean_frequencies[:,0][time_points_1.min():])} Calculated with phases')
    print(f'SubPop 2 frequency: {numpy.mean(mean_frequencies[:,1][time_points_2.min():])} Calculated with phases')
    print(f'SubPop 3 frequency: {numpy.mean(mean_frequencies[:,2][time_points_3.min():])} Calculated with phases')
    
    t1 = time.time()
    print(f'\nTempo di esecuzione del codice: {(t1-t0)/60} minuti!\n')