from kurasaka import *

import numpy as np
import itertools as it
import pandas as pd

import time
from os import path

if __name__ == "__main__":
    t0 = time.time()

    values_to_iterate = np.linspace(30., 90., 10)
    fixedval = 5.

    iterator = it.product(values_to_iterate, repeat=2)

    # list = [K12, K13, K23, K21, K31, K32]
    values_to_choose = [[val[0], fixedval, fixedval, val[1], fixedval, fixedval] for val in iterator]

    # In questo modo values_to_choose[i] è una lista di K da testare!

    num_subpop1 = 40
    num_subpop2 = 120
    num_subpop3 = 20

    print(f'Pop. 1 number of phase oscillators: {num_subpop1}')
    print(f'Pop. 2 number of phase oscillators: {num_subpop2}')
    print(f'Pop. 3 number of phase oscillators: {num_subpop3}\n')

    kuramotosakaguchi = kurasaka_oscillators(num_subpop1, num_subpop2, num_subpop3)

    list_of_variables = [
    'Val K12', 'Val K13', 'Val K23', 'Val K21', 'Val K31', 'Val K32', 
    'Sync Pop1', 'Sync Time Pop1', 'Sync Pop2', 'Sync Time Pop2', 'Sync Pop3', 'Sync Time Pop3', 'Global Sync', 
    'Freq Pop1', 'Std Freq Pop1', 'Freq Pop2', 'Std Freq Pop2', 'Freq Pop3', 'Std Freq Pop3'
    ]
    
    dictionary_of_results = {
    'Iterations' : list_of_variables
    }

    for i, values in enumerate(values_to_choose):
        t3 = time.time()

        coupconsts, omegas, alphas = kuramotosakaguchi.setmodelconstants(values)
        print(f'Coupling constants are:\n{values}\n')

        init_random = kuramotosakaguchi.setinitialconditions(clustered=False)
        times = kuramotosakaguchi.settimes(0., 10., 1000)

        phasesevolution = kuramotosakaguchi.evolve(noisy=True)
        
        syncs, ordparams = kuramotosakaguchi.findorderparameter(phasesevolution)
        
        time_points_1 = numpy.argwhere(numpy.array(syncs[0]) > 0.8)
        if time_points_1.size == 0:
            minimum_time_pop1 = 0
            print('Pop. 1 never reaches |Z|>0.8')
        else:
            minimum_time_pop1 = time_points_1.min()
            print(f'Population 1 reaches |Z|>0.80 in {minimum_time_pop1} integration points')
            
        time_points_2 = numpy.argwhere(numpy.array(syncs[1]) > 0.8)
        if time_points_2.size == 0:
            minimum_time_pop2 = 0
            print('Pop. 2 never reaches |Z|>0.8')
        else:
            minimum_time_pop2 = time_points_2.min()
            print(f'Population 2 reaches |Z|>0.80 in {minimum_time_pop2} integration points')
            
        time_points_3 = numpy.argwhere(numpy.array(syncs[2]) > 0.8)
        if time_points_3.size == 0:
            minimum_time_pop3 = 0
            print('Pop. 3 never reaches |Z|>0.8\n')
        else:
            minimum_time_pop3 = time_points_3.min()
            print(f'Population 3 reaches |Z|>0.80 in {minimum_time_pop3} integration points\n')
        
        globsync, globordparam = kuramotosakaguchi.findglobalorderparameter(ordparams)

        print(f'Mean Sync for SuPop 1: {numpy.mean(syncs[0][300:])}')
        print(f'Mean Sync for SuPop 2: {numpy.mean(syncs[1][300:])}')
        print(f'Mean Sync for SuPop 3: {numpy.mean(syncs[2][300:])}')
        print(f'Mean Global Sync: {numpy.mean(globsync[300:])}\n')

        frequencies_array, mean_frequencies, std_frequencies = kuramotosakaguchi.findperiod_phases(phasesevolution)
        print(f'SubPop 1 mean frequency: {numpy.mean(mean_frequencies[:,0][300:])} +- {numpy.mean(std_frequencies[:,0][300:])} Hz')
        print(f'SubPop 2 mean frequency: {numpy.mean(mean_frequencies[:,1][300:])} +- {numpy.mean(std_frequencies[:,1][300:])} Hz')
        print(f'SubPop 3 mean frequency: {numpy.mean(mean_frequencies[:,2][300:])} +- {numpy.mean(std_frequencies[:,2][300:])} Hz')

        syncs = [numpy.mean(syncs[0][minimum_time_pop1:]), numpy.mean(syncs[1][minimum_time_pop2:]), numpy.mean(syncs[2][minimum_time_pop3:]), numpy.mean(globsync[300:])]
        freqs = [numpy.mean(mean_frequencies[:,0][minimum_time_pop1:]), numpy.mean(mean_frequencies[:,1][minimum_time_pop2:]), numpy.mean(mean_frequencies[:,2][minimum_time_pop1:])]
        stdfreqs = [numpy.mean(std_frequencies[:,0][minimum_time_pop1:]), numpy.mean(std_frequencies[:,1][minimum_time_pop2:]), numpy.mean(std_frequencies[:,2][minimum_time_pop1:])]

        dictionary_of_results[f'Iteration {i+1}'] = values[0], values[1], values[2], values[3], values[4], values[5], syncs[0], minimum_time_pop1, syncs[1], minimum_time_pop2, syncs[2], minimum_time_pop3, syncs[3], freqs[0], stdfreqs[0], freqs[1], stdfreqs[1], freqs[2], stdfreqs[2]

        t4 = time.time()
        print(f'Iteration {i+1} finished in {(t4-t3)/60} minutes.\n')

    t1 = time.time()
    print(f'Code execution time: {(t1-t0)/60} minutes!')

    if path.exists('/home/f_mastellone/CsvFiles'):
        (pd.DataFrame.from_dict(data=dictionary_of_results, orient='index').to_csv('/home/f_mastellone/CsvFiles/data.csv', header=False))
    else:
        (pd.DataFrame.from_dict(data=dictionary_of_results, orient='index').to_csv('/Users/federicom/Desktop/data.csv', header=False))