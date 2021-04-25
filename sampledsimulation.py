from kurasaka import *

import numpy as np
import itertools as it
import pandas as pd

import time
from os import path

if __name__ == "__main__":
    t0 = time.time()

    values_to_iterate = np.linspace(10.,30.,5)
    val_K12 = 5.

    iterator = it.product(values_to_iterate, repeat=2)

    values_to_choose = [[val_K12, val[0], val[1]] for val in iterator]

    # In questo modo values_to_choose[i] è una lista di K da testare!

    num_subpop1 = 100
    num_subpop2 = 300
    num_subpop3 = 50

    print(f'Pop. 1 number of phase oscillators: {num_subpop1}')
    print(f'Pop. 2 number of phase oscillators: {num_subpop2}')
    print(f'Pop. 3 number of phase oscillators: {num_subpop3}\n')

    kuramotosakaguchi = kurasaka_oscillators(num_subpop1, num_subpop2, num_subpop3)

    dictionary_of_results = {}

    for i, list_of_values in enumerate(values_to_choose):
        coupconsts, omegas, alphas = kuramotosakaguchi.setmodelconstants(list_of_values)
        print(f'Coupling constants are:\n{list_of_values}\n')

        init_random = kuramotosakaguchi.setinitialconditions(clustered=False)
        times = kuramotosakaguchi.settimes(0., 10., 1000)

        equations = kuramotosakaguchi.kurasaka_function
        phasesevolution = kuramotosakaguchi.evolve(equations)
        syncs, ordparams = kuramotosakaguchi.findorderparameter(phasesevolution)
        globsync, globordparam = kuramotosakaguchi.findglobalorderparameter()

        print(f'Sync for SuPop 1: {numpy.mean(syncs[0][300:])}')
        print(f'Sync for SuPop 2: {numpy.mean(syncs[1][300:])}')
        print(f'Sync for SuPop 3: {numpy.mean(syncs[2][300:])}')
        print(f'Global Sync: {numpy.mean(globsync[300:])}\n')

        kuramotosakaguchi.ordparam_phase()
        frequencies = kuramotosakaguchi.findperiod()
        print(f'SubPop 1 frequency: {frequencies[0]} Calcolata con Re(Z)')
        print(f'SubPop 2 frequency: {frequencies[1]} Calcolata con Re(Z)')
        print(f'SubPop 3 frequency: {frequencies[2]} Calcolata con Re(Z)')

        list_of_sync = [numpy.mean(syncs[0][300:]), numpy.mean(syncs[1][300:]), numpy.mean(syncs[2][300:])]
        list_of_freq = [frequencies[0], frequencies[1], frequencies[2]]

        dictionary_of_results[f'Iteration {i+1} --> Val K12, Val K13, Val K23'] = list_of_values
        dictionary_of_results[f'Iteration {i+1} --> Sync Pop1, Sync Pop2, Sync Pop3, Global Sync'] = list_of_sync
        dictionary_of_results[f'Iteration {i+1} --> Freq Pop1, Freq Pop2, Freq Pop3'] = list_of_freq

        print(f'Iteration {i+1} finished.\n')

    t1 = time.time()
    print(f'Code execution time: {(t1-t0)/60} minutes!')

    if path.exists('/home/f_mastellone/CsvFiles'):
        (pd.DataFrame.from_dict(data=dictionary_of_results, orient='index').to_csv('/home/f_mastellone/CsvFiles/data.csv', header=False))
    else:
        (pd.DataFrame.from_dict(data=dictionary_of_results, orient='index').to_csv('/Users/federicom/Desktop/25iterations.csv', header=False))



