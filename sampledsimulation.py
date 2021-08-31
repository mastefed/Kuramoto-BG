from kurasaka import *

import numpy as np
import itertools as it
import pandas as pd

import time
from os import path

if __name__ == "__main__":
    t0 = time.time()

    values_to_iterate = np.linspace(0., 40., 10)
    fixedval = 5.

    iterator = it.product(values_to_iterate, repeat=2)

    #list = [K12, K13, K21, K23, K31, K32]
    values_to_choose = [[fixedval, val[0], fixedval, fixedval, fixedval, val[1]] for val in iterator]

    num_subpop1 = 40
    num_subpop2 = 120
    num_subpop3 = 20

    print(f'Pop. 1 number of phase oscillators: {num_subpop1}')
    print(f'Pop. 2 number of phase oscillators: {num_subpop2}')
    print(f'Pop. 3 number of phase oscillators: {num_subpop3}\n')

    kuramotosakaguchi = kurasaka_oscillators(num_subpop1, num_subpop2, num_subpop3)

    list_of_variables = [
    'Val K12', 'Val K13', 'Val K23', 'Val K21', 'Val K31', 'Val K32', 
    'Sync Pop1', 'Sync Time Pop1', 'Sync Pop2', 'Sync Time Pop2', 'Sync Pop3', 'Sync Time Pop3', 'Global Sync', 'Presence of Sub Delta Pop1', 'Presence of Sub Delta Pop2', 'Presence of Sub Delta Pop3', 'Integral of PSD Sync Delta Band Pop1', 'Integral of PSD Sync Delta Band Pop2', 'Integral of PSD Sync Delta Band Pop3', 'Freq Pop1', 'Std Freq Pop1', 'Freq Pop2', 'Std Freq Pop2', 'Freq Pop3', 'Std Freq Pop3'
    ]
    
    dictionary_of_results = {
    'Iterations' : list_of_variables
    }

    for i, values in enumerate(values_to_choose):
        t3 = time.time()

        coupconsts, alphas = kuramotosakaguchi.setmodelconstants(values)
        print(f'Coupling constants are:\n{values}\n')

        initial_values = kuramotosakaguchi.setinitialconditions(clustered=False)
        times = kuramotosakaguchi.settimes(0., 10., 1000)

        phasesevolution = kuramotosakaguchi.evolve(initial_values, times, noisy=True)
        
        syncs, ordparams = kuramotosakaguchi.findorderparameter(times, phasesevolution)
        
        syncSpeeds = kuramotosakaguchi.retrieveSyncSpeed(syncs)
        
        psdssync, freqsmax, _ = kuramotosakaguchi.psdofsyncs(0., 10., 1000, syncs, printPlot=False)
    
        lowDeltaDetection, deltaIntegration = kuramotosakaguchi.detectDeltas(psdssync)
        
        globsync, globordparam = kuramotosakaguchi.findglobalorderparameter(times, ordparams)

        frequencies_array, mean_frequencies, std_frequencies = kuramotosakaguchi.findFrequenciesMeanStd(1000, times, phasesevolution, syncs)
        
        syncsGathered = []
        freqsGathered = []
        stdFreqsGathered =[] 

        if syncSpeeds['Population 1 Sync Speed'] == None:
            syncsGathered.append(numpy.mean(syncs[0][300:]))
            freqsGathered.append(numpy.mean(mean_frequencies[:,0][300:]))
            stdFreqsGathered.append(numpy.mean(std_frequencies[:,0][300:]))
        else:
            minimumTimePop1 = syncSpeeds['Population 1 Sync Speed']
            syncsGathered.append(numpy.mean(syncs[0][minimumTimePop1:]))
            freqsGathered.append(numpy.mean(mean_frequencies[:,0][minimumTimePop1:]))
            stdFreqsGathered.append(numpy.mean(std_frequencies[:,0][minimumTimePop1:]))
        
        if syncSpeeds['Population 2 Sync Speed'] == None:
            syncsGathered.append(numpy.mean(syncs[1][300:]))
            freqsGathered.append(numpy.mean(mean_frequencies[:,1][300:]))
            stdFreqsGathered.append(numpy.mean(std_frequencies[:,1][300:]))
        else:
            minimumTimePop2 = syncSpeeds['Population 2 Sync Speed']
            syncsGathered.append(numpy.mean(syncs[1][minimumTimePop2:]))
            freqsGathered.append(numpy.mean(mean_frequencies[:,1][minimumTimePop2:]))
            stdFreqsGathered.append(numpy.mean(std_frequencies[:,1][minimumTimePop2:]))
            
        if syncSpeeds['Population 3 Sync Speed'] == None:
            syncsGathered.append(numpy.mean(syncs[2][300:]))
            freqsGathered.append(numpy.mean(mean_frequencies[:,2][300:]))
            stdFreqsGathered.append(numpy.mean(std_frequencies[:,2][300:]))
        else:
            minimumTimePop3 = syncSpeeds['Population 3 Sync Speed']
            syncsGathered.append(numpy.mean(syncs[2][minimumTimePop3:]))
            freqsGathered.append(numpy.mean(mean_frequencies[:,2][minimumTimePop3:]))
            stdFreqsGathered.append(numpy.mean(std_frequencies[:,2][minimumTimePop3:]))
            
        syncsGathered.append(numpy.mean(globsync[300:]))

        dictionary_of_results[f'Iteration {i+1}'] = values[0], values[1], values[2], values[3], values[4], values[5], syncsGathered[0], syncSpeeds['Population 1 Sync Speed'], syncsGathered[1], syncSpeeds['Population 2 Sync Speed'], syncsGathered[2], syncSpeeds['Population 3 Sync Speed'], syncsGathered[3], lowDeltaDetection[0], lowDeltaDetection[1], lowDeltaDetection[2], deltaIntegration[0], deltaIntegration[1], deltaIntegration[2], freqsGathered[0], stdFreqsGathered[0], freqsGathered[1], stdFreqsGathered[1], freqsGathered[2], stdFreqsGathered[2]

        t4 = time.time()
        print(f'Iteration {i+1} finished in {(t4-t3)/60} minutes.\n')

    t1 = time.time()
    print(f'Code execution time: {(t1-t0)/60} minutes!')

    if path.exists('/home/f_mastellone/CsvFiles'):
        (pd.DataFrame.from_dict(data=dictionary_of_results, orient='index').to_csv('/home/f_mastellone/CsvFiles/data.csv', header=False))
    else:
        (pd.DataFrame.from_dict(data=dictionary_of_results, orient='index').to_csv('/Users/federicom/Desktop/data.csv', header=False))