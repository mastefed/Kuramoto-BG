# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:33:47 2021

@author: Federico Vincenzo Mastellone
"""

from kurasaka import *
import time

if __name__ == "__main__":
    t0 = time.time()
    
    num_subpop1 = 100
    num_subpop2 = 300
    num_subpop3 = 50
    N1 = num_subpop1 + num_subpop2 + num_subpop3
    couplingconstants = [25., 25., 25.]

    print(f'Pop. 1 number of phase oscillators: {num_subpop1}')
    print(f'Pop. 2 number of phase oscillators: {num_subpop2}')
    print(f'Pop. 3 number of phase oscillators: {num_subpop3}\n')

    kuramotosakaguchi1 = kurasaka_oscillators(num_subpop1, num_subpop2, num_subpop3)
    coupconsts, omegas, alphas = kuramotosakaguchi1.setmodelconstants(couplingconstants)

    print(f'Coupling constants are:\n{coupconsts}\n')

    init_random = kuramotosakaguchi1.setinitialconditions(clustered=False)
    times1 = kuramotosakaguchi1.settimes(0., 10., 1000)

    equations = kuramotosakaguchi1.kurasaka_function
    phasesevolution = kuramotosakaguchi1.evolve(equations)
    syncs1, ordparams1 = kuramotosakaguchi1.findorderparameter(phasesevolution)
    globsync1, globordparam1 = kuramotosakaguchi1.findglobalorderparameter()

    print(f'Sync for SuPop 1: {numpy.mean(syncs1[0][300:])}')
    print(f'Sync for SuPop 2: {numpy.mean(syncs1[1][300:])}')
    print(f'Sync for SuPop 3: {numpy.mean(syncs1[2][300:])}')
    print(f'Global Sync: {numpy.mean(globsync1[300:])}\n')
    
    kuramotosakaguchi1.ordparam_phase()
    frequencies = kuramotosakaguchi1.findperiod()
    print(f'SubPop 1 frequency: {frequencies[0]} Calcolata con Re(Z)')
    print(f'SubPop 2 frequency: {frequencies[1]} Calcolata con Re(Z)')
    print(f'SubPop 3 frequency: {frequencies[2]} Calcolata con Re(Z)')
    
    t1 = time.time()
    print(f'\nTempo di esecuzione del codice: {(t1-t0)/60} minuti!\n')

    t2 = time.time()
    
    num_subpop1 = 40
    num_subpop2 = 120
    num_subpop3 = 20
    N2 = num_subpop1 + num_subpop2 + num_subpop3

    print(f'Pop. 1 number of phase oscillators: {num_subpop1}')
    print(f'Pop. 2 number of phase oscillators: {num_subpop2}')
    print(f'Pop. 3 number of phase oscillators: {num_subpop3}\n')

    kuramotosakaguchi2 = kurasaka_oscillators(num_subpop1, num_subpop2, num_subpop3)
    coupconsts, omegas, alphas = kuramotosakaguchi2.setmodelconstants(couplingconstants)

    print(f'Coupling constants are:\n{coupconsts}\n')

    init_random = kuramotosakaguchi2.setinitialconditions(clustered=False)
    times2 = kuramotosakaguchi2.settimes(0., 10., 1000)

    equations = kuramotosakaguchi2.kurasaka_function
    phasesevolution = kuramotosakaguchi2.evolve(equations)
    syncs2, ordparams2 = kuramotosakaguchi2.findorderparameter(phasesevolution)
    globsync2, globordparam2 = kuramotosakaguchi2.findglobalorderparameter()

    print(f'Sync for SuPop 1: {numpy.mean(syncs2[0][300:])}')
    print(f'Sync for SuPop 2: {numpy.mean(syncs2[1][300:])}')
    print(f'Sync for SuPop 3: {numpy.mean(syncs2[2][300:])}')
    print(f'Global Sync: {numpy.mean(globsync2[300:])}\n')
    
    kuramotosakaguchi2.ordparam_phase()
    frequencies = kuramotosakaguchi2.findperiod()
    print(f'SubPop 1 frequency: {frequencies[0]} Calcolata con Re(Z)')
    print(f'SubPop 2 frequency: {frequencies[1]} Calcolata con Re(Z)')
    print(f'SubPop 3 frequency: {frequencies[2]} Calcolata con Re(Z)')
    
    t3 = time.time()
    print(f'\nTempo di esecuzione del codice: {(t3-t2)/60} minuti!')

    plt.figure(f'Oscillators Sync', figsize=(13,6))
    plt.title(f'Oscillators Sync; r --> less oscillators, b --> more oscillators')
    plt.plot(times1, syncs1[0],'b' ,label='SubPop 1')
    plt.plot(times1, syncs1[1],'b' ,label='SubPop 2')
    plt.plot(times1, syncs1[2],'b' ,label='SubPop 3')
    plt.plot(times1, globsync1,'b' ,label='Global')
    
    plt.plot(times2, syncs2[0],'r' ,label='SubPop 1')
    plt.plot(times2, syncs2[1],'r' ,label='SubPop 2')
    plt.plot(times2, syncs2[2],'r' ,label='SubPop 3')
    plt.plot(times2, globsync2,'r' ,label='Global')

    plt.xlabel('Time Steps')
    plt.ylabel('R')
    plt.ylim([0.,1.])
    plt.yticks(numpy.arange(0, 1.1, step=0.1))
    plt.legend(loc='lower left')

    plt.savefig('/home/f_mastellone/Images/confronto.png')
    #plt.savefig('/Users/federicom/Desktop/confronto.png')

