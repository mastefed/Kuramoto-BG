# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 08:33:47 2021

@author: Federico Vincenzo Mastellone
"""

from kurasaka import *

if __name__ == "__main__":
    num_subpop1 = 10
    num_subpop2 = 3
    num_subpop3 = 10

    print(f'Pop. 1 number of phase oscillators: {num_subpop1}')
    print(f'Pop. 2 number of phase oscillators: {num_subpop2}')
    print(f'Pop. 3 number of phase oscillators: {num_subpop3}\n')

    kuramotosakaguchi = kurasaka_oscillators(num_subpop1, num_subpop2, num_subpop3)
    coupconsts, omegas, alphas = kuramotosakaguchi.setmodelconstants(fixed=True)

    print(f'Coupling constants are:\n{coupconsts}\n')
    print(f'Phase delay constants are:\n{alphas}\n')

    init_random = kuramotosakaguchi.setinitialconditions(clustered=False)
    times = kuramotosakaguchi.settimes(0., 10., 5000)

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

    kuramotosakaguchi.printcosineordparam(save=False, savepath='/home/f_mastellone/Images/OrderParameterOscillations.png')
    kuramotosakaguchi.printsyncparam(save=False, savepath='/home/f_mastellone/Images/SyncParameters.png')
    kuramotosakaguchi.psdofordparam(save=False, savepath='/home/f_mastellone/Images/PSD.png')
    #myanim = kuramotosakaguchi.animateoscillators()
    #kuramotosakaguchi.saveanimation(myanim, save_path='/home/f_mastellone/Images/videosimulazioneprova.mp4')