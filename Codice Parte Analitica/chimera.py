def rho(cos_ca, cos_ba, k_ac, k_ab):
    func1 = -(k_ac*cos_ca)/(k_ab*cos_ba)
    func2 = np.ma.masked_outside(func1, 0, 1)
    return func2

def drhodcosca(cos_ba, k_ac, k_ab):
    secondderivative = -(k_ac)/(k_ab*cos_ba)
    return secondderivative

def drhodcosba(cos_ca, cos_ba, k_ac, k_ab):
    firstderivative = (k_ac*cos_ca)/(k_ab*cos_ba**2)
    return firstderivative

import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    k_ab = 5
    k_ac = 15
    cos_ca, cos_ba = np.mgrid[-1:1:1000j, -1:1:1000j]
    
    fig = plt.figure('Prova')
    ax = fig.add_subplot(1,1,1)
    
    cont = ax.contourf(cos_ba, cos_ca, rho(cos_ca, cos_ba, k_ac, k_ab))
    ax.streamplot(cos_ba, cos_ca, drhodcosba(cos_ca, cos_ba, k_ac, k_ab), drhodcosca(cos_ba, k_ac, k_ab), density = 1.5, color='orange', linewidth=.3, arrowsize=1.2, arrowstyle='->')
    
    fig.colorbar(cont)
    
    plt.show()