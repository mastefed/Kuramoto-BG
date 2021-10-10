def rho(phi2phi1, phi3phi1, phi3phi2, minmax=False):
    rho2_num = ((k13*k21)/k11)*phi3phi1*phi2phi1 - k23*phi3phi2
    rho2_den = k22 - k21/k11*phi2phi1**2

    rho2 = rho2_num/rho2_den
    rho1 = -1/k11*(rho2*k12*phi2phi1 + k13*phi3phi1)
    
    rho2 = numpy.ma.masked_outside(rho2, 0, 1)
    rho1 = numpy.ma.masked_outside(rho1, 0, 1)
    
    if minmax == True:
        rho2 = numpy.ma.masked_inside(rho2, .1, .9)
        rho1 = numpy.ma.masked_inside(rho1, .1, .9)

    return rho1, rho2
    
def rhomaxmin(rho):
    rho_outside = numpy.ma.masked_inside(rho, .1, .9)
    return rho_outside
    
def drho2dc21(cp3p1, cp2p1):
    func = (k13*k21*k22*cp3p1)/(k11*(k22-(k21/k11)*cp2p1**2)**2)
    return func
    
def drho2dc31(cp2p1):
    func = k13*k21*cp2p1/((k22-(k21/k11)*cp2p1**2)*k11)
    return func
    
def drho2dc32(cp2p1):
    func = -k32/(k22-(k21/k11)*cp2p1**2)
    return func
    
def drho1dc21(cp3p1, cp2p1, cp3p2):
    rho2_num = ((k13*k21)/k11)*cp3p1*cp2p1 - k23*cp3p2
    rho2_den = k22 - k21/k11*cp2p1**2

    rho2 = rho2_num/rho2_den
    
    func = -(1/k11)*k12*(rho2 + drho2dc21(cp3p1, cp2p1)*cp2p1)
    return func
    
def drho1dc31(cp2p1):
    func = -(1/k11)*(k13 + drho2dc31(cp2p1)*k12*cp2p1)
    return func


import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


if __name__ == "__main__":
    k = 5
    
    k11 = .5
    k12 = 5
    
    k13 = k
    
    k21 = k
    
    k22 = .5
    
    k23 = k
    
    k31 = 5
    k32 = 5
    k33 = .2
    
    matrix = r'$K = \begin{bmatrix} %s & %d & %d \\ %d & %s & %d \\ %d & %d & %s  \end{bmatrix}$' % (k11, k12, k13, k21, k22, k23, k31, k32, k33)
    
    span = numpy.linspace(-1, 1, 5)
    
    cp2p1, cp3p2 = numpy.mgrid[-1:1:1000j, -1:1:1000j]
    
    container2 = []
    
    for constant in span:
        rho_1_2 = rho(cp2p1, constant, cp3p2)
        container2.append(rho_1_2[1])
        

    fig = plt.figure('rho2', figsize=(14,7))
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('font', family='serif')
    plt.suptitle(r'$\rho_2$')
    for i in range(5):
        ax2 = fig.add_subplot(2, 3, i+1)
        ax2.set_title(rf'$\cos(\phi_3 - \phi_1)$ = {round(span[i], 2)}')
        surf = ax2.contourf(cp3p2, cp2p1, container2[i])
        ax2.streamplot(cp3p2, cp2p1, drho2dc32(cp2p1), drho2dc21(span[i], cp2p1), density = 1.5, color='orange', linewidth=.3, arrowsize=1.2, arrowstyle='->')
        fig.colorbar(surf)
        ax2.set_xlabel(r'$cos(\phi_3 - \phi_2)$')
        ax2.set_ylabel(r'$cos(\phi_2 - \phi_1)$')
    
    ax2 = fig.add_subplot(2,3,6)
    ax2.axis('off')
    ax2.text(0.1, 0.5, matrix, fontsize=18)    
        
    plt.tight_layout()
    
    cp2p1, cp3p1 = numpy.mgrid[-1:1:1000j, -1:1:1000j]
    
    container1 = []
    
    for constant in span:
        rho_1_2 = rho(cp2p1, cp3p1, constant)
        container1.append(rho_1_2[0])
        
        
    fig = plt.figure('rho1', figsize=(14,7))
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('font', family='serif')
    plt.suptitle(r'$\rho_1$')
    for j in range(5):
        ax1 = fig.add_subplot(2, 3, j+1)
        ax1.set_title(fr'$\cos(\phi_3 - \phi_2)$ = {round(span[j], 2)}')
        surf = ax1.contourf(cp3p1, cp2p1, container1[j])
        ax1.streamplot(cp3p1, cp2p1, drho1dc31(cp2p1), drho1dc21(cp3p1, cp2p1, span[j]), density=1.5, linewidth=.3, color='orange', arrowsize=1.2, arrowstyle='->')
        fig.colorbar(surf)
        ax1.set_xlabel(r'$\cos(\phi_3 - \phi_1)$')
        ax1.set_ylabel(r'$\cos(\phi_2 - \phi_1)$')
        
    ax1 = fig.add_subplot(2,3,6)
    ax1.axis('off')
    ax1.text(0.1, 0.5, matrix, fontsize=18)    
        
    plt.tight_layout()
    plt.show()