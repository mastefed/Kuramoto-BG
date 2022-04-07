def rho(phi2phi1, phi3phi1, phi3phi2, k11, k12, k13, k21, k22, k23, minmax=False):
    rho2_num = ((k13*k21)/k11)*phi3phi1*phi2phi1 - k23*phi3phi2
    rho2_den = k22 - k21/k11*phi2phi1**2

    rho2 = rho2_num/rho2_den
    rho1 = -1/k11*(rho2*k12*phi2phi1 + k13*phi3phi1)
    
    rho2 = np.ma.masked_outside(rho2, 0, 1)
    rho1 = np.ma.masked_outside(rho1, 0, 1)
    
    if minmax == True:
        rho2 = np.ma.masked_inside(rho2, .1, .9)
        rho1 = np.ma.masked_inside(rho1, .1, .9)

    return rho1, rho2
    
def drho2dc21(cp3p1, cp2p1, k11, k13, k21, k22):
    func = (k13*k21*k22*cp3p1)/(k11*(k22-(k21/k11)*cp2p1**2)**2)
    return func
    
def drho2dc31(cp2p1, k11, k13, k21, k22, k32):
    func = k13*k21*cp2p1/((k22-(k21/k11)*cp2p1**2)*k11)
    return func
    
def drho2dc32(cp2p1, k11, k21, k22, k32):
    func = -k32/(k22-(k21/k11)*cp2p1**2)
    return func
    
def drho1dc21(cp3p1, cp2p1, cp3p2, k11, k12, k13, k21, k22, k23):
    rho2_num = ((k13*k21)/k11)*cp3p1*cp2p1 - k23*cp3p2
    rho2_den = k22 - k21/k11*cp2p1**2

    rho2 = rho2_num/rho2_den
    
    func = -(1/k11)*k12*(rho2 + drho2dc21(cp3p1, cp2p1, k11, k13, k21, k22)*cp2p1)
    return func
    
def drho1dc31(cp2p1, k11, k12, k13, k21, k22, k32):
    func = -(1/k11)*(k13 + drho2dc31(cp2p1, k11, k13, k21, k22, k32)*k12*cp2p1)
    return func
    
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == "__main__":

    cp2p1_1, cp3p1 = np.mgrid[-1:1:1000j, -1:1:1000j]
    cp2p1_2, cp3p2 = np.mgrid[-1:1:1000j, -1:1:1000j]
    
    cos32 = 1
    cos31 = cos32

    fig1, axes1 = plt.subplots(num=f'rhoa{cos32}', nrows=2, ncols=2, figsize=(10,8))
    plt.suptitle(r'Synchronization parameter for population a - $\rho_a$')
    
    fig2, axes2 = plt.subplots(num=f'rhob{cos31}', nrows=2, ncols=2, figsize=(10,8))
    plt.suptitle(r'Synchronization parameter for population b - $\rho_b$')
    
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    
    for i, kval in enumerate([5, 15, 25, 35]): # np.linspace(5,40,4)
    #  k = [k11, k12, k13,       [kaa, kab, kac
    #       k21, k22, k23,        kba, kbb, kbc
    #       k31, k32, k33]        kca, kcb, kcc]
        
        k = [.5, kval, kval, 
             kval, .5, kval, 
             5, 5, .2]
        
        axes1.flat[i].set_title(fr'$\cos(\psi_c - \psi_b)$ = {cos32}, $k_{{ab}}, k_{{ac}}, k_{{ba}}, k_{{bc}}={kval}$', size=12)
        axes2.flat[i].set_title(fr'$\cos(\psi_c - \psi_a)$ = {cos31}, $k_{{ab}}, k_{{ac}}, k_{{ba}}, k_{{bc}}={kval}$', size=12)
        
        rho1 = rho(cp2p1_1, cp3p1, cos32, k[0], k[1], k[2], k[3], k[4], k[5])[0]
        deriv1 = drho1dc31(cp2p1_1, k[0], k[1], k[2], k[3], k[4], k[7])
        deriv2 = drho1dc21(cp3p1, cp2p1_1, cos32, k[0], k[1], k[2], k[3], k[4], k[5])
        
        rho2 = rho(cp2p1_2, cos31, cp3p2, k[0], k[1], k[2], k[3], k[4], k[5])[1]
        deriv3 = drho2dc32(cp2p1_2, k[0], k[3], k[4], k[7])
        deriv4 = drho2dc21(cos31, cp2p1_2, k[0], k[2], k[3], k[4])
        
        surf1 = axes1.flat[i].contourf(cp3p1, cp2p1_1, rho1)
        surf2 = axes2.flat[i].contourf(cp3p2, cp2p1_2, rho2)
        stream1 = axes1.flat[i].streamplot(cp3p1, cp2p1_1, deriv1, deriv2, density=1.2, linewidth=1, color='orange', arrowsize=1.2, arrowstyle='->')
        stream2 = axes2.flat[i].streamplot(cp3p2, cp2p1_2, deriv3, deriv4, density=1.2, linewidth=1, color='orange', arrowsize=1.2, arrowstyle='->')
        
        axes1.flat[i].set_xlabel(xlabel=r'$\cos(\psi_c - \psi_a)$', size=12)
        axes1.flat[i].set_ylabel(ylabel=r'$\cos(\psi_b - \psi_a)$', size=12)
        
        axes2.flat[i].set_xlabel(xlabel=r'$cos(\psi_c - \psi_b)$', size=12)
        axes2.flat[i].set_ylabel(ylabel=r'$cos(\psi_b - \psi_a)$', size=12)
    
    vertical_spacing = 0.4
    horizontal_spacing = 0.3
    tickslabels_for_colorbar = ['0.00', '0.15', '0.30', '0.45', '0.60', '0.75', '0.90', '1.00']
    
    fig1.subplots_adjust(right=0.8, wspace=vertical_spacing, hspace=horizontal_spacing)
    cbar_ax1 = fig1.add_axes([0.85, 0.15, 0.015, 0.7])
    cbar1 = fig1.colorbar(surf1, cax=cbar_ax1)
    cbar1.set_label(label=r'\boldmath$\rho_a$ values', size=16)
    cbar1.ax.set_yticklabels(tickslabels_for_colorbar, size=12)
    
    fig2.subplots_adjust(right=0.8, wspace=vertical_spacing, hspace=horizontal_spacing)
    cbar_ax2 = fig2.add_axes([0.85, 0.15, 0.015, 0.7])
    cbar2 = fig2.colorbar(surf2, cax=cbar_ax2)
    cbar2.set_label(label=r'\boldmath$\rho_b$ values', size=16)
    cbar2.ax.set_yticklabels(tickslabels_for_colorbar, size=12)
    
    plt.show()