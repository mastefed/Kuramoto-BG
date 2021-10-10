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

if __name__ == "__main__":

    cp2p1_1, cp3p1 = np.mgrid[-1:1:1000j, -1:1:1000j]
    cp2p1_2, cp3p2 = np.mgrid[-1:1:1000j, -1:1:1000j]
    
    cos32 = 0
    cos31 = cos32

    fig1 = plt.figure(f'rhoa{cos32}', figsize=(14,7))
    plt.suptitle(r'$\rho_a$')
    fig2 = plt.figure(f'rhob{cos31}', figsize=(14,7))
    plt.suptitle(r'$\rho_b$')
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('font', family='serif')
    
    for i, kval in enumerate(np.linspace(5,40,6)):
    #  k = [k11, k12, k13,       [kaa, kab, kac
    #          k21, k22, k23,     kba, kbb, kbc
    #          k31, k32, k33]     kca, kcb, kcc]
        k = [
            .5, 5, 5, 
            kval, .5, kval, 
            5, 5, .2
            ]
        
        
        ax1 = fig1.add_subplot(2,3,i+1)
        ax2 = fig2.add_subplot(2,3,i+1)
        ax1.set_title(fr'$\cos(\phi_c - \phi_b)$ = {cos32}, $k_{{bc}}, k_{{ba}}$={kval}')
        ax2.set_title(fr'$\cos(\phi_c - \phi_a)$ = {cos31}, $k_{{bc}}, k_{{ba}}$={kval}')
        
        rho1 = rho(cp2p1_1, cp3p1, cos32, k[0], k[1], k[2], k[3], k[4], k[5])[0]
        deriv1 = drho1dc31(cp2p1_1, k[0], k[1], k[2], k[3], k[4], k[7])
        deriv2 = drho1dc21(cp3p1, cp2p1_1, cos32, k[0], k[1], k[2], k[3], k[4], k[5])
        
        rho2 = rho(cp2p1_2, cos31, cp3p2, k[0], k[1], k[2], k[3], k[4], k[5])[1]
        deriv3 = drho2dc32(cp2p1_2, k[0], k[3], k[4], k[7])
        deriv4 = drho2dc21(cos31, cp2p1_2, k[0], k[2], k[3], k[4])
        
        surf1 = ax1.contourf(cp3p1, cp2p1_1, rho1)
        surf2 = ax2.contourf(cp3p2, cp2p1_2, rho2)
        stream1 = ax1.streamplot(cp3p1, cp2p1_1, deriv1, deriv2, density=1.5, linewidth=.3, color='orange', arrowsize=1.2, arrowstyle='->')
        stream2 = ax2.streamplot(cp3p2, cp2p1_2, deriv3, deriv4, density = 1.5, color='orange', linewidth=.3, arrowsize=1.2, arrowstyle='->')
        
        fig1.colorbar(surf1)
        fig2.colorbar(surf2)
        
        ax1.set_xlabel(r'$\cos(\phi_c - \phi_a)$')
        ax1.set_ylabel(r'$\cos(\phi_b - \phi_a)$')
        
        ax2.set_xlabel(r'$cos(\phi_c - \phi_b)$')
        ax2.set_ylabel(r'$cos(\phi_b - \phi_a)$')
        
    
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()