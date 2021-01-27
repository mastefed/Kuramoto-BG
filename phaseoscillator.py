import numpy
import matplotlib.pyplot as plt
import random

from scipy.integrate import odeint

random.seed(42)

N = 3
initialvalues = numpy.zeros(N)
time_start = 0
time_end = 10
omega = numpy.zeros(N)

for i in numpy.arange(N):
    initialvalues[i] = 2*numpy.pi*random.random()
    omega[i] = 5.*random.random()

def phaseoscillators_fun(x,t):
    theta0 = x[0]
    theta1 = x[1]
    theta2 = x[2]

    dtheta0dt = omega[0] + k/N*numpy.sin(theta1 - theta0) + k/N*numpy.sin(theta2 - theta0)
    dtheta1dt = omega[1] + k/N*numpy.sin(theta0 - theta1) + k/N*numpy.sin(theta2 - theta1)
    dtheta2dt = omega[2] + k/N*numpy.sin(theta1 - theta2) + k/N*numpy.sin(theta1 - theta2)

    return [dtheta0dt, dtheta1dt, dtheta2dt]

time = numpy.linspace(time_start, time_end, 1000)

kval = [0., 0.4, 0.9]

for i, k in enumerate(kval):
    phaseoscillators_evo = odeint(phaseoscillators_fun, initialvalues, time)
    plt.figure(i)
    plt.polar(time, phaseoscillators_evo[:,0], label=f'theta0, k={k}')
    plt.polar(time, phaseoscillators_evo[:,1], label=f'theta1, k={k}')
    plt.polar(time, phaseoscillators_evo[:,2], label=f'theta2, k={k}')
    plt.legend()

plt.show()