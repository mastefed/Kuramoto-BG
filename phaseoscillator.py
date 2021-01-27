import numpy
import matplotlib.pyplot as plt
import random

from scipy.integrate import odeint

random.seed(42)

class phaseoscillators:
    def __init__(self, numberofoscillators):
        self.N = numberofoscillators
        self.initialvalues = numpy.zeros(self.N)
        self.omega = numpy.zeros(self.N)

    def printinfos(self):
        print(f'Number of Phase Oscillators: {self.N}\n')
        print(f'Start Time set at {self.time_start}\nEnd Time set at {self.time_end}')
        print(f'Initial conditions are: {self.initialvalues}\nNatural frequencies are: {self.omega}\nk: {self.k}')

    def settimes(self, time_start, time_end, time_points):
        self.time_start = time_start
        self.time_end = time_end
        self.times = numpy.linspace(self.time_start, self.time_end, time_points)

    def setinitialconditions(self):
        for i in numpy.arange(self.N):
            self.initialvalues[i] = 2*numpy.pi*random.random()
            self.omega[i] = 5.*random.random()
            self.k = random.random()

    def phaseoscillators_fun(self, x, t):
        theta0 = x[0]
        theta1 = x[1]
        theta2 = x[2]

        self.dtheta0dt = self.omega[0] + self.k/self.N*numpy.sin(theta1 - theta0) + self.k/self.N*numpy.sin(theta2 - theta0)
        self.dtheta1dt = self.omega[1] + self.k/self.N*numpy.sin(theta0 - theta1) + self.k/self.N*numpy.sin(theta2 - theta1)
        self.dtheta2dt = self.omega[2] + self.k/self.N*numpy.sin(theta1 - theta2) + self.k/self.N*numpy.sin(theta1 - theta2)

        return [self.dtheta0dt, self.dtheta1dt, self.dtheta2dt]

    def evolve(self, function):
        self.phaseoscillators_evo = odeint(function, self.initialvalues, self.times)

    def printpolar(self):
        plt.polar(self.times, self.phaseoscillators_evo[:,0], label=f'theta0, k={self.k}')
        plt.polar(self.times, self.phaseoscillators_evo[:,1], label=f'theta1, k={self.k}')
        plt.polar(self.times, self.phaseoscillators_evo[:,2], label=f'theta2, k={self.k}')
        plt.legend()
        plt.show()


threeoscillators = phaseoscillators(3)
threeoscillators.settimes(0., 10., 1000)
threeoscillators.setinitialconditions()
threeoscillators.printinfos()

threeoscillators.evolve(threeoscillators.phaseoscillators_fun)
threeoscillators.printpolar()