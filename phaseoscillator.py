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
        print(f'Number of Phase Oscillators: {self.N}')
        print(f'Start Time set at {self.time_start}\nEnd Time set at {self.time_end}')
        print(f'Initial conditions are: {self.initialvalues}\nNatural frequencies are: {self.omega}\nCoupling constant is: {self.k}\n')

    def settimes(self, time_start, time_end, time_points):
        self.time_start = time_start
        self.time_end = time_end
        self.times = numpy.linspace(self.time_start, self.time_end, time_points)

    def setinitialconditions(self, random_k=True):
        for i in range(self.N):
            self.initialvalues[i] = 2*numpy.pi*random.random()
        return self.initialvalues

    def setmodelconstants(self, random_k=True):    
        if random_k == True:
            self.k = random.random()
        elif random_k == False:
            self.k = float(input('Choose the coupling constant (0 to 1): '))

        r = float(input('Choose the maximum numbers of radians per second: '))
        for i in range(self.N):
            self.omega[i] = r*random.random()
        
        return self.k, self.omega

    def phaseoscillators_fun(self, x, t):
        self.variables = {}
        for i in range(self.N):
            self.variables[f'theta{i}'] = x[i]

        self.dthetadt = []
        for i in range(self.N):
            self.dthetadt.append(self.omega[i] + self.k/self.N*sum(numpy.sin(self.variables[f'theta{j}'] - self.variables[f'theta{i}']) for j in range(self.N)))

        return self.dthetadt

    def evolve(self, function):
        self.phaseoscillators_evo = odeint(function, self.initialvalues, self.times)
        return self.phaseoscillators_evo

    def findorderparameter(self, phases):
        self.orderparameter = []
        for i in range(len(self.times)):
            self.orderparameter.append( 1/self.N * sum(numpy.exp(complex(0,phases[i,j]))  for j in range(self.N)) )

        self.sync = []
        for i in range(len(self.orderparameter)):
            self.sync.append( numpy.sqrt(numpy.real(self.orderparameter[i])**2 + numpy.imag(self.orderparameter[i])**2) )
        
        return self.sync

    def printsyncparam(self):
        plt.figure(f'{self.N} Oscillators Sync')
        plt.title(f'Sync param for {self.N} oscillators; k={self.k}')
        plt.plot(self.times, self.sync)
        plt.xlabel('time')
        plt.ylabel('R')
        plt.ylim([0.,1.])
        plt.yticks(numpy.arange(0, 1.1, step=0.1))

    def printpolar(self):
        plt.figure(f'{self.N} Oscillators')
        for i in range(self.N):
            plt.polar(self.times, self.phaseoscillators_evo[:,i], label=f'Theta {i}')
        plt.legend()
    
    def showplots(self):
        plt.show()


thetas = phaseoscillators(15)
thetas.settimes(0., 10., 1000)
init_val = thetas.setinitialconditions()
coupconstant, natfreq = thetas.setmodelconstants(random_k=True)
thetas.printinfos()

equations = thetas.phaseoscillators_fun
phasesevolution = thetas.evolve(equations)
sync = thetas.findorderparameter(phasesevolution)

thetas.printpolar()
thetas.printsyncparam()
thetas.showplots()