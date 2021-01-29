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
        return self.times

    def setinitialconditions(self):
        for i in range(self.N):
            self.initialvalues[i] = 2*numpy.pi*random.random()
        return self.initialvalues

    def setmodelconstants(self, random_k=True):    
        if random_k == True:
            self.k = random.random()
        elif random_k == False:
            self.k = float(input('Choose the coupling constant: '))

        for i in range(self.N):
            self.omega[i] = random.gammavariate(9., .5)
        
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
            self.sync.append( numpy.sqrt(self.orderparameter[i].real**2 + self.orderparameter[i].imag**2) )
        
        return self.sync, self.orderparameter

    def printsyncparam(self):
        plt.figure(f'{self.N} Oscillators Sync')
        plt.title(f'Sync param for {self.N} oscillators; k={self.k}')
        plt.plot(self.times, self.sync)
        plt.xlabel('Time Steps')
        plt.ylabel('R')
        plt.ylim([0.,1.])
        plt.yticks(numpy.arange(0, 1.1, step=0.1))

    def printpolar(self):
        plt.figure(f'{self.N} Oscillators')
        plt.suptitle(f'Sync param for {self.N} oscillators; k={self.k}')
        for i in range(self.N):
            plt.polar(self.times, self.phaseoscillators_evo[:,i], label=f'Theta {i}')

    def animate_function(self, i):
        self.phases = self.phaseoscillators_evo[i:i+1]
        self.timestep = self.times[0:i]
        self.R = self.sync[0:i]
        
        self.imphasedict = {}
        self.rephasedict = {}

        for k in range(self.N):
            self.imphasedict[f'im_x{k}'] = numpy.exp(complex(0, self.phases[0][k])).imag
            self.rephasedict[f're_x{k}'] = numpy.exp(complex(0, self.phases[0][k])).real

        self.imagpart_ordparam = self.orderparameter[i].imag
        self.realpart_ordparam = self.orderparameter[i].real

        ticks = [-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8]

        self.ax1.clear()
        self.circ = plt.Circle((0, 0), radius=1, lw=0.3, edgecolor='k', facecolor='None')
        self.ax1.add_patch(self.circ)
        self.ax1.set_xlim(-1.2, 1.2)
        self.ax1.set_ylim(-1.2, 1.2)
        self.ax1.spines['left'].set_position('center')
        self.ax1.spines['right'].set_color('none')
        self.ax1.spines['bottom'].set_position('center')
        self.ax1.spines['top'].set_color('none')
        self.ax1.yaxis.set_ticks(ticks)
        self.ax1.xaxis.set_ticks(ticks)
        self.ax1.set_xlabel('Re', loc='right')
        self.ax1.set_ylabel('Im', loc='top')
        
        self.ax1.arrow(0., 0., self.realpart_ordparam, self.imagpart_ordparam, head_width=0.02, head_length=0.05, fc='b', ec='b', lw=1., label='Order Parameter')
        for k in range(self.N):
            self.ax1.plot(self.rephasedict[f're_x{k}'], self.imphasedict[f'im_x{k}'], 'o', ms=7.)
        self.ax1.legend()

        self.ax2.clear()
        self.ax2.set_ylim([0.,1.])
        self.ax2.set_xlim([self.time_start, self.time_end])
        self.ax2.set_xlabel('Time Steps')
        self.ax2.set_ylabel('R')

        self.ax2.plot(self.timestep, self.R, label='Sync. Parameter')
        self.ax2.legend()

    def animateoscillators(self):
        self.fig = plt.figure(f'{self.N} Oscillators Animated', figsize=(13,6))
        plt.suptitle(f'{self.N} Oscillators; k={self.k}')
        self.ax1 = plt.subplot(121)
        self.ax2 = plt.subplot(122)
        self.animated = animation.FuncAnimation(self.fig, self.animate_function, frames = len(self.phaseoscillators_evo), interval=0.1)

        return self.animated
  
    def showplots(self):
        plt.show()

    def saveanimation(self, myanimation, save_path):
        print('Video Processing...')
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='F. V. Mastellone'), bitrate=1800)
        myanimation.save(save_path, writer=writer)
        print('Such done, very wow!')

import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

from scipy.integrate import odeint

random.seed(42)

number_of_oscillators = int(input('How many oscillators do you want to simulate?: '))

thetas = phaseoscillators(number_of_oscillators)
times = thetas.settimes(0., 10., 1000)
init_val = thetas.setinitialconditions()
coupconstant, natfreq = thetas.setmodelconstants(random_k=False)

equations = thetas.phaseoscillators_fun
phasesevolution = thetas.evolve(equations)
sync, ordparam = thetas.findorderparameter(phasesevolution)

animazione = thetas.animateoscillators()
save_path = f'C:/Users/feder/Desktop/{number_of_oscillators}phaseoscillatorsk{coupconstant}.mp4'
thetas.saveanimation(animazione, save_path)