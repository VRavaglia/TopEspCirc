#r# =====================
#r#  n-MOSFET Transistor
#r# =====================

#r# This example shows how to simulate the characteristic curves of an nmos transistor.

####################################################################################################

import matplotlib.pyplot as plt

####################################################################################################

import PySpice

import PySpice.Logging.Logging as Logging

logger = Logging.setup_logging()

####################################################################################################

from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

#from PySpice.Spice.NgSpice import Server

#Server.SpiceServer.SPICE_COMMAND =\
#                    'C:/Users/fedua/Documents/Spice64/bin/ngspice.exe'

####################################################################################################

#libraries_path = find_libraries()
#spice_library = SpiceLibrary(libraries_path)
#spice_library = SpiceLibrary('./')

####################################################################################################

#r# We define a basic circuit to drive an nmos transistor using two voltage sources.
#r# The nmos transistor demonstrated in this example is a low-level device description.

#?# TODO: Write the : circuit_macros('nmos_transistor.m4')

circuit = Circuit('NMOS Transistor')
#circuit.include(spice_library['MODN'])

#circuit.include('./Libs/CMOS35.lib')
################################################################
# ATENCAO! O NGSPICE soh aceita ate o level 6
# do modelo do MOS. Se o level no .lib for 7,
# troque para 6.
# ATENCAO! O path nao pode ter o caractere
# de espaco.
################################################################
circuit.include('C:/Users/fedua/Desktop/Libs/CMOS35.lib')


# Define the DC supply voltage value
Vdd = 3

# Instanciate circuit elements
Vgate = circuit.V('gate', 'gatenode', circuit.gnd, 0@u_V)
Vdrain = circuit.V('drain', 'vdd', circuit.gnd, u_V(Vdd))
# M <name> <drain node> <gate node> <source node> <bulk/substrate node>
circuit.MOSFET(1, 'vdd', 'gatenode', circuit.gnd, circuit.gnd, model='MODN', w=1e-6,l=1e-6)

#r# We plot the characteristics :math:`Id = f(Vgs)` using a DC sweep simulation.

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.dc(Vgate=slice(0, Vdd, .001))

figure, ax = plt.subplots(figsize=(20, 10))

ax.plot(analysis['gatenode'], u_mA(-analysis.Vdrain))
ax.legend('NMOS characteristic')
ax.grid()
ax.set_xlabel('Vgs [V]')
ax.set_ylabel('Id [mA]')

#plt.tight_layout()
plt.show()

#f# save_figure('figure', 'transistor-nmos-plot.png')
