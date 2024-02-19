import PySpice.Logging.Logging as Logging
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Unit import *
from PySpice.Spice.Netlist import Circuit
import os

logger = Logging.setup_logging()

libraries_path = 'ltspice'
spice_library = SpiceLibrary(libraries_path)

circuit = Circuit('Single Neuron 100 Features')

# Define the circuit
circuit.X(1, 'LM741', 'N001', circuit.gnd, '+VCC', '-VCC', 'Vout')
circuit.R(1, 'Vout', 'N001', 1@u_kΩ)
resistor_index = 1
circuit.V(1, 'N002', circuit.gnd, 1@u_V)
voltage_source_index = 1
circuit.V(2, '+VCC', circuit.gnd, 3@u_V)
voltage_source_index += 1
circuit.V(3, circuit.gnd, '-VCC', 3@u_V)
voltage_source_index += 1

# Add n number of resistors and voltage sources
num_resistors = 100
for i in range(num_resistors):
    resistor_index += 1
    voltage_source_index += 1
    circuit.R(f'R{resistor_index}', 'N001', f'N{resistor_index + 1}', 100@u_kΩ)
    circuit.V(f'V{voltage_source_index}', f'N{resistor_index + 1}', circuit.gnd, 1@u_V)

# Include the LM741 model
circuit.include(spice_library['LM741'])

# Create a simulator and run the simulation
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.operating_point()

# Get the voltage at Vout
vout_value = float(analysis['Vout'])
print(f"Voltage at Vout: {vout_value} V")
