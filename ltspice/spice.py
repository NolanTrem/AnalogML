import csv
import PySpice.Logging.Logging as Logging
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Unit import *
from PySpice.Spice.Netlist import Circuit
import os

logger = Logging.setup_logging()

libraries_path = 'ltspice'
spice_library = SpiceLibrary(libraries_path)

# Load resistor values from CSV
def load_resistor_values(row_number):
    with open('resistor_values_layer_2_weights.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == row_number - 1:
                return [float(value) for value in row]
    return []

# Define test cases
test_cases = {
    'all_on': 1,
    'half_on': 0.5,
    'none_on': 0
}

# Specify the range of rows to use
start_row = 1
end_row = 8

# Run simulation for each row in the specified range
for row_number in range(start_row, end_row + 1):
    print(f"Row number: {row_number}")
    resistor_values = load_resistor_values(row_number)

    # Run simulation for each test case
    for test_case, voltage_factor in test_cases.items():
        print(f"  Test case: {test_case}")
        
        # Create a new circuit for each test case
        circuit = Circuit(f'Single Neuron 100 Features, {test_case}')
        
        # Define the circuit
        circuit.X(1, 'LM741', 'N001', circuit.gnd, '+VCC', '-VCC', 'Vout')
        resistor_index = 1
        circuit.V(1, 'N002', circuit.gnd, 1@u_V)
        voltage_source_index = 1
        circuit.V(2, '+VCC', circuit.gnd, 3@u_V)
        voltage_source_index += 1
        circuit.V(3, circuit.gnd, '-VCC', 3@u_V)
        voltage_source_index += 1

        # Add resistors and voltage sources based on CSV values
        reciprocal_sum = 0  # Sum of the reciprocals of resistor values
        for i, resistor_value in enumerate(resistor_values):
            resistor_index += 1
            voltage_source_index += 1
            circuit.R(f'R{resistor_index}', 'N001', f'N{resistor_index + 1}', resistor_value@u_Ω)
            reciprocal_sum += 1 / resistor_value
            voltage = 1 if i < len(resistor_values) * voltage_factor else 0
            circuit.V(f'V{voltage_source_index}', f'N{resistor_index + 1}', circuit.gnd, voltage@u_V)
        
        # Set R1 based on the equivalent resistance of the other resistors
        equivalent_resistance = 1 / reciprocal_sum
        print(f"    Setting feedback resistor to: {equivalent_resistance} Ω")
        circuit.R(1, 'Vout', 'N001', equivalent_resistance@u_Ω)

        # Include the LM741 model
        circuit.include(spice_library['LM741'])

        # Create a simulator and run the simulation
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.operating_point()

        # Get the voltage at Vout
        vout_value = float(analysis['Vout'])
        print(f"    Voltage at Vout: {vout_value} V")
