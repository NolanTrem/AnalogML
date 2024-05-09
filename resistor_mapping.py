import numpy as np
import pandas as pd

# Standard E24 resistors
resistor_values = (
    np.array(
        [
            1.0,
            1.1,
            1.2,
            1.3,
            1.5,
            1.6,
            1.8,
            2.0,
            2.2,
            2.4,
            2.7,
            3.0,
            3.3,
            3.6,
            3.9,
            4.3,
            4.7,
            5.1,
            5.6,
            6.2,
            6.8,
            7.5,
            8.2,
            9.1,
        ]
    )
    * 1e3
)
conductance_values = 1 / resistor_values
normalized_conductance_values = (conductance_values - conductance_values.min()) / (
    conductance_values.max() - conductance_values.min()
)


def find_nearest_resistor(value):
    idx = (np.abs(normalized_conductance_values - value)).argmin()
    return resistor_values[idx]


def process_csv_files(csv_files):
    for file in csv_files:
        df = pd.read_csv(file, header=None)
        # Use apply with a lambda function across each element in the DataFrame
        resistor_df = df.apply(lambda row: row.map(find_nearest_resistor), axis=1)
        resistor_df.to_csv(f"{file}_resistor_values", header=None, index=False)
        print(f"Processed {file}.")


csv_files = [
    "layer_0_weights.csv",
    "layer_0_biases.csv",
    "layer_4_weights.csv",
    "layer_4_biases.csv",
]

process_csv_files(csv_files)
