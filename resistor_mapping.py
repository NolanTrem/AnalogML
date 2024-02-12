import pandas as pd

def weight_to_resistor(value, min_resistor=300, max_resistor=100300):
    """
    Maps a quantized weight value (0 to 1) directly to a resistor value within the specified range,
    rounding to the nearest 100Ω.
    """
    resistor_value = value * (max_resistor - min_resistor) + min_resistor
    return round(resistor_value / 100) * 100

def process_csv_files(csv_files, min_resistor=300, max_resistor=100300):
    """
    Reads the CSV files containing the weights and biases and maps the values to resistor values,
    rounding to the nearest integer. Assumes CSV files do not have header rows.
    """
    for file in csv_files:
        df = pd.read_csv(file, header=None)

        resistor_df = df.applymap(
            lambda x: round(weight_to_resistor(x, min_resistor, max_resistor))
        )
        resistor_df.to_csv(f"resistor_values_{file}", header=None, index=False)

csv_files = [
    "layer_0_weights.csv",
    "layer_0_biases.csv",
    "layer_2_weights.csv",
    "layer_2_biases.csv",
]

process_csv_files(csv_files)
