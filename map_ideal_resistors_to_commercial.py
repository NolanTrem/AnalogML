import pandas as pd

def map_to_commercial_resistor(resistor_value, commercial_resistors):
    """
    Maps a resistor value to the closest commercially available resistor.
    """
    return min(commercial_resistors, key=lambda x: abs(x - resistor_value))

def process_commercial_resistors(csv_files, commercial_resistors=[10, 12, 15, 18, 22, 27, 33, 39, 47, 56]):
    """
    Reads the CSV files containing the resistor values and maps them to the closest commercially available resistors.
    Assumes CSV files do not have header rows.
    """
    for file in csv_files:
        df = pd.read_csv(f"resistor_values_{file}", header=None)

        commercial_resistor_df = df.applymap(
            lambda x: map_to_commercial_resistor(x, commercial_resistors) if x != "NULL" else "NULL"
        )
        commercial_resistor_df.to_csv(f"commercially_available_resistor_values_{file}", header=None, index=False)

csv_files = [
    "resistor_values_layer_0_weights.csv",
]

process_commercial_resistors(csv_files)
