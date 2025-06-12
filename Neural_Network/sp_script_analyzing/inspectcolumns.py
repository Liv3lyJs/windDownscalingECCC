import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a DataFrame
df = pd.read_csv('/fs/site5/eccc/cmd/x/spb001/ramp_detection/data_ramps/Nergica/Wind/Nergica_pour_E_MMV1_refHT.csv')

# Print the columns to check the names
print(df.columns)
